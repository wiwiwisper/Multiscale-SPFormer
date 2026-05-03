import functools
import gorilla
import pointgroup_ops
import spconv.pytorch as spconv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_max, scatter_mean

from spformer.utils import cuda_cast, rle_encode
from .backbone import ResidualBlock, UBlock
from .loss import Criterion
from .query_decoder import QueryDecoder


@gorilla.MODELS.register_module()
class SPFormer(nn.Module):

    def __init__(
        self,
        input_channel: int = 6,
        blocks: int = 5,
        block_reps: int = 2,
        media: int = 32,
        normalize_before=True,
        return_blocks=True,
        pool='mean',
        num_class=18,
        geo_feat_dim: int = 0,
        coarse_scale: float = 0.0,
        coarse_scale_2: float = 0.0,
        coarse_scales: list = None,
        fusion_mode: str = 'mlp',
        decoder=None,
        criterion=None,
        test_cfg=None,
        norm_eval=False,
        fix_module=[],
    ):
        super().__init__()
        self.geo_feat_dim = geo_feat_dim
        self.fusion_mode = fusion_mode

        if coarse_scales is not None:
            self.coarse_scales = coarse_scales
        elif coarse_scale_2 > 0:
            self.coarse_scales = [coarse_scale, coarse_scale_2]
        elif coarse_scale > 0:
            self.coarse_scales = [coarse_scale]
        else:
            self.coarse_scales = []

        self.num_streams = len(self.coarse_scales) + 1

        # backbone and pooling
        self.input_conv = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channel,
                media,
                kernel_size=3,
                padding=1,
                bias=False,
                indice_key='subm1',
            ))
        block = ResidualBlock
        norm_fn = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)
        block_list = [media * (i + 1) for i in range(blocks)]
        self.unet = UBlock(
            block_list,
            norm_fn,
            block_reps,
            block,
            indice_key_id=1,
            normalize_before=normalize_before,
            return_blocks=return_blocks,
        )
        self.output_layer = spconv.SparseSequential(norm_fn(media), nn.ReLU(inplace=True))
        self.pool = pool
        self.num_class = num_class

        self.fusion_layer = nn.Sequential(
            nn.Linear(media * self.num_streams, media),
            nn.LayerNorm(media),
            nn.ReLU(inplace=True),
        )

        # decoder
        self.decoder = QueryDecoder(**decoder, in_channel=media + geo_feat_dim, num_class=num_class)

        # criterion
        self.criterion = Criterion(**criterion, num_class=num_class)

        self.test_cfg = test_cfg
        self.norm_eval = norm_eval
        for module in fix_module:
            module = getattr(self, module)
            module.eval()
            for param in module.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(SPFormer, self).train(mode)
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm1d):
                    m.eval()

    def forward(self, batch, mode='loss'):
        if mode == 'loss':
            return self.loss(**batch)
        elif mode == 'predict':
            return self.predict(**batch)

    @cuda_cast
    def loss(self, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats,
             coords_float, insts, superpoints, batch_offsets):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        sp_feats, sp_coords = self.extract_feat(input, superpoints, p2v_map, coords_float, batch_offsets)
        out = self.decoder(sp_feats, batch_offsets, sp_xyz=sp_coords)

        loss, loss_dict = self.criterion(out, insts)
        return loss, loss_dict

    @cuda_cast
    def predict(self, scan_ids, voxel_coords, p2v_map, v2p_map, spatial_shape, feats,
                coords_float, insts, superpoints, batch_offsets):
        batch_size = len(batch_offsets) - 1
        voxel_feats = pointgroup_ops.voxelization(feats, v2p_map)
        input = spconv.SparseConvTensor(voxel_feats, voxel_coords.int(), spatial_shape, batch_size)

        sp_feats, sp_coords = self.extract_feat(input, superpoints, p2v_map, coords_float, batch_offsets)
        out = self.decoder(sp_feats, batch_offsets, sp_xyz=sp_coords)

        ret = self.predict_by_feat(scan_ids, out, superpoints, insts)
        return ret

    def predict_by_feat(self, scan_ids, out, superpoints, insts):
        pred_labels = out['labels']
        pred_masks = out['masks']
        pred_scores = out['scores']

        scores = F.softmax(pred_labels[0], dim=-1)[:, :-1]
        scores *= pred_scores[0]
        labels = torch.arange(
            self.num_class, device=scores.device).unsqueeze(0).repeat(self.decoder.num_query, 1).flatten(0, 1)
        scores, topk_idx = scores.flatten(0, 1).topk(self.test_cfg.topk_insts, sorted=False)
        labels = labels[topk_idx]
        labels += 1

        topk_idx = torch.div(topk_idx, self.num_class, rounding_mode='floor')
        mask_pred = pred_masks[0]
        mask_pred = mask_pred[topk_idx]
        mask_pred_sigmoid = mask_pred.sigmoid()
        # mask_pred before sigmoid()
        mask_pred = (mask_pred > 0).float()  # [n_p, M]
        mask_scores = (mask_pred_sigmoid * mask_pred).sum(1) / (mask_pred.sum(1) + 1e-6)
        scores = scores * mask_scores
        # get mask
        mask_pred = mask_pred[:, superpoints].int()

        # score_thr
        score_mask = scores > self.test_cfg.score_thr
        scores = scores[score_mask]  # (n_p,)
        labels = labels[score_mask]  # (n_p,)
        mask_pred = mask_pred[score_mask]  # (n_p, N)

        # npoint thr
        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]  # (n_p,)
        labels = labels[npoint_mask]  # (n_p,)
        mask_pred = mask_pred[npoint_mask]  # (n_p, N)

        cls_pred = labels.cpu().numpy()
        score_pred = scores.cpu().numpy()
        mask_pred = mask_pred.cpu().numpy()

        pred_instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred['scan_id'] = scan_ids[0]
            pred['label_id'] = cls_pred[i]
            pred['conf'] = score_pred[i]
            # rle encode mask to save memory
            pred['pred_mask'] = rle_encode(mask_pred[i])
            pred_instances.append(pred)

        gt_instances = insts[0].gt_instances
        return dict(scan_id=scan_ids[0], pred_instances=pred_instances, gt_instances=gt_instances)

    def create_coarse_sp(self, sp_coords, batch_offsets, scale):
        coarse_sps = torch.empty(sp_coords.size(0), dtype=torch.long, device=sp_coords.device)
        global_offset = 0
        num_batch = len(batch_offsets) - 1

        for batch_idx in range(num_batch):
            start = int(batch_offsets[batch_idx].item())
            end = int(batch_offsets[batch_idx + 1].item())
            coords_batch = sp_coords[start:end]
            grid = (coords_batch / scale).long()
            _, inverse = torch.unique(grid, return_inverse=True, dim=0)
            coarse_sps[start:end] = inverse + global_offset
            global_offset += int(inverse.max().item()) + 1

        return coarse_sps

    def compute_geo_feats(self, coords_float, superpoints, sp_coords):
        centered = coords_float - sp_coords[superpoints]
        outer = centered.unsqueeze(2) * centered.unsqueeze(1)
        cov = scatter_mean(outer.reshape(outer.shape[0], 9), superpoints, dim=0).view(-1, 3, 3)

        eigenvalues, eigenvectors = torch.linalg.eigh(cov)
        lam3, lam2, lam1 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
        lam1 = lam1.clamp(min=1e-6)

        linearity = (lam1 - lam2) / lam1
        planarity = (lam2 - lam3) / lam1
        sphericity = lam3 / lam1
        verticality = torch.abs(eigenvectors[:, 2, 2])

        geo = torch.stack([linearity, planarity, sphericity, verticality], dim=-1)
        return geo[:, :self.geo_feat_dim]

    def extract_feat(self, x, superpoints, v2p_map, coords_float, batch_offsets):
        # backbone
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        x = x.features[v2p_map.long()]  # (B*N, media)

        # superpoint pooling
        if self.pool == 'mean':
            sp_feats = scatter_mean(x, superpoints, dim=0)  # (B*M, media)
        elif self.pool == 'max':
            sp_feats, _ = scatter_max(x, superpoints, dim=0)  # (B*M, media)
        else:
            raise ValueError(f'Unsupported pool mode: {self.pool}')

        sp_coords = scatter_mean(coords_float, superpoints, dim=0)
        all_feats = [sp_feats]
        for scale in self.coarse_scales:
            coarse_sps = self.create_coarse_sp(sp_coords, batch_offsets, scale)
            coarse_feats = scatter_mean(sp_feats, coarse_sps, dim=0)
            all_feats.append(coarse_feats[coarse_sps])

        if len(all_feats) > 1:
            if len(all_feats) > 2 or self.fusion_mode == 'mlp':
                sp_feats = self.fusion_layer(torch.cat(all_feats, dim=-1))
            elif self.fusion_mode == 'add':
                sp_feats = all_feats[0] + all_feats[1]
            else:
                raise ValueError(f'Unsupported fusion mode: {self.fusion_mode}')

        if self.geo_feat_dim > 0:
            geo = self.compute_geo_feats(coords_float, superpoints, sp_coords)
            sp_feats = torch.cat([sp_feats, geo], dim=-1)

        return sp_feats, sp_coords
