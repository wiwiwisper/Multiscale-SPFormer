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
class OurSPFormer2(nn.Module):

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
        coarse_scale: float = 0.5,  # coarse-grid cell size (same unit as coords_float)
        fusion_mode: str = 'mlp',   # 'mlp': cat+Linear+LN+ReLU  |  'add': element-wise add
        decoder=None,
        criterion=None,
        test_cfg=None,
        norm_eval=False,
        fix_module=[],
    ):
        super().__init__()

        self.geo_feat_dim = geo_feat_dim
        self.coarse_scale = coarse_scale
        self.fusion_mode = fusion_mode

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

        # dual-scale fusion: cat(fine, coarse) -> media
        self.fusion_layer = nn.Sequential(
            nn.Linear(media * 2, media),
            nn.LayerNorm(media),
            nn.ReLU(inplace=True),
        )

        # decoder — in_channel expands by geo_feat_dim (same as single-scale)
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
        super(OurSPFormer2, self).train(mode)
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
        mask_pred = (mask_pred > 0).float()  # [n_p, M]
        mask_scores = (mask_pred_sigmoid * mask_pred).sum(1) / (mask_pred.sum(1) + 1e-6)
        scores = scores * mask_scores
        mask_pred = mask_pred[:, superpoints].int()

        score_mask = scores > self.test_cfg.score_thr
        scores = scores[score_mask]
        labels = labels[score_mask]
        mask_pred = mask_pred[score_mask]

        mask_pointnum = mask_pred.sum(1)
        npoint_mask = mask_pointnum > self.test_cfg.npoint_thr
        scores = scores[npoint_mask]
        labels = labels[npoint_mask]
        mask_pred = mask_pred[npoint_mask]

        cls_pred = labels.cpu().numpy()
        score_pred = scores.cpu().numpy()
        mask_pred = mask_pred.cpu().numpy()

        pred_instances = []
        for i in range(cls_pred.shape[0]):
            pred = {}
            pred['scan_id'] = scan_ids[0]
            pred['label_id'] = cls_pred[i]
            pred['conf'] = score_pred[i]
            pred['pred_mask'] = rle_encode(mask_pred[i])
            pred_instances.append(pred)

        gt_instances = insts[0].gt_instances
        return dict(scan_id=scan_ids[0], pred_instances=pred_instances, gt_instances=gt_instances)

    def create_coarse_sp(self, sp_coords, batch_offsets):
        """Assign fine SPs to coarse SPs via spatial grid hashing.

        For each batch item, discretises fine-SP centroids into a coarse
        voxel grid (cell size = self.coarse_scale), then reassigns each
        fine SP to a unique coarse SP index.

        Args:
            sp_coords:     (B*M_fine, 3)  fine SP centroids
            batch_offsets: (B+1,)         cumulative fine-SP counts

        Returns:
            coarse_sps: (B*M_fine,)  coarse SP index per fine SP (global)
        """
        B = len(batch_offsets) - 1
        coarse_sps = torch.zeros(sp_coords.shape[0], dtype=torch.long, device=sp_coords.device)
        global_offset = 0

        for b in range(B):
            start = int(batch_offsets[b].item())
            end   = int(batch_offsets[b + 1].item())
            coords_b = sp_coords[start:end]                          # (M_fine_b, 3)

            # discretise to coarse grid
            grid = (coords_b / self.coarse_scale).long()             # (M_fine_b, 3)
            _, inverse = torch.unique(grid, return_inverse=True, dim=0)
            # inverse: (M_fine_b,)  local coarse-SP index 0..M_coarse_b-1

            coarse_sps[start:end] = inverse + global_offset
            global_offset += int(inverse.max().item()) + 1

        return coarse_sps  # (B*M_fine,)

    def compute_geo_feats(self, coords_float, superpoints, sp_coords):
        """Compute per-superpoint geometric features.

        Features (in order):
            0: linearity   = (λ1-λ2)/λ1  → high for stem-like cylinders
            1: planarity   = (λ2-λ3)/λ1  → high for leaf-like flat patches
            2: sphericity  = λ3/λ1        → high for ball-like clusters
            3: verticality = |v1 · z|     → high when principal axis is vertical (stem)

        Args:
            coords_float: (B*N, 3) raw xyz coordinates
            superpoints:  (B*N,)  superpoint index per point
            sp_coords:    (B*M, 3) pre-computed superpoint centroids

        Returns:
            geo: (B*M, geo_feat_dim)
        """
        centered = coords_float - sp_coords[superpoints]               # (B*N, 3)
        outer = centered.unsqueeze(2) * centered.unsqueeze(1)          # (B*N, 3, 3)
        cov = scatter_mean(
            outer.reshape(outer.shape[0], 9), superpoints, dim=0
        ).view(-1, 3, 3)                                               # (B*M, 3, 3)

        eigenvalues, eigenvectors = torch.linalg.eigh(cov)             # ascending order
        lam3, lam2, lam1 = eigenvalues[:, 0], eigenvalues[:, 1], eigenvalues[:, 2]
        lam1 = lam1.clamp(min=1e-6)

        linearity   = (lam1 - lam2) / lam1
        planarity   = (lam2 - lam3) / lam1
        sphericity  = lam3 / lam1

        v1          = eigenvectors[:, :, 2]                            # (B*M, 3)
        verticality = torch.abs(v1[:, 2])

        geo = torch.stack([linearity, planarity, sphericity, verticality], dim=-1)  # (B*M, 4)
        return geo[:, :self.geo_feat_dim]

    def extract_feat(self, x, superpoints, v2p_map, coords_float, batch_offsets):
        # backbone
        x = self.input_conv(x)
        x, _ = self.unet(x)
        x = self.output_layer(x)
        x = x.features[v2p_map.long()]          # (B*N, media)

        # fine-scale superpoint pooling
        if self.pool == 'mean':
            sp_feats = scatter_mean(x, superpoints, dim=0)   # (B*M_fine, media)
        elif self.pool == 'max':
            sp_feats, _ = scatter_max(x, superpoints, dim=0)

        # fine-scale centroids
        sp_coords = scatter_mean(coords_float, superpoints, dim=0)  # (B*M_fine, 3)

        # coarse-scale pooling and fusion
        # 1. assign fine SPs to coarse SPs via spatial grid
        coarse_sps = self.create_coarse_sp(sp_coords, batch_offsets)  # (B*M_fine,)
        # 2. pool fine features to coarse SPs
        coarse_feats = scatter_mean(sp_feats, coarse_sps, dim=0)      # (B*M_coarse, media)
        # 3. broadcast coarse features back to fine-SP resolution
        coarse_to_fine = coarse_feats[coarse_sps]                     # (B*M_fine, media)
        # 4. fuse fine + coarse
        if self.fusion_mode == 'mlp':
            sp_feats = self.fusion_layer(
                torch.cat([sp_feats, coarse_to_fine], dim=-1)
            )
        else:  # 'add'
            sp_feats = sp_feats + coarse_to_fine                      # (B*M_fine, media)

        # geometric feature augmentation (on fine SPs)
        if self.geo_feat_dim > 0:
            geo = self.compute_geo_feats(coords_float, superpoints, sp_coords)
            sp_feats = torch.cat([sp_feats, geo], dim=-1)             # (B*M_fine, media+geo_feat_dim)

        return sp_feats, sp_coords
