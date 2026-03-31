# spformer/dataset/myplants.py
import os.path as osp
import numpy as np
import torch
import torch_scatter

from .scannetv2 import ScanNetDataset
from ..utils import Instances3D


class MyPlantsDataset(ScanNetDataset):
    """Two-class instance segmentation dataset: stem / leaf.

    Compatible with .pth (recommended) and .npz produced by your preprocess.
    Follows ScanNetV2 dataset flow so transforms/collate/evaluator work unchanged.
    """
    CLASSES = ['stem', 'leaf']
    VALID_CLASS_IDS = [0, 1]
    NYU_ID = [0, 1]
    IGNORE_INDEX = 255

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Your labels are 0/1 ideally; keep an identity mapping as a safety net.
        self.label_mapping = torch.full((4096,), self.IGNORE_INDEX, dtype=torch.long)
        self.label_mapping[0] = 0  # stem
        self.label_mapping[1] = 1  # leaf

    # -------------------------- IO --------------------------
    def load(self, filename):
        """Load one sample and return:
           xyz(float32), rgb(float32), superpoint(int), semantic_label(int), instance_label(int)

        This matches what ScanNetDataset.transform_{train,test} expect.
        Supports both .pth and .npz produced by your preprocess code.
        """
        def to_np(x, dtype):
            if torch.is_tensor(x):
                x = x.detach().cpu().numpy()
            return x.astype(dtype)

        # Preferred path: .pth produced by your preprocess
        if filename.endswith(".pth"):
            data = torch.load(filename)
            if self.with_label:
                # Official convention: (xyz, rgb, sp_index, semantic, instance)
                xyz, rgb, superpoint, semantic_label, instance_label = data
            else:
                xyz, rgb, superpoint = data
                n = (xyz.shape[0] if not torch.is_tensor(xyz)
                     else int(xyz.shape[0]))
                semantic_label = np.zeros(n, dtype=np.int32)
                instance_label = np.zeros(n, dtype=np.int32)

            xyz = to_np(xyz, np.float32)
            rgb = to_np(rgb, np.float32)
            superpoint = to_np(superpoint, np.int32)
            semantic_label = to_np(semantic_label, np.int32)
            instance_label = to_np(instance_label, np.int32)
            return xyz, rgb, superpoint, semantic_label, instance_label

        # Fallback: .npz (if you later choose to export npz)
        data = np.load(filename)
        xyz = (data['coord'] if 'coord' in data else data['xyz']).astype(np.float32)
        rgb = (data['color'] if 'color' in data else data['rgb']).astype(np.float32)
        superpoint = data['sp_index'].astype(np.int32)
        semantic_label = data['label'].astype(np.int32)
        instance_label = data['instance'].astype(np.int32)
        return xyz, rgb, superpoint, semantic_label, instance_label

    # ---------------------- Instances (GT) ----------------------
    def get_instance3D(self, instance_label, semantic_label, superpoint, scan_id):
        """Aggregate point-level semantic/instance into instance-level GT.

        Returns an Instances3D with:
          - gt_labels: [M] long, class id per instance (0..C-1)
          - gt_spmasks: [M, #superpoints] float {0,1}
          - gt_instances: [N] encoded per-point instance (for viz/eval): (cls+1)*1000 + inst+1
        """
        num_points = int(instance_label.shape[0])
        max_inst = int(instance_label.max().item()) if instance_label.numel() > 0 else -1
        num_insts = max_inst + 1

        gt_masks, gt_labels = [], []
        gt_inst = torch.zeros(num_points, dtype=torch.int64)

        for i in range(num_insts):
            idx = torch.nonzero(instance_label == i, as_tuple=False).view(-1)
            if idx.numel() == 0:
                continue

            # Majority vote on semantic labels, ignoring IGNORE_INDEX
            lab = semantic_label[idx]
            lab = lab[lab != self.IGNORE_INDEX]
            if lab.numel() == 0:
                continue
            vals, cnts = torch.unique(lab, return_counts=True)
            sem_id = int(vals[torch.argmax(cnts)].item())  # 0 or 1

            mask = torch.zeros(num_points)
            mask[idx] = 1
            gt_masks.append(mask)
            gt_labels.append(sem_id)

            # For visualization/compat with evaluator
            gt_inst[idx] = (sem_id + 1) * 1000 + i + 1

        if gt_masks:
            gt_masks = torch.stack(gt_masks, dim=0)               # [M, N]
            # superpoint must be long for scatter index
            gt_spmasks = torch_scatter.scatter_mean(
                gt_masks.float(), superpoint.long(), dim=-1
            )
            gt_spmasks = (gt_spmasks > 0.5).float()
        else:
            gt_spmasks = torch.tensor([])

        gt_labels = torch.tensor(gt_labels, dtype=torch.int64)

        inst = Instances3D(num_points, gt_instances=gt_inst.numpy())
        inst.gt_labels = gt_labels
        inst.gt_spmasks = gt_spmasks
        return inst

    # ----------------------- __getitem__ -----------------------
    def __getitem__(self, index: int):
        filename = self.filenames[index]
        scan_id = osp.basename(filename).replace(self.suffix, '')

        # Load numpy arrays
        data = self.load(filename)
        # Apply ScanNet-style transforms (crop/elastic/voxelization/…)
        data = self.transform_train(*data) if self.training else self.transform_test(*data)
        xyz, xyz_middle, rgb, superpoint, semantic_label, instance_label = data

        # Pack tensors for model/evaluator
        coord = torch.from_numpy(xyz).long()
        coord_float = torch.from_numpy(xyz_middle).float()
        feat = torch.from_numpy(rgb).float()
        superpoint = torch.from_numpy(superpoint).long()          # long index for scatter
        semantic_label = torch.from_numpy(semantic_label).long()
        instance_label = torch.from_numpy(instance_label).long()

        # ---- Normalize semantics to {0,1} if your data is {1,2} ----
        if semantic_label.numel() > 0 and int(semantic_label.min().item()) >= 1:
            semantic_label = semantic_label - 1

        # Optional safety: apply label_mapping (keeps 0->0, 1->1; others -> IGNORE)
        ok = (semantic_label >= 0) & (semantic_label < self.label_mapping.numel())
        mapped = torch.full_like(semantic_label, self.IGNORE_INDEX)
        mapped[ok] = self.label_mapping[semantic_label[ok]]
        semantic_label = mapped

        # Build instance-level GT for evaluator (ScanNet-style)
        inst = self.get_instance3D(instance_label, semantic_label, superpoint, scan_id)

        return scan_id, coord, coord_float, feat, superpoint, inst
