import argparse
import datetime
import os
import os.path as osp
import gorilla
import torch
from tqdm import tqdm

from ours2_xiaorong.dataset import build_dataloader, build_dataset
from ours2_xiaorong.evaluation import ScanNetEval
from ours2_xiaorong.model import OurSPFormer2Ablation
from ours2_xiaorong.utils import get_root_logger, save_gt_instances, save_pred_instances


def get_args():
    parser = argparse.ArgumentParser('OurSPFormer2Ablation')
    parser.add_argument('config', type=str, help='path to config file')
    parser.add_argument('checkpoint', type=str, help='path to checkpoint')
    parser.add_argument('--out', type=str, help='directory for output results')
    parser.add_argument('--no-eval', action='store_true', help='skip evaluation')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    cfg = gorilla.Config.fromfile(args.config)
    gorilla.set_random_seed(cfg.test.seed)
    logger = get_root_logger()

    model = OurSPFormer2Ablation(**cfg.model).cuda()
    logger.info(f'Load state dict from {args.checkpoint}')
    gorilla.load_checkpoint(model, args.checkpoint, strict=False)

    dataset = build_dataset(cfg.data.test, logger)
    dataloader = build_dataloader(dataset, training=False, **cfg.dataloader.test)

    results, scan_ids, pred_insts, gt_insts = [], [], [], []

    progress_bar = tqdm(total=len(dataloader))
    with torch.no_grad():
        model.eval()
        for batch in dataloader:
            result = model(batch, mode='predict')
            results.append(result)
            progress_bar.update()
        progress_bar.close()

    for res in results:
        scan_ids.append(res['scan_id'])
        pred_insts.append(res['pred_instances'])
        gt_insts.append(res['gt_instances'])

    if not args.no_eval:
        logger.info('Evaluate instance segmentation')
        scannet_eval = ScanNetEval(dataset.CLASSES)

        output_dir = args.out if args.out else '.'
        os.makedirs(output_dir, exist_ok=True)
        metrics_file_path = osp.join(output_dir, 'Evaluation_metrics.log')
        with open(metrics_file_path, 'a') as metrics_file:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            metrics_file.write(f'\n{"="*64}\n')
            metrics_file.write(f'Test Evaluation  |  Time: {timestamp}\n')
            metrics_file.write(f'Checkpoint: {args.checkpoint}\n')
            metrics_file.write(f'{"="*64}\n')
            scannet_eval.evaluate(pred_insts, gt_insts, logger, metrics_file)

    if args.out:
        logger.info('Save results')
        nyu_id = dataset.NYU_ID
        save_pred_instances(args.out, 'pred_instance', scan_ids, pred_insts, nyu_id)
        if not cfg.data.test.prefix == 'test':
            save_gt_instances(args.out, 'gt_instance', scan_ids, gt_insts, nyu_id)


if __name__ == '__main__':
    main()
