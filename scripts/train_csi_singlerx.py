"""
CSI training for single-RX GSRF models (one model per RX, no RX conditioning).

Usage:
    bash run_csi_singlerx.sh --gpu 0
    python -m scripts.train_csi_singlerx --config arguments/configs/exp_csi_singlerx.yaml --rx_idx 0
"""

import os
import json
from argparse import ArgumentParser

import torch

from arguments import ModelParams, PipelineParams, OptimizationParams, load_config
from utils.general_utils import safe_state
from scene.csi_dataset import load_csi
from scripts.train_csi_multirx import train_geometry


def main():
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="arguments/configs/exp_csi_singlerx.yaml")
    pre_args, _ = pre_parser.parse_known_args()

    yaml_cfg = load_config(pre_args.config)
    random_seed = (yaml_cfg or {}).get("random_seed", 8371)

    parser = ArgumentParser(description="CSI Training (single-RX, one model per RX)")
    parser.add_argument("--config", type=str, default="arguments/configs/exp_csi_singlerx.yaml")

    model_para_cls = ModelParams(parser, yaml_cfg=yaml_cfg)
    optimization_para_cls = OptimizationParams(parser, yaml_cfg=yaml_cfg)
    pipeline_para_cls = PipelineParams(parser, yaml_cfg=yaml_cfg)

    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--rx_idx", type=int, default=None,
                        help="Train only this RX index (default: all RXs)")

    args = parser.parse_args()

    data_dir = os.path.join(args.input_data_folder, args.dataset)
    args.source_path = data_dir
    model_path = os.path.join(args.log_base_folder, args.dataset, args.exp_name)
    os.makedirs(model_path, exist_ok=True)
    args.model_path = model_path

    safe_state(args.quiet, random_seed, torch.device(args.data_device))

    iterations = (yaml_cfg or {}).get("iterations", 30000)
    n_azimuth = getattr(args, 'n_azimuth', 72)
    n_elevation = getattr(args, 'n_elevation', 18)

    per_rx, rx_names, csi_max = load_csi(
        data_dir, ratio_train=getattr(args, 'ratio_train', 0.8), seed=random_seed,
        n_elevation=n_elevation, n_azimuth=n_azimuth,
    )
    rx_indices = [args.rx_idx] if args.rx_idx is not None else sorted(per_rx)

    print(f"\n{'='*60}")
    print("  CSI Training (single-RX, one model per RX)")
    print(f"  Data: {data_dir}")
    print(f"  Output: {model_path}")
    print(f"  RXs: {[rx_names[i] for i in rx_indices]}")
    print(f"  Iterations per RX: {iterations}")
    print(f"{'='*60}\n")

    with open(os.path.join(model_path, "config.json"), 'w') as f:
        config_dict = {k: v for k, v in vars(args).items() if not k.startswith('_')}
        config_dict['rx_names'] = rx_names
        config_dict['iterations'] = iterations
        config_dict['csi_max'] = float(csi_max)
        json.dump(config_dict, f, indent=2, default=str)

    model_args = model_para_cls.extract(args)
    model_args.num_channels_override = 52
    opt_args = optimization_para_cls.extract(args)
    opt_args.densify_until_iter = iterations // 2
    opt_args.position_lr_max_steps = iterations
    pipe_args = pipeline_para_cls.extract(args)

    for rx_idx in rx_indices:
        rx_name = rx_names[rx_idx]
        train_samples, test_samples = per_rx[rx_idx]
        rx_dir = os.path.join(model_path, rx_name)
        os.makedirs(rx_dir, exist_ok=True)
        model_args.model_path = rx_dir

        print(f"\n  [{rx_idx + 1}/{len(rx_names)}] {rx_name}: "
              f"train={len(train_samples)}, test={len(test_samples)}")
        train_geometry(model_args, opt_args, pipe_args, train_samples, rx_dir, iterations,
                       n_azimuth, n_elevation,
                       geo_save_path=os.path.join(rx_dir, f"chkpnt{iterations}.pth"), desc=rx_name)
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"  Training complete. Results: {model_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
