"""
BLE RSSI training for single-RX GSRF models (one model per gateway, no RX conditioning).

The first gateway is trained in full (geometry + FLE). Every other gateway reuses that
gateway's frozen geometry and trains its own FLE coefficients.

Usage:
    bash run_ble_singlerx.sh --gpu 0
    python -m scripts.train_ble_singlerx --config arguments/configs/exp_ble_singlerx.yaml
"""

import os
import json
import time
from random import randint
from argparse import ArgumentParser

import torch
from torch import nn
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams, load_config
from utils.general_utils import safe_state
from scene import GaussianModel
from gaussian_renderer import render_rfid
from scene.ble_dataset import load_ble_per_gateway
from scripts.train_ble_multirx import train_geometry


class EpochSampler:
    def __init__(self, n):
        self.n = n
        self.stack = []

    def __call__(self):
        if not self.stack:
            self.stack = list(range(self.n))
        return self.stack.pop(randint(0, len(self.stack) - 1))


# ---- FLE-only training on a reference geometry ----

def train_fle_only(ref_gaussians, model_args, opt_args, pipe_args, train_samples, iterations, save_path,
                   desc="FLE only (shared geometry)"):
    """Train fresh FLE coefficients on the reference gateway's frozen geometry."""

    gaussians = GaussianModel(model_args)
    for name in ('_xyz', '_scaling', '_rotation', '_attenuation'):
        setattr(gaussians, name, nn.Parameter(getattr(ref_gaussians, name).detach().clone(),
                                              requires_grad=False))
    gaussians.spatial_lr_scale = ref_gaussians.spatial_lr_scale

    n_points = gaussians._xyz.shape[0]
    features = torch.randn(n_points, gaussians.num_channels, (gaussians.max_fle_degree + 1) ** 2,
                           device="cuda") * getattr(model_args, '_fle_init_scale', 0.1)
    gaussians._features_dc = nn.Parameter(features[:, :, 0:1].transpose(1, 2).contiguous())
    gaussians._features_rest = nn.Parameter(features[:, :, 1:].transpose(1, 2).contiguous())
    gaussians.max_radii2D = torch.zeros((n_points,), device="cuda")
    gaussians.xyz_gradient_accum = torch.zeros((n_points, 1), device="cuda")
    gaussians.denom = torch.zeros((n_points, 1), device="cuda")
    gaussians.active_fle_degree = 0

    gaussians.optimizer = torch.optim.Adam([
        {'params': [gaussians._features_dc], 'lr': opt_args.feature_lr, "name": "f_dc"},
        {'params': [gaussians._features_rest],
         'lr': opt_args.feature_lr * getattr(opt_args, '_rest_lr_ratio', 1.0), "name": "f_rest"},
    ], lr=0.0, eps=1e-15)

    fle_ramp = getattr(model_args, '_fle_degree_ramp', 500)
    ema_loss = 0.0
    t_start = time.time()
    sampler = EpochSampler(len(train_samples))
    progress_bar = tqdm(range(iterations), desc=desc)

    for iteration in range(1, iterations + 1):
        if iteration % fle_ramp == 0:
            gaussians.oneup_fle_degree()

        viewpoint = train_samples[sampler()]
        gaussians.optimizer.zero_grad(set_to_none=True)

        rendered = render_rfid(viewpoint, gaussians, pipe_args)["render"]
        loss = (rendered.mean() - viewpoint.spectrum.cuda().mean()) ** 2
        loss.backward()
        gaussians.optimizer.step()

        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss
            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss:.6f}"})
                progress_bar.update(10)
            if iteration == iterations:
                progress_bar.close()

    elapsed = time.time() - t_start
    torch.save({
        'gaussians': gaussians.capture(),
        'iteration': iterations,
        'training_time_s': elapsed,
    }, save_path)
    print(f"\n  FLE trained: time={elapsed:.1f}s ({elapsed/60:.1f}min), saved {save_path}")

    return gaussians


# ---- main ----

def main():
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="arguments/configs/exp_ble_singlerx.yaml")
    pre_args, _ = pre_parser.parse_known_args()

    yaml_cfg = load_config(pre_args.config)
    random_seed = (yaml_cfg or {}).get("random_seed", 8371)

    parser = ArgumentParser(description="BLE RSSI Training (single-RX, one model per gateway)")
    parser.add_argument("--config", type=str, default="arguments/configs/exp_ble_singlerx.yaml")

    model_para_cls = ModelParams(parser, yaml_cfg=yaml_cfg)
    optimization_para_cls = OptimizationParams(parser, yaml_cfg=yaml_cfg)
    pipeline_para_cls = PipelineParams(parser, yaml_cfg=yaml_cfg)

    parser.add_argument("--quiet", action="store_true", default=False)

    args = parser.parse_args()

    data_dir = os.path.join(args.input_data_folder, args.dataset)
    args.source_path = data_dir
    model_path = os.path.join(args.log_base_folder, args.dataset, args.exp_name)
    os.makedirs(model_path, exist_ok=True)
    args.model_path = model_path

    safe_state(args.quiet, random_seed, torch.device(args.data_device))

    iterations = (yaml_cfg or {}).get("iterations", 10000)

    per_gw, gw_names = load_ble_per_gateway(
        data_dir, ratio_train=getattr(args, 'ratio_train', 0.8), seed=random_seed,
        n_elevation=getattr(args, 'n_elevation', 9), n_azimuth=getattr(args, 'n_azimuth', 36),
    )
    gw_indices = [i for i in sorted(per_gw)
                  if len(per_gw[i][0]) >= 10 and len(per_gw[i][1]) >= 5]

    print(f"\n{'='*60}")
    print("  BLE RSSI Training (single-RX, one model per gateway)")
    print(f"  Data: {data_dir}")
    print(f"  Output: {model_path}")
    print(f"  Gateways: {[gw_names[i] for i in gw_indices]}")
    print(f"  Iterations per gateway: {iterations}")
    print(f"{'='*60}\n")

    with open(os.path.join(model_path, "config.json"), 'w') as f:
        config_dict = {k: v for k, v in vars(args).items() if not k.startswith('_')}
        config_dict['gateway_names'] = gw_names
        config_dict['iterations'] = iterations
        json.dump(config_dict, f, indent=2, default=str)

    model_args = model_para_cls.extract(args)
    opt_args = optimization_para_cls.extract(args)
    opt_args.densify_until_iter = iterations // 2
    opt_args.position_lr_max_steps = iterations
    pipe_args = pipeline_para_cls.extract(args)

    ref_gaussians = None
    for gw_idx in gw_indices:
        gw_name = gw_names[gw_idx]
        train_samples, test_samples = per_gw[gw_idx]
        gw_dir = os.path.join(model_path, gw_name)
        os.makedirs(gw_dir, exist_ok=True)
        model_args.model_path = gw_dir
        ckpt_path = os.path.join(gw_dir, f"chkpnt{iterations}.pth")

        mode = "full" if ref_gaussians is None else "FLE only"
        print(f"\n  [{gw_idx + 1}/{len(gw_names)}] {gw_name} [{mode}]: "
              f"train={len(train_samples)}, test={len(test_samples)}")

        if ref_gaussians is None:
            ref_gaussians, _ = train_geometry(model_args, opt_args, pipe_args, train_samples,
                                              gw_dir, iterations, geo_save_path=ckpt_path,
                                              desc=f"{gw_name} (full)",
                                              sampler=EpochSampler(len(train_samples)))
        else:
            train_fle_only(ref_gaussians, model_args, opt_args, pipe_args, train_samples,
                           iterations, ckpt_path, desc=f"{gw_name} (FLE only)")
        torch.cuda.empty_cache()

    print(f"\n{'='*60}")
    print(f"  Training complete. Results: {model_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
