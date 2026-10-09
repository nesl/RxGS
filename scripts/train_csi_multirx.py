"""
CSI Training for distributed multi-RX (KU Leuven dataset).

Two-phase training:
  Phase 1: geometry from averaged CSI across all RXs
  Phase 2: FLE + RX conditioning (frozen geometry)

Usage:
    bash run_csi_multirx.sh --gpu 0
    python -m scripts.train_csi_multirx --config arguments/configs/exp_csi_multirx_main.yaml
"""

import os
import json
import time
from random import randint
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import yaml
import torch
from tqdm import tqdm

from arguments import ModelParams, PipelineParams, OptimizationParams, load_config
from utils.general_utils import safe_state, prepare_log_dir
from scene import Scene, GaussianModel
from gaussian_renderer.render_csi import render_csi, render_csi_rxcond
from scene.csi_dataset import load_csi
from scene.dataset_readers import SpectrumInfo
from utils.rx_conditioning import RxConditionedFLE_PropAware


# ---- Phase 1: geometry training with averaged CSI ----

def train_geometry(
    model_para_args,
    opt_args,
    pipe_args,
    train_samples,
    output_dir,
    geo_iters,
    n_azimuth,
    n_elevation,
    geo_save_path=None,
    desc="Phase 1: Geometry",
):
    """Train Gaussian geometry using averaged CSI across all RXs."""

    gaussians = GaussianModel(model_para_args)
    scene = Scene(model_para_args, gaussians)
    gaussians.training_setup(opt_args)

    ema_loss = 0.0
    t_start = time.time()
    progress_bar = tqdm(range(geo_iters), desc=desc)

    for iteration in range(1, geo_iters + 1):
        gaussians.update_learning_rate(iteration)
        fle_ramp = getattr(model_para_args, '_fle_degree_ramp', 200)
        if iteration % fle_ramp == 0:
            gaussians.oneup_fle_degree()

        viewpoint = train_samples[randint(0, len(train_samples) - 1)]
        gaussians.optimizer.zero_grad(set_to_none=True)

        render_pkg = render_csi(viewpoint, gaussians, pipe_args,
                                n_azimuth=n_azimuth, n_elevation=n_elevation)
        rendered = render_pkg["render"]  # (52, H, W)
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        pred_csi = rendered.mean(dim=(1, 2))  # (52,)
        gt_csi = viewpoint.spectrum.cuda().mean(dim=(1, 2))  # (52,)

        loss = torch.nn.functional.mse_loss(pred_csi, gt_csi)
        loss.backward()

        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss:.6f}"})
                progress_bar.update(10)
            if iteration == geo_iters:
                progress_bar.close()

            gaussians.optimizer.step()

            densify_until = getattr(opt_args, 'densify_until_iter', geo_iters // 2)
            if iteration < densify_until:
                gaussians.max_radii2D[visibility_filter] = torch.max(
                    gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(gaussians.get_xyz, visibility_filter)

                if iteration >= opt_args.densify_from_iter \
                        and iteration % opt_args.densification_interval == 0:
                    size_threshold = opt_args.raddi_size_threshold \
                        if iteration > opt_args.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt_args.densify_grad_threshold,
                                                opt_args.min_attenuation_threshold,
                                                scene.cameras_extent, size_threshold)

                if iteration % opt_args.opacity_reset_interval == 0:
                    gaussians.reset_attenuation()

    geo_time = time.time() - t_start
    geo_ckpt = geo_save_path or os.path.join(output_dir, "geometry.pth")
    os.makedirs(os.path.dirname(geo_ckpt), exist_ok=True)
    torch.save({
        'gaussians': gaussians.capture(),
        'iteration': geo_iters,
        'training_time_s': geo_time,
    }, geo_ckpt)
    print(f"\n  Geometry trained: #G={gaussians.get_xyz.shape[0]}, "
          f"time={geo_time:.1f}s, saved {geo_ckpt}")

    return gaussians, scene


# ---- Phase 2: FLE + RX conditioning ----

def train_rxcond(
    gaussians,
    scene,
    model_para_args,
    opt_args,
    pipe_args,
    train_samples,
    output_dir,
    rx_cond_args,
    film_iters,
    n_azimuth,
    n_elevation,
    resume_ckpt=None,
):
    """Freeze geometry, train FLE + RX conditioning across all RXs."""

    device = torch.device(model_para_args.data_device)

    fle_degree = model_para_args.fle_degree
    n_fle = (fle_degree + 1) ** 2
    rx_cond = RxConditionedFLE_PropAware(
        n_fle_components=n_fle,
        max_fle_degree=fle_degree,
        d_model=rx_cond_args['d_model'],
        n_freqs=rx_cond_args['n_freqs'],
        max_freq_log2=rx_cond_args.get('max_freq_log2', 5.0),
        comp_dim=rx_cond_args.get('comp_dim', 16),
        n_probe_samples=rx_cond_args.get('n_probe_samples', 16),
        num_channels=52,  # 26 subcarriers × 2 (real + imag)
    ).cuda()
    print("  RX conditioning: PropAware (global + local visibility-aware)")

    rx_cond.build_occupancy_grid(
        gaussians.get_xyz.detach(),
        gaussians.get_attenuation.detach(),
        gaussians._scaling.detach(),
        resolution=rx_cond_args.get('grid_resolution', 128),
    )

    gaussians._xyz.requires_grad_(False)
    gaussians._scaling.requires_grad_(False)
    gaussians._rotation.requires_grad_(False)
    gaussians._attenuation.requires_grad_(False)

    start_iter = 0
    if resume_ckpt:
        ckpt_data = torch.load(resume_ckpt, map_location=device)
        gaussian_params = ckpt_data['gaussians']
        gaussians._features_dc = gaussian_params[1]
        gaussians._features_rest = gaussian_params[2]
        rx_cond.load_state_dict(ckpt_data['rx_cond'])
        start_iter = ckpt_data.get('iteration', 0)
        gaussians.active_fle_degree = gaussians.max_fle_degree
        print(f"  Resumed from {resume_ckpt} (iter {start_iter})")
    else:
        gaussians.active_fle_degree = 0

    fle_lr = getattr(opt_args, 'feature_lr', 0.005)
    fle_params = [
        {'params': [gaussians._features_dc], 'lr': fle_lr, "name": "f_dc"},
        {'params': [gaussians._features_rest], 'lr': fle_lr * getattr(opt_args, '_rest_lr_ratio', 0.2), "name": "f_rest"},
    ]
    fle_optimizer = torch.optim.Adam(fle_params, lr=0.0, eps=1e-15)
    rx_cond_lr = rx_cond_args.get('lr', 1e-3)
    rx_cond_optimizer = torch.optim.Adam(rx_cond.parameters(), lr=rx_cond_lr, eps=1e-15)

    ema_loss = 0.0
    t_start = time.time()
    progress_bar = tqdm(initial=start_iter, total=film_iters, desc="Phase 2: FLE+RxCond")

    for iteration in range(start_iter + 1, film_iters + 1):
        fle_ramp = getattr(model_para_args, '_fle_degree_ramp', 200)
        if iteration % fle_ramp == 0:
            gaussians.oneup_fle_degree()

        viewpoint = train_samples[randint(0, len(train_samples) - 1)]

        fle_optimizer.zero_grad(set_to_none=True)
        rx_cond_optimizer.zero_grad(set_to_none=True)

        render_pkg = render_csi_rxcond(viewpoint, gaussians, pipe_args, rx_cond,
                                        n_azimuth=n_azimuth, n_elevation=n_elevation)
        rendered = render_pkg["render"]

        pred_csi = rendered.mean(dim=(1, 2))  # (52,)
        gt_csi = viewpoint.spectrum.cuda().mean(dim=(1, 2))  # (52,)

        loss = torch.nn.functional.mse_loss(pred_csi, gt_csi)
        loss.backward()

        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss:.6f}"})
                progress_bar.update(10)
            if iteration == film_iters:
                progress_bar.close()

            # save halfway and final checkpoints
            if iteration == film_iters // 2 or iteration == film_iters:
                elapsed = time.time() - t_start
                scene.save(iteration)
                ckpt_path = os.path.join(output_dir, f"chkpnt{iteration}.pth")
                torch.save({
                    'gaussians': gaussians.capture(),
                    'rx_cond': rx_cond.state_dict(),
                    'iteration': iteration,
                    'training_time_s': elapsed,
                }, ckpt_path)
                tag = "final" if iteration == film_iters else "halfway"
                print(f"\n  [iter {iteration}] Saved {tag} checkpoint "
                      f"(#G: {gaussians.get_xyz.shape[0]}, time: {elapsed:.1f}s)")

            fle_optimizer.step()
            rx_cond_optimizer.step()

    return gaussians, rx_cond


# ---- main ----

def main():
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="arguments/configs/exp_csi_multirx_main.yaml")
    pre_args, _ = pre_parser.parse_known_args()

    yaml_cfg = load_config(pre_args.config)
    random_seed = (yaml_cfg or {}).get("random_seed", 8371)

    parser = ArgumentParser(description="CSI Training (distributed multi-RX)")
    parser.add_argument("--config", type=str, default="arguments/configs/exp_csi_multirx_main.yaml")

    model_para_cls = ModelParams(parser, yaml_cfg=yaml_cfg)
    optimization_para_cls = OptimizationParams(parser, yaml_cfg=yaml_cfg)
    pipeline_para_cls = PipelineParams(parser, yaml_cfg=yaml_cfg)

    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--retrain_geometry", action="store_true", default=False)
    parser.add_argument("--resume", action="store_true", default=False)

    args = parser.parse_args()

    dataset_name = args.dataset
    exp_name = args.exp_name
    data_dir = os.path.join(args.input_data_folder, dataset_name)
    args.source_path = data_dir

    model_path = os.path.join(args.log_base_folder, dataset_name, exp_name)
    retrain_geometry = args.retrain_geometry
    resume = args.resume
    if not resume:
        prepare_log_dir(model_path)
    else:
        os.makedirs(model_path, exist_ok=True)
    args.model_path = model_path

    safe_state(args.quiet, random_seed, torch.device(args.data_device))

    geo_iters = (yaml_cfg or {}).get("geo_iterations", 30000)
    film_iters = (yaml_cfg or {}).get("film_iterations", 60000)
    n_azimuth = getattr(args, 'n_azimuth', 36)
    n_elevation = getattr(args, 'n_elevation', 9)


    ratio_train = getattr(args, 'ratio_train', 0.8)

    per_rx_data, rx_names, csi_max = load_csi(
        data_dir, ratio_train=ratio_train, seed=random_seed,
        n_elevation=n_elevation, n_azimuth=n_azimuth,
    )

    with open(os.path.join(data_dir, 'rx_positions.yml')) as f:
        rx_cfg = yaml.safe_load(f)
    rx_positions = np.array([rx_cfg[name] for name in rx_names], dtype=np.float32)

    train_samples, test_samples = [], []
    for rx_idx in sorted(per_rx_data.keys()):
        rx_train, rx_test = per_rx_data[rx_idx]
        train_samples.extend(rx_train)
        test_samples.extend(rx_test)

    # Phase 1 samples: per-TX CSI averaged across all RXs (RX-independent geometry).
    csi_raw = np.load(os.path.join(data_dir, 'csidata.npy'))  # (N, 8, 26) complex
    csi_norm = csi_raw / csi_max
    csi_avg = csi_norm.mean(axis=1)  # (N, 26)
    avg_re = csi_avg.real.astype(np.float32)
    avg_im = csi_avg.imag.astype(np.float32)

    tx_pos_all = pd.read_csv(os.path.join(data_dir, 'tx_pos.csv')).values.astype(np.float32)

    # render_csi ignores T_rx; rx01 position is just a placeholder.
    rx_placeholder = torch.tensor(rx_positions[0], dtype=torch.float32)

    train_indices = set(np.loadtxt(os.path.join(data_dir, 'train_index.txt'), dtype=int).tolist())

    geo_train = []
    for idx in range(len(tx_pos_all)):
        if idx not in train_indices:
            continue
        interleaved = np.empty(52, dtype=np.float32)
        interleaved[0::2] = avg_re[idx]
        interleaved[1::2] = avg_im[idx]
        spectrum = torch.from_numpy(interleaved).reshape(52, 1, 1)

        info = SpectrumInfo(
            T_rx=rx_placeholder,
            T_tx=torch.tensor(tx_pos_all[idx], dtype=torch.float32),
            spectrum=spectrum,
            spectrum_name=f"{idx + 1:05d}",
            height=n_elevation,
            width=n_azimuth,
        )
        geo_train.append(info)

    print(f"  Phase 1 geometry samples (averaged across {len(rx_names)} RXs): {len(geo_train)}")

    rx_cond_cfg = (yaml_cfg or {}).get("rx_conditioning", {})
    rx_cond_args = {
        'd_model': rx_cond_cfg.get('d_model', 64),
        'n_freqs': rx_cond_cfg.get('n_freqs', 6),
        'max_freq_log2': rx_cond_cfg.get('max_freq_log2', 5.0),
        'comp_dim': rx_cond_cfg.get('comp_dim', 16),
        'n_probe_samples': rx_cond_cfg.get('n_probe_samples', 16),
        'grid_resolution': rx_cond_cfg.get('grid_resolution', 128),
        'lr': rx_cond_cfg.get('lr', 1e-3),
    }

    print(f"\n{'='*60}")
    print("  CSI Training (distributed multi-RX)")
    print(f"  Data: {data_dir}")
    print(f"  Output: {model_path}")
    print(f"  RXs: {len(rx_names)}")
    print(f"  Train samples: {len(train_samples)} (all RXs)")
    print(f"  Phase 1 (geometry, averaged {len(rx_names)} RXs): {geo_iters} iters, {len(geo_train)} train")
    print(f"  Phase 2 (FLE+RxCond, all RXs): {film_iters} iters")
    print(f"  RX conditioning: PropAware, d_model={rx_cond_args['d_model']}")
    print(f"  CSI max (normalization): {csi_max:.4f}")
    print(f"{'='*60}\n")

    with open(os.path.join(model_path, "config.json"), 'w') as f:
        config_dict = {k: v for k, v in vars(args).items() if not k.startswith('_')}
        config_dict['rx_conditioning'] = rx_cond_args
        config_dict['rx_names'] = rx_names
        config_dict['rx_positions'] = rx_positions.tolist()
        config_dict['geo_iters'] = geo_iters
        config_dict['film_iters'] = film_iters
        config_dict['csi_max'] = float(csi_max)
        json.dump(config_dict, f, indent=2, default=str)

    model_args = model_para_cls.extract(args)
    model_args.num_channels_override = 52  # 26 subcarriers × 2 (real + imag)
    opt_args = optimization_para_cls.extract(args)
    opt_args.densify_until_iter = geo_iters // 2
    opt_args.position_lr_max_steps = geo_iters
    pipe_args = pipeline_para_cls.extract(args)

    # ---- Phase 1 ----
    device = torch.device(args.data_device)
    geo_ckpt_path = os.path.join(os.path.dirname(model_path), "geometry.pth")

    if not retrain_geometry and os.path.exists(geo_ckpt_path):
        print(f"\n  Loading existing geometry: {geo_ckpt_path}")
        gaussians = GaussianModel(model_args)
        scene = Scene(model_args, gaussians)
        geo_data = torch.load(geo_ckpt_path, map_location=device)
        gaussians.restore(geo_data['gaussians'], opt_args)
        print(f"  Loaded. #G={gaussians.get_xyz.shape[0]}, skipping Phase 1.\n")
    else:
        gaussians, scene = train_geometry(
            model_args, opt_args, pipe_args, geo_train,
            model_path, geo_iters, n_azimuth, n_elevation,
            geo_save_path=geo_ckpt_path,
        )

    # ---- Phase 2 ----
    resume_ckpt = None
    if resume:
        ckpt_files = sorted(
            [f for f in os.listdir(model_path)
             if f.startswith("chkpnt") and f.endswith(".pth")],
            key=lambda x: int(x.replace("chkpnt", "").replace(".pth", "")))
        if ckpt_files:
            resume_ckpt = os.path.join(model_path, ckpt_files[-1])
            print(f"\n  Resuming Phase 2 from {ckpt_files[-1]}")

    gaussians, rx_cond = train_rxcond(
        gaussians, scene, model_args, opt_args, pipe_args,
        train_samples, model_path, rx_cond_args, film_iters,
        n_azimuth, n_elevation, resume_ckpt=resume_ckpt,
    )

    print(f"\n{'='*60}")
    print(f"  Training complete. #Gaussians: {gaussians.get_xyz.shape[0]}")
    print(f"  Results: {model_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
