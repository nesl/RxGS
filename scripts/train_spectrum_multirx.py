import os
import json
import time
from random import randint
from argparse import ArgumentParser

import numpy as np
import torch
from tqdm import tqdm
from fused_ssim import fused_ssim

from arguments import ModelParams, PipelineParams, OptimizationParams, load_config
from utils.general_utils import safe_state, prepare_log_dir
from scene import Scene, GaussianModel
from gaussian_renderer import render_rfid, render_rfid_rxcond
from scene.spectrum_multirx import load_spectrum_all_rx
from utils.loss_utils import l1_loss, fourier_loss
from utils.rx_conditioning import RxConditionedFLE_PropAware


# ---- phase 1: geometry training with a single RX ----

def train_geometry(
    model_para_args,
    opt_args,
    pipe_args,
    train_samples,
    output_dir,
    geo_iters,
    test_samples=None,
    geo_save_path=None,
    desc="Phase 1: Geometry (single RX)",
):
    """Train Gaussian model (geometry + FLE) using a single RX."""

    gaussians = GaussianModel(model_para_args)
    scene = Scene(model_para_args, gaussians)

    gaussians.training_setup(opt_args)

    ema_loss = 0.0
    t_start = time.time()
    progress_bar = tqdm(range(geo_iters), desc=desc)

    for iteration in range(1, geo_iters + 1):
        gaussians.update_learning_rate(iteration)

        fle_ramp = getattr(model_para_args, '_fle_degree_ramp', 500)
        if iteration % fle_ramp == 0:
            gaussians.oneup_fle_degree()

        viewpoint = train_samples[randint(0, len(train_samples) - 1)]

        gaussians.optimizer.zero_grad(set_to_none=True)

        render_pkg = render_rfid(viewpoint, gaussians, pipe_args)
        rendered = render_pkg["render"]
        visibility_filter = render_pkg["visibility_filter"]
        radii = render_pkg["radii"]

        gt_spectrum = viewpoint.spectrum.cuda()

        Ll1 = l1_loss(rendered, gt_spectrum)
        pred_4d = rendered.unsqueeze(0).unsqueeze(0)
        gt_4d = gt_spectrum.unsqueeze(0).unsqueeze(0)
        ssim_loss = 1.0 - fused_ssim(pred_4d, gt_4d)
        Lfourier = fourier_loss(rendered, gt_spectrum)

        lambda_ssim = getattr(opt_args, 'lambda_dssim', 0.0)
        lambda_fourier = getattr(opt_args, 'lambda_dfourier', 0.0)
        loss = (1.0 - lambda_ssim - lambda_fourier) * Ll1 \
             + lambda_ssim * ssim_loss \
             + lambda_fourier * Lfourier
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
                                                scene.cameras_extent,
                                                size_threshold)

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
          f"time={geo_time:.1f}s ({geo_time/60:.1f}min), saved {geo_ckpt}")

    if test_samples:
        all_l1 = []
        with torch.no_grad():
            for viewpoint in test_samples:
                render_pkg = render_rfid(viewpoint, gaussians, pipe_args)
                rendered = render_pkg["render"]
                gt_spectrum = viewpoint.spectrum.cuda()
                l1 = l1_loss(rendered, gt_spectrum).item()
                all_l1.append(l1)
        l1_arr = np.array(all_l1)
        geo_result = {
            "phase": "geometry",
            "iteration": geo_iters,
            "num_test": len(test_samples),
            "num_gaussians": gaussians.get_xyz.shape[0],
            "L1_mean": round(float(l1_arr.mean()), 6),
            "L1_std": round(float(l1_arr.std()), 6),
            "L1_p50": round(float(np.percentile(l1_arr, 50)), 6),
            "L1_p90": round(float(np.percentile(l1_arr, 90)), 6),
        }
        with open(os.path.join(output_dir, "geometry_eval.json"), 'w') as f:
            json.dump(geo_result, f, indent=2)
        print(f"  Geometry test L1: {l1_arr.mean():.6f} +/- {l1_arr.std():.6f} "
              f"(median: {np.percentile(l1_arr, 50):.6f})")

    return gaussians, scene


# ---- phase 2: FLE + RX conditioning across all RXs (frozen geometry) ----

def train_film(
    gaussians,
    scene,
    model_para_args,
    opt_args,
    pipe_args,
    train_samples,
    output_dir,
    rx_cond_args,
    film_iters,
    resume_ckpt=None,
):
    """Freeze geometry, train FLE coefficients + RX conditioning across all RXs."""

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
    resume_ckpt_data = None
    if resume_ckpt:
        resume_ckpt_data = torch.load(resume_ckpt, map_location=device)
        gaussian_params = resume_ckpt_data['gaussians']
        gaussians._features_dc = gaussian_params[1]
        gaussians._features_rest = gaussian_params[2]
        rx_cond.load_state_dict(resume_ckpt_data['rx_cond'])
        start_iter = resume_ckpt_data.get('iteration', 0)
        gaussians.active_fle_degree = gaussians.max_fle_degree
        print(f"  Resumed from {resume_ckpt} (iter {start_iter})")
    else:
        gaussians.active_fle_degree = 0

    # optimizer must be built after any resume-tensor swaps above
    fle_lr = getattr(opt_args, 'feature_lr', 0.005)
    fle_params = [
        {'params': [gaussians._features_dc],   'lr': fle_lr,   "name": "f_dc"},
        {'params': [gaussians._features_rest], 'lr': fle_lr * getattr(opt_args, '_rest_lr_ratio', 0.2), "name": "f_rest"},
    ]
    fle_optimizer = torch.optim.Adam(fle_params, lr=0.0, eps=1e-15)

    rx_cond_lr = rx_cond_args.get('lr', 1e-3)
    rx_cond_optimizer = torch.optim.Adam(rx_cond.parameters(), lr=rx_cond_lr, eps=1e-15)

    if resume_ckpt_data is not None:
        if 'fle_optimizer' in resume_ckpt_data:
            fle_optimizer.load_state_dict(resume_ckpt_data['fle_optimizer'])
        if 'rx_cond_optimizer' in resume_ckpt_data:
            rx_cond_optimizer.load_state_dict(resume_ckpt_data['rx_cond_optimizer'])

    ema_loss = 0.0
    t_start = time.time()
    progress_bar = tqdm(initial=start_iter, total=film_iters, desc="Phase 2: FLE+RxCond (all RX)")

    for iteration in range(start_iter + 1, film_iters + 1):
        fle_ramp = getattr(model_para_args, '_fle_degree_ramp', 500)
        if iteration % fle_ramp == 0:
            gaussians.oneup_fle_degree()

        viewpoint = train_samples[randint(0, len(train_samples) - 1)]

        fle_optimizer.zero_grad(set_to_none=True)
        rx_cond_optimizer.zero_grad(set_to_none=True)

        render_pkg = render_rfid_rxcond(viewpoint, gaussians, pipe_args, rx_cond)
        rendered = render_pkg["render"]

        gt_spectrum = viewpoint.spectrum.cuda()

        Ll1 = l1_loss(rendered, gt_spectrum)
        pred_4d = rendered.unsqueeze(0).unsqueeze(0)
        gt_4d = gt_spectrum.unsqueeze(0).unsqueeze(0)
        ssim_loss = 1.0 - fused_ssim(pred_4d, gt_4d)
        Lfourier = fourier_loss(rendered, gt_spectrum)

        lambda_ssim = getattr(opt_args, 'lambda_dssim', 0.0)
        lambda_fourier = getattr(opt_args, 'lambda_dfourier', 0.0)
        loss = (1.0 - lambda_ssim - lambda_fourier) * Ll1 \
             + lambda_ssim * ssim_loss \
             + lambda_fourier * Lfourier
        loss.backward()

        with torch.no_grad():
            ema_loss = 0.4 * loss.item() + 0.6 * ema_loss

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss:.6f}"})
                progress_bar.update(10)
            if iteration == film_iters:
                progress_bar.close()

            fle_optimizer.step()
            rx_cond_optimizer.step()

            # save halfway and final checkpoints
            if iteration == film_iters // 2 or iteration == film_iters:
                elapsed = time.time() - t_start
                scene.save(iteration)
                ckpt_path = os.path.join(output_dir, f"chkpnt{iteration}.pth")
                torch.save({
                    'gaussians': gaussians.capture(),
                    'rx_cond': rx_cond.state_dict(),
                    'fle_optimizer': fle_optimizer.state_dict(),
                    'rx_cond_optimizer': rx_cond_optimizer.state_dict(),
                    'iteration': iteration,
                    'training_time_s': elapsed,
                }, ckpt_path)
                tag = "final" if iteration == film_iters else "halfway"
                print(f"\n  [iter {iteration}] Saved {tag} checkpoint "
                      f"(#G: {gaussians.get_xyz.shape[0]}, time: {elapsed:.1f}s)")

    return gaussians, rx_cond


# ---- main ----

def main():
    pre_parser = ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default="arguments/configs/exp_spectrum_multirx_main.yaml")
    pre_args, _ = pre_parser.parse_known_args()

    yaml_cfg = load_config(pre_args.config)
    random_seed = (yaml_cfg or {}).get("random_seed", 8371)

    parser = ArgumentParser(description="Spectrum Training (multi-RX, single model)")
    parser.add_argument("--config", type=str, default="arguments/configs/exp_spectrum_multirx_main.yaml")

    model_para_cls = ModelParams(parser, yaml_cfg=yaml_cfg)
    optimization_para_cls = OptimizationParams(parser, yaml_cfg=yaml_cfg)
    pipeline_para_cls = PipelineParams(parser, yaml_cfg=yaml_cfg)

    parser.add_argument("--quiet", action="store_true", default=False)
    parser.add_argument("--retrain_geometry", action="store_true", default=False,
                        help="Force retrain Phase 1 geometry even if checkpoint exists")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="Resume Phase 2 from latest checkpoint")

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
    geo_rx = (yaml_cfg or {}).get("geo_rx", 0)
    film_iters = (yaml_cfg or {}).get("film_iterations", 30000)


    ratio_train = getattr(args, 'ratio_train', 0.8)

    train_samples, test_samples, rx_names, rx_positions = load_spectrum_all_rx(
        data_dir, ratio_train=ratio_train, seed=random_seed,
    )

    geo_rx_name = rx_names[geo_rx]
    geo_rx_pos = rx_positions[geo_rx]
    geo_train = [s for s in train_samples
                 if np.allclose(s.T_rx.numpy(), geo_rx_pos, atol=1e-4)]
    geo_test = [s for s in test_samples
                if np.allclose(s.T_rx.numpy(), geo_rx_pos, atol=1e-4)]

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
    print("  Spectrum Training (multi-RX, 2-phase)")
    print(f"  Data: {data_dir}")
    print(f"  Output: {model_path}")
    print(f"  RXs: {len(rx_names)}")
    print(f"  Train samples: {len(train_samples)} (all RXs)")
    print(f"  Phase 1 (geometry, RX {geo_rx} '{geo_rx_name}'): {geo_iters} iters, "
          f"{len(geo_train)} train / {len(geo_test)} test")
    print(f"  Phase 2 (FLE+RxCond, all RX): {film_iters} iters")
    print(f"  Save at: [{film_iters // 2}, {film_iters}]")
    print(f"  RX conditioning: PropAware, d_model={rx_cond_args['d_model']}")
    print(f"{'='*60}\n")

    with open(os.path.join(model_path, "config.json"), 'w') as f:
        config_dict = {k: v for k, v in vars(args).items() if not k.startswith('_')}
        config_dict['rx_conditioning'] = rx_cond_args
        config_dict['rx_names'] = rx_names
        config_dict['rx_positions'] = rx_positions.tolist()
        config_dict['geo_iters'] = geo_iters
        config_dict['film_iters'] = film_iters
        config_dict['geo_rx'] = geo_rx
        json.dump(config_dict, f, indent=2, default=str)

    model_args = model_para_cls.extract(args)
    opt_args = optimization_para_cls.extract(args)
    opt_args.densify_until_iter = geo_iters // 2
    opt_args.position_lr_max_steps = geo_iters
    pipe_args = pipeline_para_cls.extract(args)

    # ---- Phase 1: geometry with single RX ----
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
            model_path, geo_iters, test_samples=geo_test,
            geo_save_path=geo_ckpt_path,
        )

    # ---- Phase 2: FLE + RX conditioning across all RXs ----
    resume_ckpt = None
    if resume:
        ckpt_files = sorted(
            [f for f in os.listdir(model_path)
             if f.startswith("chkpnt") and f.endswith(".pth")],
            key=lambda x: int(x.replace("chkpnt","").replace(".pth","")))
        if ckpt_files:
            resume_ckpt = os.path.join(model_path, ckpt_files[-1])
            print(f"\n  Resuming Phase 2 from {ckpt_files[-1]}")
        else:
            print("\n  --resume specified but no checkpoints found, starting fresh")

    gaussians, rx_cond = train_film(
        gaussians, scene, model_args, opt_args, pipe_args,
        train_samples, model_path, rx_cond_args, film_iters,
        resume_ckpt=resume_ckpt,
    )

    print(f"\n{'='*60}")
    print(f"  Training complete. #Gaussians: {gaussians.get_xyz.shape[0]}")
    print(f"  Results: {model_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
