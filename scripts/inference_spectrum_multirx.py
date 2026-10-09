"""
Spectrum Inference for multi-RX single model (with RX conditioning).

Usage:
    python -m scripts.inference_spectrum_multirx --config arguments/configs/exp_spectrum_multirx_main.yaml
"""

import os
import json
import csv
import shutil
from argparse import ArgumentParser

import numpy as np
import torch
import lpips
from fused_ssim import fused_ssim

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

from arguments import ModelParams, PipelineParams, load_config
from utils.general_utils import safe_state
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render_rfid_rxcond as render
from scene.spectrum_multirx import load_spectrum_per_rx
from utils.data_painter import paint_spectrum
from utils.rx_conditioning import RxConditionedFLE_PropAware


# ---- metrics (all on GPU) ----

def compute_mse_gpu(pred_t, gt_t):
    return float(((pred_t - gt_t) ** 2).mean().item())


def compute_psnr_gpu(pred_t, gt_t, data_range=1.0):
    mse = ((pred_t - gt_t) ** 2).mean()
    return float((10.0 * torch.log10(data_range ** 2 / mse)).item())


def compute_ssim_gpu(pred_t, gt_t):
    pred_4d = pred_t.unsqueeze(0).unsqueeze(0)
    gt_4d = gt_t.unsqueeze(0).unsqueeze(0)
    return float(fused_ssim(pred_4d, gt_4d, train=False).item())


def compute_lpips_value(pred_tensor, gt_tensor, lpips_fn):
    pred_t = pred_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    gt_t = gt_tensor.unsqueeze(0).unsqueeze(0).repeat(1, 3, 1, 1)
    pred_t = pred_t * 2.0 - 1.0
    gt_t = gt_t * 2.0 - 1.0
    with torch.no_grad():
        val = lpips_fn(pred_t, gt_t)
    return val.item()


def make_stats(arr):
    return {
        "mean": round(float(arr.mean()), 6),
        "std": round(float(arr.std()), 6),
        "min": round(float(arr.min()), 6),
        "p50": round(float(np.percentile(arr, 50)), 6),
        "p90": round(float(np.percentile(arr, 90)), 6),
        "p95": round(float(np.percentile(arr, 95)), 6),
        "max": round(float(arr.max()), 6),
    }


def evaluate_rx(
    rx_idx,
    gaussians,
    rx_cond,
    pipe_args,
    test_samples,
    rx_name,
    ckpt_path,
    output_dir,
    lpips_fn,
    n_paint=5,
    paint_seed=8371,
):
    """Evaluate one RX across all test samples."""
    l1_list, psnr_list, mse_list, ssim_list, lpips_list, names_list = [], [], [], [], [], []

    os.makedirs(output_dir, exist_ok=True)
    render_dir = os.path.join(output_dir, "rendered")
    os.makedirs(render_dir, exist_ok=True)

    # pick `n_paint` random sample indices to dump as PNG (deterministic per RX)
    rng = np.random.default_rng(paint_seed + rx_idx)
    paint_idx = set(rng.choice(len(test_samples), size=min(n_paint, len(test_samples)),
                               replace=False).tolist())

    with torch.no_grad():
        for step_idx, viewpoint in enumerate(test_samples):
            render_pkg = render(viewpoint, gaussians, pipe_args, rx_cond)
            rendered = render_pkg["render"].detach()      # GPU
            gt_spectrum = viewpoint.spectrum.cuda()       # GPU

            # all metrics on GPU; only sync to host for the scalar value
            l1_val   = float((rendered - gt_spectrum).abs().mean().item())
            mse_val  = compute_mse_gpu(rendered, gt_spectrum)
            psnr_val = compute_psnr_gpu(rendered, gt_spectrum)
            ssim_val = compute_ssim_gpu(rendered, gt_spectrum)
            lpips_val = compute_lpips_value(rendered, gt_spectrum, lpips_fn)

            l1_list.append(l1_val)
            psnr_list.append(psnr_val)
            mse_list.append(mse_val)
            ssim_list.append(ssim_val)
            lpips_list.append(lpips_val)
            names_list.append(viewpoint.spectrum_name)

            if step_idx in paint_idx:
                save_path = os.path.join(render_dir, f"{viewpoint.spectrum_name}.png")
                paint_spectrum(gt_spectrum.cpu().numpy(), rendered.cpu().numpy(),
                               psnr_val, ssim_val, save_path)

    l1_arr = np.array(l1_list)
    psnr_arr = np.array(psnr_list)
    mse_arr = np.array(mse_list)
    ssim_arr = np.array(ssim_list)
    lpips_arr = np.array(lpips_list)

    result = {
        "rx": rx_name,
        "checkpoint": ckpt_path,
        "num_test": len(psnr_arr),
        "num_gaussians": gaussians.get_xyz.shape[0],
        "L1": make_stats(l1_arr),
        "PSNR_dB": make_stats(psnr_arr),
        "MSE": make_stats(mse_arr),
        "SSIM": make_stats(ssim_arr),
        "LPIPS": make_stats(lpips_arr),
    }

    # save per-RX summary.json
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(result, f, indent=2)

    # save per-sample CSV
    with open(os.path.join(output_dir, "predictions.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["index", "name", "psnr", "mse", "ssim", "lpips"])
        for i in range(len(psnr_arr)):
            writer.writerow([i, names_list[i], f"{psnr_arr[i]:.4f}",
                             f"{mse_arr[i]:.6f}", f"{ssim_arr[i]:.4f}", f"{lpips_arr[i]:.4f}"])

    return result


def evaluate_checkpoint(
    ckpt_path,
    model_args,
    pipe_args,
    rx_cond_cfg,
    per_rx,
    rx_names,
    output_dir,
    lpips_fn,
):
    """Load one checkpoint and evaluate all RXs."""

    gaussians = GaussianModel(model_args)
    ckpt_data = torch.load(ckpt_path, map_location='cuda')
    gaussian_params = ckpt_data['gaussians']
    (gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
     gaussians._attenuation, gaussians._scaling, gaussians._rotation,
     gaussians.max_radii2D, _, _, _, gaussians.spatial_lr_scale) = gaussian_params
    gaussians.active_fle_degree = gaussians.max_fle_degree

    fle_degree = model_args.fle_degree
    n_fle = (fle_degree + 1) ** 2
    rx_cond = RxConditionedFLE_PropAware(
        n_fle_components=n_fle,
        max_fle_degree=fle_degree,
        d_model=rx_cond_cfg.get('d_model', 64),
        n_freqs=rx_cond_cfg.get('n_freqs', 6),
        max_freq_log2=rx_cond_cfg.get('max_freq_log2', 5.0),
        comp_dim=rx_cond_cfg.get('comp_dim', 16),
        n_probe_samples=rx_cond_cfg.get('n_probe_samples', 16),
    ).cuda()
    rx_cond.load_state_dict(ckpt_data['rx_cond'])
    rx_cond.eval()

    rx_cond.build_occupancy_grid(
        gaussians._xyz.detach(),
        gaussians.get_attenuation.detach(),
        gaussians._scaling.detach(),
        resolution=rx_cond_cfg.get('grid_resolution', 128),
    )

    ckpt_iter = ckpt_data.get('iteration', 'unknown')
    os.makedirs(output_dir, exist_ok=True)

    n_rx = sum(1 for rx_idx in range(len(rx_names))
               if rx_idx in per_rx and len(per_rx[rx_idx][1]) >= 1)
    print(f"\n  Checkpoint iter {ckpt_iter}: #G={gaussians.get_xyz.shape[0]}, {n_rx} RXs")

    all_results = []

    for rx_idx, rx_name in enumerate(rx_names):
        if rx_idx not in per_rx:
            continue
        _, test_samples = per_rx[rx_idx]
        if len(test_samples) < 1:
            continue

        rx_dir = os.path.join(output_dir, rx_name)
        result = evaluate_rx(
            rx_idx, gaussians, rx_cond, pipe_args, test_samples, rx_name,
            ckpt_path, rx_dir, lpips_fn
        )
        all_results.append(result)
        print(f"    {rx_name}: L1 = {result['L1']['mean']:.6f}, "
              f"PSNR = {result['PSNR_dB']['mean']:.2f} dB, "
              f"SSIM = {result['SSIM']['mean']:.4f}, "
              f"LPIPS = {result['LPIPS']['mean']:.4f}")

    # compute overall averages across RXs
    avg_l1 = np.mean([s['L1']['mean'] for s in all_results])
    avg_psnr = np.mean([s['PSNR_dB']['mean'] for s in all_results])
    avg_mse = np.mean([s['MSE']['mean'] for s in all_results])
    avg_ssim = np.mean([s['SSIM']['mean'] for s in all_results])
    avg_lpips = np.mean([s['LPIPS']['mean'] for s in all_results])

    summary = {
        "checkpoint": ckpt_path,
        "iteration": ckpt_iter,
        "num_rx": len(all_results),
        "num_gaussians": gaussians.get_xyz.shape[0],
        "overall_L1_mean": round(float(avg_l1), 6),
        "overall_PSNR_mean": round(float(avg_psnr), 2),
        "overall_MSE_mean": round(float(avg_mse), 6),
        "overall_SSIM_mean": round(float(avg_ssim), 6),
        "overall_LPIPS_mean": round(float(avg_lpips), 6),
        "per_rx": all_results,
    }

    with open(os.path.join(output_dir, "summary_all_rx.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    return summary


def main():
    parser = ArgumentParser(description="Spectrum Inference (multi-RX)")
    parser.add_argument("--config", type=str, default="arguments/configs/exp_spectrum_multirx_main.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--iter", type=int, default=None,
                        help="Evaluate this iteration (default: latest checkpoint)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Evaluate this checkpoint file (e.g. a downloaded pretrained model)")

    cmd_args = parser.parse_args()

    yaml_cfg = load_config(cmd_args.config)
    random_seed = (yaml_cfg or {}).get("random_seed", 8371)

    base_parser = ArgumentParser(add_help=False)
    base_parser.add_argument("--config", type=str, default=cmd_args.config)
    model_cls = ModelParams(base_parser, yaml_cfg=yaml_cfg)
    pipe_cls = PipelineParams(base_parser, yaml_cfg=yaml_cfg)
    base_args, _ = base_parser.parse_known_args(["--config", cmd_args.config])

    safe_state(False, random_seed, torch.device(base_args.data_device))

    data_dir = os.path.join(base_args.input_data_folder, base_args.dataset)
    base_args.source_path = data_dir

    if cmd_args.checkpoint:
        run_dir = os.path.dirname(os.path.abspath(cmd_args.checkpoint))
    else:
        run_dir = cmd_args.model_path or os.path.join(
            base_args.log_base_folder, base_args.dataset, base_args.exp_name)
    if cmd_args.output_dir:
        output_dir = cmd_args.output_dir
    elif cmd_args.checkpoint:
        output_dir = os.path.splitext(os.path.abspath(cmd_args.checkpoint))[0] + "_inference"
    else:
        output_dir = os.path.join(run_dir, "inference")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    print(f"\n{'='*60}")
    print("  Spectrum Inference (multi-RX)")
    print(f"  Model: {run_dir}")
    print(f"  Output: {output_dir}")

    # pick the latest checkpoint by default (--iter overrides)
    if cmd_args.checkpoint:
        ckpt_file = os.path.basename(cmd_args.checkpoint)
    elif cmd_args.iter:
        ckpt_file = f"chkpnt{cmd_args.iter}.pth"
    else:
        all_ckpts = sorted([f for f in os.listdir(run_dir)
                            if f.startswith("chkpnt") and f.endswith(".pth")],
                           key=lambda x: int(x.replace("chkpnt", "").replace(".pth", "")))
        if not all_ckpts:
            print(f"  No checkpoints found in {run_dir}")
            return
        ckpt_file = all_ckpts[-1]
    ckpt_path = os.path.join(run_dir, ckpt_file)
    if not os.path.exists(ckpt_path):
        print(f"  Checkpoint not found: {ckpt_path}")
        return

    print(f"  Checkpoint: {ckpt_path}")

    # load per-RX test data
    ratio_train = getattr(base_args, 'ratio_train', 0.8)
    per_rx, rx_names = load_spectrum_per_rx(data_dir, ratio_train=ratio_train, seed=random_seed)

    print(f"  Data: {data_dir}")
    print(f"  RXs: {len(rx_names)}")
    print(f"{'='*60}")

    # build args
    model_args = model_cls.extract(base_args)
    model_args.model_path = run_dir
    model_args.source_path = data_dir
    pipe_args = pipe_cls.extract(base_args)
    rx_cond_cfg = (yaml_cfg or {}).get("rx_conditioning", {})

    lpips_fn = lpips.LPIPS(net='alex').cuda()

    summary = evaluate_checkpoint(
        ckpt_path, model_args, pipe_args, rx_cond_cfg,
        per_rx, rx_names, output_dir, lpips_fn
    )

    print(f"\n{'='*60}")
    print(f"  Per-RX Summary (iter {summary['iteration']}):")
    print(f"{'='*60}")
    print(f"  {'RX':<10s}  {'L1':>10s}  {'PSNR':>8s}  {'SSIM':>8s}  {'LPIPS':>8s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in summary['per_rx']:
        print(f"  {r['rx']:<10s}  {r['L1']['mean']:>10.6f}  {r['PSNR_dB']['mean']:>8.2f}  "
              f"{r['SSIM']['mean']:>8.4f}  {r['LPIPS']['mean']:>8.4f}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'Average':<10s}  {summary['overall_L1_mean']:>10.6f}  "
          f"{summary['overall_PSNR_mean']:>8.2f}  {summary['overall_SSIM_mean']:>8.4f}  "
          f"{summary['overall_LPIPS_mean']:>8.4f}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
