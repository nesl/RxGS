"""
Spectrum inference for single-RX GSRF models (one model per RX).

Evaluates the latest checkpoint of each RX (or a specific one with --iter).

Usage:
    python -m scripts.inference_spectrum_singlerx --config arguments/configs/exp_spectrum_singlerx.yaml
"""

import os
import json
import csv
import shutil
from argparse import ArgumentParser

import numpy as np
import torch
import lpips

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision.models._utils")

from arguments import ModelParams, PipelineParams, load_config
from utils.general_utils import safe_state
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render_rfid
from scene.spectrum_multirx import load_spectrum_per_rx
from utils.data_painter import paint_spectrum
from scripts.inference_spectrum_multirx import (compute_mse_gpu, compute_psnr_gpu, compute_ssim_gpu,
                                        compute_lpips_value, make_stats)


def find_checkpoint(rx_dir, iteration=None):
    if iteration:
        path = os.path.join(rx_dir, f"chkpnt{iteration}.pth")
        return path if os.path.exists(path) else None
    ckpts = sorted([f for f in os.listdir(rx_dir) if f.startswith("chkpnt") and f.endswith(".pth")],
                   key=lambda x: int(x.replace("chkpnt", "").replace(".pth", "")))
    return os.path.join(rx_dir, ckpts[-1]) if ckpts else None


def load_gaussians(ckpt_path, model_args):
    gaussians = GaussianModel(model_args)
    ckpt_data = torch.load(ckpt_path, map_location='cuda')
    (gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
     gaussians._attenuation, gaussians._scaling, gaussians._rotation,
     gaussians.max_radii2D, _, _, _, gaussians.spatial_lr_scale) = ckpt_data['gaussians']
    gaussians.active_fle_degree = gaussians.max_fle_degree
    return gaussians, ckpt_data.get('iteration', 'unknown')


def evaluate_rx(
    rx_idx,
    gaussians,
    pipe_args,
    test_samples,
    rx_name,
    ckpt_path,
    ckpt_iter,
    output_dir,
    lpips_fn,
    n_paint=5,
    paint_seed=8371,
):
    """Evaluate one RX's model on its test samples."""
    l1_list, psnr_list, mse_list, ssim_list, lpips_list, names_list = [], [], [], [], [], []

    os.makedirs(output_dir, exist_ok=True)
    render_dir = os.path.join(output_dir, "rendered")
    os.makedirs(render_dir, exist_ok=True)

    rng = np.random.default_rng(paint_seed + rx_idx)
    paint_idx = set(rng.choice(len(test_samples), size=min(n_paint, len(test_samples)),
                               replace=False).tolist())

    with torch.no_grad():
        for step_idx, viewpoint in enumerate(test_samples):
            rendered = render_rfid(viewpoint, gaussians, pipe_args)["render"].detach()
            gt_spectrum = viewpoint.spectrum.cuda()

            l1_list.append(float((rendered - gt_spectrum).abs().mean().item()))
            mse_list.append(compute_mse_gpu(rendered, gt_spectrum))
            psnr_list.append(compute_psnr_gpu(rendered, gt_spectrum))
            ssim_list.append(compute_ssim_gpu(rendered, gt_spectrum))
            lpips_list.append(compute_lpips_value(rendered, gt_spectrum, lpips_fn))
            names_list.append(viewpoint.spectrum_name)

            if step_idx in paint_idx:
                paint_spectrum(gt_spectrum.cpu().numpy(), rendered.cpu().numpy(),
                               psnr_list[-1], ssim_list[-1], os.path.join(render_dir, f"{viewpoint.spectrum_name}.png"))

    psnr_arr = np.array(psnr_list)
    mse_arr = np.array(mse_list)
    ssim_arr = np.array(ssim_list)
    lpips_arr = np.array(lpips_list)

    result = {
        "rx": rx_name,
        "checkpoint": ckpt_path,
        "iteration": ckpt_iter,
        "num_test": len(psnr_arr),
        "num_gaussians": gaussians.get_xyz.shape[0],
        "L1": make_stats(np.array(l1_list)),
        "PSNR_dB": make_stats(psnr_arr),
        "MSE": make_stats(mse_arr),
        "SSIM": make_stats(ssim_arr),
        "LPIPS": make_stats(lpips_arr),
    }

    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(result, f, indent=2)

    with open(os.path.join(output_dir, "predictions.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["index", "name", "psnr", "mse", "ssim", "lpips"])
        for i in range(len(psnr_arr)):
            writer.writerow([i, names_list[i], f"{psnr_arr[i]:.4f}",
                             f"{mse_arr[i]:.6f}", f"{ssim_arr[i]:.4f}", f"{lpips_arr[i]:.4f}"])

    return result


def main():
    parser = ArgumentParser(description="Spectrum Inference (single-RX)")
    parser.add_argument("--config", type=str, default="arguments/configs/exp_spectrum_singlerx.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--iter", type=int, default=None,
                        help="Evaluate this iteration (default: latest checkpoint of each RX)")

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

    run_dir = cmd_args.model_path or os.path.join(
        base_args.log_base_folder, base_args.dataset, base_args.exp_name)
    output_dir = cmd_args.output_dir or os.path.join(run_dir, "inference")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    per_rx, rx_names = load_spectrum_per_rx(
        data_dir, ratio_train=getattr(base_args, 'ratio_train', 0.8), seed=random_seed)

    print(f"\n{'='*60}")
    print("  Spectrum Inference (single-RX)")
    print(f"  Model: {run_dir}")
    print(f"  Output: {output_dir}")
    print(f"  RXs: {len(rx_names)}")
    print(f"{'='*60}")

    model_args = model_cls.extract(base_args)
    model_args.model_path = run_dir
    model_args.source_path = data_dir
    pipe_args = pipe_cls.extract(base_args)

    lpips_fn = lpips.LPIPS(net='alex').cuda()

    all_results = []
    for rx_idx, rx_name in enumerate(rx_names):
        rx_dir = os.path.join(run_dir, rx_name)
        if rx_idx not in per_rx or not os.path.isdir(rx_dir):
            continue
        ckpt_path = find_checkpoint(rx_dir, cmd_args.iter)
        if ckpt_path is None:
            print(f"    {rx_name}: no checkpoint, skipped")
            continue

        gaussians, ckpt_iter = load_gaussians(ckpt_path, model_args)
        _, test_samples = per_rx[rx_idx]
        result = evaluate_rx(rx_idx, gaussians, pipe_args, test_samples, rx_name,
                             ckpt_path, ckpt_iter, os.path.join(output_dir, rx_name), lpips_fn)
        all_results.append(result)
        print(f"    {rx_name}: L1 = {result['L1']['mean']:.6f}, "
              f"PSNR = {result['PSNR_dB']['mean']:.2f} dB, "
              f"SSIM = {result['SSIM']['mean']:.4f}, "
              f"LPIPS = {result['LPIPS']['mean']:.4f}")

        del gaussians
        torch.cuda.empty_cache()

    if not all_results:
        print(f"  No checkpoints found in {run_dir}")
        return

    summary = {
        "num_rx": len(all_results),
        "overall_L1_mean": round(float(np.mean([r['L1']['mean'] for r in all_results])), 6),
        "overall_PSNR_mean": round(float(np.mean([r['PSNR_dB']['mean'] for r in all_results])), 2),
        "overall_MSE_mean": round(float(np.mean([r['MSE']['mean'] for r in all_results])), 6),
        "overall_SSIM_mean": round(float(np.mean([r['SSIM']['mean'] for r in all_results])), 6),
        "overall_LPIPS_mean": round(float(np.mean([r['LPIPS']['mean'] for r in all_results])), 6),
        "per_rx": all_results,
    }
    with open(os.path.join(output_dir, "summary_all_rx.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  {'RX':<10s}  {'L1':>10s}  {'PSNR':>8s}  {'SSIM':>8s}  {'LPIPS':>8s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")
    for r in all_results:
        print(f"  {r['rx']:<10s}  {r['L1']['mean']:>10.6f}  {r['PSNR_dB']['mean']:>8.2f}  "
              f"{r['SSIM']['mean']:>8.4f}  {r['LPIPS']['mean']:>8.4f}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*8}")
    print(f"  {'Average':<10s}  {summary['overall_L1_mean']:>10.6f}  "
          f"{summary['overall_PSNR_mean']:>8.2f}  {summary['overall_SSIM_mean']:>8.4f}  "
          f"{summary['overall_LPIPS_mean']:>8.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
