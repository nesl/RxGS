"""
CSI inference for single-RX GSRF models (one model per RX).

Evaluates the latest checkpoint of each RX (or a specific one with --iter).

Usage:
    python -m scripts.inference_csi_singlerx --config arguments/configs/exp_csi_singlerx.yaml
"""

import os
import json
import shutil
from argparse import ArgumentParser

import numpy as np
import torch

from arguments import ModelParams, PipelineParams, load_config
from utils.general_utils import safe_state
from scene.gaussian_model import GaussianModel
from gaussian_renderer.render_csi import render_csi
from scene.csi_dataset import load_csi


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


def evaluate_rx(gaussians, pipe_args, test_samples, rx_name, rx_idx, csi_max, n_azimuth, n_elevation):
    """Evaluate one RX's model on its test samples. Returns per-sample SNR."""
    all_snr = []
    all_pred_re, all_pred_im = [], []
    all_gt_re, all_gt_im = [], []

    with torch.no_grad():
        for viewpoint in test_samples:
            rendered = render_csi(viewpoint, gaussians, pipe_args,
                                  n_azimuth=n_azimuth, n_elevation=n_elevation)["render"]
            pred_csi = rendered.mean(dim=(1, 2))
            gt_csi = viewpoint.spectrum.cuda().mean(dim=(1, 2))

            pred_re = pred_csi[0::2].cpu().numpy()
            pred_im = pred_csi[1::2].cpu().numpy()
            gt_re = gt_csi[0::2].cpu().numpy()
            gt_im = gt_csi[1::2].cpu().numpy()

            pred_re_dn = pred_re * csi_max
            pred_im_dn = pred_im * csi_max
            gt_re_dn = gt_re * csi_max
            gt_im_dn = gt_im * csi_max

            err = ((pred_re_dn - gt_re_dn)**2 + (pred_im_dn - gt_im_dn)**2).sum()
            gt_pwr = (gt_re_dn**2 + gt_im_dn**2).sum()
            all_snr.append(-10 * np.log10(err / (gt_pwr + 1e-8) + 1e-10))

            all_pred_re.append(pred_re)
            all_pred_im.append(pred_im)
            all_gt_re.append(gt_re)
            all_gt_im.append(gt_im)

    snr_arr = np.array(all_snr)
    result = {
        "rx": rx_name,
        "rx_idx": rx_idx,
        "num_test": len(all_snr),
        "num_gaussians": gaussians.get_xyz.shape[0],
        "SNR_dB": {
            "mean": round(float(snr_arr.mean()), 4),
            "std": round(float(snr_arr.std()), 4),
            "min": round(float(snr_arr.min()), 4),
            "p25": round(float(np.percentile(snr_arr, 25)), 4),
            "p50": round(float(np.median(snr_arr)), 4),
            "p75": round(float(np.percentile(snr_arr, 75)), 4),
            "p90": round(float(np.percentile(snr_arr, 90)), 4),
            "p95": round(float(np.percentile(snr_arr, 95)), 4),
            "max": round(float(snr_arr.max()), 4),
        },
    }

    pred = np.array(all_pred_re) + 1j * np.array(all_pred_im)
    gt = np.array(all_gt_re) + 1j * np.array(all_gt_im)
    return result, snr_arr, pred, gt


def main():
    parser = ArgumentParser(description="CSI Inference (single-RX)")
    parser.add_argument("--config", type=str, default="arguments/configs/exp_csi_singlerx.yaml")
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

    n_azimuth = getattr(base_args, 'n_azimuth', 36)
    n_elevation = getattr(base_args, 'n_elevation', 9)

    run_dir = cmd_args.model_path or os.path.join(
        base_args.log_base_folder, base_args.dataset, base_args.exp_name)
    output_dir = cmd_args.output_dir or os.path.join(run_dir, "inference")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    per_rx, rx_names, csi_max = load_csi(
        data_dir, ratio_train=getattr(base_args, 'ratio_train', 0.8), seed=random_seed,
        n_elevation=n_elevation, n_azimuth=n_azimuth)

    model_args = model_cls.extract(base_args)
    model_args.model_path = run_dir
    model_args.source_path = data_dir
    model_args.num_channels_override = 52
    pipe_args = pipe_cls.extract(base_args)

    print(f"\n{'='*60}")
    print("  CSI Inference (single-RX)")
    print(f"  Model: {run_dir}")
    print(f"  Output: {output_dir}")
    print(f"  RXs: {len(rx_names)}")
    print(f"  CSI max (denorm): {csi_max:.4f}")
    print(f"{'='*60}")

    all_results = []
    total_err, total_gt = 0.0, 0.0

    for rx_idx, rx_name in enumerate(rx_names):
        rx_dir = os.path.join(run_dir, rx_name)
        if rx_idx not in per_rx or not os.path.isdir(rx_dir):
            continue
        _, test_samples = per_rx[rx_idx]
        if len(test_samples) < 5:
            continue
        ckpt_path = find_checkpoint(rx_dir, cmd_args.iter)
        if ckpt_path is None:
            print(f"    {rx_name}: no checkpoint, skipped")
            continue

        gaussians, ckpt_iter = load_gaussians(ckpt_path, model_args)
        result, snr_arr, pred, gt = evaluate_rx(
            gaussians, pipe_args, test_samples, rx_name, rx_idx, csi_max, n_azimuth, n_elevation)
        result["checkpoint"] = ckpt_path
        result["iteration"] = ckpt_iter
        all_results.append(result)
        print(f"    {rx_name}: SNR = {result['SNR_dB']['mean']:.2f} dB "
              f"(median {result['SNR_dB']['p50']:.2f})")

        rx_out = os.path.join(output_dir, rx_name)
        os.makedirs(rx_out, exist_ok=True)
        with open(os.path.join(rx_out, "result.json"), 'w') as f:
            json.dump(result, f, indent=2)
        np.savez(os.path.join(rx_out, "csi_results.npz"), pred=pred, gt=gt, snr_db=snr_arr)

        pred_dn = pred * csi_max
        gt_dn = gt * csi_max
        total_err += (np.abs(pred_dn - gt_dn)**2).sum()
        total_gt += (np.abs(gt_dn)**2).sum()

        del gaussians
        torch.cuda.empty_cache()

    if not all_results:
        print(f"  No checkpoints found in {run_dir}")
        return

    joint_snr = -10 * np.log10(total_err / (total_gt + 1e-8) + 1e-10)
    per_rx_snr = np.array([r['SNR_dB']['mean'] for r in all_results])

    summary = {
        "num_rx": len(all_results),
        "num_test": sum(r['num_test'] for r in all_results),
        "overall_SNR_dB": {
            "mean": round(float(per_rx_snr.mean()), 2),
            "std": round(float(per_rx_snr.std()), 2),
        },
        "joint_SNR_dB": {
            "mean": round(float(joint_snr), 2),
        },
        "per_rx": all_results,
    }
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"  {'RX':<10s}  {'SNR Mean':>10s}  {'SNR Med':>10s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}")
    for r in all_results:
        print(f"  {r['rx']:<10s}  {r['SNR_dB']['mean']:>10.2f}  {r['SNR_dB']['p50']:>10.2f}")
    print(f"\n  Overall SNR: {summary['overall_SNR_dB']['mean']:.2f} ± "
          f"{summary['overall_SNR_dB']['std']:.2f} dB  (joint: {summary['joint_SNR_dB']['mean']:.2f} dB)")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
