"""
BLE RSSI inference for single-RX GSRF models (one model per gateway).

Evaluates the latest checkpoint of each gateway (or a specific one with --iter).

Usage:
    python -m scripts.inference_ble_singlerx --config arguments/configs/exp_ble_singlerx.yaml
"""

import os
import json
import csv
import shutil
from argparse import ArgumentParser

import numpy as np
import torch

from arguments import ModelParams, PipelineParams, load_config
from utils.general_utils import safe_state
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render_rfid
from scene.ble_dataset import load_ble_per_gateway, amplitude_to_rssi
from utils.data_painter import plot_metric_cdf, plot_metric_histogram


def find_checkpoint(gw_dir, iteration=None):
    if iteration:
        path = os.path.join(gw_dir, f"chkpnt{iteration}.pth")
        return path if os.path.exists(path) else None
    ckpts = sorted([f for f in os.listdir(gw_dir) if f.startswith("chkpnt") and f.endswith(".pth")],
                   key=lambda x: int(x.replace("chkpnt", "").replace(".pth", "")))
    return os.path.join(gw_dir, ckpts[-1]) if ckpts else None


def load_gaussians(ckpt_path, model_args):
    gaussians = GaussianModel(model_args)
    ckpt_data = torch.load(ckpt_path, map_location='cuda')
    (gaussians._xyz, gaussians._features_dc, gaussians._features_rest,
     gaussians._attenuation, gaussians._scaling, gaussians._rotation,
     gaussians.max_radii2D, _, _, _, gaussians.spatial_lr_scale) = ckpt_data['gaussians']
    gaussians.active_fle_degree = gaussians.max_fle_degree
    return gaussians, ckpt_data.get('iteration', 'unknown')


def evaluate_gateway(gaussians, pipe_args, test_samples, gw_name, gw_idx):
    """Evaluate one gateway's model on its test samples."""
    all_mae, all_pred, all_gt = [], [], []

    with torch.no_grad():
        for viewpoint in test_samples:
            pred_amp = render_rfid(viewpoint, gaussians, pipe_args)["render"].mean().cpu().item()
            gt_amp = viewpoint.spectrum.mean().item()
            pred_rssi = amplitude_to_rssi(pred_amp)
            gt_rssi = amplitude_to_rssi(gt_amp)

            all_mae.append(abs(pred_rssi - gt_rssi))
            all_pred.append(pred_rssi)
            all_gt.append(gt_rssi)

    mae_arr = np.array(all_mae)
    result = {
        "gateway": gw_name,
        "gateway_idx": gw_idx,
        "num_test": len(test_samples),
        "num_gaussians": gaussians.get_xyz.shape[0],
        "MAE_dBm": {
            "mean": round(float(mae_arr.mean()), 4),
            "std": round(float(mae_arr.std()), 4),
            "min": round(float(mae_arr.min()), 4),
            "p25": round(float(np.percentile(mae_arr, 25)), 4),
            "p50": round(float(np.percentile(mae_arr, 50)), 4),
            "p75": round(float(np.percentile(mae_arr, 75)), 4),
            "p90": round(float(np.percentile(mae_arr, 90)), 4),
            "p95": round(float(np.percentile(mae_arr, 95)), 4),
            "max": round(float(mae_arr.max()), 4),
        },
    }
    return result, np.array(all_pred), np.array(all_gt), mae_arr


def main():
    parser = ArgumentParser(description="BLE RSSI Inference (single-RX)")
    parser.add_argument("--config", type=str, default="arguments/configs/exp_ble_singlerx.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--iter", type=int, default=None,
                        help="Evaluate this iteration (default: latest checkpoint of each gateway)")

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

    per_gw, gw_names = load_ble_per_gateway(
        data_dir, ratio_train=getattr(base_args, 'ratio_train', 0.8), seed=random_seed,
        n_elevation=getattr(base_args, 'n_elevation', 9), n_azimuth=getattr(base_args, 'n_azimuth', 36))

    model_args = model_cls.extract(base_args)
    model_args.model_path = run_dir
    model_args.source_path = data_dir
    pipe_args = pipe_cls.extract(base_args)

    print(f"\n{'='*60}")
    print("  BLE RSSI Inference (single-RX)")
    print(f"  Model: {run_dir}")
    print(f"  Output: {output_dir}")
    print(f"  Gateways: {len(gw_names)}")
    print(f"{'='*60}")

    all_results = []
    all_sample_mae = []

    for gw_idx, gw_name in enumerate(gw_names):
        gw_dir = os.path.join(run_dir, gw_name)
        if gw_idx not in per_gw or not os.path.isdir(gw_dir):
            continue
        _, test_samples = per_gw[gw_idx]
        if len(test_samples) < 5:
            continue
        ckpt_path = find_checkpoint(gw_dir, cmd_args.iter)
        if ckpt_path is None:
            print(f"    {gw_name}: no checkpoint, skipped")
            continue

        gaussians, ckpt_iter = load_gaussians(ckpt_path, model_args)
        result, pred_arr, gt_arr, mae_arr = evaluate_gateway(
            gaussians, pipe_args, test_samples, gw_name, gw_idx)
        result["checkpoint"] = ckpt_path
        result["iteration"] = ckpt_iter
        all_results.append(result)
        all_sample_mae.extend(mae_arr.tolist())
        print(f"    {gw_name}: MAE = {result['MAE_dBm']['mean']:.2f} dBm "
              f"(median {result['MAE_dBm']['p50']:.2f})")

        gw_out = os.path.join(output_dir, gw_name)
        os.makedirs(gw_out, exist_ok=True)
        with open(os.path.join(gw_out, "result.json"), 'w') as f:
            json.dump(result, f, indent=2)
        with open(os.path.join(gw_out, "predictions.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["index", "gt_rssi", "pred_rssi", "mae_dBm"])
            for i in range(len(mae_arr)):
                writer.writerow([i, f"{gt_arr[i]:.2f}", f"{pred_arr[i]:.2f}", f"{mae_arr[i]:.2f}"])

        del gaussians
        torch.cuda.empty_cache()

    if not all_results:
        print(f"  No checkpoints found in {run_dir}")
        return

    all_mae_arr = np.array(all_sample_mae)
    per_gw_means = np.array([r['MAE_dBm']['mean'] for r in all_results])

    summary = {
        "num_gateways": len(all_results),
        "num_test_samples_total": len(all_sample_mae),
        "overall_MAE_dBm": {
            "mean": round(float(per_gw_means.mean()), 4),
            "std": round(float(per_gw_means.std()), 4),
            "sample_min": round(float(all_mae_arr.min()), 4),
            "sample_p25": round(float(np.percentile(all_mae_arr, 25)), 4),
            "sample_p50": round(float(np.percentile(all_mae_arr, 50)), 4),
            "sample_p75": round(float(np.percentile(all_mae_arr, 75)), 4),
            "sample_p90": round(float(np.percentile(all_mae_arr, 90)), 4),
            "sample_p95": round(float(np.percentile(all_mae_arr, 95)), 4),
            "sample_max": round(float(all_mae_arr.max()), 4),
        },
        "per_gateway": all_results,
    }
    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "all_predictions.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["gateway", "index", "gt_rssi", "pred_rssi", "mae_dBm"])
        for r in all_results:
            with open(os.path.join(output_dir, r['gateway'], "predictions.csv")) as gf:
                for row in csv.DictReader(gf):
                    writer.writerow([r['gateway'], row['index'],
                                     row['gt_rssi'], row['pred_rssi'], row['mae_dBm']])

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_metric_histogram(all_mae_arr, 'MAE (dBm)', os.path.join(plots_dir, "mae_histogram.png"))
    plot_metric_cdf(all_mae_arr, 'MAE (dBm)', os.path.join(plots_dir, "mae_cdf.png"))

    print(f"\n{'='*60}")
    print(f"  {'Gateway':<10s}  {'MAE Mean':>10s}  {'MAE Med':>10s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}")
    for r in all_results:
        print(f"  {r['gateway']:<10s}  {r['MAE_dBm']['mean']:>10.2f}  {r['MAE_dBm']['p50']:>10.2f}")
    print(f"\n  Overall MAE: {summary['overall_MAE_dBm']['mean']:.2f} ± "
          f"{summary['overall_MAE_dBm']['std']:.2f} dBm")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
