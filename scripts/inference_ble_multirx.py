"""
BLE RSSI Inference for multi-RX single model (with RxConditionedFLE).

Evaluates the latest checkpoint in the model directory (or a specific one with --iter).

Usage:
    python -m scripts.inference_ble_multirx --config arguments/configs/exp_ble_multirx_main.yaml
    python -m scripts.inference_ble_multirx --config ... --iter 50000   # specific checkpoint

"""

import os
import json
import csv
import shutil
import time
from argparse import ArgumentParser

import numpy as np
import torch

from arguments import ModelParams, PipelineParams, load_config
from utils.general_utils import safe_state
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render_rfid_rxcond as render
from scene.ble_dataset import load_ble_per_gateway, amplitude_to_rssi
from utils.rx_conditioning import RxConditionedFLE_PropAware
from utils.data_painter import plot_metric_cdf, plot_metric_histogram


# ---- load checkpoint ----

def load_checkpoint(ckpt_path, model_args, rx_cond_cfg):
    """Load Gaussians and RxConditionedFLE from a checkpoint."""
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
    return gaussians, rx_cond, ckpt_iter


# ---- per-gateway evaluation ----

def evaluate_gateway(
    gaussians,
    rx_cond,
    pipe_args,
    test_samples,
    gw_name,
    gw_idx,
):
    """Evaluate one gateway's test samples using the shared model + rx_cond."""
    all_mae, all_pred, all_gt = [], [], []
    infer_times = []

    with torch.no_grad():
        for viewpoint in test_samples:
            t0 = time.time()
            render_pkg = render(viewpoint, gaussians, pipe_args, rx_cond)
            pred_amp = render_pkg["render"].mean().cpu().item()
            infer_times.append((time.time() - t0) * 1000)

            gt_amp = viewpoint.spectrum.mean().item()
            pred_rssi = amplitude_to_rssi(pred_amp)
            gt_rssi = amplitude_to_rssi(gt_amp)
            mae = abs(pred_rssi - gt_rssi)

            all_mae.append(mae)
            all_pred.append(pred_rssi)
            all_gt.append(gt_rssi)

    mae_arr = np.array(all_mae)
    pred_arr = np.array(all_pred)
    gt_arr = np.array(all_gt)
    infer_arr = np.array(infer_times)

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
        "Infer_ms": {
            "mean": round(float(infer_arr.mean()), 2),
            "std": round(float(infer_arr.std()), 2),
        },
    }
    return result, pred_arr, gt_arr, mae_arr, infer_arr


# ---- evaluate one checkpoint across all gateways ----

def evaluate_checkpoint(
    ckpt_path,
    model_args,
    pipe_args,
    rx_cond_cfg,
    per_gw,
    gw_names,
    output_dir,
):
    """Load one checkpoint and evaluate all gateways. Returns summary dict."""

    gaussians, rx_cond, ckpt_iter = load_checkpoint(ckpt_path, model_args, rx_cond_cfg)

    os.makedirs(output_dir, exist_ok=True)
    all_results = []
    all_sample_mae = []
    all_infer_ms = []

    n_gateways = sum(1 for gw_idx in range(len(gw_names))
                     if gw_idx in per_gw and len(per_gw[gw_idx][1]) >= 5)
    print(f"\n  Checkpoint iter {ckpt_iter}: #G={gaussians.get_xyz.shape[0]}, "
          f"{n_gateways} gateways")

    for gw_idx, gw_name in enumerate(gw_names):
        if gw_idx not in per_gw:
            continue
        _, test_samples = per_gw[gw_idx]
        if len(test_samples) < 5:
            continue

        result, pred_arr, gt_arr, mae_arr, infer_arr = evaluate_gateway(
            gaussians, rx_cond, pipe_args, test_samples, gw_name, gw_idx
        )

        all_results.append(result)
        all_sample_mae.extend(mae_arr.tolist())
        all_infer_ms.extend(infer_arr.tolist())
        print(f"    {gw_name}: MAE = {result['MAE_dBm']['mean']:.2f} dBm "
              f"(median {result['MAE_dBm']['p50']:.2f})")

        gw_output = os.path.join(output_dir, gw_name)
        os.makedirs(gw_output, exist_ok=True)
        with open(os.path.join(gw_output, "result.json"), 'w') as f:
            json.dump(result, f, indent=2)
        with open(os.path.join(gw_output, "predictions.csv"), 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["index", "gt_rssi", "pred_rssi", "mae_dBm"])
            for i in range(len(mae_arr)):
                writer.writerow([i, f"{gt_arr[i]:.2f}", f"{pred_arr[i]:.2f}", f"{mae_arr[i]:.2f}"])

    all_mae_arr = np.array(all_sample_mae)

    all_infer_arr = np.array(all_infer_ms) if all_infer_ms else np.array([0.0])

    # headline mean/std: across per-gateway means
    per_gw_means = np.array([r['MAE_dBm']['mean'] for r in all_results])

    summary = {
        "checkpoint": ckpt_path,
        "iteration": ckpt_iter,
        "num_gateways": len(all_results),
        "num_test_samples_total": len(all_sample_mae),
        "num_gaussians": gaussians.get_xyz.shape[0],
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
        "overall_Infer_ms": {
            "mean": round(float(all_infer_arr.mean()), 2),
            "std": round(float(all_infer_arr.std()), 2),
            "total_samples": len(all_infer_ms),
            "total_time_s": round(float(all_infer_arr.sum() / 1000), 2),
        },
        "per_gateway": all_results,
    }

    with open(os.path.join(output_dir, "summary.json"), 'w') as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "all_predictions.csv"), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["gateway", "index", "gt_rssi", "pred_rssi", "mae_dBm"])
        for r in all_results:
            gw_csv = os.path.join(output_dir, r['gateway'], "predictions.csv")
            if os.path.exists(gw_csv):
                with open(gw_csv) as gf:
                    for row in csv.DictReader(gf):
                        writer.writerow([r['gateway'], row['index'],
                                        row['gt_rssi'], row['pred_rssi'], row['mae_dBm']])

    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plot_metric_histogram(all_mae_arr, 'MAE (dBm)',
                          os.path.join(plots_dir, "mae_histogram.png"))
    plot_metric_cdf(all_mae_arr, 'MAE (dBm)',
                    os.path.join(plots_dir, "mae_cdf.png"))

    return summary


# ---- main ----

def main():
    parser = ArgumentParser(description="BLE RSSI Inference (multi-RX)")
    parser.add_argument("--config", type=str, default="arguments/configs/exp_ble_multirx_main.yaml")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Override model path (default: logs/<dataset>/<exp_name>/)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Override output directory")
    parser.add_argument("--iter", type=int, default=None,
                        help="Evaluate specific iteration only (default: latest checkpoint)")
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
    elif cmd_args.model_path:
        run_dir = cmd_args.model_path
    else:
        run_dir = os.path.join(base_args.log_base_folder, base_args.dataset, base_args.exp_name)

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
    print("  BLE RSSI Inference (multi-RX)")
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

    n_az = getattr(base_args, 'n_azimuth', 360)
    n_el = getattr(base_args, 'n_elevation', 90)
    ratio_train = getattr(base_args, 'ratio_train', 0.8)
    per_gw, gw_names = load_ble_per_gateway(data_dir, ratio_train=ratio_train, seed=random_seed,
                                             n_elevation=n_el, n_azimuth=n_az)

    print(f"  Data: {data_dir}")
    print(f"  Gateways: {len(gw_names)}")
    print(f"{'='*60}")

    model_args = model_cls.extract(base_args)
    model_args.model_path = run_dir
    model_args.source_path = data_dir
    pipe_args = pipe_cls.extract(base_args)
    rx_cond_cfg = (yaml_cfg or {}).get("rx_conditioning", {})

    summary = evaluate_checkpoint(
        ckpt_path, model_args, pipe_args, rx_cond_cfg,
        per_gw, gw_names, output_dir
    )
    mae = summary['overall_MAE_dBm']
    print(f"    Overall MAE: {mae['mean']:.2f} ± {mae['std']:.2f} dBm (sample median {mae['sample_p50']:.2f})")

    print(f"\n{'='*60}")
    print(f"  Per-gateway results (iter {summary['iteration']}):")
    print(f"{'='*60}")
    print(f"  {'Gateway':<20s}  {'MAE Mean':>10s}  {'MAE Med':>10s}  {'MAE P90':>10s}")
    print(f"  {'-'*20}  {'-'*10}  {'-'*10}  {'-'*10}")
    for r in summary['per_gateway']:
        mae = r['MAE_dBm']
        print(f"  {r['gateway']:<20s}  {mae['mean']:>10.2f}  {mae['p50']:>10.2f}  {mae['p90']:>10.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
