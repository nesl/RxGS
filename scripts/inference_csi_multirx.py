"""
CSI Inference for distributed multi-RX model.

Usage:
    python -m scripts.inference_csi_multirx --config arguments/configs/exp_csi_multirx_main.yaml
"""

import os
import json
import shutil
import time
from argparse import ArgumentParser

import numpy as np
import torch

from arguments import ModelParams, PipelineParams, load_config
from utils.general_utils import safe_state
from scene.gaussian_model import GaussianModel
from gaussian_renderer.render_csi import render_csi_rxcond as render
from scene.csi_dataset import load_csi
from utils.rx_conditioning import RxConditionedFLE_PropAware


def load_checkpoint(ckpt_path, model_args, rx_cond_cfg):
    """Load Gaussians and RxCond from checkpoint."""
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
        n_fle_components=n_fle, max_fle_degree=fle_degree,
        d_model=rx_cond_cfg.get('d_model', 64),
        n_freqs=rx_cond_cfg.get('n_freqs', 6),
        max_freq_log2=rx_cond_cfg.get('max_freq_log2', 5.0),
        comp_dim=rx_cond_cfg.get('comp_dim', 16),
        n_probe_samples=rx_cond_cfg.get('n_probe_samples', 16),
        num_channels=52,
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


def evaluate_rx(
    gaussians,
    rx_cond,
    pipe_args,
    test_samples,
    rx_name,
    rx_idx,
    csi_max,
    n_azimuth,
    n_elevation,
):
    """Evaluate one RX's test samples. Returns per-sample SNR."""
    all_snr = []
    all_pred_re, all_pred_im = [], []
    all_gt_re, all_gt_im = [], []
    infer_times = []

    with torch.no_grad():
        for viewpoint in test_samples:
            t0 = time.time()
            render_pkg = render(viewpoint, gaussians, pipe_args, rx_cond,
                                n_azimuth=n_azimuth, n_elevation=n_elevation)
            rendered = render_pkg["render"]
            pred_csi = rendered.mean(dim=(1, 2))  # (52,)
            infer_times.append((time.time() - t0) * 1000)

            gt_csi = viewpoint.spectrum.cuda().mean(dim=(1, 2))  # (52,)

            pred_re = pred_csi[0::2].cpu().numpy()  # (26,)
            pred_im = pred_csi[1::2].cpu().numpy()
            gt_re = gt_csi[0::2].cpu().numpy()
            gt_im = gt_csi[1::2].cpu().numpy()

            pred_re_dn = pred_re * csi_max
            pred_im_dn = pred_im * csi_max
            gt_re_dn = gt_re * csi_max
            gt_im_dn = gt_im * csi_max

            err = ((pred_re_dn - gt_re_dn)**2 + (pred_im_dn - gt_im_dn)**2).sum()
            gt_pwr = (gt_re_dn**2 + gt_im_dn**2).sum()
            snr = -10 * np.log10(err / (gt_pwr + 1e-8) + 1e-10)

            all_snr.append(snr)
            all_pred_re.append(pred_re)
            all_pred_im.append(pred_im)
            all_gt_re.append(gt_re)
            all_gt_im.append(gt_im)

    snr_arr = np.array(all_snr)
    infer_arr = np.array(infer_times)

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
        "Infer_ms": {
            "mean": round(float(infer_arr.mean()), 2),
            "std": round(float(infer_arr.std()), 2),
        },
    }

    pred = np.array(all_pred_re) + 1j * np.array(all_pred_im)
    gt = np.array(all_gt_re) + 1j * np.array(all_gt_im)

    return result, snr_arr, pred, gt, infer_arr


def evaluate_checkpoint(
    ckpt_path,
    model_args,
    pipe_args,
    rx_cond_cfg,
    per_rx,
    rx_names,
    csi_max,
    n_azimuth,
    n_elevation,
    output_dir,
):
    """Evaluate one checkpoint across all RXs."""
    gaussians, rx_cond, ckpt_iter = load_checkpoint(ckpt_path, model_args, rx_cond_cfg)
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    total_err, total_gt = 0.0, 0.0

    for rx_idx, rx_name in enumerate(rx_names):
        if rx_idx not in per_rx:
            continue
        _, test_samples = per_rx[rx_idx]
        if len(test_samples) < 5:
            continue

        result, snr_arr, pred, gt, infer_arr = evaluate_rx(
            gaussians, rx_cond, pipe_args, test_samples, rx_name, rx_idx,
            csi_max, n_azimuth, n_elevation
        )
        all_results.append(result)
        print(f"    {rx_name}: SNR = {result['SNR_dB']['mean']:.2f} dB "
              f"(median {result['SNR_dB']['p50']:.2f})")

        rx_dir = os.path.join(output_dir, rx_name)
        os.makedirs(rx_dir, exist_ok=True)
        with open(os.path.join(rx_dir, "result.json"), 'w') as f:
            json.dump(result, f, indent=2)
        np.savez(os.path.join(rx_dir, "csi_results.npz"),
                 pred=pred, gt=gt, snr_db=snr_arr)

        pred_dn = pred * csi_max
        gt_dn = gt * csi_max
        total_err += (np.abs(pred_dn - gt_dn)**2).sum()
        total_gt += (np.abs(gt_dn)**2).sum()

    # Joint SNR: pools |err|²/|gt|² over all samples.
    joint_snr = -10 * np.log10(total_err / (total_gt + 1e-8) + 1e-10)

    # headline mean/std: across per-RX SNR means
    per_rx_snr = np.array([r['SNR_dB']['mean'] for r in all_results])

    summary = {
        "checkpoint": ckpt_path,
        "iteration": ckpt_iter,
        "num_rx": len(all_results),
        "num_test": sum(r['num_test'] for r in all_results),
        "num_gaussians": gaussians.get_xyz.shape[0],
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

    return summary


def main():
    parser = ArgumentParser(description="CSI Inference (distributed multi-RX)")
    parser.add_argument("--config", type=str, default="arguments/configs/exp_csi_multirx_main.yaml")
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--iter", type=int, default=None)
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

    n_azimuth = getattr(base_args, 'n_azimuth', 36)
    n_elevation = getattr(base_args, 'n_elevation', 9)

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

    ratio_train = getattr(base_args, 'ratio_train', 0.8)
    per_rx, rx_names, csi_max = load_csi(
        data_dir, ratio_train=ratio_train,
        seed=random_seed, n_elevation=n_elevation, n_azimuth=n_azimuth)

    model_args = model_cls.extract(base_args)
    model_args.model_path = run_dir
    model_args.source_path = data_dir
    model_args.num_channels_override = 52
    pipe_args = pipe_cls.extract(base_args)
    rx_cond_cfg = (yaml_cfg or {}).get("rx_conditioning", {})

    print(f"\n{'='*60}")
    print("  CSI Inference (distributed multi-RX)")
    print(f"  Model: {run_dir}")
    print(f"  Output: {output_dir}")
    print(f"  RXs: {len(rx_names)}")
    print(f"  Checkpoint: {ckpt_path}")
    print(f"  CSI max (denorm): {csi_max:.4f}")
    print(f"{'='*60}")

    summary = evaluate_checkpoint(
        ckpt_path, model_args, pipe_args, rx_cond_cfg,
        per_rx, rx_names, csi_max, n_azimuth, n_elevation,
        output_dir
    )
    snr = summary['overall_SNR_dB']
    print(f"    Overall SNR: {snr['mean']:.2f} ± {snr['std']:.2f} dB  "
          f"(joint: {summary['joint_SNR_dB']['mean']:.2f} dB)")

    print(f"\n{'='*60}")
    print(f"  Per-RX results (iter {summary['iteration']}):")
    print(f"{'='*60}")
    print(f"  {'RX':<10s}  {'SNR Mean':>10s}  {'SNR Med':>10s}")
    print(f"  {'-'*10}  {'-'*10}  {'-'*10}")
    for r in summary['per_rx']:
        print(f"  {r['rx']:<10s}  {r['SNR_dB']['mean']:>10.2f}  {r['SNR_dB']['p50']:>10.2f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
