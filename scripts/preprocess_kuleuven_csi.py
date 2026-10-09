"""
Build data/csi from the KU Leuven Ultra-Dense Indoor MaMIMO CSI dataset (DIS_lab_LoS scenario).

Dataset: https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset

The 64 distributed antennas form 8 groups of 8, used as 8 RXs. Per raw sample: phase-calibrate
each antenna, average each group, denoise in the delay domain, and average 100 subcarriers into 26.
Nearby TX positions are then averaged into 6000 output positions.

Usage:
    unzip ultra_dense.zip "ultra_dense/DIS_lab_LoS/*"
    python -m scripts.preprocess_kuleuven_csi --data_dir ultra_dense/DIS_lab_LoS
"""

import os
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm


ANTENNA_GROUPS = {
    'rx01': list(range(0, 8)),
    'rx02': list(range(8, 16)),
    'rx03': list(range(16, 24)),
    'rx04': list(range(24, 32)),
    'rx05': list(range(32, 40)),
    'rx06': list(range(40, 48)),
    'rx07': list(range(48, 56)),
    'rx08': list(range(56, 64)),
}


def denoise_delay_domain(h_freq, n_keep=10):
    h_time = np.fft.ifft(h_freq)
    h_time[n_keep:] = 0
    return np.fft.fft(h_time)


def avg_subcarriers(h_100, n_out=26):
    indices = np.array_split(np.arange(len(h_100)), n_out)
    return np.array([h_100[idx].mean() for idx in indices])


def main():
    parser = ArgumentParser(description="Preprocess KU Leuven CSI (DIS_lab_LoS) into data/csi")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Path to the extracted DIS_lab_LoS folder")
    parser.add_argument("--output_dir", type=str, default="data/csi")
    parser.add_argument("--n_tx", type=int, default=6000,
                        help="Number of output TX positions (0 = keep all)")
    parser.add_argument("--subcarriers", type=int, default=26)
    parser.add_argument("--delay_taps", type=int, default=10,
                        help="Delay taps kept when denoising (of 100)")
    args = parser.parse_args()

    samples_dir = os.path.join(args.data_dir, 'samples')

    print(f"\n{'='*60}")
    print("  KU Leuven CSI Preprocessing")
    print(f"  Data: {args.data_dir}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}\n")

    ant_pos_m = np.load(os.path.join(args.data_dir, 'antenna_positions.npy')) / 1000.0
    rx_positions = {}
    for rx_name, antennas in ANTENNA_GROUPS.items():
        center = ant_pos_m[antennas].mean(axis=0)
        rx_positions[rx_name] = [float(center[0]), float(center[1]), float(center[2])]
        print(f"  {rx_name}: center=[{center[0]:.3f}, {center[1]:.3f}, {center[2]:.3f}]m")
    n_rx = len(rx_positions)

    labels_m = np.load(os.path.join(args.data_dir, 'user_positions.npy')).astype(np.float32) / 1000.0

    sample_files = sorted([f for f in os.listdir(samples_dir) if f.endswith('.npy')])
    all_indices = [int(f.replace('channel_measurement_', '').replace('.npy', ''))
                   for f in sample_files]
    n_total = len(all_indices)
    n_sub = args.subcarriers

    print(f"\nStep 1: Loading {n_total} raw samples "
          f"(keep {args.delay_taps} of 100 delay taps, 100 → {n_sub} subcarriers)")

    csi_all = np.zeros((n_total, n_rx, n_sub), dtype=np.complex64)
    for i, idx in enumerate(tqdm(all_indices, desc="Loading all")):
        h = np.load(os.path.join(samples_dir, f"channel_measurement_{idx:06d}.npy"))
        for rx_i, antennas in enumerate(ANTENNA_GROUPS.values()):
            h_group = h[antennas]
            for a in range(h_group.shape[0]):
                h_group[a] = h_group[a] * np.exp(-1j * np.angle(h_group[a, 0]))
            h_clean = denoise_delay_domain(h_group.mean(axis=0), n_keep=args.delay_taps)
            csi_all[i, rx_i, :] = avg_subcarriers(h_clean, n_sub)

    tx_pos_all = labels_m[all_indices]

    n_tx = args.n_tx if args.n_tx > 0 else n_total
    if n_tx < n_total:
        step = n_total / n_tx
        anchor_idx = [int(i * step) for i in range(n_tx)]
        _, assignments = cKDTree(tx_pos_all[anchor_idx, :2]).query(tx_pos_all[:, :2])

        csi_data = np.zeros((n_tx, n_rx, n_sub), dtype=np.complex64)
        tx_positions = np.zeros((n_tx, 3), dtype=np.float32)
        for g in range(n_tx):
            members = np.where(assignments == g)[0]
            csi_data[g] = csi_all[members].mean(axis=0)
            tx_positions[g] = tx_pos_all[members].mean(axis=0)
        print(f"\nStep 2: Averaged {n_total} → {n_tx} TX positions")
    else:
        csi_data = csi_all
        tx_positions = tx_pos_all

    os.makedirs(args.output_dir, exist_ok=True)
    np.save(os.path.join(args.output_dir, 'csidata.npy'), csi_data)
    pd.DataFrame(tx_positions, columns=['x', 'y', 'z']).to_csv(
        os.path.join(args.output_dir, 'tx_pos.csv'), index=False)
    with open(os.path.join(args.output_dir, 'rx_positions.yml'), 'w') as f:
        for rx_name, pos in rx_positions.items():
            f.write(f"{rx_name}: {[round(pos[0], 2), round(pos[1], 2), round(pos[2], 2)]}\n")
        f.write(f"num_rx: {n_rx}\n")
        f.write("source: KU Leuven Ultra Dense MaMIMO CSI Dataset\n")
        f.write("scenario: DIS_lab_LoS (distributed)\n")
        f.write("frequency: 2610000000.0\n")
        f.write("bandwidth: 20000000.0\n")
        f.write(f"n_subcarriers: {n_sub}\n")
        f.write("antenna_selection: average_group\n")

    print(f"\nSaved csidata.npy {csi_data.shape}, tx_pos.csv, rx_positions.yml to {args.output_dir}\n")


if __name__ == "__main__":
    main()
