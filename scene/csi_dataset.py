"""Dataset loader for KU Leuven distributed multi-RX CSI data.

Format: csidata.npy (N, 8, 26) complex, tx_pos.csv, rx_positions.yml.
"""

import os
import numpy as np
import pandas as pd
import yaml
import torch
from scene.dataset_readers import SpectrumInfo, split_train_test


def load_csi(
    data_dir,
    ratio_train=0.8,
    seed=8371,
    n_elevation=9,
    n_azimuth=36,
):
    """Load multi-RX CSI data with per-RX train/test splits.

    Returns:
        per_rx_data: dict {rx_idx: (train_samples, test_samples)}
        rx_names: list of rx name strings
        csi_max: normalization factor
    """
    csi_raw = np.load(os.path.join(data_dir, 'csidata.npy'))  # (N, 8, 26) complex
    num_samples = csi_raw.shape[0]

    # Normalize by max magnitude
    csi_max = np.abs(csi_raw).max()
    csi_normalized = csi_raw / csi_max
    print(f"\n  CSI normalization: max|CSI|={csi_max:.4f}")

    # Separate real and imaginary
    csi_re = csi_normalized.real.astype(np.float32)
    csi_im = csi_normalized.imag.astype(np.float32)

    # Load TX positions
    tx_pos = pd.read_csv(os.path.join(data_dir, 'tx_pos.csv')).values.astype(np.float32)

    # Load RX positions
    with open(os.path.join(data_dir, 'rx_positions.yml')) as f:
        rx_cfg = yaml.safe_load(f)

    rx_names = [k for k in rx_cfg.keys() if k.startswith('rx')]
    rx_names = sorted(rx_names)
    rx_positions = np.array([rx_cfg[name] for name in rx_names], dtype=np.float32)

    print(f"  TX range: [{tx_pos.min():.4f}, {tx_pos.max():.4f}]")
    print(f"  RX range: [{rx_positions.min():.4f}, {rx_positions.max():.4f}]")

    train_set, test_set = split_train_test(data_dir, num_samples, ratio_train, seed)

    per_rx_data = {}
    for rx_idx, rx_name in enumerate(rx_names):
        rx_pos = rx_positions[rx_idx]
        rx_train, rx_test = [], []

        for sample_idx in range(num_samples):
            # Build spectrum tensor: interleave real and imaginary
            # (26 subcarriers) → (52,) with [re0, im0, re1, im1, ...]
            re = csi_re[sample_idx, rx_idx]  # (26,)
            im = csi_im[sample_idx, rx_idx]  # (26,)
            interleaved = np.empty(52, dtype=np.float32)
            interleaved[0::2] = re
            interleaved[1::2] = im
            # Stored as (52, 1, 1), expanded to (52, H, W) during training
            spectrum = torch.from_numpy(interleaved).reshape(52, 1, 1)

            info = SpectrumInfo(
                T_rx=torch.tensor(rx_pos, dtype=torch.float32),
                T_tx=torch.tensor(tx_pos[sample_idx], dtype=torch.float32),
                spectrum=spectrum,
                spectrum_name=f"{sample_idx + 1:05d}",
                height=n_elevation,
                width=n_azimuth,
            )

            if sample_idx in train_set:
                rx_train.append(info)
            elif sample_idx in test_set:
                rx_test.append(info)

        if rx_train or rx_test:
            per_rx_data[rx_idx] = (rx_train, rx_test)

        print(f"  {rx_name}: train={len(rx_train)}, test={len(rx_test)}")

    return per_rx_data, rx_names, csi_max
