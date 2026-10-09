import os
import numpy as np
import pandas as pd
import yaml
import torch
import imageio
from scene.dataset_readers import SpectrumInfo, split_train_test


def load_spectrum_per_rx(
    data_dir,
    ratio_train=0.8,
    seed=8371,
):
    """Per-RX train/test splits. Expects rx_positions.yml, tx_pos.csv, spectrum/<tx>_<rx>.png.
    Returns ({rx_idx: (train, test)}, rx_names)."""
    with open(os.path.join(data_dir, 'rx_positions.yml')) as f:
        rx_dict = yaml.safe_load(f)['rx_positions']

    rx_names = sorted(rx_dict.keys())
    rx_positions = np.array([rx_dict[name] for name in rx_names], dtype=np.float32)

    tx_pos = pd.read_csv(os.path.join(data_dir, 'tx_pos.csv')).values.astype(np.float32)
    num_samples = tx_pos.shape[0]

    train_set, test_set = split_train_test(data_dir, num_samples, ratio_train, seed)

    spectrum_dir = os.path.join(data_dir, 'spectrum')

    per_rx_data = {}

    for rx_idx, rx_name in enumerate(rx_names):
        rx_pos = rx_positions[rx_idx]
        rx_train, rx_test = [], []

        for sample_idx in range(num_samples):
            fname = f"{sample_idx + 1:05d}_{rx_name}.png"
            fpath = os.path.join(spectrum_dir, fname)
            if not os.path.exists(fpath):
                continue

            image = imageio.imread(fpath).astype(np.float32) / 255.0
            spectrum = torch.from_numpy(image).float()
            height, width = image.shape[0], image.shape[1]

            info = SpectrumInfo(
                T_rx=torch.tensor(rx_pos, dtype=torch.float32),
                T_tx=torch.tensor(tx_pos[sample_idx], dtype=torch.float32),
                spectrum=spectrum,
                spectrum_name=f"{sample_idx + 1:05d}",
                height=height,
                width=width,
            )

            if sample_idx in train_set:
                rx_train.append(info)
            elif sample_idx in test_set:
                rx_test.append(info)

        if rx_train or rx_test:
            per_rx_data[rx_idx] = (rx_train, rx_test)

        print(f"  {rx_name}: train={len(rx_train)}, test={len(rx_test)}")

    return per_rx_data, rx_names


def load_spectrum_all_rx(
    data_dir,
    ratio_train=0.8,
    seed=8371,
):
    """All RXs mixed into single train/test sets. Returns (train, test, rx_names, rx_positions)."""
    per_rx_data, rx_names = load_spectrum_per_rx(
        data_dir, ratio_train=ratio_train, seed=seed,
    )

    with open(os.path.join(data_dir, 'rx_positions.yml')) as f:
        rx_dict = yaml.safe_load(f)['rx_positions']
    rx_positions = np.array([rx_dict[name] for name in rx_names], dtype=np.float32)

    train_all, test_all = [], []
    for rx_idx in sorted(per_rx_data.keys()):
        rx_train, rx_test = per_rx_data[rx_idx]
        train_all.extend(rx_train)
        test_all.extend(rx_test)

    print(f"\n  [Multi-RX Spectrum] Total train: {len(train_all)}, test: {len(test_all)} "
          f"across {len(per_rx_data)} RXs\n")

    return train_all, test_all, rx_names, rx_positions
