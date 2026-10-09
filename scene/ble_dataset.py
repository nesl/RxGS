import os
import numpy as np
import pandas as pd
import yaml
import torch
from scene.dataset_readers import SpectrumInfo, split_train_test


# ---- RSSI <-> amplitude conversion ----
def rssi_to_amplitude(rssi, floor=-100.0):
    return 1.0 - (rssi / floor)


def amplitude_to_rssi(amplitude, floor=-100.0):
    return floor * (1.0 - amplitude)


# ---- BLE spectrum construction ----
def make_ble_spectrum_info(
    tx_pos,
    gw_pos,
    rssi_val,
    sample_idx,
    n_elevation=9,
    n_azimuth=36,
):
    amplitude = rssi_to_amplitude(rssi_val)

    n_el, n_az = n_elevation, n_azimuth
    spectrum = torch.full((n_el, n_az), amplitude, dtype=torch.float32)

    return SpectrumInfo(
        T_rx=torch.tensor(gw_pos, dtype=torch.float32),
        T_tx=torch.tensor(tx_pos, dtype=torch.float32),
        spectrum=spectrum,
        spectrum_name=f"{sample_idx+1:05d}",
        height=n_el,
        width=n_az,
    )


# ---- per-gateway data loading and splitting ----
def load_ble_per_gateway(
    data_dir,
    ratio_train=0.8,
    seed=8371,
    n_elevation=9,
    n_azimuth=36,
):
    rssi_df = pd.read_csv(os.path.join(data_dir, 'gateway_rssi.csv'))
    rssi_values = rssi_df.values.astype(np.float32)
    gateway_names = list(rssi_df.columns)
    num_gateways = len(gateway_names)

    with open(os.path.join(data_dir, 'gateway_position.yml')) as f:
        gw_dict = yaml.safe_load(f)
    gateway_positions = np.array([gw_dict[name] for name in gateway_names], dtype=np.float32)

    tx_pos = pd.read_csv(os.path.join(data_dir, 'tx_pos.csv')).values.astype(np.float32)
    num_samples = tx_pos.shape[0]

    train_set, test_set = split_train_test(data_dir, num_samples, ratio_train, seed)

    per_gw_data = {}

    for gw_idx in range(num_gateways):
        gw_pos = gateway_positions[gw_idx]
        gw_train = []
        gw_test = []

        for sample_idx in range(num_samples):
            rssi_val = rssi_values[sample_idx, gw_idx]
            if rssi_val <= -100.0:
                continue

            info = make_ble_spectrum_info(
                tx_pos[sample_idx], gw_pos, rssi_val, sample_idx,
                n_elevation=n_elevation, n_azimuth=n_azimuth
            )

            if sample_idx in train_set:
                gw_train.append(info)
            elif sample_idx in test_set:
                gw_test.append(info)

        if gw_train or gw_test:
            per_gw_data[gw_idx] = (gw_train, gw_test)

        print(f"  {gateway_names[gw_idx]}: train={len(gw_train)}, test={len(gw_test)}")

    return per_gw_data, gateway_names


def load_ble_all_gateways(
    data_dir,
    ratio_train=0.8,
    seed=8371,
    n_elevation=9,
    n_azimuth=36,
):
    """Load BLE data with all gateways mixed into single train/test sets.

    Returns:
        train_samples: list of SpectrumInfo (all gateways mixed)
        test_samples:  list of SpectrumInfo (all gateways mixed)
        gateway_names: list of gateway name strings
        gateway_positions: (num_gw, 3) numpy array
    """
    per_gw_data, gateway_names = load_ble_per_gateway(
        data_dir, ratio_train=ratio_train,
        seed=seed, n_elevation=n_elevation, n_azimuth=n_azimuth,
    )

    with open(os.path.join(data_dir, 'gateway_position.yml')) as f:
        gw_dict = yaml.safe_load(f)
    gateway_positions = np.array([gw_dict[name] for name in gateway_names], dtype=np.float32)

    train_all, test_all = [], []
    for gw_idx in sorted(per_gw_data.keys()):
        gw_train, gw_test = per_gw_data[gw_idx]
        train_all.extend(gw_train)
        test_all.extend(gw_test)

    print(f"\n  [Multi-RX] Total train: {len(train_all)}, test: {len(test_all)} "
          f"across {len(per_gw_data)} gateways\n")

    return train_all, test_all, gateway_names, gateway_positions
