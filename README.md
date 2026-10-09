# RxGS: Receiver-Generalizable 3D Gaussian Splatting for Radio-Frequency Data Synthesis

**NeurIPS 2026**

[![Paper](https://img.shields.io/badge/arXiv-2605.24290-b31b1b.svg)](https://arxiv.org/abs/2605.24290)
[![Dataset](https://img.shields.io/badge/🤗%20Dataset-RxGS--data-yellow)](https://huggingface.co/datasets/kyang73/RxGS-data)
[![Models](https://img.shields.io/badge/🤗%20Models-RxGS--pretrained-yellow)](https://huggingface.co/kyang73/RxGS-pretrained)

Kang Yang, Mani Srivastava — University of California, Los Angeles

RxGS synthesizes RF data (RSSI, spatial spectrum, CSI) at any transmitter and any receiver in a scene with a single model. Stage I learns receiver-independent 3D Gaussian geometry; Stage II freezes it and learns directional radiance conditioned on the receiver position through a global and a local conditioning branch.

This repository contains the code to train and evaluate RxGS on the three datasets in the paper.

## Setup

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install \
    -e submodules/simple-knn \
    -e submodules/complex-gaussian-tracer \
    -e submodules/complex-gaussian-tracer-csi \
    -e submodules/complex-gaussian-tracer-multirx \
    -e submodules/fused-ssim
pip install tqdm plyfile matplotlib lpips pyyaml pandas scipy imageio "numpy<2"
```

## Data

The spatial spectrum dataset is on [🤗 kyang73/RxGS-data](https://huggingface.co/datasets/kyang73/RxGS-data) (`pip install -U huggingface_hub` provides the `hf` command):

```bash
hf download kyang73/RxGS-data spectrum_multirx.tar --repo-type dataset --local-dir data
tar -xf data/spectrum_multirx.tar -C data && rm data/spectrum_multirx.tar
```

This gives `data/spectrum_multirx/` — 90×360 spatial spectra (5,089 TXs × 21 RXs, simulated with Sionna RT).

The BLE RSSI dataset (6,000 TXs × 21 gateways) comes from [NeRF²](https://github.com/XPengZhao/NeRF2): download `BLE/rssi-dataset-1.tar.gz` from the NeRF² dataset link and place its `gateway_rssi.csv`, `tx_pos.csv` and `gateway_position.yml` under `data/ble_rssi/`.

The CSI dataset is built from the [KU Leuven Ultra-Dense Indoor MaMIMO CSI dataset](https://ieee-dataport.org/open-access/ultra-dense-indoor-mamimo-csi-dataset) (free IEEE DataPort account required). Download `ultra_dense.zip`, extract the distributed-antenna scenario, and preprocess it:

```bash
unzip ultra_dense.zip "ultra_dense/DIS_lab_LoS/*"
python -m scripts.preprocess_kuleuven_csi --data_dir ultra_dense/DIS_lab_LoS
```

This writes `data/csi/` — 52-channel complex CSI (6,000 TXs × 8 distributed RXs).

## Pretrained models

Checkpoints for the three RxGS models are on [🤗 kyang73/RxGS-pretrained](https://huggingface.co/kyang73/RxGS-pretrained):

| File                   | Dataset          | Iterations (Stage I + II) | Config                           |
|------------------------|------------------|---------------------------|----------------------------------|
| `ble_rssi.pth`         | BLE RSSI         | 30k + 100k                | `exp_ble_multirx_main.yaml`      |
| `spectrum_multirx.pth` | Spatial spectrum | 30k + 60k                 | `exp_spectrum_multirx_main.yaml` |
| `csi.pth`              | WiFi CSI         | 30k + 100k                | `exp_csi_multirx_main.yaml`      |

To evaluate them without training:

```bash
hf download kyang73/RxGS-pretrained --include "*.pth" --local-dir pretrained
python -m scripts.inference_ble_multirx      --config arguments/configs/exp_ble_multirx_main.yaml      --checkpoint pretrained/ble_rssi.pth
python -m scripts.inference_spectrum_multirx --config arguments/configs/exp_spectrum_multirx_main.yaml --checkpoint pretrained/spectrum_multirx.pth
python -m scripts.inference_csi_multirx      --config arguments/configs/exp_csi_multirx_main.yaml      --checkpoint pretrained/csi.pth
```

## Run

Each modality has one training + inference wrapper. All wrappers accept `--gpu N` and `--config <path>`. The Python scripts they call live in `scripts/` and run from the repository root as modules (e.g. `python -m scripts.train_ble_multirx --config arguments/configs/exp_ble_multirx_main.yaml`). Inference uses the latest checkpoint by default (`--iter N` on the inference script to override).

```bash
bash run_ble_multirx.sh                --gpu 0
bash run_spectrum_multirx.sh           --gpu 0
bash run_csi_multirx.sh                --gpu 0
```

Stage I geometry (Phase 1 in the code) is saved to a **dataset-level shared** path (`logs/<dataset>/geometry.pth`), and later runs on the same dataset reuse it (pass `--retrain_geometry` to train it again). Stage II (Phase 2) saves two checkpoints per run: `chkpnt{film_iters/2}.pth` (halfway) and `chkpnt{film_iters}.pth` (final).

Each script writes outputs to:
- `logs/<dataset>/<exp_name>/` — training checkpoints and config
- `logs/<dataset>/<exp_name>/inference/` — inference outputs

## Single-RX (GSRF)

The single-RX scripts train one [GSRF](https://github.com/nesl/GSRF) model per receiver without receiver conditioning:

```bash
bash run_ble_singlerx.sh               --gpu 0
bash run_spectrum_singlerx.sh          --gpu 0
bash run_csi_singlerx.sh               --gpu 0
```

Each receiver's model is saved to `logs/<dataset>/singlerx_main/<rx>/`. `scripts/train_spectrum_singlerx.py` and `scripts/train_csi_singlerx.py` accept `--rx_idx N` to train a single receiver (e.g. `python -m scripts.train_spectrum_singlerx --rx_idx 0`). For BLE, the first gateway is trained in full and the remaining gateways reuse its geometry and train only their FLE coefficients.

## Citation

```bibtex
@inproceedings{yang2026rxgs,
  title     = {RxGS: Receiver-Generalizable 3D Gaussian Splatting for Radio-Frequency Data Synthesis},
  author    = {Yang, Kang and Srivastava, Mani},
  booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
  year      = {2026}
}
```

## Acknowledgments

RxGS builds on [GSRF](https://github.com/nesl/GSRF) and [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting). The BLE RSSI data comes from [NeRF²](https://github.com/XPengZhao/NeRF2), the CSI data from the [KU Leuven Ultra-Dense Indoor MaMIMO CSI dataset](https://dx.doi.org/10.21227/nr6k-8r78), and the spatial spectrum data was simulated with [Sionna RT](https://github.com/NVlabs/sionna).

## License

This code is released under the [BSD 3-Clause License](LICENSE). The CUDA rasterizer submodules (`submodules/simple-knn`, `submodules/complex-gaussian-tracer*`) are derived from 3D Gaussian Splatting and remain subject to the Gaussian-Splatting License in their `LICENSE.md` files, which permits non-commercial research and evaluation use only. `submodules/fused-ssim` is [fused-ssim](https://github.com/rahul-goel/fused-ssim) under its MIT License. The spatial spectrum dataset on 🤗 RxGS-data is released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
