#!/bin/bash
# Train + inference for BLE RSSI single-RX (one GSRF model per gateway, no RX conditioning).
#
# Usage:
#   bash run_ble_singlerx.sh [--gpu N] [--config <path>]
#
# Examples:
#   bash run_ble_singlerx.sh
#   bash run_ble_singlerx.sh --gpu 2
#   bash run_ble_singlerx.sh --config arguments/configs/exp_ble_singlerx.yaml --gpu 0

set -e

GPU="0"
CONFIG="arguments/configs/exp_ble_singlerx.yaml"

while [[ $# -gt 0 ]]; do
    case $1 in
        --config) CONFIG="$2"; shift 2 ;;
        --gpu)    GPU="$2"; shift 2 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

if [ ! -f "$CONFIG" ]; then echo "Error: Config not found: $CONFIG"; exit 1; fi

export CUDA_VISIBLE_DEVICES="$GPU"

# ---- Training ----
echo "========================================"
echo "  Training (single-RX)"
echo "========================================"
PYTHONUNBUFFERED=1 python -m scripts.train_ble_singlerx --config "$CONFIG"

# ---- Inference ----
echo ""
echo "========================================"
echo "  Inference (single-RX)"
echo "========================================"
PYTHONUNBUFFERED=1 python -m scripts.inference_ble_singlerx --config "$CONFIG"
