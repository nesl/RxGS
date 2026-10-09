#!/bin/bash
# Train + inference for BLE RSSI multi-RX (single conditioned model across all gateways).
#
# Usage:
#   bash run_ble_multirx.sh [--gpu N] [--config <path>]
#
# Examples:
#   bash run_ble_multirx.sh
#   bash run_ble_multirx.sh --gpu 2
#   bash run_ble_multirx.sh --config arguments/configs/exp_ble_multirx_main.yaml --gpu 0

set -e

GPU="0"
CONFIG="arguments/configs/exp_ble_multirx_main.yaml"

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
echo "  Training (multi-RX)"
echo "========================================"
PYTHONUNBUFFERED=1 python -m scripts.train_ble_multirx --config "$CONFIG"

# ---- Inference ----
echo ""
echo "========================================"
echo "  Inference (multi-RX)"
echo "========================================"
PYTHONUNBUFFERED=1 python -m scripts.inference_ble_multirx \
    --config "$CONFIG"
