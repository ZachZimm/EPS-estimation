#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./venv/bin/python}"
LOG_DIR="${LOG_DIR:-artifacts/run_logs}"
REBUILD_DATASETS="${REBUILD_DATASETS:-0}"
mkdir -p "$LOG_DIR"

DATASET_RUNS=(
  "configs/expanded_run_macro_true_linear_config.json artifacts/expanded_run_macro_true_linear_dataset"
  "configs/phase_two_sector_true_linear_config.json artifacts/phase_two_sector_true_linear_dataset"
  "configs/expanded_run_macro_voltrim_true_linear_config.json artifacts/expanded_run_macro_voltrim_true_linear_dataset"
  "configs/phase_two_sector_voltrim_true_linear_config.json artifacts/phase_two_sector_voltrim_true_linear_dataset"
)

TRAINING_RUNS=(
  "configs/expanded_run_macro_true_linear_config.json artifacts/expanded_run_macro_true_linear_dataset artifacts/expanded_run_macro_true_linear_training"
  "configs/phase_two_sector_true_linear_config.json artifacts/phase_two_sector_true_linear_dataset artifacts/phase_two_sector_true_linear_training"
  "configs/expanded_run_macro_voltrim_true_linear_config.json artifacts/expanded_run_macro_voltrim_true_linear_dataset artifacts/expanded_run_macro_voltrim_true_linear_training"
  "configs/phase_two_sector_voltrim_true_linear_config.json artifacts/phase_two_sector_voltrim_true_linear_dataset artifacts/phase_two_sector_voltrim_true_linear_training"
  "configs/expanded_run_macro_true_linear_quantile_config.json artifacts/expanded_run_macro_true_linear_dataset artifacts/expanded_run_macro_true_linear_quantile_training"
  "configs/phase_two_sector_true_linear_quantile_config.json artifacts/phase_two_sector_true_linear_dataset artifacts/phase_two_sector_true_linear_quantile_training"
  "configs/expanded_run_macro_voltrim_true_linear_quantile_config.json artifacts/expanded_run_macro_voltrim_true_linear_dataset artifacts/expanded_run_macro_voltrim_true_linear_quantile_training"
  "configs/phase_two_sector_voltrim_true_linear_quantile_config.json artifacts/phase_two_sector_voltrim_true_linear_dataset artifacts/phase_two_sector_voltrim_true_linear_quantile_training"
)

run_dataset() {
  local config="$1"
  local dataset_dir="$2"
  local name ts
  name="$(basename "$config" .json)"
  ts="$(date +%Y%m%d_%H%M%S)"
  if [[ "$REBUILD_DATASETS" != "1" && -f "$dataset_dir/dataset_arrays.npz" && -f "$dataset_dir/event_metadata.csv" ]]; then
    echo "==> [$name] reusing existing dataset $dataset_dir"
    return
  fi
  echo "==> [$name] building dataset -> $dataset_dir"
  "$PYTHON_BIN" dataset_builder.py \
    --config "$config" \
    --output-dir "$dataset_dir" \
    2>&1 | tee "$LOG_DIR/${name}_dataset_${ts}.log"
}

run_training() {
  local config="$1"
  local dataset_dir="$2"
  local training_dir="$3"
  local name ts
  name="$(basename "$config" .json)"
  ts="$(date +%Y%m%d_%H%M%S)"
  echo "==> [$name] training baseline -> $training_dir"
  "$PYTHON_BIN" train_linear_baseline.py \
    --config "$config" \
    --dataset-dir "$dataset_dir" \
    --output-dir "$training_dir" \
    2>&1 | tee "$LOG_DIR/${name}_training_${ts}.log"
}

for entry in "${DATASET_RUNS[@]}"; do
  # shellcheck disable=SC2086
  run_dataset $entry
  echo
done

for entry in "${TRAINING_RUNS[@]}"; do
  # shellcheck disable=SC2086
  run_training $entry
  echo
done

echo "All true linear baseline runs completed."
