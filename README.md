# EPS Estimation

Prototype pipeline for building EPS prediction datasets and training and evaluating multiple model families against quarterly EPS announcement targets.

## What is in this repo

- `database_helper.py`: loads OHLC and financial statement data from the PostgreSQL databases used by `kairos`
- `dataset_builder.py`: builds leak-free event datasets aligned to estimated EPS publication dates
- `train_transformer.py`: trains the main neural models
- `train_linear_baseline.py`: trains the true linear baselines
- `analyze_learnable_space.py`: embedding and feature-target analysis utilities
- `viewer_server/` and `viewer_frontend/`: interactive run viewer
- `configs/`: experiment and run configuration files
- `artifacts/`: generated datasets, training outputs, analysis outputs, and logs

## Environment

This repo expects:

- a local `.env` with the same database connection settings used by `kairos`
- the project virtualenv at `./venv`
- Python dependencies installed into that venv

Typical commands use:

```bash
./venv/bin/python ...
```

## Configs

All configs now live under [`configs/`](/home/zach/wd7/projects/eps-estimation/configs).

Main families:

- expanded global runs: [`configs/expanded_run_macro_config.json`](/home/zach/wd7/projects/eps-estimation/configs/expanded_run_macro_config.json)
- sector runs: [`configs/phase_two_sector_config.json`](/home/zach/wd7/projects/eps-estimation/configs/phase_two_sector_config.json)
- volatility-trimmed runs: [`configs/expanded_run_macro_voltrim_config.json`](/home/zach/wd7/projects/eps-estimation/configs/expanded_run_macro_voltrim_config.json), [`configs/phase_two_sector_voltrim_config.json`](/home/zach/wd7/projects/eps-estimation/configs/phase_two_sector_voltrim_config.json)
- LSTM runs: `configs/*lstm*.json`
- quantile runs: `configs/*quantile*.json`
- true linear baselines: `configs/*true_linear*.json`
- older sweep configs: [`configs/experiments/`](/home/zach/wd7/projects/eps-estimation/configs/experiments)

## Build a dataset

Example:

```bash
./venv/bin/python dataset_builder.py   --config configs/expanded_run_macro_config.json   --output-dir artifacts/expanded_run_macro_dataset
```

## Train models

Neural model example:

```bash
./venv/bin/python train_transformer.py   --config configs/expanded_run_macro_config.json   --dataset-dir artifacts/expanded_run_macro_dataset   --output-dir artifacts/expanded_run_macro_training
```

True linear baseline example:

```bash
./venv/bin/python train_linear_baseline.py   --config configs/expanded_run_macro_true_linear_config.json   --dataset-dir artifacts/expanded_run_macro_true_linear_dataset   --output-dir artifacts/expanded_run_macro_true_linear_training
```

## Batch runners

Sequential suites:

- [`run_quantile_suite.sh`](/home/zach/wd7/projects/eps-estimation/run_quantile_suite.sh)
- [`run_lstm_suite.sh`](/home/zach/wd7/projects/eps-estimation/run_lstm_suite.sh)
- [`run_linear_suite.sh`](/home/zach/wd7/projects/eps-estimation/run_linear_suite.sh)
- [`run_true_linear_suite.sh`](/home/zach/wd7/projects/eps-estimation/run_true_linear_suite.sh)

Run one with:

```bash
./run_lstm_suite.sh
```

## Viewer

Start both backend and frontend with:

```bash
./start_viewer.sh
```

Default ports:

- backend: `8100`
- frontend: `4715`

The viewer indexes runs under `artifacts/*_training`, including per-sector aggregate runs and their sector breakdowns.

## Learnable-space analysis

Example:

```bash
./venv/bin/python analyze_learnable_space.py   --dataset-dir artifacts/phase_two_sector_dataset   --training-dir artifacts/phase_two_sector_training   --output-dir artifacts/learnable_space_analysis
```

## Notes

- The model targets are event-level quarterly EPS announcements, not arbitrary daily rows.
- Publication-date alignment is approximate and intentionally leak-averse.
- Current learned models should still be evaluated against simple historical baselines such as persistence and trailing mean.
