# Configuration

`src/shred_mpc/config.py` preserves the inspected defaults. The temporal CLI exposes data/template/output paths, seeds, sensors, device, epochs, scenario sizes and training count; the resolved dataclass is saved in each manifest. Change defaults only for a new experiment; do not relabel historical archives.

`--dry-run` is always safe from training/CFD. Execution needs `--execute`. Outputs live in new directories under `runs/`; resume uses saved settings. Sensor IDs refer to cell ordering, not wall taps. See `docs/method.md`.
