# Running the workflows

Run from the clean repository root. Use new run paths; commands refuse existing outputs except explicit temporal resume. Never point an output path at the original dataset or a live simulation. Importing the maintained package performs no training, CFD or directory creation.

## Dependencies and resources

Verified local environment: Python 3.12.3; NumPy 1.26.4; Matplotlib 3.10.8; pandas 2.3.3; SciPy 1.16.3; scikit-learn 1.8.0; Torch 2.9.1; PySHRED 1.0.21. These are observed pins, not universal minimum versions. `pyproject.toml` declares base, demo and research extras. Python ≥3.10 is declared; only 3.12.3 was exercised here. A clean internet installation of every research dependency was not attempted.

OpenFOAM Foundation 13 is observed, using `pimpleFoam` → `foamRun -solver incompressibleFluid`. Required CFD commands: `blockMesh`, `pimpleFoam`, `foamToVTK`, `pvpython` (ParaView 5.11.2). Load their system environment before execution. ffmpeg is optional for the animation. No CFD is needed for the result demo.

For 500 × 451 × 3,600 pressure values, a float32 array alone takes about 3.25 GB, before preprocessing, SVD, tensors and copies. The original source workspace is about 135 GB including its environment and outputs. No full-campaign runtime, GPU requirement or speedup has been measured here. Start with a separately labelled pilot, monitor memory/disk and preserve complete provenance.

## Dataset controls, CFD and export

```bash
shred-dataset plan --run-dir runs/data-pilot --cases 10 --seed 42
# EXPENSIVE: explicit solver execution in fresh owned cases
shred-dataset simulate --run-dir runs/data-pilot --execute
# Export Cp and coordinates from complete generated references
shred-dataset export --run-dir runs/data-pilot
```

Planning copies `cfd/templates/training`, records settings and creates seeded controls. Simulation writes compressed references and reuses completed ones. Export refuses an existing `CSV_files` directory. The export declares `p` to be normalized Cp. The default 500-case plan interleaves 200 ramp, 150 ramp/hold, 100 varied and 50 extended-hold histories; a ten-case pilot is debugging, not a full study. The numerical setup remains steady-state; changing it is a separate scientific experiment.

To inspect current private training exports, provide their CSV path explicitly. All 500 case directories had 451 snapshots/control rows at audit time; full numeric validation of all fields was not performed. The temporal loader checks selected cases, table timestamps, finite values and coordinate consistency. CSV sample filenames alone cannot establish missing historical physical timestamps.

## Training without evaluation CFD

```bash
shred-temporal --dry-run --run-dir runs/model-pilot   --dataset runs/data-pilot/CSV_files --dataset-limit 10 --epochs 5 --replicates 1
# EXPENSIVE: fit and save model plus transforms, without evaluation CFD
shred-temporal --resume runs/model-pilot --train-only --execute
```

Training uses the existing PySHRED architecture and sequential trajectory split, and stores weights, preprocessing transforms and training seed. Use only trusted locally generated transform pickle files. Current settings and source hashes are stored in the manifest. The dataset fingerprint is an operational path/size/mtime guard plus control-table bytes, not a cryptographic content hash of every pressure snapshot.

## Temporal and parametric forward evaluation

```bash
shred-temporal --dry-run --run-dir runs/temporal-pilot   --dataset /path/to/complete/CSV_files --dataset-limit 10 --epochs 5 --replicates 1
# EXPENSIVE: train/reload and simulate four matched reference scenarios
shred-temporal --resume runs/temporal-pilot --execute
# Optional new training seeds; reuses CFD references
shred-temporal --resume runs/temporal-pilot --trainings 3 --execute
```

The four families pair unchanged/changed future controls after static-control and changing-control histories. Static control does not prove steady flow. At the default two replicates there are eight reference scenarios. Bounds/rates are checked. Scored horizon cannot exceed trained offset. Cached references and completed result markers are reused; resume retains saved settings, except increasing the training count. Write new directories to change other parameters.

Outputs include per-position/window CSV and XLSX metrics, predicted/reference fields, input indices, controls, settings, models and transforms. Relative metrics are blank for effectively zero reference norms; absolute metrics remain available. This implemented workflow is not a completed scientific temporal validation campaign in the supplied archives.

## Historical MPC and candidate statistics

```bash
# Preparation only: complete dataset required, no CFD or training.
shred-mpc-historical --run-dir runs/historical-mpc   --dataset /path/to/complete/CSV_files --mode mpc
# EXPENSIVE: run historical pressure MPC inside that owned directory
shred-mpc-historical --run-dir runs/historical-mpc --execute
```

Use `--mode statistics` at preparation for the five-candidate validation runner. Preparation restores frozen source files into the new run, copies the MPC template, and redirects data reads/caches. The original dataset is read externally; simulation outputs stay in the owned run. A started historical run cannot be invoked again through this wrapper. This guards isolation but does not repair the historical initialization, parameter-global coupling or force convention. Full historical execution was inspected, not rerun during this preparation. Repeated retraining uses fixed data, not online learning.

The frozen scripts use their original relative imports inside the restored runtime tree. They are stored as `.py.txt` in the repository so importing the maintained package cannot launch them. Legacy velocity scripts and broken exploratory variants are documentary, not supported entry points.

## Troubleshooting

- Missing solver/ParaView: load the system environment and check the exact executable paths. Dictionary headers do not identify an installed version.
- Existing output: choose a fresh path, or use temporal `--resume` only for its own manifest.
- Incomplete CSV/control table: finish the export; do not bypass the timestamp/sample checks.
- Different mesh/order: stop and use consistent coordinates; stored sensor IDs depend on ordering.
- Unexpected Cp scale: inspect `p_raw` and pressure-convention metadata; do not normalize twice.
- Large lift relative errors near zero: report absolute errors and reference norms.
- Research dependency warnings from PySINDy/pkg_resources: observed with this environment; the focused tests still pass. No warning suppression is added.
