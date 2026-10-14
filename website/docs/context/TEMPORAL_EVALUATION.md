# Temporal and parametric forward-prediction launcher

The launcher is `shred_mpc_launcher_temporal.py`. It trains pressure SHRED once by default and evaluates four controlled histories. Additional independent trainings are optional; their evaluations reuse the same CFD reference trajectories rather than repeat CFD. This measures performance of the forward-prediction formulation, not a general SHRED reconstruction benchmark.

## Start with a plan

From the project directory:

```bash
venv/bin/python shred_mpc_launcher_temporal.py --dry-run --run-dir temporal_eval_results/pilot
```

This writes the parameter sequences and manifest without reading the training dataset, fitting a model, or running CFD. It can be used while the dataset is being generated.

After the training CSV export is complete, use a shell with OpenFOAM and ParaView available:

```bash
venv/bin/python shred_mpc_launcher_temporal.py --resume temporal_eval_results/pilot
```

The launcher checks the selected training cases for complete pressure snapshots, matching control-table lengths, and timestamps. It refuses to use the currently incomplete dataset. A failed run can be resumed with the same command; completed models, scenario references, and results are reused.

For a smaller first experiment, create a separate pilot configuration:

```bash
venv/bin/python shred_mpc_launcher_temporal.py --dry-run \
  --dataset-limit 50 --epochs 5 --replicates 1 \
  --run-dir temporal_eval_results/small_pilot
```

Resume `small_pilot` when those selected cases have complete CSV exports. This is a debugging/pilot model, not the full publication training. The default full configuration uses all 500 cases, 20 epochs, and two sequences per family, for eight CFD reference runs. `--replicates 3` covers the three default base rotation values (0, -0.04, +0.04) and generates twelve CFD runs. All CFD sequences initially start at `omega0`, currently zero; nonzero base values are reached through bounded ramps.

## Four scenarios

| Family | History at forecast start | Future forecast window |
|---|---|---|
| `same_after_static` | Constant rotation over the static conditioning period | Same rotation |
| `small_after_static` | Same prefix as `same_after_static` | Small ramp, then hold |
| `large_after_changes` | Several immediately preceding nonzero ramps | Larger ramp, then hold |
| `same_after_changes` | Same prefix as `large_after_changes` | Same rotation |

Defaults:

- 250 initial zero-rotation warm-up intervals, excluded from scoring.
- A bounded ramp to the requested base rotation, when nonzero.
- 150 constant-control intervals before the static scenarios.
- Four preceding 25-sample changes before the dynamic scenarios.
- A small future move of magnitude 0.002, and a larger move of magnitude 0.01; directions are randomly chosen from feasible directions.
- 25-sample future ramps, followed by a hold through the 50-sample forecast horizon.
- The same parameter bounds and maximum per-sample rotation change as the MPC configuration.

"Static" describes the control history. It does not assert that the flow has become steady. Increase `--warmup-samples` or `--static-samples` if flow diagnostics show a longer transient. These periods must provide the full observation history.

Control magnitudes, base rotations, repetitions, warm-up, and static duration can be set at creation time. For example:

```bash
venv/bin/python shred_mpc_launcher_temporal.py --dry-run \
  --replicates 3 --base-omegas 0 -0.04 0.04 \
  --small-move 0.002 --large-move 0.02 \
  --run-dir temporal_eval_results/wider_moves
```

A 0.02 move needs 50 samples at the current rate limit, leaving no hold inside the default 50-sample future window. The launcher rejects a move that cannot finish within the forecast horizon. A larger move here is a designed evaluation condition; it does not enlarge the MPC's candidate search radius.

## Multiple trainings

One training is sufficient for initial debugging and conditional evaluation of the trained predictor. For final repeatability results, several independent seeds help quantify sensitivity to training randomness. The launcher does not assume a universal minimum number.

```bash
venv/bin/python shred_mpc_launcher_temporal.py --resume temporal_eval_results/pilot --trainings 3
```

This retains training 0 and adds trainings 1 and 2. All use identical evaluation scenarios and cached CFD fields. Resuming uses saved experimental settings; only the training count can be extended. Use a new directory for other configuration changes. Training seeds are separated from the scenario seed and recorded per result row.

## Forward-prediction time convention

For lag length L, trained offset D, and target t, the current PySHRED implementation uses:

`Cp_hat(t) = F({[s(j-D), omega(j)] : j=t-L,...,t-1})`.

The current row is excluded by the installed library. The evaluator reproduces this convention, including the historical prefix required to avoid zero-padded evaluation windows. It passes only pressure observed up to the forecast origin, while supplying the proposed future parameter trajectory. It never feeds CFD future pressure into the model.

The forecast horizon must be no larger than the trained offset. Errors at leads 1 through 50 are positions in this fixed-offset model's deployed prediction block; they are not 50 separately trained offsets. `--horizon` changes the scored block length within that constraint, not the trained offset. It must contain the planned control ramp.

## Results and metric definitions

Each run contains:

- `manifest.json`: settings, selected dataset, environment versions, source hashes, metric definitions, and dataset fingerprint.
- `scenarios/<scenario_id>/omega_sequence.csv`, `omega.npy`, and `scenario.json`: actual controls, timestamps, forecast start, conditioning group, and parameter change.
- `scenarios/<scenario_id>/cfd_reference.npz`: the complete normalized CFD pressure trajectory, coordinates, and controls, reused across models.
- `training_data/`: a disk-backed pressure array and control/coordinate arrays, reused across training seeds. This takes additional disk space approximately equal to one float32 pressure dataset.
- `training_<id>/`: model weights, lightweight preprocessing transforms, validation history, model seed/settings, and training duration. Transforms are trusted, locally generated pickle artifacts.
- `training_<id>/results/<scenario_id>.npz`: forecast and reference pressure fields, lift trajectories, actual sensor inputs, input timestamps, and future controls.
- `forecast_errors.csv`: one row per model, scenario, and lead time.
- `window_errors.csv`: one row per complete forecast window.
- `results.xlsx`: the same two result tables, updated after each completed scenario. It needs no additional Excel-writing dependency.

Per-lead metrics include pressure absolute Euclidean L2, reference norm, relative L2, pressure RMSE, predicted/reference lift, signed lift error, and absolute lift error. Rows identify the ramp/hold part of the future trajectory. Whole-window metrics additionally include lift L2 error, relative trajectory L2, MAE, RMSE, bias, and peak absolute error. A relative metric is blank if its reference norm is effectively zero; absolute metrics remain available.

Pressure is the existing Cp convention: raw kinematic pressure minus the leftmost-cell mean, divided by 0.125. The domain metric is an unweighted cell-centre norm; no cell-volume weighting is claimed. Lift uses the project's current `lift_time_series_p` pressure-ring observable. It retains the existing factor/normalization and excludes viscous force. This choice is explicit in the manifest so legacy and future corrected force conventions are not silently mixed.

## CFD isolation and storage

The default template is `OpenFoam`, matching the training plant. Its `0`, `constant`, and `system` directories are copied into a run-local snapshot. Each scenario uses its own CFD case, complete boundary-control table, fixed sampling cadence, and solver/export logs. The active dataset-generation directories and live `MPC_OpenFoam` case are not advanced or rewritten.

Once a complete compressed CFD reference has been saved, intermediate solver time directories, VTK, and CSV exports from that owned evaluation case are removed by default. The initial and final solver states and logs remain. Use `--keep-cfd-files` at creation time to retain all intermediate files.

An incomplete or failed scenario has no completed-result JSON marker and will be retried on resume. Full result archives and existing models are not regenerated. The dataset fingerprint is checked on resume so a run cannot silently mix training data versions.

## Verification

Focused tests cover all four cases, matched prefixes, bounds/rates, exact control/pressure input indices, the real installed PySHRED window convention, model/preprocessing reloads, metrics, zero denominators, XLSX XML structure, dry-run/resume, and incomplete-dataset rejection. No full 500-case model training is part of automated verification.

Additional integration checks used a real three-snapshot, 3600-cell OpenFOAM/ParaView case and a one-epoch synthetic PySHRED training/reload. The synthetic launcher workflow also verifies adding training seeds and resuming without duplicating CFD references or result rows. These are implementation checks, not project performance results.

```bash
venv/bin/python -m unittest discover -s tests -v
```
