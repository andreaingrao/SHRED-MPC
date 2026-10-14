#!/usr/bin/env python3
"""Train a pressure SHRED once, then evaluate controlled forward forecasts.

Run --dry-run while the training dataset is being generated. Actual runs
require complete CSV exports. All evaluation CFD is isolated under run_dir.
"""
import argparse
from dataclasses import asdict
from datetime import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import numpy as np

from shred_mpc import config
from shred_mpc.temporal import (
    EvaluationSettings, atomic_json, build_forecast_input, collect_results,
    error_records, fit_engine, generate_cfd_reference, generate_omega_distribution,
    load_engine, load_training_arrays, save_engine, validate_training_dataset,
    write_scenario,
)

from shred_mpc.paths import repository_root
ROOT = repository_root()
PACKAGE = Path(__file__).resolve().parents[1]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-only", action="store_true", help="Fit/save models without generating evaluation CFD.")
    parser.add_argument("--execute", action="store_true", help="Explicitly allow training and/or CFD; otherwise use --dry-run.")
    parser.add_argument("--dry-run", action="store_true", help="Save experiment schedules; do not train or run CFD.")
    parser.add_argument("--run-dir", type=Path, help="New output directory (must not already exist).")
    parser.add_argument("--resume", type=Path, help="Continue an existing run using its saved settings and references.")
    parser.add_argument("--trainings", type=int, help="Independent model seeds; default 1. On resume this can extend the count.")
    parser.add_argument("--replicates", type=int, default=2, help="Sequences per family; total CFD cases = 4 * replicates.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-samples", type=int, default=250)
    parser.add_argument("--static-samples", type=int, default=150)
    parser.add_argument("--history-windows", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=config.meas_delta_t)
    parser.add_argument("--small-move", type=float, default=0.2 * config.MPC_OMEGA_SEARCH_RADIUS)
    parser.add_argument("--large-move", type=float, default=config.MPC_OMEGA_SEARCH_RADIUS)
    parser.add_argument("--base-omegas", nargs="+", type=float, default=[0.0, -0.04, 0.04])
    parser.add_argument("--sensors", nargs="+", type=int,
                        default=[int(value) for value in np.atleast_1d(config.sensor_ids)])
    parser.add_argument("--dataset-limit", type=int, default=config.DATASET_DIMENSIONS,
                        help="Training case count, or -1 for all configured cases. Useful for a pilot.")
    parser.add_argument("--epochs", type=int, default=config.shred_epochs)
    parser.add_argument("--device", choices=("cpu", "cuda"), default=config.shred_device)
    parser.add_argument("--dataset", type=Path, default=ROOT / "data" / "external" / "CSV_files")
    parser.add_argument("--template", type=Path, default=ROOT / "cfd" / "templates" / "training",
                        help="Read-only CFD template: 0, constant, system. Default matches the training plant.")
    parser.add_argument("--solver", default="pimpleFoam")
    parser.add_argument("--keep-cfd-files", action="store_true", help="Retain intermediate solver/VTK/CSV files as well as archives.")
    return parser.parse_args()


def dataset_fingerprint(cases, samples):
    digest = hashlib.sha256()
    for case in cases:
        digest.update((case / "omega_t_table.txt").read_bytes())
        for index in range(samples):
            path = case / f"datacsv_{index}.csv"
            stat = path.stat()
            digest.update(f"{path}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    return digest.hexdigest()


def create_run(args):
    if args.resume:
        if args.run_dir:
            raise ValueError("Use either --resume or --run-dir")
        directory = args.resume.resolve()
        manifest = json.loads((directory / "manifest.json").read_text())
        settings = EvaluationSettings(**manifest["settings"])
        if args.trainings is not None:
            if args.trainings < settings.trainings:
                raise ValueError("Cannot reduce the training count on resume")
            settings.trainings = args.trainings
            manifest["settings"] = asdict(settings)
            atomic_json(directory / "manifest.json", manifest)
        print("Resuming with the saved experiment settings.", flush=True)
        return directory, settings, manifest
    settings = EvaluationSettings(
        lags=config.shred_lags, offset=config.meas_offset, horizon=args.horizon,
        control_samples=config.CONTROL_HORIZON, warmup_samples=args.warmup_samples,
        static_samples=args.static_samples, history_windows=args.history_windows,
        replicates=args.replicates, base_omegas=tuple(args.base_omegas),
        small_move=args.small_move, large_move=args.large_move,
        omega_min=config.MPC_OMEGA_MIN, omega_max=config.MPC_OMEGA_MAX,
        initial_omega=config.omega0,
        max_increment=config.MPC_OMEGA_SEARCH_RADIUS / config.CONTROL_HORIZON,
        write_frequency=config.WRITE_FREQUENCY, solver_substeps=config.SIM_DELTA_T_fraction,
        seed=args.seed, trainings=args.trainings if args.trainings is not None else 1,
        epochs=args.epochs, compression=config.shred_pod_compression, device=args.device,
        dataset_limit=args.dataset_limit, sensors=tuple(args.sensors), solver=args.solver,
        keep_cfd_files=args.keep_cfd_files,
    )
    settings.validate()
    stamp = datetime.now(ZoneInfo("Europe/Rome")).strftime("run_%Y%m%d_%H%M%S_%f")
    directory = (args.run_dir or ROOT / "runs" / "temporal" / stamp).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    versions = {}
    for package in ("numpy", "torch", "pyshred", "pandas", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    files = sorted(PACKAGE.rglob("*.py"))
    manifest = {
        "created_at": datetime.now(ZoneInfo("Europe/Rome")).isoformat(), "settings": asdict(settings),
        "dataset": str(args.dataset.resolve()), "template": str(args.template.resolve()),
        "training_case_count": args.dataset_limit if args.dataset_limit > 0 else config.DATASET_CASES,
        "training_samples": config.TIMESTEPS + 1, "versions": versions,
        "source_hashes": {str(path.relative_to(PACKAGE)): hashlib.sha256(path.read_bytes()).hexdigest() for path in files},
        "forecast_convention": "Fixed trained offset; PySHRED windows exclude the current row. Within-block leads share no future pressure.",
        "pressure_metric": "Unweighted cell-centre Cp Euclidean L2; relative L2 and RMSE also recorded.",
        "lift_metric": "Existing lift_time_series_p pressure-ring observable, without a new normalization or viscous force term.",
        "static_definition": "Constant omega over the conditioning interval; flow may remain unsteady.",
        "cfd_reuse": "Identical references are shared across all training seeds.",
    }
    atomic_json(directory / "manifest.json", manifest)
    return directory, settings, manifest


def main():
    args = parse_args()
    if not args.dry_run and not args.execute:
        raise ValueError("Use --dry-run for a safe plan, or --execute to permit training/CFD")
    run_dir, settings, manifest = create_run(args)
    settings.validate()
    print(f"Evaluation output: {run_dir}", flush=True)
    scenarios = generate_omega_distribution(settings)
    for scenario in scenarios:
        directory = run_dir / "scenarios" / scenario["scenario_id"]
        if not directory.exists():
            write_scenario(directory, scenario, settings)
        else:
            planned = np.load(directory / "omega.npy")
            if not np.array_equal(planned, scenario["omega"]):
                raise ValueError(f"Saved scenario differs from regenerated plan: {directory}")
    print(f"Plan: {settings.trainings} training(s), {len(scenarios)} CFD sequences, "
          f"{settings.horizon} forecast positions per sequence.", flush=True)
    if args.dry_run:
        print("Schedules saved. Resume this directory when the training CSV export is complete.")
        return

    # Do not start training on a dataset currently being rebuilt by runDataSet.sh.
    cases = validate_training_dataset(manifest["dataset"], manifest["training_case_count"],
                                      manifest["training_samples"], settings.write_frequency)
    fingerprint = dataset_fingerprint(cases, manifest["training_samples"])
    if "training_dataset_fingerprint" in manifest and manifest["training_dataset_fingerprint"] != fingerprint:
        raise ValueError("Training dataset changed since this run was prepared; create a new run directory")
    manifest["training_dataset_fingerprint"] = fingerprint
    manifest["training_cases"] = [str(case) for case in cases]
    atomic_json(run_dir / "manifest.json", manifest)
    if not args.train_only:
        for command in (settings.solver, "foamToVTK", "pvpython"):
            if shutil.which(command) is None:
                raise RuntimeError(f"{command} is unavailable. Use a shell with the OpenFOAM/ParaView environment loaded.")
        template = Path(manifest["template"])
        for name in ("0", "constant", "system"):
            if not (template / name).is_dir():
                raise ValueError(f"CFD template is missing {template / name}")
        snapshot = run_dir / "cfd_template"
        if not snapshot.exists():
            snapshot.mkdir()
            for name in ("0", "constant", "system"):
                shutil.copytree(template / name, snapshot / name)
        template = snapshot

    data_dir = run_dir / "training_data"
    states = controls = points = None
    from shred_mpc.observables import lift_time_series_p
    for training_index in range(settings.trainings):
        training_id = f"training_{training_index:03d}"
        training_seed = settings.seed + 10000 + training_index
        directory = run_dir / training_id
        if (directory / "model.json").exists():
            print(f"Reloading {training_id}; training is already complete.", flush=True)
            engine = load_engine(directory, settings)
            points = np.load(data_dir / "points.npy")
        else:
            if states is None:
                data_dir.mkdir(exist_ok=True)
                cache = data_dir / "training_pressure.npy"
                if (data_dir / "complete.json").exists():
                    states = np.load(cache, mmap_mode="r")
                    points = np.load(data_dir / "points.npy")
                    controls = np.load(data_dir / "training_controls.npy")
                else:
                    states, points, controls = load_training_arrays(cases, manifest["training_samples"], data_dir)
                    if dataset_fingerprint(cases, manifest["training_samples"]) != fingerprint:
                        raise ValueError("Training CSVs changed during loading; wait for dataset export to complete")
                    atomic_json(data_dir / "complete.json", {"dataset_fingerprint": fingerprint})
            print(f"Fitting {training_id}, seed={training_seed}", flush=True)
            engine, history, seconds = fit_engine(states, controls, settings, training_seed)
            save_engine(directory, engine, settings, training_seed, seconds, history)
        if args.train_only:
            del engine
            continue
        result_dir = directory / "results"
        result_dir.mkdir(exist_ok=True)
        for scenario in scenarios:
            result_file = result_dir / f"{scenario['scenario_id']}.json"
            if result_file.exists():
                continue
            print(f"Evaluating {training_id}: {scenario['scenario_id']}", flush=True)
            scenario_dir = run_dir / "scenarios" / scenario["scenario_id"]
            started = time.perf_counter()
            pressure, cfd_points = generate_cfd_reference(
                scenario_dir, template, scenario, settings, PACKAGE / "cli" / "export_pressure.py")
            reference_seconds = time.perf_counter() - started
            if points.shape != cfd_points.shape or not np.allclose(points, cfd_points, rtol=0, atol=1e-7):
                raise ValueError("Evaluation CFD mesh differs from the training mesh")
            origin = scenario["origin_index"]
            inputs, targets, measured = build_forecast_input(pressure[:origin + 1], scenario["omega"], origin, settings)
            import torch
            if next(engine.model.parameters()).is_cuda:
                torch.cuda.synchronize()
            prediction_started = time.perf_counter()
            latent = engine.sensor_to_latent(inputs)
            reconstruction = engine.decode(latent[-settings.horizon:])
            if next(engine.model.parameters()).is_cuda:
                torch.cuda.synchronize()
            prediction = np.asarray(reconstruction["Pressure"])
            prediction_seconds = time.perf_counter() - prediction_started
            reference = pressure[origin + 1:origin + settings.horizon + 1]
            lp = lift_time_series_p(prediction, points_vec=points)
            lr = lift_time_series_p(reference, points_vec=points)
            steps, window = error_records(prediction, reference, lp, lr, scenario, settings, training_id, training_seed)
            window.update(prediction_seconds=prediction_seconds, reference_load_or_generation_seconds=reference_seconds)
            archive = result_dir / f"{scenario['scenario_id']}.npz"
            np.savez_compressed(archive, pressure_prediction=prediction, pressure_reference=reference,
                                lift_prediction=lp, lift_reference=lr, sensor_input=inputs[0],
                                target_stream_indices=targets, measured_stream_indices=measured,
                                omega_future=scenario["omega"][origin + 1:])
            atomic_json(result_file, {"steps": steps, "window": window})
            collect_results(run_dir)
        del engine
    collect_results(run_dir)
    print(f"Finished. Excel: {run_dir / 'results.xlsx'}", flush=True)
    print(f"CSV: {run_dir / 'forecast_errors.csv'} and {run_dir / 'window_errors.csv'}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Temporal evaluation failed: {error}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
