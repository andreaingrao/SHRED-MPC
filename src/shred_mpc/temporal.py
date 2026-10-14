"""Controlled forward-prediction experiments, independent of the MPC loop."""
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
import gzip
import hashlib
import json
import math
import pickle
import random
import re
import shutil
import subprocess
import time
from zipfile import ZIP_DEFLATED, ZipFile
import xml.etree.ElementTree as ET

import numpy as np

from shred_mpc.pressure import PRESSURE_DYNAMIC_Q


FAMILIES = ("same_after_static", "small_after_static", "large_after_changes", "same_after_changes")


@dataclass
class EvaluationSettings:
    lags: int = 50
    offset: int = 50
    horizon: int = 50
    control_samples: int = 25
    warmup_samples: int = 250
    static_samples: int = 150
    history_windows: int = 4
    replicates: int = 2
    base_omegas: tuple = (0.0, -0.04, 0.04)
    small_move: float = 0.002
    large_move: float = 0.01
    omega_min: float = -0.09
    omega_max: float = 0.10
    initial_omega: float = 0.0
    max_increment: float = 0.0004
    write_frequency: float = 5.0
    solver_substeps: int = 10
    seed: int = 42
    trainings: int = 1
    epochs: int = 20
    compression: int = 150
    device: str = "cpu"
    dataset_limit: int = -1
    sensors: tuple = (837, 2178, 2279, 3144, 3214, 3351, 3455)
    solver: str = "pimpleFoam"
    keep_cfd_files: bool = False

    def validate(self):
        for name in ("lags", "offset", "horizon", "control_samples", "replicates",
                     "solver_substeps", "trainings", "epochs", "compression", "history_windows"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.horizon > self.offset:
            raise ValueError("horizon must be <= the trained offset; otherwise future pressure would be required")
        if self.control_samples > self.horizon:
            raise ValueError("horizon must contain at least one complete control ramp")
        if self.warmup_samples < self.lags + self.offset:
            raise ValueError("warmup_samples must cover lags + offset")
        if self.static_samples < self.lags + self.offset:
            raise ValueError("static_samples must cover lags + offset to isolate the static-history cases")
        if self.history_windows * self.control_samples < self.lags + self.offset:
            raise ValueError("history_windows must cover lags + offset with preceding changes")
        if not self.omega_min < self.omega_max or self.max_increment <= 0 or self.write_frequency <= 0:
            raise ValueError("Invalid parameter bounds, rate, or sampling frequency")
        values = (*self.base_omegas, self.initial_omega, self.small_move, self.large_move,
                  self.max_increment, self.write_frequency)
        if not self.base_omegas or not all(math.isfinite(x) for x in values):
            raise ValueError("Parameter values must be finite and base_omegas nonempty")
        if not all(self.omega_min <= x <= self.omega_max for x in (*self.base_omegas, self.initial_omega)):
            raise ValueError("Initial and base rotations must be within bounds")
        if not 0 < self.small_move < self.large_move:
            raise ValueError("Require 0 < small_move < large_move")
        if self.large_move > self.max_increment * self.horizon + 1e-12:
            raise ValueError("large_move cannot be completed within the forecast horizon at the declared rate")
        if not self.sensors or len(set(self.sensors)) != len(self.sensors):
            raise ValueError("Provide distinct sensor IDs")
        if any(not isinstance(x, int) or x < 0 for x in self.sensors):
            raise ValueError("Sensor IDs must be nonnegative integers")


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False))
    temp.replace(path)


def _ramp(sequence, target, samples):
    start = sequence[-1]
    sequence.extend(float(start + (target - start) * step / samples) for step in range(1, samples + 1))


def _signed_move(current, magnitude, settings, rng):
    targets = [current + sign * magnitude for sign in (-1, 1)
               if settings.omega_min <= current + sign * magnitude <= settings.omega_max]
    if not targets:
        raise ValueError(f"No feasible move of magnitude {magnitude} from omega={current}")
    return rng.choice(targets)


def generate_omega_distribution(settings):
    """Generate matched static/dynamic histories and their four future cases.

    Every CFD sequence starts at initial_omega. Base rotations are reached by
    bounded ramps, then held. Static means constant control, not a claim of a
    steady flow. Dynamic histories have a nonzero move in each preceding window.
    """
    settings.validate()
    rng = random.Random(settings.seed)
    scenarios = []
    for replicate in range(settings.replicates):
        base = float(settings.base_omegas[replicate % len(settings.base_omegas)])
        static = [float(settings.initial_omega)] * (settings.warmup_samples + 1)
        if base != static[-1]:
            samples = max(settings.control_samples, math.ceil(abs(base - static[-1]) / settings.max_increment - 1e-10))
            _ramp(static, base, samples)
        static.extend([static[-1]] * settings.static_samples)
        dynamic = static.copy()
        for _ in range(settings.history_windows):
            magnitudes = sorted(set((settings.small_move, settings.large_move / 2, settings.large_move)))
            feasible = [m for m in magnitudes if m <= settings.max_increment * settings.control_samples + 1e-12
                        and any(settings.omega_min <= dynamic[-1] + sign * m <= settings.omega_max for sign in (-1, 1))]
            if not feasible:
                raise ValueError("No feasible history move; check bounds and rate")
            target = _signed_move(dynamic[-1], rng.choice(feasible), settings, rng)
            _ramp(dynamic, target, settings.control_samples)

        for family in FAMILIES:
            prefix = static if family.endswith("static") else dynamic
            sequence = prefix.copy()
            origin = len(sequence) - 1
            magnitude = settings.small_move if family == "small_after_static" else (
                settings.large_move if family == "large_after_changes" else 0.0)
            target = _signed_move(sequence[-1], magnitude, settings, rng) if magnitude else sequence[-1]
            ramp_samples = 0
            if magnitude:
                ramp_samples = max(settings.control_samples, math.ceil(magnitude / settings.max_increment - 1e-10))
                _ramp(sequence, target, ramp_samples)
            sequence.extend([float(target)] * (settings.horizon - ramp_samples))
            omega = np.asarray(sequence, dtype=np.float64)
            if np.max(np.abs(np.diff(omega))) > settings.max_increment + 1e-10:
                raise ValueError("Generated sequence exceeds the rate limit")
            scenarios.append({
                "scenario_id": f"replicate_{replicate:03d}_{family}", "family": family,
                "replicate": replicate, "base_omega": base, "origin_index": origin,
                "incoming_omega": float(prefix[-1]), "outgoing_omega": float(target),
                "delta_omega": float(target - prefix[-1]), "ramp_samples": ramp_samples,
                "history_group": f"replicate_{replicate:03d}_{'static' if family.endswith('static') else 'dynamic'}",
                "omega": omega,
            })
    return scenarios


def write_scenario(directory, scenario, settings):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    omega = scenario["omega"]
    np.save(directory / "omega.npy", omega)
    metadata = {k: v for k, v in scenario.items() if k != "omega"}
    metadata["sequence_sha256"] = hashlib.sha256(omega.tobytes()).hexdigest()
    atomic_json(directory / "scenario.json", metadata)
    with (directory / "omega_sequence.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("sample", "time_seconds", "omega", "is_forecast_target"))
        for sample, value in enumerate(omega):
            writer.writerow((sample, sample / settings.write_frequency, value, int(sample > scenario["origin_index"])))


def build_forecast_input(pressure, omega, origin, settings):
    """Pair target-time controls with offset pressure, without using future pressure.

    PySHRED windows exclude the current row. Retain lags prefix rows followed
    by horizon target rows, just as in the current pressure MPC launcher.
    """
    targets = np.arange(origin + 1 - settings.lags, origin + settings.horizon + 1)
    measured = targets - settings.offset
    if measured[0] < 0 or measured[-1] > origin:
        raise ValueError("Insufficient observed history or forecast would require future pressure")
    if origin + settings.horizon >= len(omega) or len(pressure) <= origin:
        raise ValueError("Insufficient control trajectory or observed pressure")
    ids = np.asarray(settings.sensors)
    if np.max(ids) >= pressure.shape[1]:
        raise ValueError("A sensor ID is outside the CFD field")
    inputs = np.column_stack((pressure[measured][:, ids], omega[targets]))
    return inputs[None, ...], targets, measured


def _relative(error_norm, reference_norm):
    return float(error_norm / reference_norm) if reference_norm > 1e-12 else None


def error_records(prediction, reference, lift_prediction, lift_reference, scenario, settings, training_id, training_seed):
    prediction, reference = np.asarray(prediction), np.asarray(reference)
    lp, lr = np.asarray(lift_prediction), np.asarray(lift_reference)
    if prediction.shape != reference.shape or prediction.shape[0] != settings.horizon:
        raise ValueError("Predicted and reference pressure shapes must match the forecast horizon")
    if lp.shape != (settings.horizon,) or lr.shape != lp.shape:
        raise ValueError("Lift trajectories must match the forecast horizon")
    if not all(np.all(np.isfinite(x)) for x in (prediction, reference, lp, lr)):
        raise ValueError("Non-finite prediction or CFD reference")
    error = prediction.astype(float) - reference.astype(float)
    lift_error = lp - lr
    common = {k: scenario[k] for k in ("scenario_id", "family", "replicate", "base_omega", "incoming_omega",
                                      "outgoing_omega", "delta_omega", "history_group")}
    common.update(training_id=training_id, training_seed=training_seed,
                  origin_index=scenario["origin_index"], origin_time_seconds=scenario["origin_index"] / settings.write_frequency)
    rows = []
    for i in range(settings.horizon):
        norm, ref_norm = float(np.linalg.norm(error[i])), float(np.linalg.norm(reference[i]))
        rows.append(dict(common, lead_samples=i + 1, lead_seconds=(i + 1) / settings.write_frequency,
                         target_time_seconds=(scenario["origin_index"] + i + 1) / settings.write_frequency,
                         phase="ramp" if i < scenario["ramp_samples"] else "hold",
                         pressure_error_l2=norm, pressure_reference_l2=ref_norm,
                         pressure_relative_l2=_relative(norm, ref_norm), pressure_rmse=float(np.sqrt(np.mean(error[i] ** 2))),
                         lift_prediction=float(lp[i]), lift_reference=float(lr[i]),
                         lift_error_signed=float(lift_error[i]), lift_error_absolute=float(abs(lift_error[i]))))
    norm, ref_norm = float(np.linalg.norm(error)), float(np.linalg.norm(reference))
    lift_norm, lift_ref_norm = float(np.linalg.norm(lift_error)), float(np.linalg.norm(lr))
    summary = dict(common, horizon_samples=settings.horizon, pressure_error_l2=norm,
                   pressure_reference_l2=ref_norm, pressure_relative_l2=_relative(norm, ref_norm),
                   pressure_rmse=float(np.sqrt(np.mean(error ** 2))), lift_error_l2=lift_norm,
                   lift_reference_l2=lift_ref_norm, lift_relative_l2=_relative(lift_norm, lift_ref_norm),
                   lift_mae=float(np.mean(abs(lift_error))), lift_rmse=float(np.sqrt(np.mean(lift_error ** 2))),
                   lift_bias=float(np.mean(lift_error)), lift_peak_absolute_error=float(np.max(abs(lift_error))),
                   lift_prediction_mean=float(np.mean(lp)), lift_reference_mean=float(np.mean(lr)))
    return rows, summary


def read_pressure_csv(path, raw_pressure=False):
    import pandas as pd
    frame = pd.read_csv(path)
    points = frame[["Points:0", "Points:1"]].to_numpy(dtype=float)
    source = "p_raw" if "p_raw" in frame.columns else "p"
    pressure = frame[source].to_numpy(dtype=float)
    if source == "p_raw" or raw_pressure:
        x = points[:, 0]
        left = x <= x.min() + max(1e-9, np.ptp(x) * 1e-6)
        pressure = (pressure - pressure[left].mean()) / PRESSURE_DYNAMIC_Q
    if not np.all(np.isfinite(pressure)) or not np.all(np.isfinite(points)):
        raise ValueError(f"Non-finite values in {path}")
    return pressure.astype(np.float32), points


def validate_training_dataset(root, case_count, expected_samples, write_frequency):
    root = Path(root)
    if case_count < 10:
        raise ValueError("Use at least ten cases for the existing 80/10/10 training split")
    cases = []
    for index in range(case_count):
        case = root / f"case_omega{index}"
        table = case / "omega_t_table.txt"
        if not table.exists():
            raise ValueError(f"Training dataset is incomplete: {table} is missing. Wait for dataset generation/export to finish.")
        values = [line.strip()[1:-1].split() for line in table.read_text().splitlines() if line.strip()]
        if len(values) != expected_samples:
            raise ValueError(f"{table} has {len(values)} rows; expected {expected_samples}. Dataset generation may still be in progress.")
        times = np.array([float(row[0]) for row in values])
        if not np.allclose(times, np.arange(expected_samples) / write_frequency, rtol=0, atol=1e-6):
            raise ValueError(f"Control timestamps do not match CFD sampling in {table}")
        for sample in range(expected_samples):
            path = case / f"datacsv_{sample}.csv"
            if not path.is_file() or path.stat().st_size == 0:
                raise ValueError(f"Training export is incomplete: {path}. Wait for CSV export to finish.")
        cases.append(case)
    return cases


def load_training_arrays(cases, samples, directory):
    """Use a disk-backed pressure array to avoid stacking two full dataset copies."""
    directory = Path(directory)
    cache = directory / "training_pressure.npy"
    first, points = read_pressure_csv(cases[0] / "datacsv_0.csv")
    states = np.lib.format.open_memmap(cache, mode="w+", dtype=np.float32,
                                     shape=(len(cases), samples, len(first)))
    controls = np.empty((len(cases), samples), dtype=np.float32)
    for index, case in enumerate(cases):
        print(f"Loading training case {index + 1}/{len(cases)}", flush=True)
        table = [line.strip()[1:-1].split() for line in (case / "omega_t_table.txt").read_text().splitlines() if line.strip()]
        controls[index] = [float(row[1]) for row in table]
        for sample in range(samples):
            field, field_points = read_pressure_csv(case / f"datacsv_{sample}.csv")
            if field.shape != first.shape or not np.allclose(field_points, points, rtol=0, atol=1e-7):
                raise ValueError(f"Coordinate ordering differs in {case}/datacsv_{sample}.csv")
            states[index, sample] = field
    states.flush()
    np.save(directory / "training_controls.npy", controls)
    np.save(directory / "points.npy", points)
    return states, points, controls


def fit_engine(states, controls, settings, training_seed):
    import torch
    import pyshred
    from pyshred import ParametricDataManager, SHRED, ParametricSHREDEngine
    random.seed(training_seed)
    np.random.seed(training_seed)
    torch.manual_seed(training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(training_seed)
    pyshred.set_device(settings.device)
    if max(settings.sensors) >= states.shape[-1]:
        raise ValueError("Configured sensor IDs exceed training mesh size")
    if states.shape[1] <= settings.offset + settings.lags:
        raise ValueError("Training trajectories are too short for the lag and offset")
    started = time.perf_counter()
    manager = ParametricDataManager(lags=settings.lags, train_size=0.8, val_size=0.1, test_size=0.1)
    sensors = states[:, :-settings.offset, :][:, :, list(settings.sensors)]
    parameters = controls[:, settings.offset:, None]
    manager.add_data(data=states[:, settings.offset:, :], id="Pressure", measurements=sensors, compress=settings.compression)
    manager.add_data(data=parameters, id="Parameter", measurements=parameters, compress=False)
    train, validation, _ = manager.prepare()
    model = SHRED(sequence_model="LSTM", decoder_model="MLP", latent_forecaster=None)
    history = model.fit(train_dataset=train, val_dataset=validation,
                        num_epochs=settings.epochs, patience=50, verbose=True)
    return ParametricSHREDEngine(manager, model), np.asarray(history), time.perf_counter() - started


TRANSFORM_ATTRIBUTES = ("lags", "sensor_scaler", "data_scaler", "_dataset_ids", "_dataset_lengths",
                        "_dataset_spatial_shape", "_Vt_registry", "_preSVD_scaler_registry")


def save_engine(directory, engine, settings, training_seed, training_seconds, history):
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    transforms = {name: getattr(engine.dm, name) for name in TRANSFORM_ATTRIBUTES}
    with (directory / "transforms.pkl").open("wb") as file:
        pickle.dump(transforms, file)
    weights = {name: value.detach().cpu() for name, value in engine.model.state_dict().items()}
    torch.save(weights, directory / "weights.pt")
    np.save(directory / "validation_history.npy", history)
    atomic_json(directory / "model.json", {
        "settings": asdict(settings), "training_seed": training_seed, "training_seconds": training_seconds,
        "input_size": int(engine.dm.sensor_scaler.n_features_in_),
        "output_size": int(engine.dm.data_scaler.n_features_in_),
        "architecture": "PySHRED default LSTM + MLP, no latent forecaster",
    })


def load_engine(directory, settings):
    """Load artifacts created locally by this launcher; no training arrays are pickled."""
    import torch
    import pyshred
    from pyshred import ParametricDataManager, ParametricSHREDEngine, SHRED
    directory = Path(directory)
    metadata = json.loads((directory / "model.json").read_text())
    with (directory / "transforms.pkl").open("rb") as file:
        transforms = pickle.load(file)
    manager = ParametricDataManager(lags=settings.lags)
    for name, value in transforms.items():
        setattr(manager, name, value)
    device = pyshred.set_device(settings.device)
    model = SHRED(sequence_model="LSTM", decoder_model="MLP", latent_forecaster=None)
    model.sequence.initialize(input_size=metadata["input_size"], lags=settings.lags, decoder_type="MLP")
    model.decoder.initialize(input_size=model.sequence.output_size, output_size=metadata["output_size"])
    model.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu", weights_only=True))
    model.to(device)
    return ParametricSHREDEngine(manager, model)


def prepare_cfd_case(template, case, scenario, settings):
    """Only modify an owned experiment directory, never the template or live MPC case."""
    from shred_mpc.foam import _replace_obstacle_omega_block
    template, case = Path(template), Path(case)
    case.mkdir(parents=True, exist_ok=False)
    for name in ("0", "constant", "system"):
        shutil.copytree(template / name, case / name)
    omega = scenario["omega"]
    times = np.arange(len(omega)) / settings.write_frequency
    u_path = case / "0" / "U"
    if not u_path.exists():
        u_path = case / "0" / "U.gz"
    opener = gzip.open if u_path.suffix == ".gz" else open
    with opener(u_path, "rt") as file:
        text = file.read()
    with opener(u_path, "wt") as file:
        file.write(_replace_obstacle_omega_block(text, times, omega))
    end_time = times[-1]
    delta_t = 1 / settings.write_frequency / settings.solver_substeps
    (case / "system" / "controlDict").write_text(
        'FoamFile\n{ version 2.0; format ascii; class dictionary; object controlDict; }\n'
        f'application {settings.solver};\nstartFrom startTime;\nstartTime 0;\n'
        f'stopAt endTime;\nendTime {end_time:.12g};\ndeltaT {delta_t:.12g};\n'
        f'writeControl timeStep;\nwriteInterval {settings.solver_substeps};\npurgeWrite 0;\n'
        'writeFormat ascii;\nwritePrecision 10;\nwriteCompression compressed;\n'
        'timeFormat general;\ntimePrecision 10;\nrunTimeModifiable false;\n'
    )


def _run_command(command, case, log_name):
    print(f"CFD: {' '.join(map(str, command))} ({case.name})", flush=True)
    with (case / log_name).open("w") as log:
        try:
            subprocess.run(command, cwd=case, stdout=log, stderr=subprocess.STDOUT, check=True)
        except subprocess.CalledProcessError as error:
            raise RuntimeError(f"Command failed; inspect {case / log_name}") from error


def generate_cfd_reference(directory, template, scenario, settings, exporter):
    directory = Path(directory)
    archive = directory / "cfd_reference.npz"
    if archive.exists():
        with np.load(archive) as saved:
            if not np.array_equal(saved["omega"], scenario["omega"]):
                raise ValueError("Cached CFD sequence differs from the planned sequence")
            return saved["pressure"], saved["points"]
    case = directory / "cfd_case"
    if not case.exists():
        prepare_cfd_case(template, case, scenario, settings)
    if (case / "system" / "blockMeshDict").exists():
        _run_command(["blockMesh"], case, "log.blockMesh")
    _run_command([settings.solver], case, "log.solver")
    end_time = (len(scenario["omega"]) - 1) / settings.write_frequency
    _run_command(["foamToVTK", "-useTimeName", "-fields", "(p)", "-time", f"0:{end_time:.12g}"], case, "log.foamToVTK")
    _run_command(["pvpython", str(exporter), "--samples", str(len(scenario["omega"])),
                  "--write-frequency", str(settings.write_frequency)], case, "log.csv_export")
    pressure = []
    points = None
    for sample in range(len(scenario["omega"])):
        field, current_points = read_pressure_csv(case / "CSV_data" / f"CFD_{sample}.csv", raw_pressure=True)
        if points is None:
            points = current_points
        elif not np.allclose(points, current_points, rtol=0, atol=1e-7):
            raise ValueError("CFD coordinate ordering changed over time")
        pressure.append(field)
    pressure = np.stack(pressure)
    with archive.with_suffix(".npz.tmp").open("wb") as file:
        np.savez_compressed(file, pressure=pressure, points=points, omega=scenario["omega"])
    archive.with_suffix(".npz.tmp").replace(archive)
    if not settings.keep_cfd_files:
        # Only generated intermediate artifacts in this experiment are removed.
        shutil.rmtree(case / "VTK")
        shutil.rmtree(case / "CSV_data")
        final_time = end_time
        for path in case.iterdir():
            if path.is_dir():
                try:
                    value = float(path.name)
                except ValueError:
                    continue
                if value != 0 and not math.isclose(value, final_time, rel_tol=0, abs_tol=1e-7):
                    shutil.rmtree(path)
    return pressure, points


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_excel(path, tables):
    """Write a small dependency-free XLSX workbook, with numeric cells and plain text."""
    ns = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    rel_ns = "http://schemas.openxmlformats.org/package/2006/relationships"
    ET.register_namespace("", ns)
    def xml(root):
        return ET.tostring(root, encoding="utf-8", xml_declaration=True)
    def column(index):
        result = ""
        while index:
            index, remainder = divmod(index - 1, 26)
            result = chr(65 + remainder) + result
        return result
    content = ET.Element("Types", xmlns="http://schemas.openxmlformats.org/package/2006/content-types")
    ET.SubElement(content, "Default", Extension="rels", ContentType="application/vnd.openxmlformats-package.relationships+xml")
    ET.SubElement(content, "Default", Extension="xml", ContentType="application/xml")
    ET.SubElement(content, "Override", PartName="/xl/workbook.xml", ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml")
    workbook = ET.Element(f"{{{ns}}}workbook")
    sheets = ET.SubElement(workbook, f"{{{ns}}}sheets")
    relations = ET.Element("Relationships", xmlns=rel_ns)
    root_rel = ET.Element("Relationships", xmlns=rel_ns)
    ET.SubElement(root_rel, "Relationship", Id="rId1", Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument", Target="xl/workbook.xml")
    temporary = Path(path).with_suffix(".xlsx.tmp")
    with ZipFile(temporary, "w", ZIP_DEFLATED) as archive:
        for sheet_index, (name, records) in enumerate(tables.items(), 1):
            ET.SubElement(sheets, f"{{{ns}}}sheet", name=name, sheetId=str(sheet_index),
                          attrib={"{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id": f"rId{sheet_index}"})
            ET.SubElement(relations, "Relationship", Id=f"rId{sheet_index}", Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet", Target=f"worksheets/sheet{sheet_index}.xml")
            ET.SubElement(content, "Override", PartName=f"/xl/worksheets/sheet{sheet_index}.xml", ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml")
            sheet = ET.Element(f"{{{ns}}}worksheet")
            data = ET.SubElement(sheet, f"{{{ns}}}sheetData")
            headers = list(records[0]) if records else []
            values = [headers] + [[record.get(key) for key in headers] for record in records]
            for row_index, values_row in enumerate(values, 1):
                row = ET.SubElement(data, f"{{{ns}}}row", r=str(row_index))
                for cell_index, value in enumerate(values_row, 1):
                    if value is None:
                        continue
                    cell = ET.SubElement(row, f"{{{ns}}}c", r=f"{column(cell_index)}{row_index}")
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        ET.SubElement(cell, f"{{{ns}}}v").text = str(value)
                    else:
                        cell.set("t", "inlineStr")
                        ET.SubElement(ET.SubElement(cell, f"{{{ns}}}is"), f"{{{ns}}}t").text = str(value)
            archive.writestr(f"xl/worksheets/sheet{sheet_index}.xml", xml(sheet))
        archive.writestr("[Content_Types].xml", xml(content))
        archive.writestr("_rels/.rels", xml(root_rel))
        archive.writestr("xl/workbook.xml", xml(workbook))
        archive.writestr("xl/_rels/workbook.xml.rels", xml(relations))
    temporary.replace(path)


def collect_results(run_dir):
    run_dir = Path(run_dir)
    steps, windows = [], []
    for file in sorted(run_dir.glob("training_*/results/*.json")):
        result = json.loads(file.read_text())
        steps.extend(result["steps"])
        windows.append(result["window"])
    write_csv(run_dir / "forecast_errors.csv", steps)
    write_csv(run_dir / "window_errors.csv", windows)
    write_excel(run_dir / "results.xlsx", {"Forecast_errors": steps, "Window_errors": windows})
