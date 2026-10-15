"""Pressure training and lift-following MPC helpers retained from functions.py."""
import gzip
import os
import re
import shutil
import subprocess
import tempfile
from typing import Literal

import numpy as np
import pandas as pd
import pyshred
from pyshred import ParametricDataManager, SHRED, ParametricSHREDEngine
from matplotlib import pyplot as plt

from shred_utils.input_var import *
from shred_utils.global_var import DATA_DIRECTORY
from shred_utils.pressure_preprocessing import processed_pressure_from_csv

_lift_history_pred = []
_lift_history_true = []
_lift_history_steps = []


def _num_from_text(s, pattern=r"(\d+)"):
    m = re.search(pattern, s)
    return int(m.group(1))


def _dataset_case_limit():
    if DATASET_DIMENSIONS is not None and DATASET_DIMENSIONS > 0:
        return DATASET_DIMENSIONS
    return DATASET_CASES if DATASET_CASES > 0 else None


def _select_dataset_cases(case_names):
    limit = _dataset_case_limit()
    return case_names if limit is None else case_names[:limit]


def _pressure_column_name(df, file):
    if "p" in df.columns:
        return "p"
    raise KeyError(f"Pressure column 'p' not found in {file}. Available columns: {list(df.columns)}")


def data_load_p(case_key: str = "case_omega"):

    case_names = []
    points_vec = None

    for case in os.listdir(DATA_DIRECTORY):
        if case.startswith(case_key):
            case_names.append(case)

    case_names.sort(key=lambda s: _num_from_text(s, r"case_omega(\d+)"))

    burn_states = None
    burn_parameters = None
    burn_dir = os.path.join(DATA_DIRECTORY, "burn_in")

    if os.path.isdir(burn_dir):
        burn_files = [
            f for f in os.listdir(burn_dir)
            if f.startswith("burn_in_") and f.endswith(".csv")
        ]
        burn_files.sort(key=lambda s: _num_from_text(s, r"burn_in_(\d+)\.csv"))
        burn_files = burn_files[: 2 * shred_lags]

        burn_states_list = []

        for f in burn_files:
            file_path = os.path.join(burn_dir, f)
            df = pd.read_csv(file_path)
            pressure_col = _pressure_column_name(df, file_path)
            burn_states_list.append(df[pressure_col].to_numpy(dtype=np.float32))

        if burn_states_list:
            burn_states = np.stack(burn_states_list, axis=0)
            burn_parameters = np.zeros((burn_states.shape[0],), dtype=np.float32)

    print(case_names)
    print("\n" + "-" * 38 + "\nLoading pressure DataSet into SHRED...\n" + "-" * 38 + "\n")

    states_vec = []
    parameters_vec = []

    case_names = _select_dataset_cases(case_names)

    for case in case_names:
        case_dir = os.path.join(DATA_DIRECTORY, case)
        files_list = []
        params_file = "omega_t_table.txt"

        for file in os.listdir(case_dir):
            if file.startswith("datacsv_") and file.endswith(".csv"):
                files_list.append(file)

        files_list.sort(key=lambda s: _num_from_text(s, r"datacsv_(\d+)\.csv"))

        if not files_list:
            print(f"Skipping {case_dir}: no datacsv_*.csv files found.")
            continue

        states_list = []
        for f in files_list:
            file_path = os.path.join(case_dir, f)
            df = pd.read_csv(file_path)
            pressure_col = _pressure_column_name(df, file_path)
            state = df[pressure_col].to_numpy(dtype=np.float32)
            states_list.append(state)
            if points_vec is None:
                points_vec = df[["Points:0", "Points:1"]].to_numpy()

        states_case = np.stack(states_list, axis=0)

        params_list = []
        with open(os.path.join(case_dir, params_file), "r") as f:
            for line in f:
                params_list.append(float(line.strip()[1:-1].split("\t")[-1]))

        params_case = np.stack(params_list, axis=0)

        if burn_states is not None:
            states_case = np.concatenate([burn_states, states_case], axis=0)
            params_case = np.concatenate([burn_parameters, params_case], axis=0)

        states_vec.append(states_case)
        parameters_vec.append(params_case)

    if not states_vec:
        raise FileNotFoundError(f"No pressure datacsv_*.csv files found for cases matching {case_key} in {DATA_DIRECTORY}")

    states_vec = np.stack(states_vec, axis=0)
    parameters_vec = np.stack(parameters_vec, axis=0)

    print(parameters_vec)
    print(parameters_vec.shape)
    print(
        "\n"
        + "-" * 24
        + "\nPressure data loaded, dimensions: \n"
        + "-" * 24
        + f"\n [states_vec] = {states_vec.shape}"
        + f"\n [points_vec] = {points_vec.shape}"
        + f"\n [parameters_vec] = {parameters_vec.shape} \n"
        + "-" * 24
    )

    return states_vec, points_vec, parameters_vec


def sensor_maker_p(states_vec, points_vec):

    print("\n" + "-" * 42 + "\nExtracting pressure sensor measurements...\n" +  "-"* 42 + "\n")

    measurements_vec = states_vec[..., sensor_ids]

    print("\n" + "-" * 23 + "\nPressure sensors loaded\n" +  "-"* 23 + "\n")

    return measurements_vec


def shred_training_p(states_vec, measurements_vec, parameters_time_tensor):

    print(states_vec.shape)
    print(measurements_vec.shape)
    print(parameters_time_tensor.shape)

    T_total = states_vec.shape[1]

    states_p = states_vec[:, meas_offset:, :]
    measurements_p = measurements_vec[:, 0 : T_total - meas_offset, :]
    parameters_time_sensor = np.asarray(parameters_time_tensor[:, meas_offset:T_total])[..., None]

    device = pyshred.set_device(shred_device)

    manager_pod = ParametricDataManager(
        lags=shred_lags,
        train_size=0.8,
        val_size=0.1,
        test_size=0.1,
    )

    manager_pod.add_data(
        data=states_p,
        id="Pressure",
        measurements=measurements_p,
        compress=shred_pod_compression
    )

    manager_pod.add_data(
        data=parameters_time_sensor,
        id="Parameter",
        measurements=parameters_time_sensor,
        compress=False
    )

    train_dataset, val_dataset, test_dataset = manager_pod.prepare()

    model = SHRED(
        sequence_model="LSTM",
        decoder_model="MLP",
        latent_forecaster=None,
    )

    val_errors = model.fit(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        num_epochs=shred_epochs,
        patience=50,
        verbose=True
    )

    print("Train MSE:", model.evaluate(train_dataset))
    print("Val   MSE:", model.evaluate(val_dataset))
    print("Test  MSE:", model.evaluate(test_dataset))

    return manager_pod, model, parameters_time_tensor


def _load_reference_points(n_points=None):
    case_dir = os.path.join(DATA_DIRECTORY, "case_omega0")
    sample_file = os.path.join(case_dir, "datacsv_0.csv")

    if not os.path.isfile(sample_file):
        case_names = [
            case for case in os.listdir(DATA_DIRECTORY)
            if case.startswith("case_omega")
            and os.path.isdir(os.path.join(DATA_DIRECTORY, case))
        ]
        case_names.sort(key=lambda s: _num_from_text(s, r"case_omega(\d+)"))
        if not case_names:
            raise FileNotFoundError(f"No case_omega folders found in {DATA_DIRECTORY}")

        case_dir = os.path.join(DATA_DIRECTORY, case_names[0])
        files_list = [
            file for file in os.listdir(case_dir)
            if file.startswith("datacsv_") and file.endswith(".csv")
        ]
        files_list.sort(key=lambda s: _num_from_text(s, r"datacsv_(\d+)\.csv"))
        if not files_list:
            raise FileNotFoundError(f"No datacsv_*.csv files found in {case_dir}")
        sample_file = os.path.join(case_dir, files_list[0])

    points = pd.read_csv(sample_file)[["Points:0", "Points:1"]].to_numpy(dtype=float)
    if n_points is not None and points.shape[0] != n_points:
        raise ValueError(
            f"point count mismatch: input has {n_points} points, "
            f"but {sample_file} has {points.shape[0]}"
        )
    return points


def _pressure_lift_geometry(points):
    points = np.asarray(points, dtype=float)
    n_points = points.shape[0]
    cache = getattr(_pressure_lift_geometry, "_cache", {})

    if n_points not in cache:
        center = np.asarray((0.0, 0.0), dtype=float)
        radius = np.linalg.norm(points - center, axis=1)
        sorted_ids = np.argsort(radius)
        sorted_radius = radius[sorted_ids]

        search_count = min(240, sorted_radius.size)
        radius_gap = np.diff(sorted_radius[:search_count])
        positive_gap = radius_gap[radius_gap > 1e-12]
        gap_scale = np.median(positive_gap[: min(60, positive_gap.size)]) if positive_gap.size else 0.0
        gap_threshold = max(0.05, 8.0 * gap_scale)
        ring_breaks = np.flatnonzero(radius_gap > gap_threshold)
        ring_count = int(ring_breaks[0] + 1) if ring_breaks.size else min(120, n_points)

        surface_ids = sorted_ids[:ring_count]
        theta = np.mod(np.arctan2(points[surface_ids, 1], points[surface_ids, 0]), 2.0 * np.pi)
        theta_order = np.argsort(theta)
        surface_ids = surface_ids[theta_order]
        theta = theta[theta_order]

        dtheta_prev = np.mod(theta - np.roll(theta, 1), 2.0 * np.pi)
        dtheta_next = np.mod(np.roll(theta, -1) - theta, 2.0 * np.pi)
        theta_weights = 0.5 * (dtheta_prev + dtheta_next)
        sin_theta = np.sin(theta)

        x = points[:, 0]
        left_x = np.min(x)
        x_tol = max(1e-9, np.ptp(x) * 1e-6)
        left_ids = np.flatnonzero(x <= left_x + x_tol)
        if left_ids.size == 0:
            left_ids = np.asarray([int(np.argmin(x))])

        cache[n_points] = surface_ids, theta_weights, sin_theta, left_ids
        _pressure_lift_geometry._cache = cache

    return cache[n_points]


def lift_time_series_p(state_p, points_vec=None):
    sp = np.asarray(state_p, dtype=float)
    if sp.ndim == 1:
        sp = sp[None, :]
    elif sp.ndim > 2:
        sp = sp.reshape(-1, sp.shape[-1])

    n_points = sp.shape[1]
    points = _load_reference_points(n_points) if points_vec is None else np.asarray(points_vec, dtype=float)
    if points.shape[0] != n_points:
        raise ValueError(
            f"point count mismatch: pressure input has {n_points} points, "
            f"but points_vec has {points.shape[0]}"
        )

    surface_ids, theta_weights, sin_theta, _ = _pressure_lift_geometry(points)
    cp = sp[:, surface_ids]

    # state_p is already (p - p_inf) / q_inf. Positive lift is upward;
    # pressure acts opposite to the outward normal.
    return -np.sum(cp * sin_theta[None, :] * theta_weights[None, :], axis=1)


def lift_generated_p(state_p, points_vec=None) -> float:
    return float(lift_time_series_p(state_p, points_vec=points_vec).sum())


def csv_data_load_p(file, preprocess=False):

    if preprocess:
        state = np.asarray(processed_pressure_from_csv(file), dtype=np.float32)
    else:
        data = pd.read_csv(file)
        pressure_col = _pressure_column_name(data, file)
        state = data[pressure_col].to_numpy(dtype=np.float32)

    return state


def pressure_cfd_extractor(omegas=None, count=None, preprocess=True, csv_dir="MPC_OpenFoam/CSV_data"):

    states = []

    if count is None:
        if omegas is None:
            raise ValueError("pressure_cfd_extractor needs either omegas or count.")
        count = len(omegas)

    for i in range(count):

        state_dir = os.path.join(csv_dir, f"CFD_{i}.csv")

        state = csv_data_load_p(state_dir, preprocess=preprocess)

        states.append(state)

    states = np.stack(states, axis=0)

    return states, states


def reset_shred_lift_comp_plot_history():
    """Reset stored lift traces used by shred_lift_comp_plot."""
    _lift_history_pred.clear()
    _lift_history_true.clear()
    _lift_history_steps.clear()


def _update_lift_history(timestep, lift_pred, lift_true, reset_history=False):

    if reset_history:
        reset_shred_lift_comp_plot_history()

    if _lift_history_steps and timestep < _lift_history_steps[-1]:
        reset_shred_lift_comp_plot_history()

    _lift_history_steps.append(float(timestep))
    _lift_history_pred.append(float(lift_pred))
    _lift_history_true.append(float(lift_true))

    lift_steps = np.asarray(_lift_history_steps, dtype=float)
    lift_pred_hist = np.asarray(_lift_history_pred, dtype=float)
    lift_true_hist = np.asarray(_lift_history_true, dtype=float)

    return lift_steps, lift_pred_hist, lift_true_hist


def shred_lift_comp_plot_p(x, y, x_meas, y_meas, p_true, p_pred, val_min, val_max, par_true, par_pred, timestep, type: Literal["show", "save"], desired_lift=None, reset_history: bool = False):

    par_true = float(np.asarray(par_true).ravel()[0])
    par_pred = float(np.asarray(par_pred).ravel()[0])

    if desired_lift is None:
        desired_lift = LIFT_TARGET

    p_true = np.asarray(p_true, dtype=float).reshape(-1)
    p_pred = np.asarray(p_pred, dtype=float).reshape(-1)
    points_vec = np.stack([np.asarray(x, dtype=float), np.asarray(y, dtype=float)], axis=-1)

    if val_min == val_max:
        val_max = val_min + 1e-12

    lift_pred = lift_generated_p(p_pred, points_vec=points_vec)
    lift_true = lift_generated_p(p_true, points_vec=points_vec)

    lift_steps, lift_pred_hist, lift_true_hist = _update_lift_history(
        timestep=timestep,
        lift_pred=lift_pred,
        lift_true=lift_true,
        reset_history=reset_history,
    )

    lift_diff = lift_pred_hist - lift_true_hist

    p_delta = abs(p_pred - p_true)
    err = float(np.linalg.norm(p_pred - p_true))
    true_norm = float(np.linalg.norm(p_true))
    err_abs = err / true_norm if true_norm != 0 else np.nan

    fig = plt.figure(figsize=(16, 9), constrained_layout=True)
    gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.5])

    ax_cfd = fig.add_subplot(gs[0, 0])
    scatt0 = ax_cfd.scatter(
        x, y, c=p_true, cmap=cmap, s=size, edgecolor="none", vmin=val_min, vmax=val_max
    )
    ax_cfd.scatter(x_meas, y_meas, c=circle_colour, s=circle_size, edgecolor="none")
    ax_cfd.set_aspect("equal")
    ax_cfd.axis("off")
    ax_cfd.set_title(f"CFD normalized pressure, omega = {par_true:.3f}, time = {timestep}")

    ax_pred = fig.add_subplot(gs[1, 0])
    scatt1 = ax_pred.scatter(
        x, y, c=p_pred, cmap=cmap, s=size, edgecolor="none", vmin=val_min, vmax=val_max
    )
    ax_pred.scatter(x_meas, y_meas, c=circle_colour, s=circle_size, edgecolor="none")
    ax_pred.set_aspect("equal")
    ax_pred.axis("off")
    ax_pred.set_title(f"SHRED normalized pressure, omega = {par_pred:.3f}, time = {timestep}")

    fig.colorbar(mappable=scatt1, ax=ax_pred, label="(p - p_inf) / q_inf", orientation='horizontal')

    ax_err_map = fig.add_subplot(gs[2, 0])
    err_vmax = max(float(np.max(p_delta)), 1e-12)
    scatt2 = ax_err_map.scatter(
        x, y, c=p_delta, cmap="Greys", s=size, edgecolor="none", vmin=0, vmax=err_vmax
    )
    ax_err_map.scatter(x_meas, y_meas, c=circle_colour, s=circle_size, edgecolor="none")
    ax_err_map.set_aspect("equal")
    ax_err_map.axis("off")
    ax_err_map.set_title(
        f"Pressure error: \n[error]= {err:.3f}, [absolute error]= {err_abs:.3f}\n"
        f"[Lift predicted]= {lift_pred:.3f}, [actual lift]= {lift_true:.3f}"
    )

    ax_lift = fig.add_subplot(gs[0:2, 1])
    ax_lift.plot(lift_steps, lift_true_hist, label="CFD lift", color="tab:blue")
    ax_lift.plot(lift_steps, lift_pred_hist, label="Pred lift", color="tab:orange")
    _plot_desired_lift(ax_lift, lift_steps, desired_lift, reference_steps=lift_steps)
    ax_lift.scatter(lift_steps[-1], lift_true_hist[-1], color="tab:blue", s=25, zorder=3)
    ax_lift.scatter(lift_steps[-1], lift_pred_hist[-1], color="tab:orange", s=25, zorder=3)
    ax_lift.set_ylabel("Lift")
    ax_lift.set_title("Pressure lift over time")
    ax_lift.grid(True, alpha=0.3)
    ax_lift.legend()

    ax_lift_err = fig.add_subplot(gs[2, 1], sharex=ax_lift)
    ax_lift_err.plot(lift_steps, lift_diff, label="Pred - CFD", color="tab:red")
    ax_lift_err.axhline(0.0, color="gray", linewidth=0.8)
    ax_lift_err.scatter(lift_steps[-1], lift_diff[-1], color="tab:red", s=25, zorder=3)
    ax_lift_err.set_xlabel("Timestep")
    ax_lift_err.set_ylabel("Lift error")
    ax_lift_err.set_title("Pressure lift prediction error")
    ax_lift_err.grid(True, alpha=0.3)
    ax_lift_err.legend()

    if type == "show":
        plt.show()
    elif type == "save":
        os.makedirs("shred_images_p", exist_ok=True)
        plt.savefig(f"shred_images_p/{timestep}_{par_true}_.png", dpi=150)
        plt.close('all')

        print(f"[{timestep}]")

    return


def _plot_desired_lift(ax, x, desired_lift, reference_steps=None):
    """Plot a scalar setpoint or a setpoint schedule at its reference steps."""
    desired_lift = np.asarray(desired_lift, dtype=float).reshape(-1)

    if desired_lift.size == 1:
        ax.axhline(
            desired_lift[0],
            label="Desired lift",
            color="gray",
            linestyle="--",
            linewidth=1.5,
        )
        return

    if reference_steps is None:
        if desired_lift.size != len(x):
            raise ValueError(
                "A desired-lift schedule must have one value per plotted point "
                "when reference_steps is not provided."
            )
        desired_lift_plot = desired_lift
    else:
        reference_steps = np.asarray(reference_steps, dtype=int)
        if reference_steps.size != len(x):
            raise ValueError("reference_steps must have one value per plotted point.")
        if reference_steps.size and (
            reference_steps.min() < 0 or reference_steps.max() >= desired_lift.size
        ):
            raise ValueError("The desired-lift schedule does not cover all reference steps.")
        desired_lift_plot = desired_lift[reference_steps]

    # 'post' holds the old objective until the next reference step, where the
    # new objective becomes active.
    ax.step(
        x,
        desired_lift_plot,
        where="post",
        label="Desired lift",
        color="gray",
        linestyle="--",
        linewidth=1.5,
    )


def lift_time_plot(lift_vec_pred, lift_vec_cfd, type: Literal["show", "save"], desired_lift=None, out_dir="shred_images"):
    lift_vec_pred = np.asarray(lift_vec_pred, dtype=float)
    lift_vec_cfd = np.asarray(lift_vec_cfd, dtype=float)

    n = min(lift_vec_pred.shape[0], lift_vec_cfd.shape[0])
    if n == 0:
        print("lift_time_plot: empty lift vectors, nothing to plot.")
        return
    if lift_vec_pred.shape[0] != lift_vec_cfd.shape[0]:
        print(f"lift_time_plot: clipping to {n} points (pred={lift_vec_pred.shape[0]}, cfd={lift_vec_cfd.shape[0]}).")

    if desired_lift is None:
        desired_lift = LIFT_TARGET

    reference_steps = shred_lags + np.arange(n) * CONTROL_HORIZON
    x = reference_steps / WRITE_FREQUENCY
    lift_pred = lift_vec_pred[:n]
    lift_cfd = lift_vec_cfd[:n]
    relative_lift_error = np.full_like(lift_cfd, np.nan, dtype=float)
    nonzero_cfd = np.abs(lift_cfd) > np.finfo(float).eps
    relative_lift_error[nonzero_cfd] = (
        100.0
        * (lift_pred[nonzero_cfd] - lift_cfd[nonzero_cfd])
        / np.abs(lift_cfd[nonzero_cfd])
    )

    fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True, constrained_layout=True)
    ax0, ax1 = axs

    ax0.plot(x, lift_cfd, label="CFD lift", color="tab:blue")
    ax0.plot(x, lift_pred, label="Pred lift", color="tab:orange")
    _plot_desired_lift(ax0, x, desired_lift, reference_steps=reference_steps)
    ax0.set_ylabel("Lift")
    ax0.set_title("Lift over time")
    ax0.legend()
    ax0.grid(True, alpha=0.3)

    ax1.plot(x, relative_lift_error, label="(Pred - CFD) / |CFD|", color="tab:red")
    ax1.axhline(0.0, color="gray", linewidth=0.8)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Relative lift error (%)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    if type == "show":
        plt.show()
    elif type == "save":
        os.makedirs(out_dir, exist_ok=True)
        plt.savefig(f"{out_dir}/lift_compare.png", dpi=200)
        plt.close('all')


def lift_time_plot_p(lift_vec_pred, lift_vec_cfd, type: Literal["show", "save"], desired_lift=None, out_dir="shred_images_p"):
    return lift_time_plot(
        lift_vec_pred=lift_vec_pred,
        lift_vec_cfd=lift_vec_cfd,
        type=type,
        desired_lift=desired_lift,
        out_dir=out_dir,
    )


def _foam_float(value):
    text = f"{float(value):.12g}"
    return "0" if text == "-0" else text


def _foam_time_directories(case_dir):
    times = []
    for name in os.listdir(case_dir):
        path = os.path.join(case_dir, name)
        if not os.path.isdir(path):
            continue
        try:
            times.append((float(name), name))
        except ValueError:
            continue
    times.sort(key=lambda item: item[0])
    return times


def _foam_field_path(case_dir, time_name, field_name):
    plain = os.path.join(case_dir, time_name, field_name)
    compressed = plain + ".gz"
    if os.path.isfile(plain):
        return plain
    if os.path.isfile(compressed):
        return compressed
    raise FileNotFoundError(f"Could not find {field_name} or {field_name}.gz in {case_dir}/{time_name}")


def _foam_field_validation_error(case_dir, time_name, field_name):
    """Return None for a readable field, otherwise a short failure reason."""
    try:
        path = _foam_field_path(case_dir, time_name, field_name)
    except FileNotFoundError:
        return f"missing {field_name}"
    if os.path.getsize(path) == 0:
        return f"empty {os.path.basename(path)}"
    try:
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rb") as field_file:
            while field_file.read(1024 * 1024):
                pass
    except (OSError, EOFError) as error:
        return f"unreadable {os.path.basename(path)} ({error})"
    return None


def prepare_openfoam_resume_checkpoint(
    case_dir="MPC_OpenFoam",
    required_fields=("p", "U", "phi"),
):
    """Quarantine incomplete trailing times and return the latest valid time.

    An interrupted solver can leave the numerically latest directory only
    partially written. OpenFOAM's ``latestTime`` would still select it and then
    fail while reading the missing fields. Incomplete trailing directories are
    moved under ``incomplete_time_backups`` so recovery is reversible.
    """
    case_dir = os.path.abspath(case_dir)
    times = _foam_time_directories(case_dir)
    if not times:
        raise FileNotFoundError(f"No OpenFOAM time directories found in {case_dir}")

    incomplete_trailing_times = []
    latest_complete = None
    for time_value, time_name in reversed(times):
        errors = [
            error
            for field_name in required_fields
            if (
                error := _foam_field_validation_error(
                    case_dir,
                    time_name,
                    field_name,
                )
            )
            is not None
        ]
        if not errors:
            latest_complete = (time_value, time_name)
            break
        incomplete_trailing_times.append((time_name, errors))

    if latest_complete is None:
        details = "; ".join(
            f"{time_name}: {', '.join(errors)}"
            for time_name, errors in incomplete_trailing_times
        )
        raise FileNotFoundError(
            f"No complete OpenFOAM checkpoint with {required_fields} in "
            f"{case_dir}. Checked: {details}"
        )

    if incomplete_trailing_times:
        backup_parent = os.path.join(case_dir, "incomplete_time_backups")
        os.makedirs(backup_parent, exist_ok=True)
        backup_dir = tempfile.mkdtemp(prefix="recovered_", dir=backup_parent)
        for time_name, errors in reversed(incomplete_trailing_times):
            source = os.path.join(case_dir, time_name)
            destination = os.path.join(backup_dir, time_name)
            shutil.move(source, destination)
            print(
                f"Quarantined incomplete OpenFOAM time {time_name} to "
                f"{destination}: {', '.join(errors)}"
            )

    return latest_complete


def _read_foam_field(path):
    if path.endswith(".gz"):
        with gzip.open(path, "rt") as f:
            return f.read()
    with open(path, "r") as f:
        return f.read()


def _write_foam_field(path, text):
    if path.endswith(".gz"):
        with gzip.open(path, "wt") as f:
            f.write(text)
        return
    with open(path, "w") as f:
        f.write(text)


def _omega_table_block(omega_times, omega_values):
    lines = [
        "        omega",
        "        {",
        "            type            table;",
        "            values",
        f"            {len(omega_values)}",
        "            (",
    ]
    for time, omega in zip(omega_times, omega_values):
        lines.append(f"            ({_foam_float(time)} {_foam_float(omega)})")
    lines.extend([
        "            )",
        "            ;",
        "        }",
    ])
    return lines


def _replace_obstacle_omega_block(field_text, omega_times, omega_values):
    lines = field_text.splitlines()
    obstacle_idx = None

    for i, line in enumerate(lines):
        if line.strip() == "obstacle":
            obstacle_idx = i
            break

    if obstacle_idx is None:
        raise ValueError("Could not find obstacle boundary in U field.")

    omega_idx = None
    value_idx = None
    for i in range(obstacle_idx + 1, len(lines)):
        stripped = lines[i].strip()
        keyword = stripped.split(None, 1)[0] if stripped else ""
        if keyword == "omega":
            omega_idx = i
        elif omega_idx is not None and keyword == "value":
            value_idx = i
            break
        elif omega_idx is not None and keyword == "frontAndBack":
            break

    if omega_idx is None or value_idx is None:
        raise ValueError("Could not isolate omega block inside obstacle boundary.")

    new_lines = lines[:omega_idx] + _omega_table_block(omega_times, omega_values) + lines[value_idx:]
    return "\n".join(new_lines) + "\n"


def _write_resume_omega_table(case_dir, time_name, omega_times, omega_values):
    u_path = _foam_field_path(case_dir, time_name, "U")
    u_text = _read_foam_field(u_path)
    u_text = _replace_obstacle_omega_block(u_text, omega_times, omega_values)
    _write_foam_field(u_path, u_text)


def _clear_generated_files(case_dir):
    for rel_dir, prefixes in (("VTK", ()), ("CSV_data", ("CFD_",))):
        directory = os.path.join(case_dir, rel_dir)
        if not os.path.isdir(directory):
            continue
        for name in os.listdir(directory):
            path = os.path.join(directory, name)
            if not os.path.isfile(path):
                continue
            if rel_dir == "VTK" and name.endswith(".vtk"):
                os.remove(path)
            elif rel_dir == "CSV_data" and name.endswith(".csv") and name.startswith(prefixes):
                os.remove(path)


def _OpenFoam_launcher_profile_with_csv(omegas, omega_future, csv_script="CSV_doer.py"):
    case_dir = "MPC_OpenFoam"
    omegas = np.asarray(omegas, dtype=float)
    omega_future = np.asarray(omega_future, dtype=float)

    if omegas.ndim != 1 or omegas.size == 0:
        raise ValueError(f"omegas must be a non-empty 1D sequence, got shape {omegas.shape}")
    if omega_future.ndim != 1 or omega_future.size == 0:
        raise ValueError(
            f"omega_future must be a non-empty 1D sequence, got shape {omega_future.shape}"
        )
    if not np.all(np.isfinite(omega_future)):
        raise ValueError("omega_future contains non-finite values")

    initial_only = _foam_time_directories(case_dir)[-1][0] == 0.0
    latest_time, latest_time_name = prepare_openfoam_resume_checkpoint(
        case_dir, required_fields=("p", "U") if initial_only else ("p", "U", "phi")
    )

    write_dt = 1.0 / WRITE_FREQUENCY
    solver_dt = write_dt / SIM_DELTA_T_fraction
    end_time = latest_time + omega_future.size * write_dt

    current_omega = float(omegas[-1])
    omegas = np.concatenate([omegas, omega_future])

    omega_times = latest_time + np.arange(omega_future.size + 1) * write_dt
    omega_values = np.concatenate([[current_omega], omega_future])

    with open("MPC_OpenFoam/system/timesteps.txt", "w") as f:
        f.write(f"{SIM_DELTA_T_fraction}")

    with open("MPC_OpenFoam/system/total_time.txt", "w") as f:
        f.write(_foam_float(end_time))

    with open("MPC_OpenFoam/0/omega_sequence.txt", "w") as f:
        for time, omega in zip(omega_times, omega_values):
            f.write(f"({_foam_float(time)}\t{_foam_float(omega)})\n")

    _write_resume_omega_table(case_dir, latest_time_name, omega_times, omega_values)

    vtk_start_step = int(round((latest_time + write_dt) / solver_dt))
    vtk_end_step = int(round(end_time / solver_dt))
    vtk_time_range = f"{_foam_float(latest_time + write_dt)}:{_foam_float(end_time)}"

    cwd = os.getcwd()

    try:
        os.chdir(case_dir)
        _clear_generated_files(".")
        csv_env = os.environ.copy()
        csv_env["MPC_VTK_START_STEP"] = str(vtk_start_step)
        csv_env["MPC_VTK_END_STEP"] = str(vtk_end_step)
        subprocess.run(["pimpleFoam"], check=True)
        subprocess.run(["foamToVTK", "-time", vtk_time_range], check=True)
        subprocess.run(["pvpython", csv_script], check=True, env=csv_env)
    finally:
        os.chdir(cwd)

    return omegas


def _OpenFoam_launcher_with_csv(omegas, par_optimum, control_horizon=CONTROL_HORIZON, csv_script="CSV_doer.py"):
    if control_horizon <= 0:
        raise ValueError(f"control_horizon must be positive, got {control_horizon}")

    omegas = np.asarray(omegas, dtype=float)
    if omegas.ndim != 1 or omegas.size == 0:
        raise ValueError(f"omegas must be a non-empty 1D sequence, got shape {omegas.shape}")

    omega_line = np.linspace(
        float(omegas[-1]),
        float(par_optimum),
        control_horizon + 1,
    )[1:]
    return _OpenFoam_launcher_profile_with_csv(
        omegas=omegas,
        omega_future=omega_line,
        csv_script=csv_script,
    )


def OpenFoam_launcher_p(omegas, par_optimum, control_horizon=CONTROL_HORIZON):
    return _OpenFoam_launcher_with_csv(
        omegas=omegas,
        par_optimum=par_optimum,
        control_horizon=control_horizon,
        csv_script="CSV_doer_p.py",
    )


def DataLoad_Training_p(case_key: str):

    #### DATA LOAD ####

    states_vec, points_vec, parameters_time_tensor      = data_load_p(case_key=case_key)
    # states_vec: [cases, timesteps, cells]
    # points_vec: [cells, dimensions]
    # parameters_time_tensor: [cases, timesteps, parameter]

    measurements_vec_training                           = sensor_maker_p(states_vec=states_vec, points_vec=points_vec)
    # [cases, timesteps, sensor_ids]


    #### TRAINING ####

    manager_pod, model, parameters_time_tensor          = shred_training_p(states_vec=states_vec, measurements_vec=measurements_vec_training, parameters_time_tensor=parameters_time_tensor)


    #### EXPORT ####

    engine_pod                                          = ParametricSHREDEngine(manager_pod, model)
    state_0 = states_vec[0, ...]

    return engine_pod, state_0, points_vec, manager_pod
