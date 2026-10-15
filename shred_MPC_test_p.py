"""Train the pressure model, or run the original lift-following cylinder MPC."""
import argparse
import json
import os
import pickle
import shutil
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import pyshred
from pyshred import ParametricDataManager, ParametricSHREDEngine, SHRED

from shred_utils.input_var import *
import shred_utils.input_var as input_var
import shred_utils.functions as functions
from shred_utils.functions import (
    DataLoad_Training_p, OpenFoam_launcher_p, pressure_cfd_extractor,
    lift_time_series_p, lift_generated_p, shred_lift_comp_plot_p, lift_time_plot_p,
)

ROOT = Path(__file__).resolve().parent
TRANSFORMS = ("lags", "sensor_scaler", "data_scaler", "_dataset_ids", "_dataset_lengths",
              "_dataset_spatial_shape", "_Vt_registry", "_preSVD_scaler_registry")


def model_settings():
    return {name: getattr(input_var, name) for name in
            ("sensor_ids", "meas_offset", "meas_delta_t", "shred_lags", "shred_pod_compression")}


def save_model(directory, engine, states, points):
    # Store inference transforms rather than pickling the complete training dataset.
    with (directory / "transforms.pkl").open("wb") as file:
        pickle.dump({name: getattr(engine.dm, name) for name in TRANSFORMS}, file)
    torch.save({name: value.detach().cpu() for name, value in engine.model.state_dict().items()},
               directory / "weights.pt")
    np.savez_compressed(directory / "initial_history.npz", states=states, points=points)
    metadata = {"settings": model_settings(),
                "input_size": int(engine.dm.sensor_scaler.n_features_in_),
                "output_size": int(engine.dm.data_scaler.n_features_in_)}
    (directory / "model.json").write_text(json.dumps(metadata, indent=2) + "\n")
    shutil.copy2(ROOT / "shred_utils/input_var.py", directory / "input_var.py")


def load_model(directory):
    # Load only model directories created locally by this script (pickle is not sandboxed).
    metadata = json.loads((directory / "model.json").read_text())
    if metadata["settings"] != model_settings():
        raise ValueError("Model settings differ from input_var.py; restore the saved training settings.")
    with (directory / "transforms.pkl").open("rb") as file:
        transforms = pickle.load(file)
    manager = ParametricDataManager(lags=shred_lags)
    for name, value in transforms.items():
        setattr(manager, name, value)
    device = pyshred.set_device(shred_device)
    model = SHRED(sequence_model="LSTM", decoder_model="MLP", latent_forecaster=None)
    model.sequence.initialize(input_size=metadata["input_size"], lags=shred_lags, decoder_type="MLP")
    model.decoder.initialize(input_size=model.sequence.output_size, output_size=metadata["output_size"])
    model.load_state_dict(torch.load(directory / "weights.pt", map_location="cpu", weights_only=True))
    model.to(device)
    with np.load(directory / "initial_history.npz") as initial:
        states, points = initial["states"], initial["points"]
    return ParametricSHREDEngine(manager, model), states, points


def run_mpc(engine_pod, states_vec, points_vec):
    # points_vec = [n_points, 2]

    x = points_vec[..., 0]
    y = points_vec[..., 1]

    # sensor coordinates, for graphic purposes

    print(f"sensors: {sensor_ids}")

    x_meas = x[sensor_ids]
    y_meas = y[sensor_ids]


    # Rolling pressure measurements and control history

    measurements = []
    parameters_vec = []
    parameters_cfd_sequence = []

    lift_pred = []
    lift_real = []


    parameters_vec.append(omega0)

    for _ in range(shred_lags):
        parameters_cfd_sequence.append(omega0)

    cwd = os.getcwd()

    t = shred_lags

    if CONTROL_HORIZON <= 0 or CONTROL_HORIZON > meas_delta_t:
        raise ValueError(f"CONTROL_HORIZON must be in [1, {meas_delta_t}], got {CONTROL_HORIZON}")
    if MPC_COST_HORIZON < CONTROL_HORIZON or MPC_COST_HORIZON > meas_delta_t:
        raise ValueError(
            f"MPC_COST_HORIZON must be in [{CONTROL_HORIZON}, {meas_delta_t}], "
            f"got {MPC_COST_HORIZON}"
        )
    if MPC_OMEGA_GRID_SIZE < 3 or MPC_OMEGA_GRID_SIZE % 2 == 0:
        raise ValueError(f"MPC_OMEGA_GRID_SIZE must be an odd integer >= 3, got {MPC_OMEGA_GRID_SIZE}")
    if MPC_OMEGA_SEARCH_RADIUS <= 0:
        raise ValueError(f"MPC_OMEGA_SEARCH_RADIUS must be positive, got {MPC_OMEGA_SEARCH_RADIUS}")
    if MPC_OMEGA_MIN >= MPC_OMEGA_MAX:
        raise ValueError(f"Invalid omega bounds [{MPC_OMEGA_MIN}, {MPC_OMEGA_MAX}]")
    if MPC_LIFT_DEADBAND < 0:
        raise ValueError(f"MPC_LIFT_DEADBAND must be non-negative, got {MPC_LIFT_DEADBAND}")
    if MPC_OSCILLATION_WEIGHT < 0 or MPC_CONTROL_MOVE_WEIGHT < 0:
        raise ValueError("MPC objective weights must be non-negative")

    os.makedirs("shred_images_p", exist_ok=True)

    #### MPC CYCLE ####

    while len(lift_real) < N_cycles:

        LIFT_T = LIFT_TARGET[t]
        
        os.chdir(cwd)

        measurements = states_vec[-(shred_lags + meas_delta_t):, sensor_ids]  # (T, Ns)
        meas_flat = measurements

        print(f"pressure measurements = {measurements}")

        lift_mean_vec = []
        lift_std_vec = []
        lift_applied_mean_vec = []
        tracking_cost_vec = []
        oscillation_cost_vec = []
        move_cost_vec = []
        total_cost_vec = []
        pressure_future_candidates = []
        parameter_future_candidates = []

        #### LIFT PREDICTIVE EVALUATION ####

        current_omega = float(parameters_cfd_sequence[-1])
        possible_omegas = np.linspace(
            current_omega - MPC_OMEGA_SEARCH_RADIUS,
            current_omega + MPC_OMEGA_SEARCH_RADIUS,
            MPC_OMEGA_GRID_SIZE,
        )
        possible_omegas = np.clip(possible_omegas, MPC_OMEGA_MIN, MPC_OMEGA_MAX)
        possible_omegas = np.unique(np.round(possible_omegas, OMEGA_TABLE_DECIMALS))

        dir = t + CONTROL_HORIZON

        os.makedirs(f"shred_images_p/{dir}", exist_ok=True)

        for i in range(len(possible_omegas)):

            omega = float(possible_omegas[i])

            omega_apply = np.linspace(current_omega, omega, CONTROL_HORIZON + 1)[1:]
            omega_hold = np.full(meas_delta_t - CONTROL_HORIZON, omega)
            omega_future = np.concatenate([omega_apply, omega_hold])
            omega_vec = np.concatenate([parameters_cfd_sequence[-shred_lags:], omega_future])

            print(f"MEAS_DIM = {measurements.shape}, OMEGA_DIM = {omega_vec.shape}")
            sensors = np.concatenate([meas_flat, omega_vec[:, None]], axis=-1)[None, ...]

            print(f"SENSORS_DIM = {sensors.shape}")

            latent = engine_pod.sensor_to_latent(sensors)
            recon = engine_pod.decode(latent)

            p_pred = np.asarray(recon["Pressure"])
            par_pred = np.asarray(recon["Parameter"])
            if p_pred.ndim > 2 and p_pred.shape[0] == 1:
                p_pred = p_pred[0]
            if par_pred.ndim > 2 and par_pred.shape[0] == 1:
                par_pred = par_pred[0]

            print(f"p_pred {p_pred.shape}")

            p_future = p_pred[-meas_delta_t:, ...]
            parameter_future = par_pred[-meas_delta_t:, ...]
            pressure_future_candidates.append(p_future)
            parameter_future_candidates.append(parameter_future)

            lift_trajectory = lift_time_series_p(
                p_future[:MPC_COST_HORIZON, ...],
                points_vec=points_vec,
            )
            lift_mean = float(np.mean(lift_trajectory))
            lift_std = float(np.std(lift_trajectory))
            applied_lift_mean = float(np.mean(lift_trajectory[:CONTROL_HORIZON]))

            absolute_tracking_error = abs(lift_mean - LIFT_T)
            tracking_cost = max(absolute_tracking_error - MPC_LIFT_DEADBAND, 0.0)
            oscillation_cost = MPC_OSCILLATION_WEIGHT * lift_std
            normalized_move = abs(omega - current_omega) / MPC_OMEGA_SEARCH_RADIUS
            move_cost = MPC_CONTROL_MOVE_WEIGHT * normalized_move
            total_cost = tracking_cost + oscillation_cost + move_cost

            lift_mean_vec.append(lift_mean)
            lift_std_vec.append(lift_std)
            lift_applied_mean_vec.append(applied_lift_mean)
            tracking_cost_vec.append(tracking_cost)
            oscillation_cost_vec.append(oscillation_cost)
            move_cost_vec.append(move_cost)
            total_cost_vec.append(total_cost)

        lift_mean_vec = np.asarray(lift_mean_vec)
        lift_std_vec = np.asarray(lift_std_vec)
        lift_applied_mean_vec = np.asarray(lift_applied_mean_vec)
        tracking_cost_vec = np.asarray(tracking_cost_vec)
        oscillation_cost_vec = np.asarray(oscillation_cost_vec)
        move_cost_vec = np.asarray(move_cost_vec)
        total_cost_vec = np.asarray(total_cost_vec)

        fig, (ax_lift, ax_cost) = plt.subplots(2, 1, figsize=(8, 8), sharex=True, constrained_layout=True)
        ax_lift.errorbar(
            possible_omegas,
            lift_mean_vec,
            yerr=lift_std_vec,
            fmt='ro',
            capsize=3,
            label="Predicted mean +/- std",
        )
        ax_lift.axhline(LIFT_T, color="black", linestyle="--", label="Desired lift")
        ax_lift.axhspan(
            LIFT_T - MPC_LIFT_DEADBAND,
            LIFT_T + MPC_LIFT_DEADBAND,
            color="green",
            alpha=0.12,
            label="Tracking deadband",
        )
        ax_lift.set_ylabel("Lift")
        ax_lift.grid(True, alpha=0.3)
        ax_lift.legend()

        ax_cost.plot(possible_omegas, total_cost_vec, 'ko-', label="Total cost")
        ax_cost.plot(possible_omegas, tracking_cost_vec, label="Tracking")
        ax_cost.plot(possible_omegas, oscillation_cost_vec, label="Oscillation")
        ax_cost.plot(possible_omegas, move_cost_vec, label="Control move")
        ax_cost.set_xlabel("Candidate terminal omega")
        ax_cost.set_ylabel("Cost")
        ax_cost.grid(True, alpha=0.3)
        ax_cost.legend()
        fig.savefig(f"shred_images_p/{t}.png", dpi=200)
        plt.close(fig)

        for omega, mean_lift, std_lift, total_cost in zip(
            possible_omegas,
            lift_mean_vec,
            lift_std_vec,
            total_cost_vec,
        ):
            print(
                f"candidate omega={omega:.6f}: mean lift={mean_lift:.6f}, "
                f"lift std={std_lift:.6f}, total cost={total_cost:.6f}"
            )

        id_omega_choice = int(np.argmin(total_cost_vec))
        pressure_chosen_future = pressure_future_candidates[id_omega_choice]
        parameter_chosen_future = parameter_future_candidates[id_omega_choice]

        # Compare like with like later: CFD advances only CONTROL_HORIZON samples.
        lift_pred.append(lift_applied_mean_vec[id_omega_choice])

        parameters_vec.append(float(possible_omegas[id_omega_choice]))

        print(
            f"MPC choice: omega={possible_omegas[id_omega_choice]:.6f}, "
            f"mean lift={lift_mean_vec[id_omega_choice]:.6f}, "
            f"lift std={lift_std_vec[id_omega_choice]:.6f}, "
            f"cost={total_cost_vec[id_omega_choice]:.6f}"
        )


        #### COMPARATIVE PLOT CREATION ####

        parameters_cfd_sequence = OpenFoam_launcher_p(parameters_cfd_sequence, parameters_vec[-1], control_horizon=CONTROL_HORIZON)
        os.chdir(cwd)

        p_cfd, states_vec_cfd = pressure_cfd_extractor(count=CONTROL_HORIZON, preprocess=True)
        p_cfd_new = p_cfd
        parameter_cfd_new = parameters_cfd_sequence[-CONTROL_HORIZON:]
        timestep_new = len(parameters_cfd_sequence) - CONTROL_HORIZON

        print(f"states_vec: {states_vec.shape}, states_vec_cfd: {states_vec_cfd.shape}")
        states_vec = np.concatenate([states_vec, states_vec_cfd], axis=0)

        lift_cfd = lift_generated_p(p_cfd_new, points_vec=points_vec) / CONTROL_HORIZON

        lift_real.append(lift_cfd)

        print(f"STATES_VEC = = = = = = : {states_vec.shape}")

        print(len(parameters_vec))

        plot_values = np.concatenate([
            pressure_chosen_future[:CONTROL_HORIZON, ...].reshape(-1),
            p_cfd_new.reshape(-1),
        ])
        val_min = float(np.min(plot_values))
        val_max = float(np.max(plot_values))

        for i in range(CONTROL_HORIZON):

            shred_lift_comp_plot_p(
                x=x,
                y=y,
                x_meas=x_meas,
                y_meas=y_meas,
                p_pred=pressure_chosen_future[i, :],
                p_true=p_cfd_new[i, :],
                val_min=val_min,
                val_max=val_max,
                par_true=parameter_cfd_new[i],
                par_pred=parameter_chosen_future[i],
                timestep=timestep_new + i,
                desired_lift=LIFT_TARGET,
                type='save',
            )

        t = timestep_new + CONTROL_HORIZON


    lift_time_plot_p(lift_vec_cfd=lift_real, lift_vec_pred=lift_pred, desired_lift=LIFT_TARGET, type='save')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-only", action="store_true", help="Train and save; do not run CFD.")
    parser.add_argument("--model", type=Path, required=True, help="New training output or an existing local model directory.")
    parser.add_argument("--dataset", type=Path, default=ROOT / "OpenFoam/CSV_files", help="Training CSV directory.")
    parser.add_argument("--run-dir", type=Path, help="New output directory for MPC; must not already exist.")
    args = parser.parse_args()
    model_dir = args.model.resolve()
    if args.train_only:
        if args.run_dir:
            parser.error("--run-dir applies only to MPC.")
        if model_dir.exists():
            parser.error(f"Model output already exists: {model_dir}; choose a new name.")
        dataset = args.dataset.resolve()
        if not dataset.is_dir():
            parser.error(f"Dataset missing: {dataset}; run bash OpenFoam/runDataSet.sh first.")
        functions.DATA_DIRECTORY = str(dataset)
        model_dir.mkdir(parents=True, exist_ok=False)
        engine, states, points, _ = DataLoad_Training_p("case_omega")
        save_model(model_dir, engine, states, points)
        print(f"Saved model, transforms and initial history to {model_dir}")
        return
    if args.run_dir is None:
        parser.error("MPC needs --run-dir runs/lift-01 (a new directory).")
    run_dir = args.run_dir.resolve()
    if run_dir.exists():
        parser.error(f"Run output already exists: {run_dir}; choose a new name.")
    if not (model_dir / "model.json").is_file():
        parser.error(f"Model missing or incomplete: {model_dir}; run --train-only first.")
    missing = [name for name in ("pimpleFoam", "foamToVTK", "pvpython") if shutil.which(name) is None]
    if missing:
        parser.error("Load the OpenFOAM/ParaView environment; missing: " + ", ".join(missing))
    if N_cycles <= 0 or shred_lags + N_cycles * CONTROL_HORIZON > len(LIFT_TARGET):
        parser.error("N_cycles must be positive and fit within the configured LIFT_TARGET history.")
    engine, states, points = load_model(model_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    shutil.copytree(ROOT / "MPC_OpenFoam", run_dir / "MPC_OpenFoam")
    shutil.copy2(ROOT / "shred_utils/input_var.py", run_dir / "input_var.py")
    (run_dir / "run.json").write_text(json.dumps({"model": str(model_dir), "lift_targets": LIFT_TARGET_VALUES}, indent=2) + "\n")
    previous = Path.cwd()
    try:
        os.chdir(run_dir)
        run_mpc(engine, states, points)
    finally:
        os.chdir(previous)


if __name__ == "__main__":
    main()
