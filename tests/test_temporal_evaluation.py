from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import numpy as np

from shred_mpc.temporal import (
    EvaluationSettings, FAMILIES, build_forecast_input, collect_results, error_records,
    generate_omega_distribution, load_engine, save_engine, validate_training_dataset,
    write_excel, write_scenario, atomic_json,
)


class TemporalEvaluationTests(unittest.TestCase):
    def test_scenarios_match_histories_and_cases(self):
        settings = EvaluationSettings(replicates=3)
        scenarios = generate_omega_distribution(settings)
        self.assertEqual(len(scenarios), 12)
        for replicate in range(3):
            group = {row["family"]: row for row in scenarios if row["replicate"] == replicate}
            for first, second in (("same_after_static", "small_after_static"),
                                  ("same_after_changes", "large_after_changes")):
                origin = group[first]["origin_index"]
                self.assertEqual(origin, group[second]["origin_index"])
                np.testing.assert_array_equal(group[first]["omega"][:origin + 1], group[second]["omega"][:origin + 1])
            for family in FAMILIES:
                row = group[family]
                omega, origin = row["omega"], row["origin_index"]
                np.testing.assert_array_equal(omega[:settings.warmup_samples + 1], settings.initial_omega)
                self.assertEqual(len(omega) - origin - 1, settings.horizon)
                self.assertLessEqual(np.max(np.abs(np.diff(omega))), settings.max_increment + 1e-10)
                self.assertGreaterEqual(omega.min(), settings.omega_min)
                self.assertLessEqual(omega.max(), settings.omega_max)
                if family.endswith("static"):
                    np.testing.assert_array_equal(omega[origin - settings.static_samples:origin + 1], row["incoming_omega"])
                else:
                    self.assertGreater(np.max(np.abs(np.diff(omega[origin - 100:origin + 1]))), 0)
                expected = settings.small_move if family == "small_after_static" else (
                    settings.large_move if family == "large_after_changes" else 0)
                self.assertAlmostEqual(abs(row["delta_omega"]), expected)

    def test_reproducible_schedules(self):
        settings = EvaluationSettings()
        first, second = generate_omega_distribution(settings), generate_omega_distribution(settings)
        for a, b in zip(first, second):
            np.testing.assert_array_equal(a["omega"], b["omega"])
        third = generate_omega_distribution(replace(settings, seed=123))
        self.assertTrue(any(not np.array_equal(a["omega"], b["omega"]) for a, b in zip(first, third)))

    def test_larger_changes_use_longer_ramps(self):
        settings = EvaluationSettings(large_move=0.02)
        scenario = next(x for x in generate_omega_distribution(settings) if x["family"] == "large_after_changes")
        self.assertEqual(scenario["ramp_samples"], 50)
        self.assertAlmostEqual(abs(scenario["delta_omega"]), 0.02)

    def test_forecast_input_is_causal_and_matches_offset(self):
        settings = EvaluationSettings(sensors=(0, 2))
        origin = 300
        pressure = np.column_stack((np.arange(origin + 1), np.arange(origin + 1) + 1000,
                                    np.arange(origin + 1) + 2000))
        omega = np.arange(origin + settings.horizon + 1) / 10000
        inputs, targets, measured = build_forecast_input(pressure, omega, origin, settings)
        self.assertEqual(inputs.shape, (1, 100, 3))
        np.testing.assert_array_equal(targets, np.arange(251, 351))
        np.testing.assert_array_equal(measured, np.arange(201, 301))
        np.testing.assert_array_equal(inputs[0, :, 0], measured)
        np.testing.assert_array_equal(inputs[0, :, 2], omega[targets])
        with self.assertRaises(ValueError):
            build_forecast_input(pressure, omega, origin, replace(settings, horizon=51))

    def test_library_windows_exclude_current_row(self):
        from pyshred.processor.utils import generate_lagged_sensor_measurements_rom
        settings = EvaluationSettings(sensors=(0,))
        pressure = np.arange(301)[:, None].astype(float)
        inputs, _, _ = build_forecast_input(pressure, np.zeros(351), 300, settings)
        lagged = generate_lagged_sensor_measurements_rom(inputs, settings.lags)
        self.assertEqual(lagged[-50, -1, 0], 250)
        self.assertEqual(lagged[-1, -1, 0], 299)

    def test_metrics_and_zero_reference_denominators(self):
        settings = EvaluationSettings()
        scenario = generate_omega_distribution(settings)[0]
        reference = np.ones((50, 3))
        rows, summary = error_records(reference + 2, reference, np.ones(50), np.zeros(50),
                                      scenario, settings, "training_000", 42)
        self.assertAlmostEqual(rows[0]["pressure_error_l2"], np.sqrt(12))
        self.assertAlmostEqual(rows[0]["pressure_relative_l2"], 2)
        self.assertEqual(summary["lift_mae"], 1)
        self.assertEqual(summary["lift_rmse"], 1)
        self.assertIsNone(summary["lift_relative_l2"])
        self.assertEqual(rows[-1]["lead_seconds"], 10)

    def test_excel_and_result_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            results = root / "training_000" / "results"
            results.mkdir(parents=True)
            atomic_json(results / "scenario.json", {"steps": [{"scenario_id": "a", "error": 1.2}],
                                                    "window": {"scenario_id": "a", "error": None}})
            collect_results(root)
            self.assertIn("1.2", (root / "forecast_errors.csv").read_text())
            with ZipFile(root / "results.xlsx") as archive:
                for file in archive.namelist():
                    ET.fromstring(archive.read(file))
                sheet = archive.read("xl/worksheets/sheet1.xml").decode()
                self.assertIn("1.2", sheet)
                self.assertIn("inlineStr", sheet)
            write_excel(root / "plain_text.xlsx", {"Text": [{"name": "=1+1"}]})
            with ZipFile(root / "plain_text.xlsx") as archive:
                self.assertNotIn("<f>", archive.read("xl/worksheets/sheet1.xml").decode())

    def test_incomplete_training_dataset_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "incomplete"):
                validate_training_dataset(directory, 10, 451, 5)

    def test_model_and_scalers_round_trip(self):
        import pyshred
        from pyshred import ParametricDataManager, ParametricSHREDEngine, SHRED
        import torch
        torch.set_num_threads(1)
        pyshred.set_device("cpu")
        settings = replace(EvaluationSettings(), lags=5, offset=5, horizon=5,
                           control_samples=5, compression=2, sensors=(0, 1))
        rng = np.random.default_rng(2)
        data = rng.normal(size=(10, 20, 4))
        controls = rng.normal(size=(10, 20, 1))
        manager = ParametricDataManager(lags=5)
        manager.add_data(data=data, id="Pressure", measurements=data[:, :, :2], compress=2)
        manager.add_data(data=controls, id="Parameter", measurements=controls, compress=False)
        manager.prepare()
        model = SHRED(sequence_model="LSTM", decoder_model="MLP", latent_forecaster=None)
        model.sequence.initialize(input_size=3, lags=5, decoder_type="MLP")
        model.decoder.initialize(input_size=model.sequence.output_size, output_size=3)
        engine = ParametricSHREDEngine(manager, model)
        inputs = rng.normal(size=(1, 10, 3))
        expected = engine.decode(engine.sensor_to_latent(inputs))["Pressure"]
        with tempfile.TemporaryDirectory() as directory:
            save_engine(directory, engine, settings, 12, 0.1, np.array([0.2]))
            restored = load_engine(directory, settings)
            actual = restored.decode(restored.sensor_to_latent(inputs))["Pressure"]
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-6)

    def test_cli_dry_run_and_resume(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            run_dir = Path(directory) / "run"
            command = [sys.executable, "-m", "shred_mpc.cli.temporal"]
            first = subprocess.run(command + ["--dry-run", "--replicates", "1", "--run-dir", str(run_dir)],
                                   capture_output=True, text=True, check=True)
            self.assertIn("4 CFD sequences", first.stdout)
            self.assertEqual(len(list((run_dir / "scenarios").iterdir())), 4)
            self.assertFalse(any(run_dir.glob("training_*")))
            subprocess.run(command + ["--dry-run", "--resume", str(run_dir), "--trainings", "2"],
                           capture_output=True, text=True, check=True)
            self.assertEqual(json.loads((run_dir / "manifest.json").read_text())["settings"]["trainings"], 2)


if __name__ == "__main__":
    unittest.main()
