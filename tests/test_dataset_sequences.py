import collections
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from shred_mpc import config
from shred_mpc.dataset import dataset_profile_for_run, random_dataSet_generator


class DatasetSequenceTests(unittest.TestCase):
    def test_full_campaign_matches_counts_and_current_split(self):
        expected = {
            "ramp": config.DATASET_RAMP_CASES,
            "ramp_hold": config.DATASET_RAMP_HOLD_CASES,
            "varied": config.DATASET_VARIED_CASES,
            "extended_hold": config.DATASET_EXTENDED_HOLD_CASES,
        }
        profiles = [dataset_profile_for_run(i)[1] for i in range(config.DATASET_CASES)]
        self.assertEqual(collections.Counter(profiles), expected)
        # Each sequential partition should retain all four profile families.
        for start, end in ((0, 400), (400, 450), (450, 500)):
            self.assertEqual(set(profiles[start:end]), set(expected))

    def test_all_500_sequences_have_valid_lengths_bounds_and_rates(self):
        rate = config.MPC_OMEGA_SEARCH_RADIUS / config.CONTROL_HORIZON
        for run_index in range(config.DATASET_CASES):
            duration, profile, _ = dataset_profile_for_run(run_index)
            omega = random_dataSet_generator(
                config.TIMESTEPS, duration, config.omega0, profile, seed=12000 + run_index
            )
            with self.subTest(run_index=run_index, profile=profile):
                self.assertEqual(len(omega), config.TIMESTEPS + 1)
                self.assertEqual(omega[:config.shred_lags], [config.omega0] * config.shred_lags)
                self.assertGreaterEqual(min(omega), config.MPC_OMEGA_MIN)
                self.assertLessEqual(max(omega), config.MPC_OMEGA_MAX)
                self.assertLessEqual(max(abs(b - a) for a, b in zip(omega, omega[1:])), rate + 1e-12)

    def test_ramp_hold_has_constant_second_half(self):
        omega = random_dataSet_generator(449, 25, 0, "ramp_hold", seed=1)
        for start in range(config.shred_lags, len(omega), config.meas_delta_t):
            ramp_end = start + config.CONTROL_HORIZON - 1
            if ramp_end < len(omega):
                hold = omega[ramp_end + 1:start + config.meas_delta_t]
                self.assertEqual(hold, [omega[ramp_end]] * len(hold))
        self.assertNotEqual(omega[config.shred_lags + 24], 0)

    def test_varied_profile_supports_wider_moves_at_same_rate(self):
        with patch.object(config, "DATASET_VARIED_RAMP_SAMPLES", (100,)), \
             patch.object(config, "DATASET_VARIED_HOLD_SAMPLES", (0,)):
            omega = random_dataSet_generator(149, 25, 0, "varied", seed=2)
        self.assertGreater(abs(omega[-1]), config.MPC_OMEGA_SEARCH_RADIUS)
        self.assertLessEqual(abs(omega[-1]), 4 * config.MPC_OMEGA_SEARCH_RADIUS)
        self.assertAlmostEqual(omega[config.shred_lags], omega[-1] / 100)

    def test_extended_hold_and_short_final_segment(self):
        with patch.object(config, "DATASET_EXTENDED_HOLD_SAMPLES", (100,)):
            omega = random_dataSet_generator(178, 25, 0, "extended_hold", seed=1)
        self.assertEqual(omega[75:175], [omega[74]] * 100)
        self.assertEqual(len(omega), 179)
        self.assertLessEqual(abs(omega[-1] - omega[-2]), 0.0004 + 1e-12)

    def test_seeded_histories_repeat_and_other_seeds_differ(self):
        for profile in ("ramp", "ramp_hold", "varied", "extended_hold"):
            args = (450, 25, 0, profile)
            self.assertEqual(random_dataSet_generator(*args, seed=11),
                             random_dataSet_generator(*args, seed=11))
            self.assertNotEqual(random_dataSet_generator(*args, seed=11),
                                random_dataSet_generator(*args, seed=12))

    def test_boundary_starts_do_not_create_invalid_moves(self):
        for start in (config.MPC_OMEGA_MIN, config.MPC_OMEGA_MAX):
            for profile in ("ramp", "ramp_hold", "varied", "extended_hold"):
                omega = random_dataSet_generator(450, 25, start, profile, seed=4)
                self.assertGreaterEqual(min(omega), config.MPC_OMEGA_MIN)
                self.assertLessEqual(max(omega), config.MPC_OMEGA_MAX)

    def test_cli_writes_complete_table_and_preserves_stdout(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "omega.txt"
            result = subprocess.run(
                [sys.executable, "-m", "shred_mpc.cli.control_table", "3", "--seed", "42",
                 "--output", str(output)],
                cwd=directory, check=True, capture_output=True, text=True,
            )
            table = output.read_text()
        self.assertEqual(result.stdout, table)
        rows = table.splitlines()
        self.assertEqual(len(rows), config.TIMESTEPS + 1)
        self.assertEqual(rows[0], "(0\t0)")
        self.assertAlmostEqual(float(rows[-1][1:-1].split()[0]),
                               config.TIMESTEPS / config.WRITE_FREQUENCY)
        self.assertIn("seed=45", result.stderr)

    def test_invalid_configuration_and_indices_are_rejected(self):
        for run_index in (-1, config.DATASET_CASES):
            with self.assertRaises(ValueError):
                dataset_profile_for_run(run_index)
        with self.assertRaises(ValueError):
            random_dataSet_generator(450, 0, 0)
        with self.assertRaises(ValueError):
            random_dataSet_generator(450, 25, 1)
        with self.assertRaises(ValueError):
            random_dataSet_generator(450, 25, 0, "unknown")


if __name__ == "__main__":
    unittest.main()
