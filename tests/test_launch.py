"""Small launch/persistence checks; synthetic inputs are not research results."""
import contextlib
import importlib.util
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch
from pyshred import ParametricSHREDEngine
import shred_mpc_launcher as launch
from shred_utils import functions, input_var

ROOT = Path(__file__).resolve().parents[1]


class LaunchTests(unittest.TestCase):
    def test_training_checkpoint_reloads_same_predictions(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(41)
        states = rng.normal(size=(10, 16, 8)).astype(np.float32)
        controls = rng.normal(size=(10, 16)).astype(np.float32)
        settings = dict(sensor_ids=[0, 2], shred_lags=3, meas_offset=2,
                        meas_delta_t=2, shred_pod_compression=2, shred_epochs=1)
        with contextlib.ExitStack() as stack, tempfile.TemporaryDirectory() as directory:
            for module in (functions, input_var, launch):
                for name, value in settings.items():
                    if hasattr(module, name):
                        stack.enter_context(patch.object(module, name, value))
            manager, model, _ = functions.shred_training_p(states, states[..., [0, 2]], controls)
            engine = ParametricSHREDEngine(manager, model)
            sensors = np.concatenate([states[0:1, :8, :][..., [0, 2]],
                                      controls[0:1, :8, None]], axis=-1)
            before = engine.decode(engine.sensor_to_latent(sensors))['Pressure']
            output = Path(directory)
            launch.save_model(output, engine, states[0], np.zeros((8, 2)))
            restored, history, points = launch.load_model(output)
            after = restored.decode(restored.sensor_to_latent(sensors))['Pressure']
            np.testing.assert_allclose(after, before, rtol=1e-6, atol=1e-7)
            np.testing.assert_array_equal(history, states[0])
            self.assertEqual(points.shape, (8, 2))
            with patch.object(input_var, 'sensor_ids', [1, 3]):
                with self.assertRaisesRegex(ValueError, 'settings differ'):
                    launch.load_model(output)

    def test_existing_outputs_are_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            for args in (['--train-only', '--model', directory],
                         ['--model', 'missing-model', '--run-dir', directory]):
                result = subprocess.run([sys.executable, str(ROOT / 'shred_mpc_launcher.py'), *args],
                                        cwd=directory, capture_output=True, text=True)
                self.assertEqual(result.returncode, 2)
                self.assertIn('already exists', result.stderr)
                self.assertEqual(list(Path(directory).iterdir()), [])

    def test_initial_checkpoint_does_not_need_generated_phi(self):
        with tempfile.TemporaryDirectory() as directory:
            initial = Path(directory) / '0'
            initial.mkdir()
            for name in ('p', 'U'):
                (initial / name).write_text('initial field')
            self.assertEqual(functions.prepare_openfoam_resume_checkpoint(
                directory, required_fields=('p', 'U')), (0.0, '0'))
            with self.assertRaises(FileNotFoundError):
                functions.prepare_openfoam_resume_checkpoint(directory)

    def test_imports_do_not_write_to_working_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            code = "import shred_mpc_launcher, DataSet_to_CSV, test2; import shred_utils.functions"
            environment = {**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONDONTWRITEBYTECODE': '1'}
            subprocess.run([sys.executable, '-c', code], cwd=directory, env=environment, check=True,
                           capture_output=True, text=True)
            self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == '__main__':
    unittest.main()
