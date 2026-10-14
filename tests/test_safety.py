import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from shred_mpc.observables import lift_time_series_p

class SafetyTests(unittest.TestCase):
    def test_legacy_lift_convention_and_coordinate_cache(self):
        theta=np.linspace(0,2*np.pi,120,endpoint=False)
        points=2.5*np.column_stack([np.cos(theta),np.sin(theta)])
        self.assertAlmostEqual(float(lift_time_series_p(np.ones(120),points)[0]),0,places=12)
        self.assertAlmostEqual(float(lift_time_series_p(np.sin(theta),points)[0]),-np.pi,places=12)
        rotated=2.5*np.column_stack([np.cos(theta+np.pi/2),np.sin(theta+np.pi/2)])
        self.assertAlmostEqual(float(lift_time_series_p(np.sin(theta),rotated)[0]),0,places=12)

    def test_imports_do_not_create_files_or_launch_work(self):
        code='import shred_mpc, shred_mpc.config, shred_mpc.dataset, shred_mpc.pressure, shred_mpc.temporal, shred_mpc.foam, shred_mpc.observables, shred_mpc.cli.temporal, shred_mpc.cli.control_table, shred_mpc.cli.dataset, shred_mpc.cli.historical, shred_mpc.cli.export_pressure'
        with tempfile.TemporaryDirectory() as d:
            subprocess.run([sys.executable,'-B','-c',code],cwd=d,check=True)
            self.assertEqual(list(Path(d).iterdir()),[])

    def test_expensive_launcher_requires_execute(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'run'
            result=subprocess.run([sys.executable,'-m','shred_mpc.cli.temporal','--run-dir',str(out)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertIn('--execute',result.stderr)
            self.assertFalse(out.exists())

    def test_control_table_cannot_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'table';out.write_text('preserve')
            result=subprocess.run([sys.executable,'-m','shred_mpc.cli.control_table','--output',str(out)],capture_output=True,text=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(out.read_text(),'preserve')

    def test_dataset_plan_is_isolated(self):
        with tempfile.TemporaryDirectory() as d:
            out=Path(d)/'dataset'
            command=[sys.executable,'-m','shred_mpc.cli.dataset','plan','--run-dir',str(out),'--cases','2','--seed','42']
            subprocess.run(command,check=True,capture_output=True)
            self.assertTrue((out/'template/constant/polyMesh/points.gz').exists())
            self.assertEqual(len(list((out/'cases').iterdir())),2)
            self.assertFalse(list(out.rglob('log.solver')))
            second=subprocess.run(command,capture_output=True)
            self.assertNotEqual(second.returncode,0)

if __name__=='__main__':unittest.main()
