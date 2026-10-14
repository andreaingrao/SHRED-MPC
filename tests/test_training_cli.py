"""Small synthetic integration test; never used as research performance evidence."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np

@unittest.skipUnless(importlib.util.find_spec('pyshred'),'research extra not installed')
class TrainingCLIIntegration(unittest.TestCase):
    def test_train_only_fits_persists_and_resumes_without_cfd(self):
        import torch
        from shred_mpc import config
        from shred_mpc.cli.temporal import main
        old_threads=torch.get_num_threads();torch.set_num_threads(1)
        try:
            with tempfile.TemporaryDirectory() as d:
                base=Path(d);data=base/'data';rng=np.random.default_rng(31)
                for case in range(10):
                    folder=data/f'case_omega{case}';folder.mkdir(parents=True)
                    (folder/'omega_t_table.txt').write_text(''.join(f'({j/5}\t0)\n' for j in range(12)))
                    for j in range(12):
                        np.savetxt(folder/f'datacsv_{j}.csv',np.column_stack([rng.normal(size=6),np.arange(6),np.ones(6)]),delimiter=',',header='p,Points:0,Points:1',comments='')
                out=base/'run'
                args=['shred-temporal','--execute','--train-only','--run-dir',str(out),'--dataset',str(data),'--dataset-limit','10','--epochs','1','--replicates','1','--sensors','0','2']
                with patch.multiple(config,shred_lags=3,meas_offset=3,meas_delta_t=3,CONTROL_HORIZON=2,TIMESTEPS=11,shred_pod_compression=2),patch('sys.argv',args),patch('shred_mpc.cli.temporal.shutil.which',side_effect=AssertionError('CFD preflight reached')),contextlib.redirect_stdout(io.StringIO()):
                    main()
                model=out/'training_000/weights.pt';before=model.read_bytes()
                self.assertTrue((out/'training_000/transforms.pkl').exists())
                self.assertFalse((out/'cfd_template').exists())
                with patch('sys.argv',['shred-temporal','--resume',str(out),'--train-only','--execute']),contextlib.redirect_stdout(io.StringIO()):main()
                self.assertEqual(model.read_bytes(),before)
                self.assertFalse(list(out.rglob('log.solver')))
        finally:torch.set_num_threads(old_threads)

if __name__=='__main__':unittest.main()
