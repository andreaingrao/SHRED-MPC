"""Prepare an isolated copy of the historical MPC runner; execution is explicit.

This preserves historical scientific limitations. It is not the temporal evaluator.
"""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys

from shred_mpc.paths import repository_root
ROOT = repository_root()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--dataset',type=Path)
    p.add_argument('--mode',choices=['mpc','statistics'],default='mpc')
    p.add_argument('--execute',action='store_true')
    a=p.parse_args(); run=a.run_dir.resolve()
    if not a.execute:
        if a.dataset is None or not a.dataset.is_dir():
            p.error('--dataset must identify an existing CSV dataset (read only)')
        # Reject incomplete data before any model fit, using current defaults.
        from shred_mpc import config
        from shred_mpc.temporal import validate_training_dataset
        validate_training_dataset(a.dataset,config.DATASET_CASES,config.TIMESTEPS+1,config.WRITE_FREQUENCY)
        run.mkdir(parents=True,exist_ok=False)
        source=ROOT/'archive/historical-pressure'
        for path in source.rglob('*.py.txt'):
            out=run/path.relative_to(source).with_suffix(''); out.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,out)
        shutil.copytree(ROOT/'cfd/templates/mpc',run/'MPC_OpenFoam',dirs_exist_ok=True)
        # Runtime data paths are explicit. Caches and CFD output stay in this owned directory.
        (run/'shred_utils/global_var.py').write_text(
            'import os\nfrom shred_utils.input_var import *\n'
            f'DATA_DIRECTORY = {str(a.dataset.resolve())!r}\n'
            'NUMPY_DIRECTORY_NAME = "NumPy_files"\nNUMPY_DIRECTORY = "cache"\n'
            'ALTERNATIVE_DATA_DIR = "CSV_files"\nALTERNATIVE_NUMPY_DIR = "cache/alternative"\n')
        (run/'cache').mkdir()
        (run/'historical-run.json').write_text(json.dumps({'mode':a.mode,'dataset':str(a.dataset.resolve())},indent=2))
        print(f'Prepared {run}; no training or CFD. Inspect docs/workflows.md before --execute.')
        return
    settings=json.loads((run/'historical-run.json').read_text())
    for name in ['pimpleFoam','foamToVTK','pvpython']:
        if not shutil.which(name):p.error(f'{name} is unavailable')
    if (run/'EXECUTION_STARTED').exists():
        p.error('This historical run has already started; prepare a new directory')
    (run/'EXECUTION_STARTED').write_text('Historical run explicitly invoked. Do not reuse this directory.\n')
    script='shred_MPC_test_p.py' if settings['mode']=='mpc' else 'shred_test_p_stat.py'
    subprocess.run([sys.executable,script],cwd=run,check=True)


if __name__=='__main__':
    main()
