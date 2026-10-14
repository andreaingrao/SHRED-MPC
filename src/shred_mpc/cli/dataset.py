"""Plan, simulate, or export a new dataset without touching the source cases."""
import argparse
import json
from pathlib import Path
import shutil
import numpy as np
from shred_mpc import config
from shred_mpc.dataset import dataset_profile_for_run, random_dataSet_generator, generate_file
from shred_mpc.temporal import EvaluationSettings, atomic_json, generate_cfd_reference

from shred_mpc.paths import repository_root
ROOT = repository_root()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['plan', 'simulate', 'export'])
    parser.add_argument('--run-dir', type=Path, required=True)
    parser.add_argument('--template', type=Path, default=ROOT/'cfd/templates/training')
    parser.add_argument('--cases', type=int, default=config.DATASET_CASES)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    run = args.run_dir.resolve()
    if args.phase == 'plan':
        if not 1 <= args.cases <= config.DATASET_CASES:
            parser.error(f'--cases must be in [1, {config.DATASET_CASES}]')
        for part in ['0','constant','system']:
            if not (args.template/part).is_dir():
                parser.error(f'Template is missing {part}')
        run.mkdir(parents=True, exist_ok=False)
        (run/'template').mkdir()
        for part in ['0','constant','system']:
            shutil.copytree(args.template/part, run/'template'/part)
        atomic_json(run/'dataset.json', {'cases':args.cases, 'seed':args.seed,
                    'samples':config.TIMESTEPS+1, 'write_frequency':config.WRITE_FREQUENCY,
                    'settings':EvaluationSettings().__dict__})
        for i in range(args.cases):
            duration, profile, _ = dataset_profile_for_run(i)
            omega = random_dataSet_generator(config.TIMESTEPS,duration,config.omega0,profile,seed=args.seed+i)
            case = run/'cases'/f'case_omega{i}'
            case.mkdir(parents=True)
            np.save(case/'omega.npy', omega)
            generate_file(case/'omega_t_table.txt',omega,1/config.WRITE_FREQUENCY)
            atomic_json(case/'profile.json',{'profile':profile,'seed':args.seed+i})
        print(f'Planned {args.cases} cases in {run}; no CFD executed.')
        return
    manifest=json.loads((run/'dataset.json').read_text())
    settings=EvaluationSettings(**manifest['settings'])
    if args.phase == 'simulate':
        if not args.execute:
            parser.error('CFD execution requires --execute')
        for command in ['blockMesh',settings.solver,'foamToVTK','pvpython']:
            if not shutil.which(command):
                parser.error(f'{command} unavailable; load OpenFOAM/ParaView environment')
        for i in range(manifest['cases']):
            case=run/'cases'/f'case_omega{i}'
            generate_cfd_reference(case,run/'template',{'omega':np.load(case/'omega.npy')},settings,
                                   Path(__file__).with_name('export_pressure.py'))
    else:
        # Preflight before creating any export; complete references are required.
        for i in range(manifest['cases']):
            if not (run/'cases'/f'case_omega{i}'/'cfd_reference.npz').is_file():
                parser.error(f'Missing reference for case {i}; complete simulation first')
        out=run/'CSV_files'
        out.mkdir(exist_ok=False)
        for i in range(manifest['cases']):
            case=run/'cases'/f'case_omega{i}'; dest=out/case.name; dest.mkdir()
            shutil.copy2(case/'omega_t_table.txt',dest/'omega_t_table.txt')
            with np.load(case/'cfd_reference.npz') as data:
                for t,p in enumerate(data['pressure']):
                    np.savetxt(dest/f'datacsv_{t}.csv',np.column_stack([p,data['points']]),
                               delimiter=',',header='p,Points:0,Points:1',comments='')
        atomic_json(out/'pressure-convention.json',{'p':'Cp = (kinematic p - leftmost-cell mean)/0.125',
                     'source':'cfd_reference.npz', 'raw_pressure_in_csv':False})
        print(f'Exported {out}; existing exports are never overwritten.')


if __name__ == '__main__':
    main()
