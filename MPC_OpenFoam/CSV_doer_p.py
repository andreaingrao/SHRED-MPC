"""Export pressure at cell centres for the current MPC segment (pvpython)."""
import os
from pathlib import Path


def _timestep_key(path):
    return float(path.stem.rsplit('_', 1)[1])


def main():
    from paraview.simple import LegacyVTKReader, GetAnimationScene, CellCenters, SaveData, Delete

    base_dir = Path(__file__).resolve().parent
    files = sorted((base_dir / 'VTK').glob('*.vtk'), key=_timestep_key)
    start, end = os.environ.get('MPC_VTK_START_STEP'), os.environ.get('MPC_VTK_END_STEP')
    if start is not None and end is not None:
        files = [p for p in files if float(start) <= _timestep_key(p) <= float(end)]
    if not files:
        raise FileNotFoundError(f"No VTK files for the requested segment in {base_dir / 'VTK'}")
    reader = LegacyVTKReader(FileNames=[str(p) for p in files])
    GetAnimationScene().UpdateAnimationUsingDataTimeSteps()
    centers = CellCenters(Input=reader)
    output = base_dir / 'CSV_data'
    output.mkdir(exist_ok=True)
    SaveData(str(output / 'CFD.csv'), proxy=centers,
             FieldAssociation='Point Data', AddMetaData=1, WriteTimeSteps=1,
             WriteTimeStepsSeparately=1, ChooseArraysToWrite=1, PointDataArrays=['p'])
    # ParaView omits the index when the reader has only one timestep.
    if (output / 'CFD.csv').exists():
        (output / 'CFD.csv').rename(output / 'CFD_0.csv')
    Delete(centers)
    Delete(reader)
    print('Pressure CSV export complete.')


if __name__ == '__main__':
    main()
