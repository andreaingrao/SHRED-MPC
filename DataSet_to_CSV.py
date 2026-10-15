"""Export dataset pressure at cell centres using ParaView (run with pvpython)."""
import re
from pathlib import Path

from shred_utils.input_var import DATASET_CASES
from shred_utils.pressure_preprocessing import preprocess_pressure_csv_directory


def numeric_key(path):
    return int(re.search(r"_(\d+)\.vtk$", path.name).group(1))


def main():
    from paraview.simple import LegacyVTKReader, GetAnimationScene, CellCenters, SaveData, Delete

    dataset_path = Path(__file__).resolve().parent / "OpenFoam"
    cases = sorted((p for p in dataset_path.glob("case_omega*") if p.is_dir()),
                   key=lambda p: int(p.name.removeprefix("case_omega")))[:DATASET_CASES]
    if not cases:
        raise FileNotFoundError(f"No case_omega folders found in {dataset_path}")
    for case in cases:
        files = sorted((case / "VTK").glob(f"{case.name}_*.vtk"), key=numeric_key)
        if not files:
            raise FileNotFoundError(f"No VTK files found in {case / 'VTK'}")
        output = dataset_path / "CSV_files" / case.name
        if list(output.glob('datacsv_*.csv')):
            raise FileExistsError(f"Pressure export already exists: {output}")
        output.mkdir(parents=True, exist_ok=True)
        reader = LegacyVTKReader(FileNames=[str(p) for p in files])
        GetAnimationScene().UpdateAnimationUsingDataTimeSteps()
        centers = CellCenters(Input=reader)
        SaveData(str(output / 'datacsv.csv'), proxy=centers,
                 FieldAssociation='Point Data', AddMetaData=1, WriteTimeSteps=1,
                 WriteTimeStepsSeparately=1, ChooseArraysToWrite=1, PointDataArrays=['p'])
        preprocess_pressure_csv_directory(output)
        Delete(centers)
        Delete(reader)
        print(f"Exported {case.name}")


if __name__ == '__main__':
    main()
