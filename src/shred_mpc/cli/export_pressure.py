"""Headless ParaView export for the owned temporal-evaluation CFD case."""
import argparse
from pathlib import Path




def main():
    from paraview.simple import CellCenters, Delete, LegacyVTKReader, SaveData
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, required=True)
    parser.add_argument("--write-frequency", type=float, required=True)
    args = parser.parse_args()
    case = Path.cwd()
    files = []
    for path in (case / "VTK").glob("*.vtk"):
        try:
            stamp = float(path.stem.rsplit("_", 1)[1])
        except ValueError:
            continue
        files.append((stamp, path))
    files.sort(key=lambda item: item[0])
    if len(files) != args.samples:
        raise ValueError(f"Expected {args.samples} VTK samples, found {len(files)}")
    for index, (stamp, _) in enumerate(files):
        if abs(stamp - index / args.write_frequency) > 1e-7:
            raise ValueError(f"Unexpected VTK time {stamp} at sample {index}; expected {index / args.write_frequency}")
    reader = LegacyVTKReader(FileNames=[str(path) for _, path in files])
    centers = CellCenters(Input=reader)
    output = case / "CSV_data"
    output.mkdir(exist_ok=True)
    try:
        SaveData(str(output / "CFD.csv"), proxy=centers, FieldAssociation="Point Data",
                 AddMetaData=1, WriteTimeSteps=1, WriteTimeStepsSeparately=1,
                 ChooseArraysToWrite=1, PointDataArrays=["p"])
    finally:
        Delete(centers)
        Delete(reader)
    for sample in range(args.samples):
        if not (output / f"CFD_{sample}.csv").is_file():
            raise ValueError(f"Missing exported sample CFD_{sample}.csv")


if __name__ == "__main__":
    main()
