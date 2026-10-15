from pathlib import Path

# Default data location; the launcher also accepts --dataset.
DATA_DIRECTORY = str(Path(__file__).resolve().parents[1] / "OpenFoam" / "CSV_files")
