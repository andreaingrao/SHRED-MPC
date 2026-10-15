"""Write the existing dataset time settings when executed by runDataSet.sh."""
import sys
from pathlib import Path

script_dir = Path(__file__).resolve().parent
sys.path.insert(0, str(script_dir.parent))
from shred_utils.input_var import DATASET_CASES, TIMESTEPS, WRITE_FREQUENCY, SIM_DELTA_T_fraction


def main():
    values = {
        "end_time.txt": round(float(TIMESTEPS / WRITE_FREQUENCY), 1),
        "delta_t.txt": round(float(1 / WRITE_FREQUENCY / SIM_DELTA_T_fraction), 3),
        "write_interval.txt": round(float(SIM_DELTA_T_fraction), 3),
    }
    for name, value in values.items():
        (script_dir / "system" / name).write_text(str(value))
    print(DATASET_CASES)


if __name__ == "__main__":
    main()
