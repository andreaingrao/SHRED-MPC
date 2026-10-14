import argparse
import sys
from pathlib import Path

from shred_mpc.dataset import (
    dataset_profile_for_run,
    generate_file,
    random_dataSet_generator,
)
from shred_mpc import config


FILE_NAME = "0/omega_t_table.txt"


def main():
    parser = argparse.ArgumentParser(description="Generate a random dataset omega trajectory.")
    parser.add_argument("run_index", nargs="?", type=int, default=0)
    parser.add_argument("--seed", type=int, default=config.DATASET_SEED,
                        help="Base seed; the run index is added to reproduce individual cases.")
    parser.add_argument("--output", type=Path, default=Path(FILE_NAME))
    args = parser.parse_args()

    ramp_samples, profile, profile_name = dataset_profile_for_run(args.run_index)
    seed = None if args.seed is None else args.seed + args.run_index
    omegas = random_dataSet_generator(
        timesteps=config.TIMESTEPS,
        swap_deltaT=ramp_samples,
        omega0=config.omega0,
        profile=profile,
        seed=seed,
    )
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}; choose a new output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    generate_file(args.output, omegas, timestep_delta=1 / config.WRITE_FREQUENCY)
    print(
        f"Generated {profile_name} profile: run={args.run_index}, "
        f"omega0={config.omega0}, samples={len(omegas)}, seed={seed}",
        file=sys.stderr,
    )
    # Preserve the table-only stdout contract used by shell callers.
    sys.stdout.write(args.output.read_text())


if __name__ == "__main__":
    main()
