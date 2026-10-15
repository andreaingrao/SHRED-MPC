import random

from shred_utils import input_var as config


def _round_omega(value):
    value = round(float(value), config.OMEGA_TABLE_DECIMALS)
    return 0.0 if value == -0.0 else value


def _format_table_float(value, decimals):
    value = round(float(value), decimals)
    if value == -0.0:
        value = 0.0
    text = f"{value:.{decimals}f}"
    return text.rstrip("0").rstrip(".") if "." in text else text


def dataset_profile_for_run(run_index):
    """Interleave the four families while preserving the configured counts."""
    families = (
        ("ramp", config.DATASET_RAMP_CASES),
        ("ramp_hold", config.DATASET_RAMP_HOLD_CASES),
        ("varied", config.DATASET_VARIED_CASES),
        ("extended_hold", config.DATASET_EXTENDED_HOLD_CASES),
    )
    counts = [count for _, count in families]
    if any(not isinstance(count, int) or count < 0 for count in counts):
        raise ValueError("Dataset family counts must be non-negative integers")
    total = sum(counts)
    run_index = 0 if run_index is None else run_index
    if not isinstance(run_index, int) or not 0 <= run_index < total:
        raise ValueError(f"run index {run_index} is outside DATASET_CASES={total}")

    assigned = [0] * len(families)
    for index in range(run_index + 1):
        available = [i for i, count in enumerate(counts) if assigned[i] < count]
        choice = max(available, key=lambda i: counts[i] * (index + 1) - assigned[i] * total)
        assigned[choice] += 1
    profile = families[choice][0]
    return config.CONTROL_HORIZON, profile, profile


def _validate_durations(values, name, allow_zero=False):
    minimum = 0 if allow_zero else 1
    if not values or any(not isinstance(value, int) or value < minimum for value in values):
        raise ValueError(f"{name} must contain integers >= {minimum}")


def _random_terminal_omega(current, ramp_samples, rng):
    """Choose a feasible random move; longer ramps allow proportionately larger moves."""
    midpoint = config.MPC_OMEGA_GRID_SIZE // 2
    max_move = config.MPC_OMEGA_SEARCH_RADIUS * ramp_samples / config.CONTROL_HORIZON
    spacing = max_move / midpoint
    candidates = []
    for index in range(config.MPC_OMEGA_GRID_SIZE):
        target = _round_omega(current + (index - midpoint) * spacing)
        if (
            config.MPC_OMEGA_MIN <= target <= config.MPC_OMEGA_MAX
            and abs(target - current) <= max_move + 1e-12
        ):
            candidates.append(target)
    return rng.choice(candidates)


def random_dataSet_generator(timesteps, swap_deltaT, omega0, profile="ramp", *, seed=None):
    """Generate random bounded control histories on the complete CFD time grid.

    ``timesteps`` is the total number of written intervals, including the
    initial constant history. Return timesteps + 1 values, from time zero to
    the CFD end time. The first shred_lags values equal omega0.

    ``swap_deltaT`` is the base ramp duration. Families are successive ramps,
    ramp-and-hold, varied-duration ramps, and extended holds between ramps.
    The old ``hold`` name is an alias for extended_hold; all moves are ramps.
    A last segment may be truncated at the end time, preserving its slope.
    No trajectory deduplication or coverage-based selection is performed.
    """
    if not isinstance(timesteps, int) or timesteps < 0:
        raise ValueError("timesteps must be a non-negative integer")
    if not isinstance(swap_deltaT, int) or swap_deltaT <= 0:
        raise ValueError("swap_deltaT must be a positive integer")
    if not isinstance(config.shred_lags, int) or config.shred_lags <= 0:
        raise ValueError("shred_lags must be a positive integer")
    if config.CONTROL_HORIZON <= 0 or config.MPC_OMEGA_SEARCH_RADIUS <= 0:
        raise ValueError("CONTROL_HORIZON and MPC_OMEGA_SEARCH_RADIUS must be positive")
    if config.MPC_OMEGA_GRID_SIZE < 3 or config.MPC_OMEGA_GRID_SIZE % 2 == 0:
        raise ValueError("MPC_OMEGA_GRID_SIZE must be an odd integer >= 3")
    if config.MPC_OMEGA_MIN >= config.MPC_OMEGA_MAX:
        raise ValueError("Invalid omega bounds")
    if not config.MPC_OMEGA_MIN <= float(omega0) <= config.MPC_OMEGA_MAX:
        raise ValueError("omega0 must lie within the omega bounds")

    profile = "extended_hold" if profile == "hold" else profile
    if profile not in ("ramp", "ramp_hold", "varied", "extended_hold"):
        raise ValueError(f"Unknown dataset profile: {profile}")
    if profile == "ramp_hold" and config.meas_delta_t < swap_deltaT:
        raise ValueError("meas_delta_t must be at least the ramp duration")
    if profile == "varied":
        _validate_durations(config.DATASET_VARIED_RAMP_SAMPLES, "DATASET_VARIED_RAMP_SAMPLES")
        _validate_durations(config.DATASET_VARIED_HOLD_SAMPLES, "DATASET_VARIED_HOLD_SAMPLES", True)
    if profile == "extended_hold":
        _validate_durations(config.DATASET_EXTENDED_HOLD_SAMPLES, "DATASET_EXTENDED_HOLD_SAMPLES")

    rng = random if seed is None else random.Random(seed)
    length = timesteps + 1
    omega = [_round_omega(omega0)] * min(config.shred_lags, length)
    max_increment = config.MPC_OMEGA_SEARCH_RADIUS / config.CONTROL_HORIZON
    rounding_tolerance = 10 ** (-config.OMEGA_TABLE_DECIMALS)

    while len(omega) < length:
        if profile == "varied":
            ramp_samples = rng.choice(config.DATASET_VARIED_RAMP_SAMPLES)
            hold_samples = rng.choice(config.DATASET_VARIED_HOLD_SAMPLES)
        else:
            ramp_samples = swap_deltaT
            if profile == "ramp_hold":
                hold_samples = config.meas_delta_t - ramp_samples
            elif profile == "extended_hold":
                hold_samples = rng.choice(config.DATASET_EXTENDED_HOLD_SAMPLES)
            else:
                hold_samples = 0

        start = omega[-1]
        target = _random_terminal_omega(start, ramp_samples, rng)
        for step in range(1, min(ramp_samples, length - len(omega)) + 1):
            value = _round_omega(start + (target - start) * step / ramp_samples)
            if abs(value - omega[-1]) > max_increment + rounding_tolerance:
                raise ValueError("Rounded omega ramp exceeds the configured rate limit")
            omega.append(value)
        omega.extend([omega[-1]] * min(hold_samples, length - len(omega)))

    return omega


def generate_file(filename, omegas, timestep_delta, start_time=0.0):
    with open(filename, "w") as file:
        for index, omega in enumerate(omegas):
            time = start_time + index * timestep_delta
            time_text = _format_table_float(time, 6)
            omega_text = _format_table_float(omega, config.OMEGA_TABLE_DECIMALS)
            file.write(f"({time_text}\t{omega_text})\n")
