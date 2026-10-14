import csv
from pathlib import Path


FREESTREAM_U = 0.5
PRESSURE_DYNAMIC_Q = 0.5 * FREESTREAM_U**2
PRESSURE_COLUMN = "p"
RAW_PRESSURE_COLUMN = "p_raw"
X_COLUMN = "Points:0"


def pressure_difference_normalized(pressure_values, x_values, q_inf=PRESSURE_DYNAMIC_Q, x_tolerance=None):
    pressure = [float(value) for value in pressure_values]
    x = [float(value) for value in x_values]

    if len(pressure) != len(x):
        raise ValueError(f"pressure/x length mismatch: {len(pressure)} vs {len(x)}")
    if not pressure:
        raise ValueError("cannot preprocess empty pressure field")
    if abs(q_inf) < 1e-12:
        raise ValueError("q_inf must be non-zero")

    x_min = min(x)
    x_range = max(x) - x_min
    tolerance = x_tolerance if x_tolerance is not None else max(1e-9, x_range * 1e-6)
    left_ids = [idx for idx, value in enumerate(x) if value <= x_min + tolerance]

    if not left_ids:
        left_ids = [min(range(len(x)), key=lambda idx: x[idx])]

    p_inf = sum(pressure[idx] for idx in left_ids) / len(left_ids)
    normalized = [(value - p_inf) / q_inf for value in pressure]

    return normalized, p_inf


def _read_csv_rows(csv_path):
    with open(csv_path, "r", newline="") as file:
        reader = csv.DictReader(file)
        rows = list(reader)
        fieldnames = reader.fieldnames

    if fieldnames is None:
        raise ValueError(f"{csv_path} has no CSV header")

    return fieldnames, rows


def _pressure_source_column(fieldnames, pressure_column=PRESSURE_COLUMN, raw_column=RAW_PRESSURE_COLUMN, source_column=None):
    if source_column is not None:
        if source_column not in fieldnames:
            raise KeyError(f"Pressure source column '{source_column}' not found. Available columns: {fieldnames}")
        return source_column

    if raw_column is not None and raw_column in fieldnames:
        return raw_column
    if pressure_column in fieldnames:
        return pressure_column

    raise KeyError(f"Pressure column '{pressure_column}' not found. Available columns: {fieldnames}")


def processed_pressure_from_csv(
    csv_path,
    pressure_column=PRESSURE_COLUMN,
    x_column=X_COLUMN,
    q_inf=PRESSURE_DYNAMIC_Q,
    raw_column=RAW_PRESSURE_COLUMN,
    source_column=None,
):
    fieldnames, rows = _read_csv_rows(csv_path)

    pressure_source = _pressure_source_column(
        fieldnames=fieldnames,
        pressure_column=pressure_column,
        raw_column=raw_column,
        source_column=source_column,
    )
    if x_column not in fieldnames:
        raise KeyError(f"X coordinate column '{x_column}' not found in {csv_path}. Available columns: {fieldnames}")

    pressure_values = [row[pressure_source] for row in rows]
    x_values = [row[x_column] for row in rows]
    processed, _ = pressure_difference_normalized(
        pressure_values=pressure_values,
        x_values=x_values,
        q_inf=q_inf,
    )

    return processed


def preprocess_pressure_csv(
    csv_path,
    pressure_column=PRESSURE_COLUMN,
    x_column=X_COLUMN,
    output_column=PRESSURE_COLUMN,
    q_inf=PRESSURE_DYNAMIC_Q,
    raw_column=RAW_PRESSURE_COLUMN,
    preserve_raw=True,
):
    csv_path = Path(csv_path)
    fieldnames, rows = _read_csv_rows(csv_path)

    pressure_source = _pressure_source_column(
        fieldnames=fieldnames,
        pressure_column=pressure_column,
        raw_column=raw_column if preserve_raw else None,
    )
    if x_column not in fieldnames:
        raise KeyError(f"X coordinate column '{x_column}' not found in {csv_path}. Available columns: {fieldnames}")

    if preserve_raw and raw_column is not None and raw_column not in fieldnames:
        fieldnames.append(raw_column)
        for row in rows:
            row[raw_column] = row[pressure_source]
        pressure_source = raw_column

    pressure_values = [row[pressure_source] for row in rows]
    x_values = [row[x_column] for row in rows]
    processed, _ = pressure_difference_normalized(
        pressure_values=pressure_values,
        x_values=x_values,
        q_inf=q_inf,
    )

    if output_column not in fieldnames:
        fieldnames.append(output_column)

    for row, value in zip(rows, processed):
        row[output_column] = f"{value:.12g}"

    tmp_path = csv_path.with_suffix(csv_path.suffix + ".tmp")
    with open(tmp_path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    tmp_path.replace(csv_path)


def preprocess_pressure_csv_directory(
    directory,
    pattern="datacsv_*.csv",
    pressure_column=PRESSURE_COLUMN,
    x_column=X_COLUMN,
    output_column=PRESSURE_COLUMN,
    q_inf=PRESSURE_DYNAMIC_Q,
    raw_column=RAW_PRESSURE_COLUMN,
    preserve_raw=True,
):
    directory = Path(directory)
    files = sorted(directory.glob(pattern), key=lambda path: _numeric_suffix(path))

    if not files:
        raise FileNotFoundError(f"No files matching {pattern} found in {directory}")

    for csv_file in files:
        preprocess_pressure_csv(
            csv_path=csv_file,
            pressure_column=pressure_column,
            x_column=x_column,
            output_column=output_column,
            q_inf=q_inf,
            raw_column=raw_column,
            preserve_raw=preserve_raw,
        )

    return files


def _numeric_suffix(path):
    stem = Path(path).stem
    try:
        return int(stem.rsplit("_", 1)[1])
    except (IndexError, ValueError):
        return 0
