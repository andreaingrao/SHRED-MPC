"""Pure boundary-control formatting extracted from the historical workflow."""
import re
import numpy as np

def _foam_float(value):
    text = f"{float(value):.12g}"
    return "0" if text == "-0" else text

def _omega_table_block(omega_times, omega_values):
    lines = [
        "        omega",
        "        {",
        "            type            table;",
        "            values",
        f"            {len(omega_values)}",
        "            (",
    ]
    for time, omega in zip(omega_times, omega_values):
        lines.append(f"            ({_foam_float(time)} {_foam_float(omega)})")
    lines.extend([
        "            )",
        "            ;",
        "        }",
    ])
    return lines

def _replace_obstacle_omega_block(field_text, omega_times, omega_values):
    lines = field_text.splitlines()
    obstacle_idx = None

    for i, line in enumerate(lines):
        if line.strip() == "obstacle":
            obstacle_idx = i
            break

    if obstacle_idx is None:
        raise ValueError("Could not find obstacle boundary in U field.")

    omega_idx = None
    value_idx = None
    for i in range(obstacle_idx + 1, len(lines)):
        stripped = lines[i].strip()
        keyword = stripped.split(None, 1)[0] if stripped else ""
        if keyword == "omega":
            omega_idx = i
        elif omega_idx is not None and keyword == "value":
            value_idx = i
            break
        elif omega_idx is not None and keyword == "frontAndBack":
            break

    if omega_idx is None or value_idx is None:
        raise ValueError("Could not isolate omega block inside obstacle boundary.")

    new_lines = lines[:omega_idx] + _omega_table_block(omega_times, omega_values) + lines[value_idx:]
    return "\n".join(new_lines) + "\n"
