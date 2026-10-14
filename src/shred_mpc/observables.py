"""Historical pressure-ring observable, not total or diameter-normalized lift.

Arithmetic retained; provide explicit coordinates. See docs/method.md.
"""
import numpy as np

def _pressure_lift_geometry(points):
    points = np.asarray(points, dtype=float)
    n_points = points.shape[0]
    cache = getattr(_pressure_lift_geometry, "_cache", {})
    key = (points.shape, points.tobytes())

    if key not in cache:
        center = np.asarray((0.0, 0.0), dtype=float)
        radius = np.linalg.norm(points - center, axis=1)
        sorted_ids = np.argsort(radius)
        sorted_radius = radius[sorted_ids]

        search_count = min(240, sorted_radius.size)
        radius_gap = np.diff(sorted_radius[:search_count])
        positive_gap = radius_gap[radius_gap > 1e-12]
        gap_scale = np.median(positive_gap[: min(60, positive_gap.size)]) if positive_gap.size else 0.0
        gap_threshold = max(0.05, 8.0 * gap_scale)
        ring_breaks = np.flatnonzero(radius_gap > gap_threshold)
        ring_count = int(ring_breaks[0] + 1) if ring_breaks.size else min(120, n_points)

        surface_ids = sorted_ids[:ring_count]
        theta = np.mod(np.arctan2(points[surface_ids, 1], points[surface_ids, 0]), 2.0 * np.pi)
        theta_order = np.argsort(theta)
        surface_ids = surface_ids[theta_order]
        theta = theta[theta_order]

        dtheta_prev = np.mod(theta - np.roll(theta, 1), 2.0 * np.pi)
        dtheta_next = np.mod(np.roll(theta, -1) - theta, 2.0 * np.pi)
        theta_weights = 0.5 * (dtheta_prev + dtheta_next)
        sin_theta = np.sin(theta)

        x = points[:, 0]
        left_x = np.min(x)
        x_tol = max(1e-9, np.ptp(x) * 1e-6)
        left_ids = np.flatnonzero(x <= left_x + x_tol)
        if left_ids.size == 0:
            left_ids = np.asarray([int(np.argmin(x))])

        cache[key] = surface_ids, theta_weights, sin_theta, left_ids
        _pressure_lift_geometry._cache = cache

    return cache[key]

def lift_time_series_p(state_p, points_vec=None):
    sp = np.asarray(state_p, dtype=float)
    if sp.ndim == 1:
        sp = sp[None, :]
    elif sp.ndim > 2:
        sp = sp.reshape(-1, sp.shape[-1])

    n_points = sp.shape[1]
    if points_vec is None:
        raise ValueError("Explicit points_vec is required; no implicit dataset reads")
    points = np.asarray(points_vec, dtype=float)
    if points.shape[0] != n_points:
        raise ValueError(
            f"point count mismatch: pressure input has {n_points} points, "
            f"but points_vec has {points.shape[0]}"
        )

    surface_ids, theta_weights, sin_theta, _ = _pressure_lift_geometry(points)
    cp = sp[:, surface_ids]

    # state_p is already (p - p_inf) / q_inf. Positive lift is upward;
    # pressure acts opposite to the outward normal.
    return -np.sum(cp * sin_theta[None, :] * theta_weights[None, :], axis=1)

def lift_generated_p(state_p, points_vec=None) -> float:
    return float(lift_time_series_p(state_p, points_vec=points_vec).sum())
