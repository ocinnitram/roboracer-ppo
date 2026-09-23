# SPDX-License-Identifier: LGPL-3.0-only
# Modified from TUMFTM trajectory_planning_helpers and global_racetrajectory_optimization; see THIRD_PARTY_NOTICES.md.
"""Minimum-curvature raceline, then wall widths along that line."""
from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np
import pandas as pd
import quadprog
import yaml
from scipy import interpolate, optimize
from scipy.interpolate import splprep
from scipy.signal import savgol_filter

from sim.env import VEHICLE_YAML
from trackgen.paths import TrackPaths
from trackgen.plotting import draw_map, new_axes, show_plot
from trackgen.stages.centreline import splev_xy

Walls = tuple[np.ndarray, float, tuple[float, float]]


def load_vehicle() -> dict[str, float]:
    """Load the car parameters from sim/vehicle.yaml."""
    data = yaml.safe_load(VEHICLE_YAML.read_text())
    return {str(key): float(value) for key, value in data.items()}


_CAR = load_vehicle()
STEPSIZE_OPTS = {
    "stepsize_prep": 0.1,
    "stepsize_reg": 0.05,
    "stepsize_interp_after_opt": 0.005,
}
REG_SMOOTH_OPTS = {"k_reg": 3, "s_reg": 5.0}
VEH_WIDTH = _CAR["ego_width_m"]
V_MAX = _CAR["max_speed_mps"]
# Lateral-acceleration cap of the printed lap estimate only; vehicle.yaml has no such limit.
A_MAX_LAT = 2.5
WIDTH_OPT_MINCURV = 0.9
KAPPA_BOUND = 0.8
MINCURV_STEP_M = 0.15
MINCURV_MAX_ITERS = 40
MINCURV_TOL_M = 0.1
MINCURV_GRID_M = 0.25
RAYCAST_RESOLUTION = 0.05
RAYCAST_SMOOTHING = 0.1
WALL_STEP_M = 0.01
WALL_MAX_M = 3.0


def import_track(file_path: str) -> np.ndarray:
    """Load a 10-column line table as [x, y, w_r, w_l], each width 3 cm narrower."""
    data = np.loadtxt(file_path, comments='#', delimiter=',', skiprows=1)
    reftrack_imp = np.column_stack((data[:, 0:2], data[:, 7] - 0.03, data[:, 6] - 0.03))
    w_tr_min = np.amin(reftrack_imp[:, 2] + reftrack_imp[:, 3])
    if w_tr_min < VEH_WIDTH + 0.1:
        print(f"WARNING: Min track width {w_tr_min:.2f}m is close to vehicle width {VEH_WIDTH:.2f}m")
    return reftrack_imp


def _raycast_widths(track: np.ndarray, normvec: np.ndarray, walls: Walls) -> None:
    """Overwrite track[:, 2:4] (w_r along +normal, w_l along -normal) with ray-cast free distances."""
    # Cap the inner wall short of the centre of curvature so 1 - n*kappa stays positive.
    free, resolution, origin = walls
    closed = np.vstack((track[:, :2], track[0, :2]))
    el = np.sqrt(np.sum(np.diff(closed, axis=0) ** 2, axis=1))
    kappa = calc_curv_num(path=track[:, :2], el_lengths=el)
    for i in range(len(track)):
        w_r, w_l = _wall_distances(track[i, :2], normvec[i], free, resolution, origin)
        safe_radius = 0.95 / (abs(float(kappa[i])) + 1e-6)
        if kappa[i] > 0.0:
            w_l = min(w_l, safe_radius)
        elif kappa[i] < 0.0:
            w_r = min(w_r, safe_radius)
        track[i, 2] = w_r
        track[i, 3] = w_l


def prep_track(reftrack_imp: np.ndarray, reg_smooth_opts: dict, stepsize_opts: dict, walls: Walls) -> tuple:
    """Smooth the centreline, fit closed splines, ray-cast its widths, and return normals."""
    reftrack_interp = spline_approximation(track=reftrack_imp,
                                           k_reg=reg_smooth_opts["k_reg"],
                                           s_reg=reg_smooth_opts["s_reg"],
                                           stepsize_prep=stepsize_opts["stepsize_prep"],
                                           stepsize_reg=stepsize_opts["stepsize_reg"])

    refpath_interp_cl = np.vstack((reftrack_interp[:, :2], reftrack_interp[0, :2]))
    coeffs_x_interp, coeffs_y_interp, a_interp, normvec_normalized_interp = calc_splines(refpath_interp_cl)

    # Smoothing moves the reference off the skeleton in hairpins, so its widths no longer bound the free space.
    _raycast_widths(reftrack_interp, normvec_normalized_interp, walls)

    if check_normals_crossing(track=reftrack_interp, normvec_normalized=normvec_normalized_interp, horizon=10):
        print("WARNING: At least one pair of normals is crossed! This may lead to unexpected results.", file=sys.stderr)

    return reftrack_interp, normvec_normalized_interp, a_interp, coeffs_x_interp, coeffs_y_interp


def normalize_psi(psi: np.ndarray) -> np.ndarray:
    """Wrap heading into [-pi, pi)."""
    psi_out = np.sign(psi) * np.mod(np.abs(psi), 2 * math.pi)
    psi_out[psi_out >= math.pi] -= 2 * math.pi
    psi_out[psi_out < -math.pi] += 2 * math.pi
    return psi_out


def create_raceline(refline: np.ndarray,
                    normvectors: np.ndarray,
                    alpha: np.ndarray,
                    stepsize_interp: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply lateral shifts and linearly resample the closed racing line; return xy, s and segment lengths."""
    raceline = refline + np.expand_dims(alpha, 1) * normvectors
    raceline_cl = np.vstack((raceline, raceline[0]))
    el_lengths_raceline = np.sqrt(np.sum(np.power(np.diff(raceline_cl, axis=0), 2), axis=1))
    s_raceline_nodes = np.cumsum(np.insert(el_lengths_raceline, 0, 0.0))
    total_dist_raceline = s_raceline_nodes[-1]

    num_interp_points = max(2, math.ceil(total_dist_raceline / stepsize_interp))
    s_raceline_interp_cl = np.linspace(0.0, total_dist_raceline, num_interp_points)
    x_interp = np.interp(s_raceline_interp_cl, s_raceline_nodes, raceline_cl[:, 0], period=total_dist_raceline)
    y_interp = np.interp(s_raceline_interp_cl, s_raceline_nodes, raceline_cl[:, 1], period=total_dist_raceline)

    raceline_interp = np.column_stack((x_interp, y_interp))[:-1]
    raceline_interp_cl = np.vstack((raceline_interp, raceline_interp[0]))
    el_lengths_interp = np.sqrt(np.sum(np.power(np.diff(raceline_interp_cl, axis=0), 2), axis=1))
    return raceline_interp, s_raceline_interp_cl[:-1], el_lengths_interp


def side_of_line(a: np.ndarray, b: np.ndarray, z: np.ndarray) -> float:
    """Return the side of point z relative to segment a->b (0 on the line)."""
    return float(np.sign((b[0] - a[0]) * (z[1] - a[1]) - (b[1] - a[1]) * (z[0] - a[0])))


def interp_track(track: np.ndarray, stepsize: float) -> np.ndarray:
    """Linearly resample every column of a closed track to a given spacing."""
    track_cl = np.vstack((track, track[0]))
    el_lengths_cl = np.sqrt(np.sum(np.power(np.diff(track_cl[:, :2], axis=0), 2), axis=1))
    el_lengths_cl[el_lengths_cl < 1e-9] = 1e-9
    dists_cum_cl = np.cumsum(el_lengths_cl)
    dists_cum_cl = np.insert(dists_cum_cl, 0, 0.0)
    total_dist = dists_cum_cl[-1]
    no_points_interp_cl = math.ceil(total_dist / stepsize) + 1
    dists_interp_cl = np.linspace(0.0, total_dist, no_points_interp_cl)
    track_interp_cl = np.zeros((no_points_interp_cl, track_cl.shape[1]))
    for i in range(track_cl.shape[1]):
        track_interp_cl[:, i] = np.interp(dists_interp_cl, dists_cum_cl, track_cl[:, i])
    return track_interp_cl[:-1]


def calc_spline_lengths(coeffs_x: np.ndarray, coeffs_y: np.ndarray, no_interp_points: int = 15) -> np.ndarray:
    """Estimate each cubic-spline segment length."""
    no_splines = coeffs_x.shape[0]
    spline_lengths = np.zeros(no_splines)
    t_steps = np.linspace(0.0, 1.0, no_interp_points)
    spl_coords = np.zeros((no_interp_points, 2))
    t_pow = np.vstack([t_steps**0, t_steps**1, t_steps**2, t_steps**3]).T
    for i in range(no_splines):
        spl_coords[:, 0] = t_pow @ coeffs_x[i, :]
        spl_coords[:, 1] = t_pow @ coeffs_y[i, :]
        diffs = np.diff(spl_coords, axis=0)
        seg_lengths = np.sqrt(np.sum(diffs**2, axis=1))
        spline_lengths[i] = np.sum(seg_lengths)
    return spline_lengths


def calc_splines(path: np.ndarray) -> tuple:
    """Fit closed cubic splines through a path whose last point repeats the first; return coeffs, system matrix, unit normals."""
    lengths = np.sqrt(np.sum(np.power(np.diff(path, axis=0), 2), axis=1))
    lengths = np.append(lengths, lengths[0])
    no_splines = path.shape[0] - 1
    lengths[lengths < 1e-9] = 1e-9
    scaling = lengths[:-1] / lengths[1:]
    M = np.zeros((no_splines * 4, no_splines * 4))
    b_x = np.zeros((no_splines * 4, 1))
    b_y = np.zeros((no_splines * 4, 1))
    template_M = np.array([[1,0,0,0,0,0,0,0],[1,1,1,1,0,0,0,0],[0,1,2,3,0,-1,0,0],[0,0,2,6,0,0,-2,0]])
    for i in range(no_splines):
        j = i * 4
        if i < no_splines - 1:
            M[j: j + 4, j: j + 8] = template_M
            M[j + 2, j + 5] *= scaling[i]
            M[j + 3, j + 6] *= math.pow(scaling[i], 2)
        else:
            M[j: j + 2, j: j + 4] = [[1,0,0,0],[1,1,1,1]]
        b_x[j: j + 2] = [[path[i,0]], [path[i + 1,0]]]
        b_y[j: j + 2] = [[path[i,1]], [path[i + 1,1]]]
    # Closure: heading and curvature continuous across the last spline into the first.
    M[-2, 1] = scaling[-1]
    M[-2, -3:] = [-1, -2, -3]
    M[-1, 2] = 2 * math.pow(scaling[-1], 2)
    M[-1, -2:] = [-2, -6]
    x_les = np.squeeze(np.linalg.solve(M, b_x))
    y_les = np.squeeze(np.linalg.solve(M, b_y))
    coeffs_x = np.reshape(x_les, (no_splines, 4))
    coeffs_y = np.reshape(y_les, (no_splines, 4))
    normvec = np.stack((coeffs_y[:, 1], -coeffs_x[:, 1]), axis=1)
    norm = np.linalg.norm(normvec, axis=1)
    normvec_normalized = np.zeros_like(normvec)
    valid_norm_indices = norm > 1e-9
    normvec_normalized[valid_norm_indices] = normvec[valid_norm_indices] / norm[valid_norm_indices, np.newaxis]
    if np.any(~valid_norm_indices):
        print(f"Warning: {np.sum(~valid_norm_indices)} zero-length or undefined tangent segments found in calc_splines. Normals set to [0,0].")
    return coeffs_x, coeffs_y, M, normvec_normalized


def check_normals_crossing(track: np.ndarray, normvec_normalized: np.ndarray, horizon: int = 10) -> bool:
    """Return True if neighbouring track normals cross within the track width."""
    no_points = track.shape[0]
    if no_points < 3:
        return False
    if horizon >= no_points / 2:
        horizon = int(no_points / 2) - 1
    if horizon < 1:
        return False
    les_mat = np.zeros((2, 2))
    idx_list = list(range(0, no_points))
    idx_list = idx_list[-horizon:] + idx_list + idx_list[:horizon]
    for idx in range(no_points):
        idx_neighbours = idx_list[idx:idx + 2 * horizon + 1]
        del idx_neighbours[horizon]
        idx_neighbours = np.array(idx_neighbours)
        if np.linalg.norm(normvec_normalized[idx]) < 1e-9:
            continue
        cross_prods = np.cross(normvec_normalized[idx], normvec_normalized[idx_neighbours])
        is_collinear_b = np.isclose(cross_prods, 0.0, atol=1e-6)
        idx_neighbours_rel = idx_neighbours[~is_collinear_b]
        for idx_comp in idx_neighbours_rel:
            const = track[idx_comp, :2] - track[idx, :2]
            les_mat[:, 0] = normvec_normalized[idx]
            les_mat[:, 1] = -normvec_normalized[idx_comp]
            try:
                lambdas = np.linalg.solve(les_mat, const)
            except np.linalg.LinAlgError:
                continue
            w_r_idx = track[idx, 2]
            w_l_idx = track[idx, 3]
            w_r_comp = track[idx_comp, 2]
            w_l_comp = track[idx_comp, 3]
            eps = 1e-6
            if (-w_l_idx - eps <= lambdas[0] <= w_r_idx + eps and
                    -w_l_comp - eps <= lambdas[1] <= w_r_comp + eps):
                # Normals meeting at their own base points are neighbours, not a crossing.
                if abs(lambdas[0]) > 1e-3 and abs(lambdas[1]) > 1e-3:
                    return True
    return False


def dist_to_p(t_glob: float, path_coeffs: tuple, p: np.ndarray) -> float:
    """Distance from p to the spline point at parameter t."""
    t_eval = np.clip(t_glob, 0.0, 1.0)
    s = np.array(interpolate.splev(t_eval, path_coeffs)).flatten()
    return float(np.linalg.norm(p - s))


def spline_approximation(track: np.ndarray, k_reg: int = 3, s_reg: float = 10, stepsize_prep: float = 1.0,
                         stepsize_reg: float = 3.0) -> np.ndarray:
    """Fit a smooth closed spline and interpolate track widths onto it."""
    track_interp = interp_track(track=track, stepsize=stepsize_prep)
    track_interp_cl = np.vstack((track_interp[:, :2], track_interp[0, :2]))

    track_cl = np.vstack((track, track[0]))
    no_points_track_cl = track_cl.shape[0]
    el_lengths_cl = np.sqrt(np.sum(np.power(np.diff(track_cl[:, :2], axis=0), 2), axis=1))
    el_lengths_cl[el_lengths_cl < 1e-9] = 1e-9
    dists_cum_cl = np.cumsum(el_lengths_cl)
    dists_cum_cl = np.insert(dists_cum_cl, 0, 0.0)
    total_len_orig = dists_cum_cl[-1]

    u_param = np.linspace(0, 1, len(track_interp_cl))
    tck_cl, _u = interpolate.splprep([track_interp_cl[:, 0], track_interp_cl[:, 1]], u=u_param, k=k_reg, s=s_reg, per=1)

    no_points_lencalc_cl = max(100, math.ceil(total_len_orig) * 4)
    path_smoothed_lencalc = np.array(interpolate.splev(np.linspace(0.0, 1.0, no_points_lencalc_cl), tck_cl)).T
    len_path_smoothed_est = np.sum(np.sqrt(np.sum(np.power(np.diff(path_smoothed_lencalc, axis=0), 2), axis=1)))

    no_points_reg_cl = math.ceil(len_path_smoothed_est / stepsize_reg) + 1
    t_reg = np.linspace(0.0, 1.0, no_points_reg_cl)
    path_smoothed = np.array(interpolate.splev(t_reg, tck_cl)).T[:-1]

    # Project every original point onto the smoothed spline to carry its widths over.
    dists_cl = np.zeros(no_points_track_cl)
    closest_point_cl = np.zeros((no_points_track_cl, 2))
    closest_t_glob_cl = np.zeros(no_points_track_cl)
    t_glob_guess_cl = dists_cum_cl / total_len_orig
    for i in range(no_points_track_cl):
        res = optimize.minimize(dist_to_p, x0=t_glob_guess_cl[i], args=(tck_cl, track_cl[i, :2]),
                                method='Nelder-Mead', bounds=[(0, 1)])
        if res.success:
            closest_t_glob_cl[i] = res.x[0]
            dists_cl[i] = res.fun
        else:
            print(f"Warning: Closest point optimization failed for index {i}. Using guess.")
            closest_t_glob_cl[i] = t_glob_guess_cl[i]
            dists_cl[i] = dist_to_p(closest_t_glob_cl[i], tck_cl, track_cl[i, :2])
        closest_point_cl[i] = interpolate.splev(closest_t_glob_cl[i], tck_cl)

    sides = np.zeros(no_points_track_cl - 1)
    for i in range(no_points_track_cl - 1):
        sides[i] = side_of_line(a=track_cl[i, :2], b=track_cl[i + 1, :2], z=closest_point_cl[i])
    sides_cl = np.hstack((sides, sides[0]))
    w_tr_right_new_cl = track_cl[:, 2] + sides_cl * dists_cl
    w_tr_left_new_cl = track_cl[:, 3] - sides_cl * dists_cl
    closest_t_glob_cl = np.mod(np.unwrap(closest_t_glob_cl * 2 * np.pi) / (2 * np.pi), 1.0)
    sort_indices = np.argsort(closest_t_glob_cl)
    closest_t_glob_cl_sorted = closest_t_glob_cl[sort_indices]
    unique_t = np.unique(closest_t_glob_cl_sorted)
    w_tr_right_unique = np.array([np.mean(w_tr_right_new_cl[sort_indices][closest_t_glob_cl_sorted == t]) for t in unique_t])
    w_tr_left_unique = np.array([np.mean(w_tr_left_new_cl[sort_indices][closest_t_glob_cl_sorted == t]) for t in unique_t])

    w_tr_right_smoothed_cl = np.interp(t_reg, unique_t, w_tr_right_unique, period=1.0)
    w_tr_left_smoothed_cl = np.interp(t_reg, unique_t, w_tr_left_unique, period=1.0)
    return np.column_stack((path_smoothed, w_tr_right_smoothed_cl[:-1], w_tr_left_smoothed_cl[:-1]))


def calc_curv_num(path: np.ndarray, el_lengths: np.ndarray, stepsize_psi_preview: float = 1.0,
                  stepsize_psi_review: float = 1.0, stepsize_curv_preview: float = 2.0,
                  stepsize_curv_review: float = 2.0) -> np.ndarray:
    """Numerically estimate curvature along a closed path from finite heading differences."""
    no_points = path.shape[0]
    avg_step = np.mean(el_lengths)
    ind_prev_psi = min(max(1, round(stepsize_psi_preview / avg_step)), no_points - 1)
    ind_rev_psi = min(max(1, round(stepsize_psi_review / avg_step)), no_points - 1)
    ind_prev_curv = min(max(1, round(stepsize_curv_preview / avg_step)), no_points - 1)
    ind_rev_curv = min(max(1, round(stepsize_curv_review / avg_step)), no_points - 1)

    path_padded = np.vstack((path[-ind_rev_psi:], path, path[:ind_prev_psi]))
    dx = path_padded[ind_rev_psi+ind_prev_psi:, 0] - path_padded[:-ind_rev_psi-ind_prev_psi, 0]
    dy = path_padded[ind_rev_psi+ind_prev_psi:, 1] - path_padded[:-ind_rev_psi-ind_prev_psi, 1]
    psi = normalize_psi(np.arctan2(dy, dx))

    psi_padded = np.concatenate((psi[-ind_rev_curv:], psi, psi[:ind_prev_curv]))
    delta_psi = normalize_psi(psi_padded[ind_rev_curv+ind_prev_curv:] - psi_padded[:-ind_rev_curv-ind_prev_curv])

    s_cum = np.cumsum(el_lengths)
    s_cum = np.insert(s_cum, 0, 0.0)
    s_rev = s_cum[-1] - s_cum[::-1][1:]
    s_rev = np.insert(s_rev, len(s_rev), 0.0)
    s_padded = np.concatenate((s_rev[-ind_rev_curv-1:-1], s_cum[:-1], s_cum[-1] + s_cum[1:ind_prev_curv+1]))

    delta_s = s_padded[ind_rev_curv+ind_prev_curv:] - s_padded[:-ind_rev_curv-ind_prev_curv]
    delta_s[np.abs(delta_s) < 1e-6] = 1e-6
    return delta_psi / delta_s


def opt_min_curv(reftrack: np.ndarray,
                 normvectors: np.ndarray,
                 A: np.ndarray,
                 kappa_bound: float,
                 w_veh: float) -> np.ndarray:
    """Solve the minimum-curvature lateral-shift QP on a closed track."""
    no_points = reftrack.shape[0]
    no_splines = no_points
    if no_points != normvectors.shape[0]:
        raise RuntimeError("Array size of reftrack should be the same as normvectors!")
    if no_points * 4 != A.shape[0] or A.shape[0] != A.shape[1]:
        raise RuntimeError("Spline equation system matrix A has wrong dimensions!")

    # Extraction matrices pick the b_i (gradient) and c_i (curvature) coefficients out of the spline solution.
    A_ex_b = np.zeros((no_points, no_splines * 4), dtype=int)
    for i in range(no_splines):
        A_ex_b[i, i * 4 + 1] = 1
    A_ex_c = np.zeros((no_points, no_splines * 4), dtype=int)
    for i in range(no_splines):
        A_ex_c[i, i * 4 + 2] = 2

    A_inv = np.linalg.inv(A)
    T_c = np.matmul(A_ex_c, A_inv)

    M_x = np.zeros((no_splines * 4, no_points))
    M_y = np.zeros((no_splines * 4, no_points))
    for i in range(no_splines):
        j = i * 4
        if i < no_points - 1:
            M_x[j, i] = normvectors[i, 0]
            M_x[j + 1, i + 1] = normvectors[i + 1, 0]
            M_y[j, i] = normvectors[i, 1]
            M_y[j + 1, i + 1] = normvectors[i + 1, 1]
        else:
            M_x[j, i] = normvectors[i, 0]
            M_x[j + 1, 0] = normvectors[0, 0]
            M_y[j, i] = normvectors[i, 1]
            M_y[j + 1, 0] = normvectors[0, 1]

    q_x = np.zeros((no_splines * 4, 1))
    q_y = np.zeros((no_splines * 4, 1))
    for i in range(no_splines):
        j = i * 4
        if i < no_points - 1:
            q_x[j, 0] = reftrack[i, 0]
            q_x[j + 1, 0] = reftrack[i + 1, 0]
            q_y[j, 0] = reftrack[i, 1]
            q_y[j + 1, 0] = reftrack[i + 1, 1]
        else:
            q_x[j, 0] = reftrack[i, 0]
            q_x[j + 1, 0] = reftrack[0, 0]
            q_y[j, 0] = reftrack[i, 1]
            q_y[j + 1, 0] = reftrack[0, 1]

    x_prime = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_x)
    y_prime = np.eye(no_points, no_points) * np.matmul(np.matmul(A_ex_b, A_inv), q_y)

    x_prime_sq = np.power(x_prime, 2)
    y_prime_sq = np.power(y_prime, 2)
    x_prime_y_prime = -2 * np.matmul(x_prime, y_prime)

    curv_den = np.power(x_prime_sq + y_prime_sq, 1.5)
    curv_part = np.divide(1, curv_den, out=np.zeros_like(curv_den), where=curv_den != 0)
    curv_part_sq = np.power(curv_part, 2)

    P_xx = np.matmul(curv_part_sq, y_prime_sq)
    P_yy = np.matmul(curv_part_sq, x_prime_sq)
    P_xy = np.matmul(curv_part_sq, x_prime_y_prime)

    T_nx = np.matmul(T_c, M_x)
    T_ny = np.matmul(T_c, M_y)

    H_x = np.matmul(T_nx.T, np.matmul(P_xx, T_nx))
    H_xy = np.matmul(T_ny.T, np.matmul(P_xy, T_nx))
    H_y = np.matmul(T_ny.T, np.matmul(P_yy, T_ny))
    H = H_x + H_xy + H_y
    H = (H + H.T) / 2

    f_x = 2 * np.matmul(np.matmul(q_x.T, T_c.T), np.matmul(P_xx, T_nx))
    f_xy = np.matmul(np.matmul(q_x.T, T_c.T), np.matmul(P_xy, T_ny)) \
           + np.matmul(np.matmul(q_y.T, T_c.T), np.matmul(P_xy, T_nx))
    f_y = 2 * np.matmul(np.matmul(q_y.T, T_c.T), np.matmul(P_yy, T_ny))
    f = np.squeeze(f_x + f_xy + f_y)

    Q_x = np.matmul(curv_part, y_prime)
    Q_y = np.matmul(curv_part, x_prime)
    E_kappa = np.matmul(Q_y, T_ny) - np.matmul(Q_x, T_nx)
    k_kappa_ref = np.matmul(Q_y, np.matmul(T_c, q_y)) - np.matmul(Q_x, np.matmul(T_c, q_x))

    con_ge = np.ones((no_points, 1)) * kappa_bound - k_kappa_ref
    con_le = -(np.ones((no_points, 1)) * -kappa_bound - k_kappa_ref)
    con_stack = np.append(con_ge, con_le)

    dev_max_right = reftrack[:, 2] - w_veh / 2
    dev_max_left = reftrack[:, 3] - w_veh / 2
    if np.any(-dev_max_right > dev_max_left) or np.any(-dev_max_left > dev_max_right):
        raise RuntimeError("Problem not solvable, track might be too small to run with current safety distance!")

    G = np.vstack((np.eye(no_points), -np.eye(no_points), E_kappa, -E_kappa))
    h = np.append(dev_max_right, dev_max_left)
    h = np.append(h, con_stack)

    # quadprog minimises 1/2 x'Hx - a'x s.t. C'x >= b and reads only the lower triangle of H.
    return quadprog.solve_qp(H, -f, -G.T, -h, 0)[0]


def compute_path_length(path_xy: np.ndarray) -> float:
    """Return the length of a closed polyline."""
    path_closed = np.vstack([path_xy, path_xy[0]])
    distances = np.sqrt(np.sum(np.power(np.diff(path_closed, axis=0), 2), axis=1))
    return float(np.sum(distances))


def _wall_distances(
    point: np.ndarray,
    normal: np.ndarray,
    free: np.ndarray,
    resolution: float,
    origin: tuple[float, float],
) -> tuple[float, float]:
    """Free distance left and right of a point along its normal, less half a cell so it never over-reports."""
    height, width = free.shape

    def blocked(x: float, y: float) -> bool:
        col = int((x - origin[0]) / resolution)
        row = height - 1 - int((y - origin[1]) / resolution)
        return row < 0 or row >= height or col < 0 or col >= width or not free[row, col]

    def free_distance(direction: np.ndarray) -> float:
        d = 0.0
        while d < WALL_MAX_M and not blocked(point[0] + d * direction[0], point[1] + d * direction[1]):
            d += WALL_STEP_M
        return max(0.0, d - 0.5 * resolution)

    return free_distance(normal), free_distance(-normal)


def _solve_shift(paths: TrackPaths, walls: Walls) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return the centreline, final reference, its normals, and the minimum-curvature shifts on it."""
    reftrack_imp = import_track(str(paths.centreline_csv))
    # The QP is dense in the point count; raycast_table resamples the final line finely anyway.
    iter_steps = {**STEPSIZE_OPTS, "stepsize_reg": MINCURV_GRID_M}
    ref, normvec, a_interp, _cx, _cy = prep_track(
        reftrack_imp=reftrack_imp,
        reg_smooth_opts=dict(REG_SMOOTH_OPTS),
        stepsize_opts=iter_steps,
        walls=walls,
    )
    # The QP linearises curvature, so a full shift folds hairpins; take bounded trust-region steps instead.
    for iteration in range(MINCURV_MAX_ITERS):
        alpha = opt_min_curv(reftrack=ref, normvectors=normvec, A=a_interp,
                             kappa_bound=KAPPA_BOUND, w_veh=WIDTH_OPT_MINCURV)
        shift = float(np.max(np.abs(alpha)))
        print(f"  min-curvature step {iteration + 1}: max shift {shift:.3f} m", flush=True)
        if shift <= MINCURV_TOL_M or iteration == MINCURV_MAX_ITERS - 1:
            # Keep the final shift inside the trust region too.
            alpha = np.clip(alpha, -MINCURV_STEP_M, MINCURV_STEP_M)
            break
        step = np.clip(alpha, -MINCURV_STEP_M, MINCURV_STEP_M)
        moved = np.column_stack(
            (ref[:, :2] + step[:, None] * normvec, ref[:, 2] - step, ref[:, 3] + step)
        )
        ref = interp_track(moved, MINCURV_GRID_M)
        _cx, _cy, a_interp, normvec = calc_splines(np.vstack((ref[:, :2], ref[0, :2])))
        _raycast_widths(ref, normvec, walls)
    return reftrack_imp, ref, normvec, alpha


def lap_time(raceline_xy: np.ndarray, el_lengths: np.ndarray) -> float:
    """Return the curvature-limited lap-time estimate of a closed line."""
    kappa = calc_curv_num(path=raceline_xy, el_lengths=el_lengths)
    vx = np.minimum(V_MAX, np.sqrt(A_MAX_LAT / (np.abs(kappa) + 1e-6)))
    return float(np.sum(el_lengths / np.maximum(vx, 1e-3)))


def raycast_table(
    xy: np.ndarray,
    free: np.ndarray,
    resolution: float,
    origin: tuple[float, float],
) -> pd.DataFrame:
    """Fit a closed spline to xy, resample it, and measure the wall clearance at every station."""
    x_raw = xy[:, 0].copy()
    y_raw = xy[:, 1].copy()
    # A periodic fit needs the last point to equal the first exactly.
    if np.hypot(x_raw[0] - x_raw[-1], y_raw[0] - y_raw[-1]) < RAYCAST_RESOLUTION:
        x_raw, y_raw = x_raw[:-1], y_raw[:-1]
    x_raw = np.append(x_raw, x_raw[0])
    y_raw = np.append(y_raw, y_raw[0])
    tck, _u = splprep([x_raw, y_raw], s=RAYCAST_SMOOTHING, per=True, k=3)
    u_test = np.linspace(0, 1, len(x_raw) * 10)
    x_t, y_t = splev_xy(u_test, tck)
    total_len = float(np.sum(np.hypot(np.diff(x_t), np.diff(y_t))))
    num_points = int(total_len / RAYCAST_RESOLUTION)
    u_new = np.linspace(0, 1, num_points)[:-1]
    x_new, y_new = splev_xy(u_new, tck)
    s_new = np.concatenate((np.array([0.0]), np.cumsum(np.hypot(np.diff(x_new), np.diff(y_new)))))
    dx, dy = splev_xy(u_new, tck, der=1)
    d2x, d2y = splev_xy(u_new, tck, der=2)
    track_yaw = np.arctan2(dy, dx)
    kappa = (dx * d2y - dy * d2x) / ((dx ** 2 + dy ** 2) ** 1.5)
    kappa = np.clip(kappa, -0.7, 0.7)
    kappa = np.asarray(savgol_filter(kappa, window_length=41, polyorder=3), dtype=float)
    n_x, n_y = -dy, dx
    norm = np.hypot(n_x, n_y)
    n_x = n_x / norm
    n_y = n_y / norm
    w_l_raw = np.zeros(len(x_new))
    w_r_raw = np.zeros(len(x_new))
    for i in range(len(x_new)):
        w_l_raw[i], w_r_raw[i] = _wall_distances(
            np.array([x_new[i], y_new[i]]),
            np.array([n_x[i], n_y[i]]),
            free,
            resolution,
            origin,
        )
    # Cap the inner width at 0.95 of the radius so 1 - n*kappa stays positive.
    safe_radius = 0.95 / (np.abs(kappa) + 1e-6)
    w_l_safe = np.where(kappa > 0.0, np.minimum(w_l_raw, safe_radius), w_l_raw)
    w_r_safe = np.where(kappa < 0.0, np.minimum(w_r_raw, safe_radius), w_r_raw)
    return pd.DataFrame(
        {
            "x": x_new,
            "y": y_new,
            "s": s_new,
            "s_norm": s_new / s_new[-1],
            "track_yaw_rad": track_yaw,
            "kappa": kappa,
            "w_l": w_l_safe,
            "w_r": w_r_safe,
            "n_x": n_x,
            "n_y": n_y,
        }
    )


def draw_walls(ax: Any, df: pd.DataFrame, label: str) -> None:
    """Draw the wall clearances of a 10-column line table on ax."""
    ax.plot(df["x"] + df["n_x"] * df["w_l"], df["y"] + df["n_y"] * df["w_l"], "g-", lw=0.8, label=label)
    ax.plot(df["x"] - df["n_x"] * df["w_r"], df["y"] - df["n_y"] * df["w_r"], "g-", lw=0.8)


def run(paths: TrackPaths, direction: str, plot: bool) -> None:
    """Write raceline.csv from the centreline."""
    if not paths.centreline_csv.exists():
        raise FileNotFoundError(f"missing centreline: {paths.centreline_csv}")
    walls = paths.occupancy()
    print("solving minimum curvature")
    ref_imp, ref_mc, norm_mc, alpha_mc = _solve_shift(paths, walls)
    raceline_xy, _s_rl, el_cl = create_raceline(
        refline=ref_mc[:, :2],
        normvectors=norm_mc,
        alpha=alpha_mc,
        stepsize_interp=STEPSIZE_OPTS["stepsize_interp_after_opt"],
    )
    print(f"Raceline Length: {compute_path_length(raceline_xy):.2f} m")
    if plot:
        fig, ax = new_axes((12, 12))
        draw_map(ax, *walls)
        ax.plot(ref_imp[:, 0], ref_imp[:, 1], color="gold", ls="--", lw=1.2, label="centreline")
        ax.plot(raceline_xy[:, 0], raceline_xy[:, 1], "b-", lw=1.5, label="minimum curvature")
        ax.scatter([raceline_xy[0, 0]], [raceline_xy[0, 1]], c="lime", s=80, zorder=5, label="s = 0")
        ax.set_title(f"minimum curvature ({direction}, curvature-limited time {lap_time(raceline_xy, el_cl):.2f}s)")
        ax.legend(loc="upper right")
        show_plot(fig, True)

    table = raycast_table(raceline_xy, *walls)
    table.to_csv(paths.raceline_csv, index=False, float_format="%.4f")
    print(f"wrote {paths.raceline_csv}")
    if plot:
        fig, ax = new_axes((12, 12))
        draw_map(ax, *walls)
        ax.plot(table["x"], table["y"], "b-", lw=1.5, label="raceline")
        draw_walls(ax, table, "wall clearances")
        ax.scatter([table["x"][0]], [table["y"][0]], c="lime", s=80, zorder=5, label="s = 0")
        ax.set_title("raceline with wall widths")
        ax.legend(loc="upper right")
        show_plot(fig, True)
