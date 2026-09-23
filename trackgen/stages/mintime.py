# SPDX-License-Identifier: LGPL-3.0-only
# Modified from TUMFTM global_racetrajectory_optimization and trajectory_planning_helpers; see THIRD_PARTY_NOTICES.md.
"""Minimum-time raceline about the minimum-curvature raceline via a CasADi/IPOPT collocation NLP."""
from __future__ import annotations

import math
import time
from typing import Any

import casadi as ca
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from trackgen.paths import TrackPaths
from trackgen.plotting import draw_map, show_plot
from trackgen.stages.raceline import (
    REG_SMOOTH_OPTS,
    STEPSIZE_OPTS,
    calc_curv_num,
    calc_spline_lengths,
    calc_splines,
    create_raceline,
    draw_walls,
    import_track,
    load_vehicle,
    prep_track,
    raycast_table,
)

# Preview/review distances for numerical curvature on the reference line.
CURV_CALC_OPTS = {
    "d_preview_curv": 0.2,
    "d_review_curv": 0.2,
}

# Vehicle half-width used as a clearance margin in the lateral-deviation bounds.
WIDTH_OPT = 0.45
# Smoothness weight on successive steering commands.
PENALTY_DELTA = 10.0
# Smoothness weight on successive longitudinal force samples.
PENALTY_F = 0.01
# Constant tyre-road friction; D = F_z * mue.
MUE = 0.5
# Bound accelerations to these safe-trajectory limits.
SAFE_TRAJ = True
AX_POS_SAFE = 9.51
AX_NEG_SAFE = -9.51
AY_SAFE = 7.5
# Target time between collocation nodes.
CONTROL_DT = 0.07

# Quadratic aero drag coefficient.
DRAGCOEFF = 0.075
# Front-axle lift coefficient (downforce ~ v^2).
LIFTCOEFF_FRONT = 0.045
# Rear-axle lift coefficient (downforce ~ v^2).
LIFTCOEFF_REAR = 0.075
# Front share of braking force.
K_BRAKE_FRONT = 0.6
# Front share of driving force.
K_DRIVE_FRONT = 0.5
# Front share of roll moment.
K_ROLL = 0.5
# Speed-loop time constant used as tau_v.
T_DRIVE = 0.1
# Peak drive force (fallback for a_max).
F_DRIVE_MAX = 40.0
# Left-right wheel track; vehicle.yaml only has body width.
TRACK_WIDTH_M = 0.296
# Rolling-resistance coefficient.
C_ROLL = 0.013
# Nominal Pacejka load.
F_Z0 = 300.0
B_FRONT = 10.0
C_FRONT = 2.5
EPS_FRONT = -0.1
E_FRONT = 1.0
B_REAR = 10.0
C_REAR = 2.5
EPS_REAR = -0.1
E_REAR = 1.0

_NX = 8
_NU = 3
_D_COLLOC = 3


def _dm_float(value: object) -> float:
    """Convert a CasADi DM/numeric scalar to float."""
    arr = np.asarray(value, dtype=float).reshape(-1)
    return float(arr[0])


def _concat(items: list[Any]) -> np.ndarray:
    """Flatten mixed bound/guess fragments into one float vector."""
    return np.concatenate([np.asarray(item, dtype=float).reshape(-1) for item in items])


def compute_curvature(path: np.ndarray) -> np.ndarray:
    """Return Menger curvature at each interior point of an open polyline."""
    curvatures = np.zeros(len(path))
    if len(path) < 3:
        return curvatures
    for i in range(1, len(path) - 1):
        p1, p2, p3 = path[i - 1], path[i], path[i + 1]
        a = float(np.linalg.norm(p3 - p2))
        b = float(np.linalg.norm(p1 - p3))
        c = float(np.linalg.norm(p2 - p1))
        if a < 1e-9 or b < 1e-9 or c < 1e-9:
            continue
        if abs((a + b) - c) < 1e-9 or abs((b + c) - a) < 1e-9 or abs((c + a) - b) < 1e-9:
            continue
        semi = (a + b + c) / 2.0
        area_arg = semi * (semi - a) * (semi - b) * (semi - c)
        area = 0.0 if area_arg < 0.0 else math.sqrt(area_arg)
        denom = a * b * c
        curvatures[i] = np.inf if denom < 1e-12 else 4.0 * area / denom
    curvatures[0] = curvatures[1]
    curvatures[-1] = curvatures[-2]
    return curvatures


def compute_vp(path: np.ndarray, vmax: float = 4.0, amax_lat: float = 2.5, eps: float = 1e-6) -> np.ndarray:
    """Return a curvature-limited speed at each point of path."""
    curvatures = compute_curvature(path)
    vp = np.zeros_like(curvatures)
    for i in range(len(vp)):
        vp[i] = min(vmax, math.sqrt(amax_lat / (abs(curvatures[i]) + eps)))
    return vp


def resample_track_for_control_dt(
    reftrack: np.ndarray,
    control_dt: float,
    vmax: float,
    amax_lat: float = 7.5,
) -> np.ndarray:
    """Resample a closed [x, y, w_r, w_l] track to a nominal control period."""
    if control_dt <= 0:
        return reftrack
    refline = reftrack[:, :2]
    w_tr_r = reftrack[:, 2]
    w_tr_l = reftrack[:, 3]
    refline_cl = np.vstack((refline, refline[0]))
    ds = np.sqrt(np.sum(np.diff(refline_cl, axis=0) ** 2, axis=1))
    s_cum = np.cumsum(np.insert(ds, 0, 0.0))
    total_s = float(s_cum[-1])
    v_est = np.maximum(compute_vp(refline_cl, vmax=vmax, amax_lat=amax_lat), 0.1)
    dt = ds / (0.5 * (v_est[:-1] + v_est[1:]))
    t_cum = np.cumsum(np.insert(dt, 0, 0.0))
    total_t = float(t_cum[-1])
    t_new = np.arange(0.0, total_t, control_dt)
    if len(t_new) == 0 or t_new[-1] < total_t - 1e-9:
        t_new = np.append(t_new, total_t)
    s_nodes = s_cum[:-1]
    s_new = np.interp(t_new, t_cum, s_cum, period=total_s)
    if s_new[-1] >= total_s - 1e-9 or s_new[-1] <= s_new[0] + 1e-9:
        s_new = s_new[:-1]
        t_new = t_new[:-1]
    x_new = np.interp(s_new, s_nodes, refline[:, 0], period=total_s)
    y_new = np.interp(s_new, s_nodes, refline[:, 1], period=total_s)
    wr_new = np.interp(s_new, s_nodes, w_tr_r, period=total_s)
    wl_new = np.interp(s_new, s_nodes, w_tr_l, period=total_s)
    return np.column_stack((x_new, y_new, wr_new, wl_new))


def _mintime_pars(car: dict[str, float]) -> dict[str, Any]:
    """Assemble the NLP parameter dictionaries from vehicle yaml and constants."""
    veh = {
        "wheelbase_front": car["length_front_m"],
        "wheelbase_rear": car["length_rear_m"],
        "wheelbase": car["length_front_m"] + car["length_rear_m"],
        "track_width_front": TRACK_WIDTH_M,
        "track_width_rear": TRACK_WIDTH_M,
        "cog_z": car["cog_height_m"],
        "I_z": car["inertia_kgm2"],
        "liftcoeff_front": LIFTCOEFF_FRONT,
        "liftcoeff_rear": LIFTCOEFF_REAR,
        "k_brake_front": K_BRAKE_FRONT,
        "k_drive_front": K_DRIVE_FRONT,
        "k_roll": K_ROLL,
        "f_drive_max": F_DRIVE_MAX,
        "delta_max": car["steer_lock_cmd_rad"],
        "max_steering_rate": car["steer_rate_max_radps"],
        "tau_delta": car["tau_delta_s"],
        "tau_delta_dot": car["tau_delta_dot_s"],
        "tau_a": car["tau_accel_s"],
        "t_sd": car["steer_transport_s"],
        "delta_th": car["steer_delta_th_rad"],
        "steer_gamma": car["steer_gamma"],
        "a_max": car["max_accel_mps2"],
    }
    return {
        "curv_calc_opts": dict(CURV_CALC_OPTS),
        "veh_params": {
            "v_max": car["max_speed_mps"],
            "mass": car["mass_kg"],
            "dragcoeff": DRAGCOEFF,
            "g": car["gravity_mps2"],
        },
        "optim_opts_mintime": {
            "width_opt": WIDTH_OPT,
            "penalty_delta": PENALTY_DELTA,
            "penalty_F": PENALTY_F,
            "mue": MUE,
            "safe_traj": SAFE_TRAJ,
            "ax_pos_safe": AX_POS_SAFE,
            "ax_neg_safe": AX_NEG_SAFE,
            "ay_safe": AY_SAFE,
            "control_dt": CONTROL_DT,
        },
        "vehicle_params_mintime": veh,
        "tire_params_mintime": {
            "c_roll": C_ROLL,
            "f_z0": F_Z0,
            "B_front": B_FRONT,
            "C_front": C_FRONT,
            "eps_front": EPS_FRONT,
            "E_front": E_FRONT,
            "B_rear": B_REAR,
            "C_rear": C_REAR,
            "eps_rear": EPS_REAR,
            "E_rear": E_REAR,
        },
    }


def opt_mintime_core(
    reftrack: np.ndarray,
    coeffs_x: np.ndarray,
    coeffs_y: np.ndarray,
    normvectors: np.ndarray,
    pars: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve the minimum-time collocation NLP; return lateral offsets, speeds, and the resampled reference with its normals."""
    opt = pars["optim_opts_mintime"]
    veh = pars["vehicle_params_mintime"]
    control_dt = opt.get("control_dt", None)
    if control_dt is not None and float(control_dt) > 0:
        reftrack = resample_track_for_control_dt(
            reftrack=reftrack,
            control_dt=float(control_dt),
            vmax=float(pars["veh_params"]["v_max"]),
            amax_lat=float(opt.get("ay_safe", 7.5)),
        )
        refpath_cl = np.vstack((reftrack[:, :2], reftrack[0, :2]))
        coeffs_x, coeffs_y, _a, normvectors = calc_splines(refpath_cl)

    spline_lengths_refline = calc_spline_lengths(coeffs_x=coeffs_x, coeffs_y=coeffs_y)
    kappa_refline = calc_curv_num(
        path=reftrack[:, :2],
        el_lengths=spline_lengths_refline,
        stepsize_curv_preview=pars["curv_calc_opts"]["d_preview_curv"],
        stepsize_curv_review=pars["curv_calc_opts"]["d_review_curv"],
    )
    kappa_refline_cl = np.append(kappa_refline, kappa_refline[0])
    w_tr_left_cl = np.append(reftrack[:, 3], reftrack[0, 3])
    w_tr_right_cl = np.append(reftrack[:, 2], reftrack[0, 2])
    s_opt = np.cumsum(np.insert(spline_lengths_refline, 0, 0.0))

    wheelbase_total = float(veh.get("wheelbase", veh["wheelbase_front"] + veh["wheelbase_rear"]))
    if wheelbase_total < 1e-6:
        wheelbase_total = 1e-6
    delta_guess = np.clip(
        np.arctan(wheelbase_total * kappa_refline_cl),
        -float(veh["delta_max"]),
        float(veh["delta_max"]),
    )

    s_opt_list = [float(v) for v in s_opt.tolist()]
    kappa_interp = ca.interpolant("kappa_interp", "linear", [s_opt_list], kappa_refline_cl.tolist())
    w_tr_left_interp = ca.interpolant("w_tr_left_interp", "linear", [s_opt_list], w_tr_left_cl.tolist())
    w_tr_right_interp = ca.interpolant("w_tr_right_interp", "linear", [s_opt_list], w_tr_right_cl.tolist())

    d = _D_COLLOC
    tau = np.append(0, ca.collocation_points(d, "legendre"))
    c_mat = np.zeros((d + 1, d + 1))
    d_vec = np.zeros(d + 1)
    b_vec = np.zeros(d + 1)
    for j in range(d + 1):
        p = np.poly1d([1.0])
        for r in range(d + 1):
            if r != j:
                p *= np.poly1d([1.0, -tau[r]]) / (tau[j] - tau[r])
        d_vec[j] = p(1.0)
        p_der = np.polyder(p)
        for r in range(d + 1):
            c_mat[j, r] = p_der(tau[r])
        b_vec[j] = np.polyint(p)(1.0)

    nx = _NX
    nu = _NU
    v_n = ca.SX.sym("v_n")
    v_s = 50.0
    v = v_s * v_n
    beta_n = ca.SX.sym("beta_n")
    beta_s = 0.5
    beta = beta_s * beta_n
    omega_z_n = ca.SX.sym("omega_z_n")
    omega_z_s = 2.0
    omega_z = omega_z_s * omega_z_n
    n_n = ca.SX.sym("n_n")
    n_s = 5.0
    n = n_s * n_n
    xi_n = ca.SX.sym("xi_n")
    xi_s = 1.0
    xi = xi_s * xi_n
    delta_n = ca.SX.sym("delta_n")
    delta_s = 0.5
    delta = delta_s * delta_n
    delta_dot_n = ca.SX.sym("delta_dot_n")
    sv_max = float(veh.get("max_steering_rate", 3.2))
    delta_dot_s = sv_max
    delta_dot = delta_dot_s * delta_dot_n
    a_max = float(veh.get("a_max", veh.get("f_drive_max", 40.0) / pars["veh_params"]["mass"]))
    a_n = ca.SX.sym("a_n")
    a_s = a_max
    a = a_s * a_n
    x = ca.vertcat(v_n, beta_n, omega_z_n, n_n, xi_n, delta_n, delta_dot_n, a_n)
    x_s = np.array([v_s, beta_s, omega_z_s, n_s, xi_s, delta_s, delta_dot_s, a_s])

    delta_cmd_n = ca.SX.sym("delta_cmd_n")
    delta_cmd_s = delta_s
    delta_cmd = delta_cmd_s * delta_cmd_n
    v_cmd_n = ca.SX.sym("v_cmd_n")
    v_cmd_s = v_s
    v_cmd = v_cmd_s * v_cmd_n
    gamma_y_n = ca.SX.sym("gamma_y_n")
    gamma_y_s = 5000.0
    gamma_y = gamma_y_s * gamma_y_n
    u = ca.vertcat(delta_cmd_n, v_cmd_n, gamma_y_n)

    tire = pars["tire_params_mintime"]
    grav = float(pars["veh_params"]["g"])
    mass = float(pars["veh_params"]["mass"])
    kappa = ca.SX.sym("kappa")
    s_current = ca.SX.sym("s_current")
    f_xdrag = float(pars["veh_params"]["dragcoeff"]) * v**2
    f_xroll = float(tire["c_roll"]) * mass * grav
    f_zstat_f = 0.5 * mass * grav * float(veh["wheelbase_rear"]) / wheelbase_total
    f_zstat_r = 0.5 * mass * grav * float(veh["wheelbase_front"]) / wheelbase_total
    f_zlift_f = 0.5 * float(veh["liftcoeff_front"]) * v**2
    f_zlift_r = 0.5 * float(veh["liftcoeff_rear"]) * v**2

    long_force = mass * a
    long_load_transfer = 0.5 * float(veh["cog_z"]) / wheelbase_total * (long_force - f_xdrag - f_xroll)
    f_zdyn_fl = -long_load_transfer - float(veh["k_roll"]) * gamma_y
    f_zdyn_fr = -long_load_transfer + float(veh["k_roll"]) * gamma_y
    f_zdyn_rl = +long_load_transfer - (1.0 - float(veh["k_roll"])) * gamma_y
    f_zdyn_rr = +long_load_transfer + (1.0 - float(veh["k_roll"])) * gamma_y
    f_z_fl = f_zstat_f + f_zlift_f + f_zdyn_fl
    f_z_fr = f_zstat_f + f_zlift_f + f_zdyn_fr
    f_z_rl = f_zstat_r + f_zlift_r + f_zdyn_rl
    f_z_rr = f_zstat_r + f_zlift_r + f_zdyn_rr

    eps_atan = 1e-6
    tw_f = float(veh["track_width_front"])
    tw_r = float(veh["track_width_rear"])
    lf = float(veh["wheelbase_front"])
    lr = float(veh["wheelbase_rear"])
    alpha_fl = delta - ca.atan2(
        v * ca.sin(beta) + lf * omega_z,
        ca.fmax(v * ca.cos(beta) - 0.5 * tw_f * omega_z, eps_atan),
    )
    alpha_fr = delta - ca.atan2(
        v * ca.sin(beta) + lf * omega_z,
        ca.fmax(v * ca.cos(beta) + 0.5 * tw_f * omega_z, eps_atan),
    )
    alpha_rl = ca.atan2(
        -v * ca.sin(beta) + lr * omega_z,
        ca.fmax(v * ca.cos(beta) - 0.5 * tw_r * omega_z, eps_atan),
    )
    alpha_rr = ca.atan2(
        -v * ca.sin(beta) + lr * omega_z,
        ca.fmax(v * ca.cos(beta) + 0.5 * tw_r * omega_z, eps_atan),
    )
    mue = float(opt["mue"])
    f_z_fl_pos = ca.fmax(f_z_fl, 1.0)
    f_z_fr_pos = ca.fmax(f_z_fr, 1.0)
    f_z_rl_pos = ca.fmax(f_z_rl, 1.0)
    f_z_rr_pos = ca.fmax(f_z_rr, 1.0)
    fz0_pos = max(float(tire.get("f_z0", 1.0)), 1.0)
    f_y_fl = (
        mue
        * f_z_fl_pos
        * (1.0 + float(tire["eps_front"]) * f_z_fl_pos / fz0_pos)
        * ca.sin(
            float(tire["C_front"])
            * ca.atan(
                float(tire["B_front"]) * alpha_fl
                - float(tire["E_front"]) * (float(tire["B_front"]) * alpha_fl - ca.atan(float(tire["B_front"]) * alpha_fl))
            )
        )
    )
    f_y_fr = (
        mue
        * f_z_fr_pos
        * (1.0 + float(tire["eps_front"]) * f_z_fr_pos / fz0_pos)
        * ca.sin(
            float(tire["C_front"])
            * ca.atan(
                float(tire["B_front"]) * alpha_fr
                - float(tire["E_front"]) * (float(tire["B_front"]) * alpha_fr - ca.atan(float(tire["B_front"]) * alpha_fr))
            )
        )
    )
    f_y_rl = (
        mue
        * f_z_rl_pos
        * (1.0 + float(tire["eps_rear"]) * f_z_rl_pos / fz0_pos)
        * ca.sin(
            float(tire["C_rear"])
            * ca.atan(
                float(tire["B_rear"]) * alpha_rl
                - float(tire["E_rear"]) * (float(tire["B_rear"]) * alpha_rl - ca.atan(float(tire["B_rear"]) * alpha_rl))
            )
        )
    )
    f_y_rr = (
        mue
        * f_z_rr_pos
        * (1.0 + float(tire["eps_rear"]) * f_z_rr_pos / fz0_pos)
        * ca.sin(
            float(tire["C_rear"])
            * ca.atan(
                float(tire["B_rear"]) * alpha_rr
                - float(tire["E_rear"]) * (float(tire["B_rear"]) * alpha_rr - ca.atan(float(tire["B_rear"]) * alpha_rr))
            )
        )
    )

    blend = 0.5 * (1.0 + ca.tanh(a / 0.1))
    k_front = float(veh["k_brake_front"]) + blend * (float(veh["k_drive_front"]) - float(veh["k_brake_front"]))
    f_x_total = mass * a + f_xdrag + f_xroll
    eps_cos = 1e-3
    f_x_front_total = k_front * f_x_total / ca.fmax(ca.cos(delta), eps_cos)
    f_x_rear_total = (1.0 - k_front) * f_x_total
    f_x_fl = 0.5 * f_x_front_total
    f_x_fr = 0.5 * f_x_front_total
    f_x_rl = 0.5 * f_x_rear_total
    f_x_rr = 0.5 * f_x_rear_total

    sum_fx = f_x_rl + f_x_rr + (f_x_fl + f_x_fr) * ca.cos(delta) - (f_y_fl + f_y_fr) * ca.sin(delta) - f_xdrag - f_xroll
    sum_fy = (f_x_fl + f_x_fr) * ca.sin(delta) + f_y_rl + f_y_rr + (f_y_fl + f_y_fr) * ca.cos(delta)
    ax = sum_fx / mass
    ay = sum_fy / mass

    delta_th = float(veh.get("delta_th", 0.033))
    steer_gamma = float(veh.get("steer_gamma", 0.622))
    tau_delta = float(veh.get("tau_delta", 0.075)) + float(veh.get("t_sd", 0.060))
    tau_delta_dot = float(veh.get("tau_delta_dot", 0.027))
    delta_target = delta_cmd**3 / ((delta_cmd**2 + delta_th**2) * (1.0 + steer_gamma * delta_cmd**2))
    delta_dot_des = (delta_target - delta) / tau_delta
    delta_ddot = (delta_dot_des - delta_dot) / tau_delta_dot

    tau_v = T_DRIVE
    v_cmd_max = float(pars["veh_params"]["v_max"])
    a_max_idm = a_max
    a_des = ca.fmin(ca.fmax((v_cmd - v) / tau_v, -a_max_idm), a_max_idm)
    tau_a_dyn = float(veh.get("tau_a", 0.061))
    da = (a_des - a) / tau_a_dyn

    sf = (1.0 - n * kappa) / ca.fmax(v * ca.cos(xi + beta), 0.1)
    dv = sf * (ax * ca.cos(beta) + ay * ca.sin(beta))
    dbeta = sf * (-omega_z + (ay * ca.cos(beta) - ax * ca.sin(beta)) / ca.fmax(v, 0.1))
    term_rr_rl_x = (f_x_rr - f_x_rl) * tw_r / 2.0
    term_fr_fl_x = (f_x_fr - f_x_fl) * ca.cos(delta) * tw_f / 2.0
    term_fl_fr_y = (f_y_fl - f_y_fr) * ca.sin(delta) * tw_f / 2.0
    term_front_y = (f_y_fl + f_y_fr) * ca.cos(delta) * lf
    term_front_x = (f_x_fl + f_x_fr) * ca.sin(delta) * lf
    term_rear_y = -(f_y_rl + f_y_rr) * lr
    domega_z = (sf / float(veh["I_z"])) * (
        term_rr_rl_x + term_fr_fl_x + term_fl_fr_y + term_front_y + term_front_x + term_rear_y
    )
    dn = sf * v * ca.sin(xi + beta)
    dxi = sf * omega_z - kappa
    d_delta = sf * delta_dot
    d_delta_dot = sf * delta_ddot
    d_a = sf * da
    dx = ca.vertcat(dv, dbeta, domega_z, dn, dxi, d_delta, d_delta_dot, d_a) / x_s

    delta_cmd_min = -float(veh["delta_max"]) / delta_cmd_s
    delta_cmd_max = float(veh["delta_max"]) / delta_cmd_s
    v_cmd_min = 0.0 / v_cmd_s
    v_cmd_max_n = v_cmd_max / v_cmd_s
    gamma_y_min = -np.inf
    gamma_y_max = np.inf
    v_min = 1.0 / v_s
    v_max = float(pars["veh_params"]["v_max"]) / v_s
    beta_min = -0.4 / beta_s
    beta_max = 0.4 / beta_s
    omega_limit_est = float(veh["delta_max"]) * float(pars["veh_params"]["v_max"]) / wheelbase_total
    omega_z_min = -omega_limit_est * 1.5 / omega_z_s
    omega_z_max = +omega_limit_est * 1.5 / omega_z_s
    xi_min = -0.5 * np.pi / xi_s
    xi_max = 0.5 * np.pi / xi_s
    delta_min = -float(veh["delta_max"]) / delta_s
    delta_max = float(veh["delta_max"]) / delta_s
    delta_dot_min = -sv_max / delta_dot_s
    delta_dot_max = sv_max / delta_dot_s
    a_min = -a_max / a_s
    a_max_n = a_max / a_s
    v_guess = min(5.0, float(pars["veh_params"]["v_max"])) / v_s

    f_dyn = ca.Function("f_dyn", [x, u, kappa, s_current], [dx, sf], ["x", "u", "kappa", "s_current"], ["dx", "sf"])
    f_fx = ca.Function("f_fx", [x, u, s_current], [f_x_fl, f_x_fr, f_x_rl, f_x_rr])
    f_fy = ca.Function("f_fy", [x, u, s_current], [f_y_fl, f_y_fr, f_y_rl, f_y_rr])
    f_fz = ca.Function("f_fz", [x, u, s_current], [f_z_fl, f_z_fr, f_z_rl, f_z_rr])
    f_a = ca.Function("f_a", [x, u, s_current], [ax, ay])

    w: list[Any] = []
    w0: list[np.ndarray] = []
    lbw: list[np.ndarray] = []
    ubw: list[np.ndarray] = []
    g: list[Any] = []
    lbg: list[Any] = []
    ubg: list[Any] = []
    j_cost: Any = 0
    dt_opt_list: list[Any] = []
    delta_p: list[Any] = []
    f_p: list[Any] = []

    xk_sym = ca.MX.sym("X0", nx)
    w.append(xk_sym)
    s0 = float(s_opt[0])
    n_min_0 = (-_dm_float(w_tr_right_interp(s0)) + float(opt["width_opt"]) / 2.0) / n_s
    n_max_0 = (_dm_float(w_tr_left_interp(s0)) - float(opt["width_opt"]) / 2.0) / n_s
    lbw.append(np.array([v_min, beta_min, omega_z_min, n_min_0, xi_min, delta_min, delta_dot_min, a_min]))
    ubw.append(np.array([v_max, beta_max, omega_z_max, n_max_0, xi_max, delta_max, delta_dot_max, a_max_n]))
    w0_init = np.array([v_guess, 0.0, 0.0, (n_min_0 + n_max_0) / 2.0, 0.0, delta_guess[0] / delta_s, 0.0, 0.0])
    w0.append(w0_init)

    n_interval = len(s_opt) - 1
    h_steps = np.diff(s_opt)
    xk: Any = xk_sym
    for k in range(n_interval):
        uk_sym = ca.MX.sym(f"U_{k}", nu)
        w.append(uk_sym)
        lbw.append(np.array([delta_cmd_min, v_cmd_min, gamma_y_min]))
        ubw.append(np.array([delta_cmd_max, v_cmd_max_n, gamma_y_max]))
        v_cmd_guess = min(v_guess, v_cmd_max_n)
        delta_cmd_guess = float(np.clip(delta_guess[k] / delta_cmd_s, delta_cmd_min, delta_cmd_max))
        w0.append(np.array([delta_cmd_guess, v_cmd_guess, 0.0]))

        xc_sym: list[Any] = []
        lbw_coll = np.array([v_min, beta_min, omega_z_min, -np.inf, xi_min, delta_min, delta_dot_min, a_min])
        ubw_coll = np.array([v_max, beta_max, omega_z_max, +np.inf, xi_max, delta_max, delta_dot_max, a_max_n])
        w0_coll = np.array([v_guess, 0.0, 0.0, w0_init[3], 0.0, delta_guess[k + 1] / delta_s, 0.0, 0.0])
        for j in range(d):
            xkj = ca.MX.sym(f"X_{k}_{j}", nx)
            xc_sym.append(xkj)
            w.append(xkj)
            lbw.append(lbw_coll)
            ubw.append(ubw_coll)
            w0.append(w0_coll)

        xk_end = d_vec[0] * xk
        hk = float(h_steps[k])
        sk = float(s_opt[k])
        sf_interval_sum: Any = 0
        for j in range(1, d + 1):
            xp = c_mat[0, j] * xk
            for r in range(d):
                xp = xp + c_mat[r + 1, j] * xc_sym[r]
            s_coll = sk + tau[j] * hk
            kappa_coll = kappa_interp(s_coll)
            fj, qj = f_dyn(xc_sym[j - 1], uk_sym, kappa_coll, s_coll)
            g.append(hk * fj - xp)
            lbg.append([0.0] * nx)
            ubg.append([0.0] * nx)
            xk_end = xk_end + d_vec[j] * xc_sym[j - 1]
            j_cost = j_cost + b_vec[j] * qj * hk
            sf_interval_sum = sf_interval_sum + b_vec[j] * qj * hk
        dt_opt_list.append(sf_interval_sum)

        xk_next = ca.MX.sym(f"X_{k + 1}", nx)
        w.append(xk_next)
        s_next = float(s_opt[k + 1])
        n_min_k1 = (-_dm_float(w_tr_right_interp(s_next)) + float(opt["width_opt"]) / 2.0) / n_s
        n_max_k1 = (_dm_float(w_tr_left_interp(s_next)) - float(opt["width_opt"]) / 2.0) / n_s
        lbw.append(np.array([v_min, beta_min, omega_z_min, n_min_k1, xi_min, delta_min, delta_dot_min, a_min]))
        ubw.append(np.array([v_max, beta_max, omega_z_max, n_max_k1, xi_max, delta_max, delta_dot_max, a_max_n]))
        w0.append(np.array([v_guess, 0.0, 0.0, (n_min_k1 + n_max_k1) / 2.0, 0.0, delta_guess[k + 1] / delta_s, 0.0, 0.0]))

        g.append(xk_end - xk_next)
        lbg.append([0.0] * nx)
        ubg.append([0.0] * nx)

        f_x_flk, f_x_frk, f_x_rlk, f_x_rrk = f_fx(xk_next, uk_sym, s_next)
        f_y_flk, f_y_frk, f_y_rlk, f_y_rrk = f_fy(xk_next, uk_sym, s_next)
        f_z_flk, f_z_frk, f_z_rlk, f_z_rrk = f_fz(xk_next, uk_sym, s_next)
        axk, ayk = f_a(xk_next, uk_sym, s_next)
        mue_k = max(mue, 0.1)
        fz_fl_pos_k = ca.fmax(f_z_flk, 1.0)
        fz_fr_pos_k = ca.fmax(f_z_frk, 1.0)
        fz_rl_pos_k = ca.fmax(f_z_rlk, 1.0)
        fz_rr_pos_k = ca.fmax(f_z_rrk, 1.0)
        g.append((f_x_flk / (mue_k * fz_fl_pos_k)) ** 2 + (f_y_flk / (mue_k * fz_fl_pos_k)) ** 2)
        g.append((f_x_frk / (mue_k * fz_fr_pos_k)) ** 2 + (f_y_frk / (mue_k * fz_fr_pos_k)) ** 2)
        g.append((f_x_rlk / (mue_k * fz_rl_pos_k)) ** 2 + (f_y_rlk / (mue_k * fz_rl_pos_k)) ** 2)
        g.append((f_x_rrk / (mue_k * fz_rr_pos_k)) ** 2 + (f_y_rrk / (mue_k * fz_rr_pos_k)) ** 2)
        lbg.append([0.0] * 4)
        ubg.append([1.001] * 4)

        delta_k = xk_next[5] * delta_s
        sum_fy_k = (f_x_flk + f_x_frk) * ca.sin(delta_k) + f_y_rlk + f_y_rrk + (f_y_flk + f_y_frk) * ca.cos(delta_k)
        track_avg = (tw_f + tw_r) / 2.0
        g.append(sum_fy_k * float(veh["cog_z"]) / track_avg - uk_sym[2] * gamma_y_s)
        lbg.append([-1e-4])
        ubg.append([1e-4])

        if opt["safe_traj"]:
            ax_pos_safe = opt.get("ax_pos_safe", None)
            ax_neg_safe = opt.get("ax_neg_safe", None)
            ay_safe = opt.get("ay_safe", None)
            if ax_pos_safe is not None:
                g.append(axk)
                lbg.append([-np.inf])
                ubg.append([float(ax_pos_safe)])
            if ax_neg_safe is not None:
                g.append(-axk)
                lbg.append([-np.inf])
                ubg.append([-float(ax_neg_safe)])
            if ay_safe is not None:
                g.append(ayk)
                g.append(-ayk)
                lbg.append([-np.inf, -np.inf])
                ubg.append([float(ay_safe), float(ay_safe)])

        delta_p.append(uk_sym[0] * delta_cmd_s)
        f_p.append(xk_next[7] * a_s * mass)
        xk = xk_next

    g.append(w[0] - xk)
    lbg.append([0.0] * nx)
    ubg.append([0.0] * nx)

    delta_p_v = ca.vertcat(*delta_p)
    f_p_v = ca.vertcat(*f_p)
    j_cost = j_cost + float(opt["penalty_delta"]) * ca.sumsqr(ca.diff(ca.vertcat(delta_p_v, delta_p_v[0])))
    j_cost = j_cost + float(opt["penalty_F"]) * ca.sumsqr(ca.diff(ca.vertcat(f_p_v, f_p_v[0])))

    w_vec = ca.vertcat(*w)
    g_vec = ca.vertcat(*g)
    w0_vec = _concat(w0)
    lbw_vec = _concat(lbw)
    ubw_vec = _concat(ubw)
    lbg_vec = _concat(lbg)
    ubg_vec = _concat(ubg)
    n_vars = int(w_vec.shape[0])
    n_cons = int(g_vec.shape[0])

    nlp = {"f": j_cost, "x": w_vec, "g": g_vec}
    opts = {
        "expand": True,
        "verbose": False,
        "print_time": False,
        "ipopt.print_level": 0,
        "ipopt.max_iter": 1000,
        "ipopt.tol": 1e-4,
        "ipopt.acceptable_tol": 1e-3,
        "ipopt.hessian_approximation": "limited-memory",
        "ipopt.sb": "yes",
    }
    solver = ca.nlpsol("solver", "ipopt", nlp, opts)
    t0 = time.perf_counter()
    sol = solver(x0=w0_vec, lbx=lbw_vec, ubx=ubw_vec, lbg=lbg_vec, ubg=ubg_vec)
    elapsed = time.perf_counter() - t0
    stats = solver.stats()
    status = str(stats.get("return_status", "Unknown"))
    if status not in {"Solve_Succeeded", "Solved_To_Acceptable_Level"}:
        raise SystemExit(f"mintime failed: {status}")

    sol_w = sol["x"]
    sol_np = np.array(sol_w).reshape(-1)
    x_opt = np.zeros((n_interval + 1, nx))
    x_opt[0, :] = sol_np[0:nx] * x_s
    for k in range(n_interval):
        start = nx + k * (nu + d * nx + nx) + nu + d * nx
        x_opt[k + 1, :] = sol_np[start : start + nx] * x_s
    dt_fun = ca.Function("dt_eval", [w_vec], [ca.vertcat(*dt_opt_list)])
    dt_num = np.array(dt_fun(sol_w)).reshape(-1)
    t_opt = np.cumsum(np.insert(dt_num, 0, 0.0))
    t_nlp = float(t_opt[-1])
    print(
        f"mintime: {n_vars} vars, {n_cons} constraints, {elapsed:.3f}s, {status}, "
        f"Minimum Laptime {t_nlp:.3f}s"
    )
    return x_opt[:-1, 3], x_opt[:-1, 0], reftrack, normvectors


def _interp_velocity(
    v_opt: np.ndarray,
    reftrack: np.ndarray,
    s_raceline: np.ndarray,
) -> np.ndarray:
    """Interpolate optimiser-grid speed onto the high-resolution racing line."""
    path_cl = np.vstack((reftrack[:, :2], reftrack[0, :2]))
    el_opt = np.sqrt(np.sum(np.power(np.diff(path_cl, axis=0), 2), axis=1))
    s_grid = np.cumsum(np.insert(el_opt, 0, 0.0))
    if len(v_opt) == len(s_grid) - 1:
        mid = (s_grid[:-1] + s_grid[1:]) / 2.0
        return np.interp(s_raceline, mid, v_opt, left=float(v_opt[0]), right=float(v_opt[-1]))
    if len(v_opt) == len(s_grid):
        return np.interp(s_raceline, s_grid, v_opt, left=float(v_opt[0]), right=float(v_opt[-1]))
    raise SystemExit(f"mintime velocity length {len(v_opt)} does not match grid {len(s_grid)}")


def _lap_time(v_ref: np.ndarray, el_lengths: np.ndarray) -> float:
    """Integrate ds/v along the interpolated racing line."""
    safe_v = np.maximum(v_ref, 1e-3)
    if len(el_lengths) == len(safe_v):
        return float(np.sum(el_lengths / safe_v))
    t = 0.0
    for k in range(len(safe_v)):
        ds = float(el_lengths[k]) if k < len(el_lengths) else 0.0
        t += ds / float(safe_v[k])
    return t


def _plot(paths: TrackPaths, table: pd.DataFrame, lap_time: float) -> None:
    """Show the racing line coloured by speed with its walls on the map, and speed over station."""
    from matplotlib.collections import LineCollection

    free, resolution, origin = paths.occupancy()
    xy = table[["x", "y"]].to_numpy()
    s = table["s"].to_numpy()
    v_ref = table["v_ref"].to_numpy()
    fig, (ax_map, ax_v) = plt.subplots(
        2, 1, figsize=(12, 12), gridspec_kw={"height_ratios": [3, 1]}
    )
    draw_map(ax_map, free, resolution, origin)
    draw_walls(ax_map, table, "ray-cast walls")
    points = xy.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    line = LineCollection(segments, cmap="turbo", linewidths=3.0, zorder=3)
    line.set_array(v_ref[:-1])
    ax_map.add_collection(line)
    ax_map.scatter([xy[0, 0]], [xy[0, 1]], c="lime", s=60, zorder=5, label="s = 0")
    fig.colorbar(line, ax=ax_map, fraction=0.03, pad=0.02, label="v_ref [m/s]")
    ax_map.set_title(f"minimum-time line, lap {lap_time:.2f} s")
    ax_map.legend(loc="upper right")
    ax_v.plot(s, v_ref, "k-", lw=1.2)
    ax_v.set_xlim(0.0, float(s[-1]))
    ax_v.set_ylim(0.0, float(np.max(v_ref)) * 1.1)
    ax_v.set_xlabel("s [m]")
    ax_v.set_ylabel("v_ref [m/s]")
    ax_v.grid(True, alpha=0.3)
    fig.tight_layout()
    show_plot(fig, True)


def run(paths: TrackPaths, plot: bool, vmax: float | None = None) -> float:
    """Write raceline_mintime.csv and return the interpolated-profile lap time."""
    # The Frenet OCP needs a reference without crossing normals, which the skeleton centreline lacks in hairpins.
    if not paths.raceline_csv.exists():
        raise FileNotFoundError(f"missing raceline: {paths.raceline_csv}; run the raceline stage")
    car = load_vehicle()
    if vmax is not None:
        car["max_speed_mps"] = float(vmax)
    pars = _mintime_pars(car)
    walls = paths.occupancy()
    reftrack_interp, normvec, _a, coeffs_x, coeffs_y = prep_track(
        reftrack_imp=import_track(str(paths.raceline_csv)),
        reg_smooth_opts=dict(REG_SMOOTH_OPTS),
        stepsize_opts=dict(STEPSIZE_OPTS),
        walls=walls,
    )
    alpha_n, v_opt, reftrack, normvectors = opt_mintime_core(
        reftrack=reftrack_interp,
        coeffs_x=coeffs_x,
        coeffs_y=coeffs_y,
        normvectors=normvec,
        pars=pars,
    )
    alpha_opt = -alpha_n
    if len(alpha_opt) != len(reftrack):
        s_alpha = np.linspace(0.0, 1.0, len(alpha_opt))
        path_cl = np.vstack((reftrack[:, :2], reftrack[0, :2]))
        el = np.sqrt(np.sum(np.power(np.diff(path_cl, axis=0), 2), axis=1))
        s_ref = np.cumsum(np.insert(el, 0, 0.0))[:-1]
        denom = float(s_ref[-1]) if len(s_ref) and s_ref[-1] > 1e-6 else 1.0
        alpha_opt = np.interp(s_ref / denom, s_alpha, alpha_opt)
    raceline_xy, s_rl, el_cl = create_raceline(
        refline=reftrack[:, :2],
        normvectors=normvectors,
        alpha=alpha_opt,
        stepsize_interp=float(STEPSIZE_OPTS["stepsize_interp_after_opt"]),
    )
    v_ref = _interp_velocity(v_opt, reftrack, s_rl)
    t_final = _lap_time(v_ref, el_cl)
    table = raycast_table(raceline_xy, *walls)
    lap_len = float(s_rl[-1] + el_cl[-1]) if len(el_cl) else 1.0
    table["v_ref"] = np.interp(table["s"].to_numpy(), s_rl, v_ref, period=lap_len)
    table.to_csv(paths.raceline_mintime_csv, index=False, float_format="%.4f")
    print(f"wrote {paths.raceline_mintime_csv}")
    if plot:
        _plot(paths, table, t_final)
    return t_final
