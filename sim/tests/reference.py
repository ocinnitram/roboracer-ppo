"""Float64 reference of the plant dynamics: writes fixtures.h for test_dynamics.c."""

from pathlib import Path

import numpy as np

# Tire-curve constants, solved once.
LAT_PEAK_SCALE = 1.20
_C_CURVE = 2.0 * (np.pi - np.arcsin(1.0 / LAT_PEAK_SCALE)) / np.pi
ALPHA_PK = 0.16
_B_R = np.tan(np.pi / (2.0 * _C_CURVE)) / ALPHA_PK

SIGMA_FY = 0.03
_SIGMA_V_FLOOR = 0.5
TAU_BETA_KIN = 0.15

# DT used by the C cascade (get_inputs hardcodes DT_SIM).
DT = 0.001

# Plant constants matching the C fixtures.
P = dict(
    mu=0.42, C_Sf=4.718, C_Sr=5.4562, lf=0.15875, lr=0.17145, h=0.074,
    m=3.74, I=0.04712, s_min=-0.4189, s_max=0.4189, sv_min=-3.2, sv_max=3.2,
    v_switch=7.319, v_min=-10.0, v_max=10.0,
    tau_v=0.118, tau_a=0.061, a_max=3.48, t_vd=0.020, t_sd=0.060,
    tau_delta=0.075, tau_delta_dot=0.027, steer_gamma=0.622, delta_th=0.033,
)


# Steering and acceleration constraints.
def steering_constraint(steering_angle, steering_velocity, s_min, s_max, sv_min, sv_max):
    if (steering_angle <= s_min and steering_velocity <= 0) or (steering_angle >= s_max and steering_velocity >= 0):
        steering_velocity = 0.
    elif steering_velocity <= sv_min:
        steering_velocity = sv_min
    elif steering_velocity >= sv_max:
        steering_velocity = sv_max
    return steering_velocity


def accl_constraints(vel, accl, v_switch, a_max, v_min, v_max):
    if vel > v_switch:
        pos_limit = a_max * v_switch / vel
    else:
        pos_limit = a_max
    if (vel <= v_min and accl <= 0) or (vel >= v_max and accl >= 0):
        accl = 0.
    elif accl <= -a_max:
        accl = -a_max
    elif accl >= pos_limit:
        accl = pos_limit
    return accl


def vehicle_dynamics_ks(x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I,
                        s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max):
    lwb = lf + lr
    u = np.array([steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max),
                  accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max)])
    f = np.array([x[3]*np.cos(x[4]),
                  x[3]*np.sin(x[4]),
                  u[0],
                  u[1],
                  x[3]/lwb*np.tan(x[2])])
    return f


# Axle forces and lagged-force dynamics.
def _axle_forces(Fx_total, F_yf0, F_yr0, mu, Fz_f, Fz_r, S):
    Fx_f = Fx_total * Fz_f / (Fz_f + Fz_r)
    Fx_r = Fx_total - Fx_f
    cap_f = S * mu * Fz_f
    cap_r = S * mu * Fz_r
    n_f = np.sqrt(Fx_f**2 + F_yf0**2)
    n_r = np.sqrt(Fx_r**2 + F_yr0**2)
    s_f = cap_f / n_f if n_f > cap_f else 1.0
    s_r = cap_r / n_r if n_r > cap_r else 1.0
    return Fx_f*s_f, F_yf0*s_f, Fx_r*s_r, F_yr0*s_r, Fx_f*s_f + Fx_r*s_r


def _steady_state_lateral_forces(x, accl, mu, C_Sf, C_Sr, lf, lr, h, m, S, C_curve, B_r):
    """Pure-slip Pacejka lateral forces at the current state (the lag targets)."""
    g = 9.81
    v = x[3]
    if v*np.cos(x[6]) < 0.5:
        return 0.0, 0.0
    L = lf + lr
    vx = v*np.cos(x[6])
    vy_f = v*np.sin(x[6]) + lf*x[5]
    vy_r = v*np.sin(x[6]) - lr*x[5]
    alpha_f = x[2] - np.arctan2(vy_f, vx)
    alpha_r = -np.arctan2(vy_r, vx)
    Fz_f = m*(g*lr - accl*h)/L
    Fz_r = m*(g*lf + accl*h)/L
    D_f = S*mu*Fz_f
    D_r = S*mu*Fz_r
    B_f = B_r * C_Sf/C_Sr
    Fy_f = D_f*np.sin(C_curve*np.arctan(B_f*alpha_f))
    Fy_r = D_r*np.sin(C_curve*np.arctan(B_r*alpha_r))
    return Fy_f, Fy_r


def _dynamics_with_lagged_forces(x, u_init, mu, C_Sf, C_Sr, lf, lr, h, m, I,
                                 s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max,
                                 F_yf0, F_yr0, S):
    from math import sin, cos
    u = np.array([
        steering_constraint(x[2], u_init[0], s_min, s_max, sv_min, sv_max),
        accl_constraints(x[3], u_init[1], v_switch, a_max, v_min, v_max),
    ])

    # Low-speed kinematic branch.
    if x[3] * np.cos(x[6]) < 0.5:
        lwb = lf + lr
        x_ks = x[0:5]
        f_ks = vehicle_dynamics_ks(
            x_ks, u, mu, C_Sf, C_Sr, lf, lr, h, m, I,
            s_min, s_max, sv_min, sv_max, v_switch, a_max, v_min, v_max,
        )
        return np.hstack((
            f_ks,
            np.array([
                u[1] / lwb * np.tan(x[2]) + x[3] / (lwb * np.cos(x[2]) ** 2) * u[0],
                -x[6] / TAU_BETA_KIN,
            ]),
        ))

    # Dynamic branch.
    vx = x[3]*cos(x[6])
    vy = x[3]*sin(x[6])
    g = 9.81
    L = lf + lr
    Fz_f = m*(g*lr - u[1]*h)/L
    Fz_r = m*(g*lf + u[1]*h)/L
    Fx_f, F_yf, Fx_r, F_yr, _ = _axle_forces(m*u[1], F_yf0, F_yr0, mu, Fz_f, Fz_r, S)
    cd = cos(x[2])
    sd = sin(x[2])
    vx_dot = (Fx_f*cd - F_yf*sd + Fx_r)/m + vy*x[5]
    vy_dot = (Fx_f*sd + F_yf*cd + F_yr)/m - vx*x[5]
    r_dot = (lf*(F_yf*cd + Fx_f*sd) - lr*F_yr) / I
    v_dot = (vx*vx_dot + vy*vy_dot) / x[3]
    beta_dot = (vx*vy_dot - vy*vx_dot) / x[3]**2
    X_dot = x[3]*cos(x[6]+x[4])
    Y_dot = x[3]*sin(x[6]+x[4])
    return np.array([X_dot, Y_dot, u[0], v_dot, x[5], r_dot, beta_dot])


# Actuator cascade.
def better_pid(speed, delayed_delta_cmd, current_speed, current_steer,
               current_accel, current_delta_dot, dt, min_sv, max_sv,
               tau_v, tau_a, a_max, tau_delta, tau_delta_dot,
               delta_th, steer_gamma):
    delayed_delta_cmd_sq = delayed_delta_cmd * delayed_delta_cmd
    delayed_delta_cmd_cb = delayed_delta_cmd_sq * delayed_delta_cmd
    delta_th_sq = delta_th * delta_th
    delta_target = delayed_delta_cmd_cb / ((delayed_delta_cmd_sq + delta_th_sq)*(1 + steer_gamma * delayed_delta_cmd_sq))
    delta_dot_des = (delta_target - current_steer) / tau_delta
    delta_ddot = (delta_dot_des - current_delta_dot) / tau_delta_dot
    sv = current_delta_dot + dt * delta_ddot
    if sv > max_sv:
        sv = max_sv
    elif sv < min_sv:
        sv = min_sv
    a_des = (speed - current_speed) / tau_v
    if a_des > a_max:
        a_des = a_max
    elif a_des < -a_max:
        a_des = -a_max
    accl = current_accel + dt * ((a_des - current_accel) / tau_a)
    if accl > a_max:
        accl = a_max
    elif accl < -a_max:
        accl = -a_max
    return accl, sv


# Full-trajectory reference: Euler-lag the pure-slip forces outside RK4,
# then RK4 the 7-state with the forces held constant across substages.
def reference_trajectory(x0, steer_cmd, vel_cmd, dt, n_steps, check_steps):
    x = np.array(x0, dtype=float)
    F_yf = F_yr = 0.0
    accel = delta_dot = 0.0
    S, C_curve, B_r = LAT_PEAK_SCALE, _C_CURVE, _B_R
    out = {}
    for step in range(1, n_steps + 1):
        accl, sv = better_pid(
            vel_cmd, steer_cmd, x[3], x[2], accel, delta_dot, dt,
            P['sv_min'], P['sv_max'], P['tau_v'], P['tau_a'], P['a_max'],
            P['tau_delta'], P['tau_delta_dot'], P['delta_th'], P['steer_gamma'],
        )
        accel, delta_dot = accl, sv
        # First-order relaxation lag, Euler, OUTSIDE RK4.
        F_yf_ss, F_yr_ss = _steady_state_lateral_forces(
            x, accl, P['mu'], P['C_Sf'], P['C_Sr'], P['lf'], P['lr'], P['h'], P['m'],
            S, C_curve, B_r)
        vx_now = max(x[3]*np.cos(x[6]), _SIGMA_V_FLOOR)
        lag_alpha = min(1.0, dt / (SIGMA_FY / vx_now))
        F_yf += lag_alpha * (F_yf_ss - F_yf)
        F_yr += lag_alpha * (F_yr_ss - F_yr)
        # RK4 with held forces.
        u = np.array([sv, accl])
        k1 = _dynamics_with_lagged_forces(x,            u, *dyn_args(), F_yf, F_yr, S)
        k2 = _dynamics_with_lagged_forces(x + dt*k1/2,  u, *dyn_args(), F_yf, F_yr, S)
        k3 = _dynamics_with_lagged_forces(x + dt*k2/2,  u, *dyn_args(), F_yf, F_yr, S)
        k4 = _dynamics_with_lagged_forces(x + dt*k3,    u, *dyn_args(), F_yf, F_yr, S)
        x = x + dt*(k1 + 2*k2 + 2*k3 + k4)/6.0
        # yaw wrap to [0, 2pi) (compare angularly in C)
        if   x[4] > 2*np.pi: x[4] -= 2*np.pi
        elif x[4] < 0:       x[4] += 2*np.pi
        if step in check_steps:
            out[step] = x.copy()
    return out


# Test cases
# State layout is [X, Y, delta, v, psi, r, beta]; fy_f, fy_r are held lagged forces.
# (X, Y, delta, v, psi, r, beta, fy_f, fy_r, sv_raw, accl_raw)
DYN_CASES = [
    ("ks_standstill",   [0,0, 0.0, 0.0, 0.0, 0.0, 0.0,  0.0,  0.0,  0.2,  2.0]),
    ("ks_lowspeed_beta",[1,2, 0.2, 0.30, 0.5, 0.4, 0.35, 0.0,  0.0,  0.5,  1.0]),
    ("dyn_straightish", [0,0, 0.05, 4.0, 0.3, 0.2, 0.02, 5.0,  8.0,  0.3,  1.5]),
    ("dyn_hardturn",    [3,1, 0.30, 6.0, 1.0, 1.5, 0.15, 20.0, 30.0,-0.5, -2.0]),
    ("dyn_saturate",    [0,0, 0.40, 8.0, 0.0, 2.0, 0.20, 60.0, 90.0, 1.0,  5.0]),
]

# (current_steer, current_speed, current_accel, current_delta_dot,
#  delayed_delta_cmd, delayed_speed_cmd)
CASCADE_CASES = [
    ("from_rest", [0.0, 0.0, 0.0, 0.0,   0.2,  3.0]),
    ("midspeed",  [0.1, 4.0, 1.0, 0.3,  -0.15, 6.0]),
    ("decel",     [0.3, 8.0, -1.0, -0.2, 0.0,  0.0]),
    ("accl_clip", [0.0, 0.0, 0.0, 0.0,   0.4, 10.0]),
]


def dyn_args():
    return (P['mu'], P['C_Sf'], P['C_Sr'], P['lf'], P['lr'], P['h'], P['m'], P['I'],
            P['s_min'], P['s_max'], P['sv_min'], P['sv_max'], P['v_switch'],
            P['a_max'], P['v_min'], P['v_max'])


def f(v):
    """Format any python number as a valid round-trippable C float literal."""
    return repr(float(v)) + "f"   # repr(float) always has '.' or 'e' -> valid


def main():
    lines = []
    w = lines.append
    w("// AUTO-GENERATED by reference.py — do not edit by hand.")
    w("// Ground truth from tests/reference.py, computed in float64.")
    w("// The C test checks its float32 results against these.")
    w("#ifndef FIXTURES_H")
    w("#define FIXTURES_H")
    w("")
    # Emit one fixture builder per struct. Only the fields the dynamics and
    # cascade math read are set; the rest stay 0.
    w("static Chassis fixture_chassis(void){")
    w("    Chassis chas = {0};")
    w(f"    chas.m={f(P['m'])}; chas.inertia={f(P['I'])};")
    w(f"    chas.lf={f(P['lf'])}; chas.lr={f(P['lr'])}; chas.h={f(P['h'])};")
    w("    chas.gravity=9.81f; chas.sigma_fy=0.03f;")
    w("    chas.sigma_v_floor=0.5f; chas.tau_beta_kin=0.15f;")
    w("    return chas;")
    w("}")
    w("")
    w("static Limits fixture_limits(void){")
    w("    Limits lim = {0};")
    w(f"    lim.s_min={f(P['s_min'])}; lim.s_max={f(P['s_max'])}; lim.sv_min={f(P['sv_min'])}; lim.sv_max={f(P['sv_max'])};")
    w(f"    lim.v_switch={f(P['v_switch'])}; lim.a_max={f(P['a_max'])}; lim.v_min={f(P['v_min'])}; lim.v_max={f(P['v_max'])};")
    w(f"    lim.tau_v={f(P['tau_v'])}; lim.tau_a={f(P['tau_a'])}; lim.t_vd={f(P['t_vd'])}; lim.t_sd={f(P['t_sd'])};")
    w(f"    lim.tau_delta={f(P['tau_delta'])}; lim.tau_delta_dot={f(P['tau_delta_dot'])};")
    w(f"    lim.steer_gamma={f(P['steer_gamma'])}; lim.delta_th={f(P['delta_th'])};")
    w("    return lim;")
    w("}")
    w("")
    w("static Tyre fixture_tyre(void){")
    w("    Tyre tyre = {0};")
    w(f"    tyre.mu={f(P['mu'])}; tyre.c_sf={f(P['C_Sf'])}; tyre.c_sr={f(P['C_Sr'])};")
    w(f"    tyre.s={f(LAT_PEAK_SCALE)}; tyre.c_curve={f(_C_CURVE)}; tyre.b_r={f(_B_R)};")
    w("    return tyre;")
    w("}")
    w("")

    # ---- dynamics derivative cases ----
    w("typedef struct {")
    w("    const char* name;")
    w("    float x,y,delta,v,yaw,r,beta,fy_f,fy_r;")
    w("    float sv, accl;")
    w("    float exp[7];   // [x,y,delta,v,yaw,r,beta]")
    w("} DynCase;")
    w("static const DynCase DYN_CASES[] = {")
    for name, c in DYN_CASES:
        x = np.array([c[0], c[1], c[2], c[3], c[4], c[5], c[6]], dtype=float)
        F_yf0, F_yr0 = c[7], c[8]
        u = np.array([c[9], c[10]], dtype=float)
        dz = _dynamics_with_lagged_forces(x, u, *dyn_args(), F_yf0, F_yr0, LAT_PEAK_SCALE)
        st = ", ".join(f(v) for v in c[:9])
        sv_accl = f"{f(c[9])}, {f(c[10])}"
        exp = ", ".join(f(v) for v in dz)
        w(f'    {{"{name}", {st}, {sv_accl}, {{{exp}}}}},')
    w("};")
    w(f"static const int N_DYN_CASES = {len(DYN_CASES)};")
    w("")

    # ---- cascade cases ----
    w("typedef struct {")
    w("    const char* name;")
    w("    float cur_steer, cur_speed, cur_accel, cur_delta_dot;")
    w("    float delayed_delta_cmd, delayed_speed_cmd;")
    w("    float exp_sv, exp_accl;")
    w("} CascadeCase;")
    w("static const CascadeCase CASCADE_CASES[] = {")
    for name, c in CASCADE_CASES:
        cur_steer, cur_speed, cur_accel, cur_dd, dcmd, scmd = c
        accl, sv = better_pid(
            scmd, dcmd, cur_speed, cur_steer, cur_accel, cur_dd, DT,
            P['sv_min'], P['sv_max'], P['tau_v'], P['tau_a'], P['a_max'],
            P['tau_delta'], P['tau_delta_dot'], P['delta_th'], P['steer_gamma'],
        )
        vals = ", ".join(f(v) for v in c)
        w(f'    {{"{name}", {vals}, {f(sv)}, {f(accl)}}},')
    w("};")
    w(f"static const int N_CASCADE_CASES = {len(CASCADE_CASES)};")
    w("")

    # ---- full-trajectory checkpoints ----
    # Open-loop hold: steady steer + speed demand from a moving start, long
    # enough to engage the dynamic branch, slip saturation, and the tyre lag.
    TRAJ_X0 = [0.0, 0.0, 0.0, 3.0, 0.0, 0.0, 0.0]   # [X,Y,delta,v,psi,r,beta]
    TRAJ_STEER, TRAJ_VEL = 0.15, 4.0
    TRAJ_DT, TRAJ_NSTEPS = 0.001, 500
    TRAJ_CHECK_STEPS = [100, 200, 300, 400, 500]
    traj = reference_trajectory(TRAJ_X0, TRAJ_STEER, TRAJ_VEL, TRAJ_DT, TRAJ_NSTEPS,
                             set(TRAJ_CHECK_STEPS))
    w("typedef struct { int step; float st[7]; } TrajCheck;  // [x,y,delta,v,yaw,r,beta]")
    w(f"static const float TRAJ_X0[7] = {{{', '.join(f(v) for v in TRAJ_X0)}}};")
    w(f"static const float TRAJ_STEER_CMD = {f(TRAJ_STEER)};")
    w(f"static const float TRAJ_VEL_CMD   = {f(TRAJ_VEL)};")
    w(f"static const int   TRAJ_NSTEPS    = {TRAJ_NSTEPS};")
    w("static const TrajCheck TRAJ_CHECKS[] = {")
    for s in TRAJ_CHECK_STEPS:
        st = ", ".join(f(v) for v in traj[s])
        w(f"    {{{s}, {{{st}}}}},")
    w("};")
    w(f"static const int N_TRAJ_CHECKS = {len(TRAJ_CHECK_STEPS)};")
    w("")
    w("#endif // FIXTURES_H")

    out = Path(__file__).resolve().parent / "fixtures.h"
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {out}  ({len(DYN_CASES)} dyn cases, {len(CASCADE_CASES)} cascade cases)")
    print(f"pacejka: s={LAT_PEAK_SCALE}  c_curve={_C_CURVE:.6f}  b_r={_B_R:.6f}")


if __name__ == "__main__":
    main()
