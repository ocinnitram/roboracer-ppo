/* Numeric gate for the plant against fixtures.h from reference.py.
 *
 *   T1  dynamics()    : per-state derivative
 *   T2  get_inputs()  : actuator cascade
 *   T3  rk4_step()    : integrator self-convergence (dt=1e-3 vs dt=1e-4)
 *   T4  tick loop     : get_inputs + rk4_step trajectory vs the reference
 */
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "roboracer.h"
#include "fixtures.h"

#define ATOL 1e-3   /* absolute floor: distinguishes real bugs from f32 noise */
#define RTOL 1e-4   /* relative: derivatives span O(1)..O(100)               */

static int g_fail = 0;
static int g_checks = 0;

static const char *DZ_NAMES[7] = {"x_dot", "y_dot", "delta_dot", "v_dot",
                                  "yaw_dot", "r_dot", "beta_dot"};

static void check(const char *test, const char *name, const char *comp,
                  double got, double ref) {
    g_checks++;
    double err = fabs(got - ref);
    double tol = ATOL + RTOL * fabs(ref);
    if (err > tol) {
        g_fail++;
        printf("  \x1b[31mFAIL\x1b[0m %-7s %-16s %-9s got=% .6g  ref=% .6g  |err|=%.3g (tol %.3g)\n",
               test, name, comp, got, ref, err, tol);
    }
}

/* ---- T1: dynamics() derivative ---- */
static void test_dynamics(void) {
    printf("\nT1  dynamics() derivative vs reference\n");
    Chassis chas = fixture_chassis();
    Limits  lim  = fixture_limits();
    Tyre    tyre = fixture_tyre();
    int fail0 = g_fail;
    for (int i = 0; i < N_DYN_CASES; i++) {
        const DynCase *c = &DYN_CASES[i];
        State z = {0};
        z.x = c->x; z.y = c->y; z.delta = c->delta; z.v = c->v;
        z.yaw = c->yaw; z.r = c->r; z.beta = c->beta;
        z.fy_f = c->fy_f; z.fy_r = c->fy_r;
        Inputs u = {c->sv, c->accl};
        State dz = dynamics(z, u, &chas, &lim, &tyre);
        double got[7] = {dz.x, dz.y, dz.delta, dz.v, dz.yaw, dz.r, dz.beta};
        for (int k = 0; k < 7; k++)
            check("T1", c->name, DZ_NAMES[k], got[k], c->exp[k]);
    }
    printf("  %s (%d cases)\n", g_fail == fail0 ? "\x1b[32mok\x1b[0m" : "\x1b[31mhad failures\x1b[0m",
           N_DYN_CASES);
}

/* ---- T2: get_inputs() cascade ---- */
static void test_cascade(void) {
    printf("\nT2  get_inputs() cascade vs better_pid\n");
    int fail0 = g_fail;
    for (int i = 0; i < N_CASCADE_CASES; i++) {
        const CascadeCase *c = &CASCADE_CASES[i];
        Roboracer env = {0};
        env.limits = fixture_limits();
        env.state.delta = c->cur_steer;
        env.state.v = c->cur_speed;
        env.act.accel = c->cur_accel;
        env.act.delta_dot = c->cur_delta_dot;
        Inputs in = get_inputs(&env, c->delayed_delta_cmd, c->delayed_speed_cmd);
        check("T2", c->name, "sv", in.sv, c->exp_sv);
        check("T2", c->name, "accl", in.accl, c->exp_accl);
    }
    printf("  %s (%d cases)\n", g_fail == fail0 ? "\x1b[32mok\x1b[0m" : "\x1b[31mhad failures\x1b[0m",
           N_CASCADE_CASES);
}

/* Integrate a fixed (sv, accl) hold for `secs` at step `dt`, return final pos. */
static void integrate(Chassis *chas, Limits *lim, Tyre *tyre, double dt, double secs,
                      double *out_x, double *out_y) {
    State z = {0};
    z.v = 3.0f;                 /* start in the dynamic branch (v*cos b > 0.5) */
    Inputs u = {0.1f, 2.0f};    /* gentle steer rate + accel */
    long n = (long)llround(secs / dt);
    for (long i = 0; i < n; i++) {
        z = rk4_step(z, u, (float)dt, chas, lim, tyre);
        z.yaw = normalise_yaw(z.yaw);
    }
    *out_x = z.x;
    *out_y = z.y;
}

/* ---- T3: RK4 self-convergence ---- */
static void test_rk4_convergence(void) {
    printf("\nT3  rk4_step() convergence (1ms vs 0.1ms over 1s)\n");
    Chassis chas = fixture_chassis();
    Limits  lim  = fixture_limits();
    Tyre    tyre = fixture_tyre();
    double xc, yc, xf, yf;
    integrate(&chas, &lim, &tyre, 1e-3, 1.0, &xc, &yc);
    integrate(&chas, &lim, &tyre, 1e-4, 1.0, &xf, &yf);
    double err = hypot(xc - xf, yc - yf);
    double target = 1e-4; /* 0.1 mm */
    g_checks++;
    printf("  pos @1ms  = (% .6f, % .6f)\n", xc, yc);
    printf("  pos @0.1ms= (% .6f, % .6f)\n", xf, yf);
    printf("  |err| = %.4g m (%.4f mm)   target < %.3g m\n", err, err * 1e3, target);
    if (err > target || !isfinite(err)) {
        g_fail++;
        printf("  \x1b[31mFAIL\x1b[0m integrator not converging at 1ms\n");
    } else {
        printf("  \x1b[32mok\x1b[0m\n");
    }
}

/* T4: C per-tick loop against the reference trajectory (Euler-lag forces
 * outside RK4). TRAJ_TOL absorbs the drift between the two lag schemes. */
#define TRAJ_TOL 5e-3   /* 5 mm over 0.5 s */
static void test_trajectory(void) {
    printf("\nT4  full-trajectory C vs reference (0.5 s hold, lag schemes differ)\n");
    Chassis chas = fixture_chassis();
    Limits  lim  = fixture_limits();
    Tyre    tyre = fixture_tyre();
    Roboracer env = {0};
    env.chassis = chas; env.limits = lim; env.tyre = tyre;
    env.state.x = TRAJ_X0[0]; env.state.y = TRAJ_X0[1]; env.state.delta = TRAJ_X0[2];
    env.state.v = TRAJ_X0[3]; env.state.yaw = TRAJ_X0[4];
    env.state.r = TRAJ_X0[5]; env.state.beta = TRAJ_X0[6];

    int ci = 0, fail0 = g_fail;
    double max_pos = 0, max_v = 0, max_yaw = 0;
    for (int step = 1; step <= TRAJ_NSTEPS; step++) {
        Inputs in = get_inputs(&env, TRAJ_STEER_CMD, TRAJ_VEL_CMD);
        env.act.delta_dot = in.sv;
        env.act.accel = in.accl;
        env.state = rk4_step(env.state, in, DT_SIM, &env.chassis, &env.limits, &env.tyre);
        env.state.yaw = normalise_yaw(env.state.yaw);

        if (ci < N_TRAJ_CHECKS && TRAJ_CHECKS[ci].step == step) {
            const float *st = TRAJ_CHECKS[ci].st;
            double dpos = hypot(env.state.x - st[0], env.state.y - st[1]);
            double dyaw = fabs(atan2(sin(env.state.yaw - st[4]), cos(env.state.yaw - st[4])));
            double dv   = fabs(env.state.v - st[3]);
            if (dpos > max_pos) max_pos = dpos;
            if (dyaw > max_yaw) max_yaw = dyaw;
            if (dv   > max_v)   max_v   = dv;
            g_checks += 3;
            if (dpos > TRAJ_TOL || dyaw > TRAJ_TOL || dv > TRAJ_TOL) {
                g_fail++;
                printf("  \x1b[31mFAIL\x1b[0m step %d  dpos=%.4g m  dyaw=%.4g  dv=%.4g\n",
                       step, dpos, dyaw, dv);
            }
        }
    }
    printf("  max over %d checkpoints: dpos=%.4g m (%.3f mm)  dyaw=%.4g rad  dv=%.4g m/s\n",
           N_TRAJ_CHECKS, max_pos, max_pos * 1e3, max_yaw, max_v);
    printf("  %s\n", g_fail == fail0 ? "\x1b[32mok\x1b[0m" : "\x1b[31mhad failures\x1b[0m");
}

int main(void) {
    printf("roboracer dynamics faithfulness tests\n");
    printf("=====================================\n");
    test_dynamics();
    test_cascade();
    test_rk4_convergence();
    test_trajectory();
    printf("\n-------------------------------------\n");
    if (g_fail == 0)
        printf("\x1b[32mALL PASSED\x1b[0m (%d checks)\n", g_checks);
    else
        printf("\x1b[31m%d / %d checks FAILED\x1b[0m\n", g_fail, g_checks);
    return g_fail ? 1 : 0;
}
