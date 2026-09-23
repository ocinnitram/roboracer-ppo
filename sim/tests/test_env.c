/* Env behaviour between the physics (test_dynamics.c) and the policy:
 *
 *   E1  raytrace()           : beams hit walls (not through them), stay in [-1,1]
 *   E2  get_reward()         : progress/collision/lap sign + scale
 *   E3  check_done()         : on-track vs wall vs off-map collision logic
 *   E4  get_s_norm()         : raceline progress lookup + monotonicity
 *   E5  construct_observation: 26-float obs layout / passthrough
 *   E6  c_step() clamps      : out-of-range actions clipped to [-1,1] in obs
 *   E7  lap counting         : finish-line wrap counts a lap; the last lap is COMPLETE
 *   E8  raytrace_obs()       : observed beams bit-identical to the full scan
 *   E9  static rectangle     : world pose, lap wrap, lidar, collision, isolation
 *   E10 c_set_obstacles()    : policy lidar on a reset env includes the new boxes
 *
 * Uses the same track blob as test_edt (EDT_BLOB_PATH).
 */
#include <stdio.h>
#include <math.h>
#include <string.h>

#include "roboracer.h"
#include "fixtures_env.h"   // EDT_BLOB_PATH

#define ATOL 1e-4f

static int g_fail = 0;
static int g_checks = 0;

static void expect(const char *grp, const char *what, int cond) {
    g_checks++;
    if (!cond) { g_fail++; printf("  \x1b[31mFAIL\x1b[0m %-4s %s\n", grp, what); }
}
static void near(const char *grp, const char *what, float got, float ref, float tol) {
    g_checks++;
    if (fabsf(got - ref) > tol) {
        g_fail++;
        printf("  \x1b[31mFAIL\x1b[0m %-4s %-22s got=% .6g ref=% .6g |err|=%.3g\n",
               grp, what, got, ref, fabsf(got - ref));
    }
}

/* Populate a realistic env and load the track blob. */
static void setup_env(Roboracer *env) {
    memset(env, 0, sizeof(*env));
    if (load_track_blob(env, EDT_BLOB_PATH) != 0) {
        printf("\x1b[31mFATAL\x1b[0m load_track_blob failed\n");
        exit(1);
    }

    env->chassis = (Chassis){ .m=3.74f, .inertia=0.04712f, .lf=0.15875f, .lr=0.17145f,
        .h=0.074f, .ego_l=0.5f, .ego_w=0.29f, .r_inscribed=0.145f,
        .r_circumscribed=0.289f, .lidar_mount_offset=0.10355f,
        .gravity=9.81f, .sigma_fy=0.03f, .sigma_v_floor=0.5f, .tau_beta_kin=0.15f };

    env->limits = (Limits){ .s_min=-0.4189f, .s_max=0.4189f, .sv_min=-3.2f, .sv_max=3.2f,
        .v_switch=7.319f, .a_max=3.48f, .v_min=0.0f, .v_max=4.0f,
        .tau_v=0.118f, .tau_a=0.061f, .t_vd=0.020f, .t_sd=0.060f,
        .tau_delta=0.075f, .tau_delta_dot=0.027f, .steer_gamma=0.622f, .delta_th=0.033f,
        .yaw_max=3.142f, .r_max=2.618f };

    env->tyre = (Tyre){ .mu=0.42f, .c_sf=4.718f, .c_sr=5.4562f };
    set_tyre_curve(&env->tyre, 1.20f, 0.16f);

    env->perception.fov = 4.712389f;
    env->perception.max_range = 10.0f;
    env->perception.n_beams = 811;
    env->perception.angle_min = -env->perception.fov / 2.0f;
    env->perception.angle_inc = env->perception.fov / (env->perception.n_beams - 1);
    init_beam_tables(&env->perception);

    env->time = (Timing){ .control_period=67, .compute_latency=6, .compute_latency_sd=0.0f };
    env->reward = (Reward){ .progress=20.0f, .collision=-1.0f, .lap_completion_bonus=1.0f };
    env->lap_cap = LAPS_TO_FINISH;
    env->optimal_lap_time = 7.17f;

    init(env);  // actuator delay-buffer lengths from t_sd/t_vd
}

/* Full-resolution scan: every native beam through the same tracer as the policy beams. */
static void raytrace(Roboracer *env, float *lidar) {
    float cpsi = cosf(env->state.yaw), spsi = sinf(env->state.yaw);
    float ox = env->state.x + env->chassis.lidar_mount_offset * cpsi;
    float oy = env->state.y + env->chassis.lidar_mount_offset * spsi;
    for (int i = 0; i < env->perception.n_beams; i++)
        lidar[i] = trace_one_beam(ox, oy, cpsi, spsi, env->perception.beam_cos[i],
                                  env->perception.beam_sin[i], env);
}

/* place the car on raceline index i (free-space, faces along the line) */
static void place_on_rline(Roboracer *env, int i) {
    const float *row = rline_row(&env->track, i);
    env->state = (State){0};
    env->state.x   = row[RL_X];
    env->state.y   = row[RL_Y];
    env->state.yaw = row[RL_YAW];
}

/* ---- E1: raytrace hits walls, never through them ---- */
static void test_raytrace(void) {
    printf("\nE1  raytrace() beams hit walls (regression: not through them)\n");
    Roboracer env; setup_env(&env);
    float lidar[MAX_BEAMS];
    const int idxs[] = {0, 80, 160, 240, 320, 400};
    int n_hit_total = 0;
    int f0 = g_fail;

    for (int ii = 0; ii < 6; ii++) {
        place_on_rline(&env, idxs[ii]);
        raytrace(&env, lidar);

        float cpsi = cosf(env.state.yaw), spsi = sinf(env.state.yaw);
        float ox = env.state.x + env.chassis.lidar_mount_offset * cpsi;
        float oy = env.state.y + env.chassis.lidar_mount_offset * spsi;
        float clear0 = bilinear_sample_edt(&env.track, ox, oy);
        expect("E1", "lidar origin on track (clearance>0)", clear0 > 0.0f);

        for (int b = 0; b < env.perception.n_beams; b++) {
            float nrm = lidar[b];
            if (!isfinite(nrm) || nrm < -1.0001f || nrm > 1.0001f) {
                expect("E1", "beam normalized in [-1,1]", 0); break;
            }
            float range = (nrm + 1.0f) * 0.5f * env.perception.max_range;
            // beam world direction = yaw + (angle_min + b*angle_inc)
            float c = cpsi*env.perception.beam_cos[b] - spsi*env.perception.beam_sin[b];
            float s = spsi*env.perception.beam_cos[b] + cpsi*env.perception.beam_sin[b];

            // lower bound: cannot hit a wall closer than the origin clearance
            if (range < clear0 - 0.06f) {
                expect("E1", "no premature hit (range >= clearance)", 0);
            }
            // through-wall regression: if it claims a hit, that point must be at
            // (or inside) a wall, NOT out in open space.
            if (range < env.perception.max_range * 0.999f) {
                n_hit_total++;
                float hx = ox + range * c, hy = oy + range * s;
                float dhit = bilinear_sample_edt(&env.track, hx, hy);
                if (dhit > 0.15f) {   // open space at the claimed hit => through-wall
                    expect("E1", "hit point is at a wall (not open space)", 0);
                }
            }
        }
    }
    expect("E1", "some beams hit (enclosed track)", n_hit_total > 0);
    printf("  %d beams hit across 6 poses   %s\n", n_hit_total,
           g_fail == f0 ? "\x1b[32mok\x1b[0m" : "\x1b[31mhad failures\x1b[0m");
    unload_track_blob(&env);
}

/* ---- E2: reward sign + scale ---- */
static void test_reward(void) {
    printf("\nE2  get_reward() sign + scale\n");
    Reward r = { .progress=20.0f, .collision=-1.0f, .lap_completion_bonus=1.0f };
    int f0 = g_fail;
    near("E2", "forward progress",      get_reward(&r, 0.01f, NOT_DONE,  false, 1.0f, 1.0f), 0.20f, ATOL);
    near("E2", "backward clamped to 0", get_reward(&r, -0.5f, NOT_DONE,  false, 1.0f, 1.0f), 0.00f, ATOL);
    near("E2", "collision penalty",     get_reward(&r, 0.00f, COLLISION, false, 1.0f, 1.0f), -1.00f, ATOL);
    near("E2", "progress + lap bonus",  get_reward(&r, 0.01f, NOT_DONE,  true,  1.0f, 1.0f), 1.20f, ATOL);
    near("E2", "complete = bonus only", get_reward(&r, 0.00f, COMPLETE,  true,  1.0f, 1.0f), 1.00f, ATOL);
    printf("  %s\n", g_fail == f0 ? "\x1b[32mok\x1b[0m" : "\x1b[31mhad failures\x1b[0m");
}

/* ---- E3: collision detection ---- */
static void test_collision(void) {
    printf("\nE3  check_done() on-track / wall / off-map\n");
    Roboracer env; setup_env(&env);
    int f0 = g_fail;

    place_on_rline(&env, 0);
    expect("E3", "on raceline => NOT_DONE", check_done(&env) == NOT_DONE);

    State off = {0}; off.x = 1e6f; off.y = 1e6f;
    env.state = off;
    expect("E3", "off-map => COLLISION", check_done(&env) == COLLISION);

    // find an interior cell strictly inside a wall (signed EDT < 0) and aim there
    int W = env.track.W, H = env.track.H;
    int found = 0;
    for (int iy = 1; iy < H-1 && !found; iy++)
        for (int ix = 1; ix < W-1 && !found; ix++)
            if (env.track.edt[iy*W + ix] < -0.001f) {
                State w = {0};
                w.x = env.track.origin_x + ix * env.track.res;
                w.y = env.track.origin_y + iy * env.track.res;
                env.state = w;
                expect("E3", "inside wall => COLLISION", check_done(&env) == COLLISION);
                found = 1;
            }
    expect("E3", "found a wall cell to test", found);
    printf("  %s\n", g_fail == f0 ? "\x1b[32mok\x1b[0m" : "\x1b[31mhad failures\x1b[0m");
    unload_track_blob(&env);
}

/* ---- E4: raceline progress lookup ---- */
static void test_progress(void) {
    printf("\nE4  get_s_norm() lookup + monotonic\n");
    Roboracer env; setup_env(&env);
    int f0 = g_fail;

    int idxs[] = {0, 100, 300};
    for (int k = 0; k < 3; k++) {
        int i = idxs[k];
        place_on_rline(&env, i);
        int ndx = i;
        float s = get_s_norm(&env.state, &env.track, &ndx);
        near("E4", "s_norm == rline s at point", s, rline_row(&env.track, i)[RL_S_NORM], 1e-3f);
        expect("E4", "nearest index is the point", ndx == i);
    }
    // s_norm increases along the line (away from the wrap at the end)
    int mono = 1;
    for (int i = 0; i < 50; i++)
        if (rline_row(&env.track, i + 1)[RL_S_NORM] <= rline_row(&env.track, i)[RL_S_NORM]) mono = 0;
    expect("E4", "s_norm monotonic increasing along line", mono);
    printf("  %s\n", g_fail == f0 ? "\x1b[32mok\x1b[0m" : "\x1b[31mhad failures\x1b[0m");
    unload_track_blob(&env);
}

/* ---- E5: observation construction ---- */
static void test_observation(void) {
    printf("\nE5  construct_observation() 26-float layout\n");
    Roboracer env; setup_env(&env);
    float obs[26]; memset(obs, 0, sizeof(obs));
    env.observations = obs;
    int f0 = g_fail;

    Observation o = {0};
    o.odom.s_norm = 0.11f; o.odom.r_norm = 0.22f; o.odom.v_norm = 0.33f; o.odom.yaw_norm = 0.44f;
    o.prev_delta = 0.55f; o.prev_v = 0.66f;
    for (int i = 0; i < MAX_BEAMS; i++) o.lidar[i] = i * 0.001f;

    construct_observation(&env, &o);

    near("E5", "obs[0] s_norm",   obs[0], 0.11f, ATOL);
    near("E5", "obs[1] r_norm",   obs[1], 0.22f, ATOL);
    near("E5", "obs[2] v_norm",   obs[2], 0.33f, ATOL);
    near("E5", "obs[3] yaw_norm", obs[3], 0.44f, ATOL);
    for (int b = 0; b < NUM_LIDAR_DOWNSAMPLE; b++)
        near("E5", "obs lidar subsample", obs[4 + b],
             o.lidar[obs_beam_index((int)env.perception.n_beams, b)], ATOL);
    near("E5", "obs[24] prev_delta", obs[24], 0.55f, ATOL);
    near("E5", "obs[25] prev_v",     obs[25], 0.66f, ATOL);
    printf("  %s\n", g_fail == f0 ? "\x1b[32mok\x1b[0m" : "\x1b[31mhad failures\x1b[0m");
    env.observations = NULL;  // don't let unload touch our stack buffer
    unload_track_blob(&env);
}

/* ---- E6: c_step action clamps ---- */
static void test_clamps(void) {
    printf("\nE6  c_step() clamps out-of-range actions into obs\n");
    Roboracer env; setup_env(&env);
    float obs[26], act[2], rew[1], term[1];
    env.observations = obs; env.actions = act; env.rewards = rew; env.terminals = term;
    int f0 = g_fail;

    c_reset(&env);

    // way past +1: the prev-action obs channels hold this step's action,
    // clamped, and the step must not have early-terminated.
    env.actions[0] = 5.0f; env.actions[1] = -3.0f;
    c_step(&env);
    near("E6", "obs prev_delta clamped to +1", obs[24], 1.0f, ATOL);
    near("E6", "obs prev_v clamped to -1",     obs[25], -1.0f, ATOL);
    expect("E6", "step did not early-terminate", env.tick > 0);

    // way below -1
    env.actions[0] = -9.0f; env.actions[1] = 4.0f;
    c_step(&env);
    near("E6", "obs prev_delta clamped to -1", obs[24], -1.0f, ATOL);
    near("E6", "obs prev_v clamped to +1",     obs[25],  1.0f, ATOL);

    int finite = 1;
    for (int i = 0; i < 26; i++) if (!isfinite(obs[i])) finite = 0;
    expect("E6", "all 26 obs finite after steps", finite);
    printf("  %s\n", g_fail == f0 ? "\x1b[32mok\x1b[0m" : "\x1b[31mhad failures\x1b[0m");

    env.observations = NULL; env.actions = NULL; env.rewards = NULL; env.terminals = NULL;
    unload_track_blob(&env);
}

/* ---- E7: lap-wrap counting + COMPLETE terminal ---- */
static void test_lap_counting(void) {
    printf("\nE7  c_step() finish-line wrap -> lap count / COMPLETE\n");
    Roboracer env; setup_env(&env);
    float obs[26], act[2], rew[1], term[1];
    env.observations = obs; env.actions = act; env.rewards = rew; env.terminals = term;
    int f0 = g_fail;
    int last = env.track.n_rline - 1;

    // Single wrap: car sits at the start (s_norm~0) with rl_ndx at the last
    // raceline point (s_norm~1). The first odom tick sees ds ~ -1 => finish line.
    c_reset(&env);
    env.rl_ndx = last;
    env.tick = 4000;
    env.lap_start_tick = 0;
    env.current_lap = 0;
    env.actions[0] = 0.0f; env.actions[1] = -1.0f;   // v_cmd=0 -> stays put, no crash
    c_step(&env);
    expect("E7", "wrap increments current_lap 0->1", env.current_lap == 1);
    expect("E7", "single lap is not terminal", env.terminals[0] == 0);
    expect("E7", "lap-completion bonus paid", env.rewards[0] > 0.9f);

    // Final wrap: one lap short of the finish -> wrapping triggers COMPLETE,
    // which sets the terminal and resets (so we assert via the terminal flag).
    c_reset(&env);
    env.rl_ndx = last;
    env.tick = 4000;
    env.lap_start_tick = 0;
    env.current_lap = LAPS_TO_FINISH - 1;
    env.actions[0] = 0.0f; env.actions[1] = -1.0f;
    c_step(&env);
    expect("E7", "completing final lap => terminal", env.terminals[0] == 1);

    printf("  %s\n", g_fail == f0 ? "\x1b[32mok\x1b[0m" : "\x1b[31mhad failures\x1b[0m");
    env.observations = NULL; env.actions = NULL; env.rewards = NULL; env.terminals = NULL;
    unload_track_blob(&env);
}

/* ---- E8: raytrace_obs() matches raytrace() at the 20 observed beams ---- */
/* The policy path traces only its 20 beams; they must equal the full scan bit for bit. */
static void test_raytrace_obs_equiv(void) {
    printf("\nE8  raytrace_obs() == raytrace() at the 20 observed beams\n");
    Roboracer env; setup_env(&env);
    float full[MAX_BEAMS], obs[MAX_BEAMS];
    const int idxs[] = {0, 80, 160, 240, 320, 400};
    int f0 = g_fail;

    for (int ii = 0; ii < 6; ii++) {
        place_on_rline(&env, idxs[ii]);
        raytrace(&env, full);
        raytrace_obs(&env, obs);
        for (int b = 0; b < NUM_LIDAR_DOWNSAMPLE; b++) {
            int i = obs_beam_index((int)env.perception.n_beams, b);
            // exact equality: same helper, same inputs -> no tolerance needed
            expect("E8", "obs beam == full beam (bit-identical)", obs[i] == full[i]);
        }
    }
    printf("  %s\n", g_fail == f0 ? "\x1b[32mok\x1b[0m" : "\x1b[31mhad failures\x1b[0m");
    unload_track_blob(&env);
}

/* ---- E9: track-frame static rectangle ---- */
#define E9_EGO_NDX 22        /* start of a 3 m near-straight (max |kappa| 0.074/m) */
#define E9_V_EGO 2.0f        /* [m/s] */
#define E9_HL 0.25f          /* [m] half length of the test rectangle */
#define E9_HW 0.145f         /* [m] half width */

/* Ego on raceline index i at speed v; rl_ndx synced so progress reads the
 * true station. */
static void park_ego(Roboracer *env, int i, float v) {
    place_on_rline(env, i);
    env->state.v = v;
    env->rl_ndx = i;
    get_s_norm(&env->state, &env->track, &env->rl_ndx);
}

static float station_of(const Roboracer *env, int i) {
    return rline_row(&env->track, i)[RL_S];
}

static Obstacle rect_at(float s, float n) {
    return (Obstacle){ .s = s, .n = n, .half_length = E9_HL, .half_width = E9_HW };
}

static void test_obstacles(void) {
    printf("\nE9  static rectangle: world pose, lap wrap, lidar, collision, isolation\n");
    Roboracer env, other;
    setup_env(&env);
    setup_env(&other);
    int f0 = g_fail;
    park_ego(&env, E9_EGO_NDX, E9_V_EGO);
    park_ego(&other, E9_EGO_NDX, E9_V_EGO);
    const float s_ego = station_of(&env, E9_EGO_NDX);
    const float lap = env.track.lap_length;

    /* The world pose is the racing-line point at s, offset along the left normal. */
    const float s_box = station_of(&env, E9_EGO_NDX + 30);
    Obstacle box = rect_at(s_box, 0.3f);
    c_set_obstacles(&env, &box, 1);
    float lx, ly, lyaw;
    rline_pose_at_s(&env.track, s_box, &lx, &ly, &lyaw);
    near("E9", "cx = line - n sin(yaw)", env.obstacles[0].cx, lx - 0.3f*sinf(lyaw), 1e-4f);
    near("E9", "cy = line + n cos(yaw)", env.obstacles[0].cy, ly + 0.3f*cosf(lyaw), 1e-4f);
    near("E9", "yaw is the track yaw", env.obstacles[0].yaw, lyaw, 1e-4f);

    /* Stations outside [0, lap) wrap onto the lap at install. */
    Obstacle past_seam = rect_at(lap + 0.5f, 0.0f);
    c_set_obstacles(&env, &past_seam, 1);
    near("E9", "station past the seam wraps", env.obstacles[0].s, 0.5f, 1e-4f);
    Obstacle before_start = rect_at(-0.3f, 0.0f);
    c_set_obstacles(&env, &before_start, 1);
    near("E9", "negative station wraps", env.obstacles[0].s, lap - 0.3f, 1e-4f);

    /* Lidar: a rectangle on the line ahead; the central beam (angle 0)
     * returns the distance from the lidar origin to its near face. */
    Obstacle ahead = rect_at(station_of(&env, E9_EGO_NDX + 20), 0.0f);
    c_set_obstacles(&env, &ahead, 1);
    float scan[MAX_BEAMS], clear[MAX_BEAMS];
    raytrace(&env, scan);
    raytrace(&other, clear);
    int centre = ((int)env.perception.n_beams - 1) / 2;
    float range = (scan[centre] + 1.0f) * 0.5f * env.perception.max_range;
    float expected = (ahead.s - s_ego) - env.chassis.lidar_mount_offset - E9_HL;
    near("E9", "central beam returns the near face", range, expected, 0.02f);
    expect("E9", "the other env's scan sees no rectangle", clear[centre] > scan[centre] + 0.05f);
    expect("E9", "clearance ahead is no collision", check_done(&env) == NOT_DONE);

    /* SAT collision: the rectangle centred on the ego. */
    Obstacle on_ego = rect_at(s_ego, 0.0f);
    c_set_obstacles(&env, &on_ego, 1);
    expect("E9", "overlap with the rectangle is a collision", check_done(&env) == COLLISION);

    printf("  %s\n", g_fail == f0 ? "\x1b[32mok\x1b[0m" : "\x1b[31mhad failures\x1b[0m");
    unload_track_blob(&env);
    unload_track_blob(&other);
}

/* ---- E10: set_obstacles rebuilds policy lidar on a reset env ---- */
static void test_set_obstacles_refreshes_obs(void) {
    printf("\nE10 set_obstacles refreshes policy lidar on a reset env\n");
    Roboracer env; setup_env(&env);
    float obs[26], act[2], rew[1], term[1];
    env.observations = obs; env.actions = act; env.rewards = rew; env.terminals = term;
    int f0 = g_fail;

    c_reset(&env);
    float lidar_empty[NUM_LIDAR_DOWNSAMPLE];
    memcpy(lidar_empty, obs + 4, sizeof(lidar_empty));

    float s0 = rline_row(&env.track, 0)[RL_S];
    Obstacle crate = rect_at(s0 + 5.0f, 0.0f);
    c_set_obstacles(&env, &crate, 1);

    int changed = 0;
    for (int b = 0; b < NUM_LIDAR_DOWNSAMPLE; b++)
        if (obs[4 + b] != lidar_empty[b]) changed = 1;
    expect("E10", "policy lidar changes after set_obstacles", changed);

    float lidar_after_set[NUM_LIDAR_DOWNSAMPLE];
    memcpy(lidar_after_set, obs + 4, sizeof(lidar_after_set));
    c_reset(&env);
    for (int b = 0; b < NUM_LIDAR_DOWNSAMPLE; b++)
        expect("E10", "matches reset-after-set lidar", obs[4 + b] == lidar_after_set[b]);

    printf("  %s\n", g_fail == f0 ? "\x1b[32mok\x1b[0m" : "\x1b[31mhad failures\x1b[0m");
    env.observations = NULL; env.actions = NULL; env.rewards = NULL; env.terminals = NULL;
    unload_track_blob(&env);
}

int main(void) {
    printf("roboracer env behaviour tests\n");
    printf("=============================\n");
    test_raytrace();
    test_reward();
    test_collision();
    test_progress();
    test_observation();
    test_clamps();
    test_lap_counting();
    test_raytrace_obs_equiv();
    test_obstacles();
    test_set_obstacles_refreshes_obs();
    printf("\n-----------------------------\n");
    if (g_fail == 0) printf("\x1b[32mALL PASSED\x1b[0m (%d checks)\n", g_checks);
    else             printf("\x1b[31m%d / %d checks FAILED\x1b[0m\n", g_fail, g_checks);
    return g_fail ? 1 : 0;
}
