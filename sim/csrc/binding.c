/* Observation/action sizes, kwargs parsing and the per-env rr_* ABI. */
#include "roboracer.h"
#define OBS_SIZE 26          // odom(4) + lidar(20) + prev_action(2)
#define NUM_ATNS 2           // steer, speed

#define Env Roboracer
#include "vecenv.h"

int my_init(Env* env, Dict* kwargs, const char* blob_path) {  // Read kwargs and load the blob; 0 on success.
    env->limits.s_min = dict_get(kwargs, "s_min")->value;
    env->limits.s_max = dict_get(kwargs, "s_max")->value;
    env->limits.sv_min = dict_get(kwargs, "sv_min")->value;
    env->limits.sv_max = dict_get(kwargs, "sv_max")->value;
    env->limits.v_switch = dict_get(kwargs, "v_switch")->value;
    env->limits.a_max = dict_get(kwargs, "a_max")->value;
    env->limits.v_min = dict_get(kwargs, "v_min")->value;
    env->limits.v_max = dict_get(kwargs, "v_max")->value;
    env->limits.tau_v = dict_get(kwargs, "tau_v")->value;
    env->limits.tau_a = dict_get(kwargs, "tau_a")->value;
    env->limits.t_vd = dict_get(kwargs, "t_vd")->value;
    env->limits.t_sd = dict_get(kwargs, "t_sd")->value;
    env->limits.tau_delta = dict_get(kwargs, "tau_delta")->value;
    env->limits.tau_delta_dot = dict_get(kwargs, "tau_delta_dot")->value;
    env->limits.steer_gamma = dict_get(kwargs, "steer_gamma")->value;
    env->limits.delta_th = dict_get(kwargs, "delta_th")->value;
    env->limits.yaw_max = dict_get(kwargs, "yaw_max")->value;
    env->limits.r_max = dict_get(kwargs, "r_max")->value;

    env->chassis.m = dict_get(kwargs, "m")->value;
    env->chassis.inertia = dict_get(kwargs, "i")->value;
    env->chassis.lf = dict_get(kwargs, "lf")->value;
    env->chassis.lr = dict_get(kwargs, "lr")->value;
    env->chassis.h = dict_get(kwargs, "h")->value;
    env->chassis.ego_l = dict_get(kwargs, "ego_l")->value;
    env->chassis.ego_w = dict_get(kwargs, "ego_w")->value;
    env->chassis.r_circumscribed = dict_get(kwargs, "r_circumscribed")->value;
    env->chassis.r_inscribed = dict_get(kwargs, "r_inscribed")->value;
    env->chassis.lidar_mount_offset = dict_get(kwargs, "lidar_mount_offset")->value;
    env->chassis.gravity = dict_get(kwargs, "gravity")->value;
    env->chassis.sigma_fy = dict_get(kwargs, "sigma_fy")->value;
    env->chassis.sigma_v_floor = dict_get(kwargs, "sigma_v_floor")->value;
    env->chassis.tau_beta_kin = dict_get(kwargs, "tau_beta_kin")->value;

    env->tyre.mu = dict_get(kwargs, "mu")->value;
    env->tyre.c_sf = dict_get(kwargs, "c_sf")->value;
    env->tyre.c_sr = dict_get(kwargs, "c_sr")->value;
    set_tyre_curve(&env->tyre, dict_get(kwargs, "tyre_s")->value, dict_get(kwargs, "alpha_pk")->value);

    env->time.control_period = (int)dict_get(kwargs, "control_period")->value;
    env->time.compute_latency = (int)dict_get(kwargs, "compute_latency")->value;
    env->time.compute_latency_sd = (float)dict_get(kwargs, "compute_latency_sd")->value;
    env->sim2real = (int)dict_get(kwargs, "sim2real")->value; // gates timing jitter, tyre draws and sensor noise

    env->perception.fov = dict_get(kwargs, "lidar_fov")->value;
    env->perception.max_range = dict_get(kwargs, "lidar_max_range")->value;
    env->perception.n_beams = dict_get(kwargs, "lidar_num_scans")->value;
    env->perception.angle_min = -env->perception.fov / 2.0f;
    env->perception.angle_inc = env->perception.fov / (env->perception.n_beams - 1);
    init_beam_tables(&env->perception);

    // Odometry std is in normalised obs units; lidar terms are metres, scaled to the 2*range/max-1 encoding.
    env->noise.odom_std[0] = dict_get(kwargs, "obs_noise_s_norm")->value;
    env->noise.odom_std[1] = dict_get(kwargs, "obs_noise_yaw_rate")->value;
    env->noise.odom_std[2] = dict_get(kwargs, "obs_noise_vel")->value;
    env->noise.odom_std[3] = dict_get(kwargs, "obs_noise_yaw")->value;
    float mr = env->perception.max_range;
    env->noise.lidar_bias_amp   = dict_get(kwargs, "lidar_bias")->value       * 2.0f / mr;
    env->noise.lidar_stoch_norm = dict_get(kwargs, "lidar_stochastic")->value * 2.0f / mr;

    env->reward.progress = dict_get(kwargs, "progress_reward")->value;
    env->reward.collision = dict_get(kwargs, "collision_reward")->value;
    env->reward.lap_completion_bonus = dict_get(kwargs, "lap_completion_reward")->value;

    env->optimal_lap_time = dict_get(kwargs, "optimal_lap_time")->value;
    env->lap_cap = LAPS_TO_FINISH;
    env->tick_cap = 0;

    init(env);     // actuator delay lengths, now that limits.t_sd/t_vd are set
    return c_init(env, blob_path);
}

static Env* rr_env(void* vec_p, int env_idx){
    StaticVec* vec = (StaticVec*)vec_p;
    if (env_idx < 0 || env_idx >= vec->size) return NULL;
    return &((Env*)vec->envs)[env_idx];
}

// Env i's struct, for the optional renderer library.
void* rr_env_ptr(void* vec_p, int env_idx){ return rr_env(vec_p, env_idx); }

void rr_seed(void* vec_p, int env_idx, unsigned int seed){
    Env* env = rr_env(vec_p, env_idx);
    if (env) env->rng = seed | 1u;   // xorshift needs a nonzero state
}

// Live lap state, valid between steps; the in-step auto-reset clears it.
int rr_current_lap(void* vec_p, int env_idx){
    Env* env = rr_env(vec_p, env_idx);
    return env ? (int)env->current_lap : -1;
}
float rr_last_lap_time(void* vec_p, int env_idx){
    Env* env = rr_env(vec_p, env_idx);
    return env ? env->last_lap_time : -1.0f;
}

// Outcome of the last finished episode, latched before its auto-reset.
int rr_done_laps(void* vec_p, int env_idx){
    Env* env = rr_env(vec_p, env_idx);
    return env ? (int)env->done_lap : -1;
}
float rr_done_lap_time(void* vec_p, int env_idx){
    Env* env = rr_env(vec_p, env_idx);
    return env ? env->done_lap_time : -1.0f;
}
int rr_done_kind(void* vec_p, int env_idx){
    Env* env = rr_env(vec_p, env_idx);
    return env ? (int)env->done_kind : -1;
}

// True plant state: x, y, yaw, delta, v, r, beta, accel, delta_dot, fy_f, fy_r.
void rr_vehicle_state(void* vec_p, int env_idx, float* out){
    Env* env = rr_env(vec_p, env_idx);
    if (!env) return;
    out[0] = env->state.x;
    out[1] = env->state.y;
    out[2] = env->state.yaw;
    out[3] = env->state.delta;
    out[4] = env->state.v;
    out[5] = env->state.r;
    out[6] = env->state.beta;
    out[7] = env->act.accel;
    out[8] = env->act.delta_dot;
    out[9] = env->state.fy_f;
    out[10] = env->state.fy_r;
}

// Laps before a COMPLETE terminal (at least 1); persists across auto-resets.
void rr_set_lap_cap(void* vec_p, int env_idx, int laps){
    Env* env = rr_env(vec_p, env_idx);
    if (env) env->lap_cap = laps < 1 ? 1 : laps;
}

// Episode cap in 1 ms ticks (<= 0 uses MAX_TICKS); persists across auto-resets.
void rr_set_tick_cap(void* vec_p, int env_idx, int tick_cap){
    Env* env = rr_env(vec_p, env_idx);
    if (env) env->tick_cap = tick_cap;
}

// Reset one env without latching a terminal.
void rr_reset_env(void* vec_p, int env_idx){
    Env* env = rr_env(vec_p, env_idx);
    if (env) c_reset(env);
}

// Track-frame rectangles packed as [s, n, half_length, half_width] per obstacle;
// the layout persists across auto-resets and the current observation is rebuilt to include it.
#define OBSTACLE_PACK_FLOATS 4
void rr_set_obstacles(void* vec_p, int env_idx, const float* data, int n){
    Env* env = rr_env(vec_p, env_idx);
    if (!env) return;
    if (n < 0) n = 0;
    if (n > MAX_OBSTACLES) n = MAX_OBSTACLES;
    Obstacle obstacles[MAX_OBSTACLES];
    for (int i = 0; i < n; i++){
        const float* row = data + i*OBSTACLE_PACK_FLOATS;
        obstacles[i] = (Obstacle){
            .s = row[0], .n = row[1], .half_length = row[2], .half_width = row[3],
        };
    }
    c_set_obstacles(env, obstacles, n);
}
