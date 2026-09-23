/* Pacejka plant, LiDAR, obstacles, and the reset/step hooks. */

#pragma once

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <fcntl.h>
#include <stdio.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>
#include "tyre_dr_basin.h"

#define DT_SIM 0.001f
#define DELAY_MAX 100
#define MAX_TICKS 45000 // episode cap in 1 ms ticks when no per-env tick cap is set
#define NUM_LIDAR_DOWNSAMPLE 20
#define EPS_LIDAR_HIT 0.01f
#define CAP_RAYTRACE_ITER 128
#define WINDOW_S 5
#define MAX_BEAMS 811
#define LAPS_TO_FINISH 2 // default lap cap: out-lap + flying lap
#define MAX_OBSTACLES 16 // rectangles overlaid on the static EDT

// Map 20 policy beams to full-scan indices identically in C and NumPy.
#define OBS_BEAM_SCALE 10000
static const int OBS_BEAM_FRAC[NUM_LIDAR_DOWNSAMPLE] = {
    // right side: -135.00 -108.75  -82.50  -56.25 deg
    0,    972,  1944, 2917,
    // front block: -30.00 .. +30.00 deg, 5.4545 deg apart
    3889, 4091, 4293, 4495, 4697, 4899, 5101, 5303, 5505, 5707, 5909, 6111,
    // left side:  +56.25  +82.50 +108.75 +135.00 deg
    7083, 8056, 9028, 10000
};
static inline int obs_beam_index(int n_beams, int b){
    return (OBS_BEAM_FRAC[b] * (n_beams - 1) + OBS_BEAM_SCALE / 2) / OBS_BEAM_SCALE;
}

// Odometry arrival period [ms]: fixed at ODOM_NOM, jittered under sim2real.
#define ODOM_NOM    20.0f
#define ODOM_SD      2.0f
#define ODOM_MIN     15
#define ODOM_MAX     25

// Episode outcome flags
#define NOT_DONE 0
#define COLLISION 1
#define COMPLETE 2

typedef struct {
    float delta_dot, accel;
    float v_cmd[DELAY_MAX], delta_cmd[DELAY_MAX];
    int head_delta;
    int head_v;
    size_t n_delta;
    size_t n_v;
    float prev_delta_cmd, prev_v_cmd;
} Actuator;

// Saturated tyre forces per axle
typedef struct {
     float fx_f, fy_f, fx_r, fy_r;
} AxleForces;

typedef struct {
    float x, y, delta, v, yaw, r, beta, fy_f, fy_r;
} State;

typedef struct {
    float fy_f, fy_r;
} LateralForces;

typedef struct{
float mu, c_sf, c_sr, s, c_curve, b_r;
} Tyre;

// Pacejka shape C and rear stiffness B from the peak scale S and the peak slip angle.
static void set_tyre_curve(Tyre *tyre, float s, float alpha_pk){
    tyre->s = s;
    tyre->c_curve = 2.0f*((float)M_PI - asinf(1.0f/s)) / (float)M_PI;
    tyre->b_r = tanf((float)M_PI/(2.0f*tyre->c_curve)) / alpha_pk;
}

typedef struct {
    float progress, collision, lap_completion_bonus;
} Reward;

typedef struct {
    float m, inertia, lf, lr, h, ego_l, ego_w;  // integrator, RK4, collision
    float r_circumscribed, r_inscribed;
    float lidar_mount_offset; // lidar offset from the CoG
    float gravity, sigma_fy, sigma_v_floor, tau_beta_kin;
} Chassis;

typedef struct{
    float s_min,s_max, sv_min,sv_max, v_switch, a_max, v_min,v_max,
          tau_v,tau_a, t_vd,t_sd, tau_delta,tau_delta_dot, steer_gamma, delta_th,
          yaw_max, r_max;
} Limits; // get_inputs + normalization

typedef struct{
     float fov, max_range;
     float n_beams;
     float angle_min;
     float angle_inc;
     float beam_cos[MAX_BEAMS];  // cos(phi_i), body frame
     float beam_sin[MAX_BEAMS];  // sin(phi_i)
} Perception;

typedef struct{
    float sv, accl;
} Inputs;

// Integer 1 ms tick counts.
typedef struct{
    int control_period;
    int compute_latency;
    float compute_latency_sd; // [ms] spread drawn under sim2real, 0 = fixed
    int pending_compute_latency; // drawn with the observation, consumed by the next step
} Timing;

// Observation (sensor) noise — applied at obs read-time only, never to dynamics.
typedef struct{
    float odom_std[4];        // indexed to obs[0..3]: s_norm, r(yaw_rate), v, yaw
    float lidar_bias_amp;     // per-episode U(-amp,+amp) half-width, normalised units
    float lidar_stoch_norm;   // per-step per-beam std, normalised units
    float lidar_bias_norm;    // this episode's drawn bias
} Noise;

// Normalised states for the RL
typedef struct {
    float s_norm, yaw_norm, r_norm, v_norm;
} Odom;

// The observation for the agent
typedef struct{
    float lidar[MAX_BEAMS];
    Odom odom;
    float prev_delta, prev_v;
} Observation;

typedef struct{
    void* blob_base; size_t blob_len; float* edt;
    float* rline; int W, H, n_rline; float res, origin_x, origin_y;
    float lap_length;   // [m] last raceline station plus the closing segment
} Track;

// Columns stored for each raceline sample in the track blob.
enum RlineCol { RL_X = 0, RL_Y, RL_S, RL_S_NORM, RL_YAW, RL_KAPPA, RL_W_L, RL_W_R, RL_N_X, RL_N_Y,
                RL_COLS };

static inline const float* rline_row(const Track *t, int i){ return t->rline + i * RL_COLS; }

// Lap length [m]: the last raceline station plus the segment closing the loop.
static float rline_lap_length(const Track *t){
    const float *last = rline_row(t, t->n_rline - 1);
    const float *first = rline_row(t, 0);
    return last[RL_S] + hypotf(first[RL_X] - last[RL_X], first[RL_Y] - last[RL_Y]);
}

// Track-frame rectangle; its world pose is computed once when installed.
typedef struct {
    float s, n;         // station [m] and lateral offset [m, +left]
    float half_length;  // [m] along the track
    float half_width;   // [m] across
    float cx, cy, yaw;  // world pose
} Obstacle;

typedef struct Roboracer {
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
    int tick;
    int rl_ndx; // cached nearest raceline index
    unsigned int rng;
    int8_t current_lap;
    int lap_cap;
    int tick_cap;         // <= 0 uses MAX_TICKS
    int lap_start_tick;   // tick at the last finish-line crossing (0 at episode start)
    float last_lap_time;  // seconds of the most recently completed lap (0 if none)
    // Latched at terminal, BEFORE the auto-reset wipes current_lap/last_lap_time,
    // so Python can read the finished episode's outcome after c_step returns.
    int8_t done_lap;
    float done_lap_time;
    int8_t done_kind;      // latched: 0 tick-cap, COLLISION, COMPLETE

    float optimal_lap_time;
    State state;
    Tyre tyre;
    Chassis chassis;
    Limits limits;
    Actuator act;
    Timing time;
    Track track;
    Perception perception;
    Reward reward;
    int sim2real;
    int next_odom_tick;
    float odom_cache_s, odom_cache_v, odom_cache_yaw, odom_cache_r; // last odometry arrival
    Noise noise;
    Obstacle obstacles[MAX_OBSTACLES]; // installed layout, kept across auto-resets
    int n_obstacles;
} Roboracer;

static void init(Roboracer *env){
    env->n_obstacles = 0;
    int n_delta = (int)ceil((double)env->limits.t_sd / DT_SIM);
    int n_v =  (int)ceil((double)env->limits.t_vd / DT_SIM);
    assert(n_delta >= 0 && n_delta <= DELAY_MAX);
    assert(n_v >= 0 && n_v <= DELAY_MAX);
    env->act.n_delta = n_delta;
    env->act.n_v = n_v;
    unsigned int z = (env->rng + 0x9E3779B9u);
    z = (z ^ (z>>16)) * 0x85EBCA6Bu;
    z = (z ^ (z>>13)) * 0xC2B2AE35u;
    env->rng = (z ^ (z>>16)) | 1u;   // ensure nonzero
}

// One RNG stream per env drives every sim2real draw, so the draw order is part of the behaviour.
static inline unsigned int xorshift32(unsigned int *s){
    unsigned int x=*s; x^=x<<13; x^=x>>17; x^=x<<5; *s = x ? x : 0x1u; return *s;
}
static inline float rng_uniform(unsigned int *s){          // [0,1)
    return (xorshift32(s) >> 8) * (1.0f/16777216.0f);
}
static inline float rng_normal(unsigned int *s){           // Box-Muller (one of two)
    float u1 = fmaxf(1e-7f, rng_uniform(s)), u2 = rng_uniform(s);
    return sqrtf(-2.0f*logf(u1)) * cosf(2.0f*(float)M_PI*u2);
}
static inline int sample_norm_int(unsigned int *s, float mean, float sd, int lo, int hi){
    int p = (int)lroundf(mean + sd*rng_normal(s));
    return p < lo ? lo : (p > hi ? hi : p);
}

// Draw the compute latency the next control step will use.
static inline void schedule_next_decision_timing(Roboracer *env){
    env->time.pending_compute_latency = (env->sim2real && env->time.compute_latency_sd > 0.0f)
        ? sample_norm_int(&env->rng, (float)env->time.compute_latency, env->time.compute_latency_sd,
                          0, env->time.control_period - 1)
        : env->time.compute_latency;
    assert(env->time.pending_compute_latency < env->time.control_period);
}

static Inputs make_inputs(float sv, float accl){
    return (Inputs){sv, accl};
}

static void init_beam_tables(Perception *p){
     for (int i = 0; i < p->n_beams; i++){
         float phi = p->angle_min + i * p->angle_inc;   // body-frame beam angle
         p->beam_cos[i] = cosf(phi);
         p->beam_sin[i] = sinf(phi);
     }
 }

// Scale each axle's combined force back onto its Kamm (friction) circle.
static AxleForces axle_forces(float fx_total, float f_yf0, float f_yr0, float mu, float fz_f, float fz_r, float s) {
    float fx_f = fx_total * fz_f/(fz_f +fz_r);
    float fx_r = fx_total - fx_f;
    float cap_f = s * mu * fz_f;
    float cap_r = s * mu * fz_r;
    float n_f = sqrtf(fx_f*fx_f + f_yf0*f_yf0);
    float n_r = sqrtf(fx_r*fx_r + f_yr0*f_yr0);
    float s_f = (n_f > cap_f) ? cap_f / n_f : 1.0f;
    float s_r = (n_r >  cap_r) ? cap_r / n_r : 1.0f;
    AxleForces out;
    out.fx_f = fx_f*s_f;
    out.fy_f = f_yf0 * s_f;
    out.fx_r = fx_r*s_r;
    out.fy_r = f_yr0 * s_r;
    return out;
}

// Steady-state pure-slip Pacejka lateral forces at the current state.
static LateralForces calculate_lateral_forces(State z, float accl, Chassis *chas, Tyre *tyre){
    float cb = cosf(z.beta);
    float vx = z.v * cb;
    if (vx < chas->sigma_v_floor) return (LateralForces){0.f, 0.f};

    float vsb = z.v * sinf(z.beta);
    float vy_f = vsb + chas->lf * z.r;
    float vy_r = vsb - chas->lr * z.r;
    float alpha_f = z.delta - atan2f(vy_f, vx);
    float alpha_r = -atan2f(vy_r, vx);

    // Normal loads with longitudinal force transfer
    const float wb = chas->lf + chas->lr;
    float fz_f = chas->m * (chas->gravity * chas->lr - accl *chas->h) / wb;
    float fz_r = chas->m * (chas->gravity * chas->lf + accl *chas->h) / wb;

    // Pure slip lateral forces
    float d_f = tyre->s * tyre->mu * fz_f;
    float d_r = tyre->s * tyre->mu * fz_r;
    float b_f = tyre->b_r * tyre->c_sf/tyre->c_sr; // front stiffness from the rear via the cornering-stiffness ratio
    float fy_f = d_f * sinf(tyre->c_curve * atanf(b_f * alpha_f));
    float fy_r = d_r * sinf(tyre->c_curve * atanf(tyre->b_r * alpha_r));

    return (LateralForces) {fy_f, fy_r};
}

// Clamp the steering rate to its limits and stop at the steering lock.
static void steering_constraint(float delta, float *sv, const Limits *l) {
    if (((delta <= l->s_min) && (*sv <= 0)) || ((delta >= l->s_max) && (*sv >= 0))){
        *sv = 0.0f;
    } else if (*sv < l->sv_min) {
        *sv =l->sv_min;
    } else if (*sv > l->sv_max) {
        *sv = l->sv_max;
    }
}

// Clamp acceleration, power-limited above v_switch and zero at the speed limits.
static void accl_constraint(float v, float *accl, const Limits *lim){
    float pos_limit = (v > lim->v_switch) ? (lim->a_max * lim->v_switch / v) : lim->a_max;

    if (((v <= lim->v_min) && (*accl <= 0)) || ((v >= lim->v_max) && (*accl >=0))){
        *accl = 0;
    } else if (*accl <= -lim->a_max){
        *accl = -lim->a_max;
    } else if (*accl >= pos_limit){
        *accl = pos_limit;
    }
}

// Kinematic single-track model; beta relaxes to zero.
static void vehicle_dynamics_ks(State *dz, State z, Inputs *inputs, Chassis *chas){
    float wb = chas->lr + chas->lf;

    dz->x = z.v * cosf(z.yaw);
    dz->y = z.v * sinf(z.yaw);
    dz->delta = inputs->sv;
    dz->v = inputs->accl;
    dz->yaw = z.v / wb * tanf(z.delta);
    dz->r = inputs->accl / wb * tanf(z.delta) + z.v / (wb * cosf(z.delta)*cosf(z.delta)) * inputs->sv;
    dz->beta = -z.beta / chas->tau_beta_kin;
}

// State derivative: kinematic below sigma_v_floor, dynamic single-track with lagged tyre forces above.
static State dynamics(State z, Inputs inputs, Chassis *chas, const Limits *lim, Tyre *tyre){
    steering_constraint(z.delta, &inputs.sv, lim);
    accl_constraint(z.v, &inputs.accl, lim);

    State dz = {0};
    float v_cb = z.v * cosf(z.beta);  // longitudinal body-frame speed, reused below
    LateralForces ss = calculate_lateral_forces(z, inputs.accl, chas, tyre);
    float tau = chas->sigma_fy / fmaxf(v_cb, chas->sigma_v_floor); // relaxation-length lag
    dz.fy_f = (ss.fy_f - z.fy_f) / tau;
    dz.fy_r = (ss.fy_r - z.fy_r) / tau;

    if (v_cb < chas->sigma_v_floor){
        vehicle_dynamics_ks(&dz, z, &inputs, chas);
        return dz;
    }

    float vx = v_cb, vy = z.v * sinf(z.beta);
    float wb = chas->lf +chas->lr;
    float fz_f = chas->m * (chas->gravity * chas->lr - inputs.accl * chas->h)/wb;
    float fz_r = chas->m * (chas->gravity * chas->lf + inputs.accl * chas->h)/wb;

    // Combined-slip saturation of the longitudinal demand
    AxleForces af = axle_forces(chas->m * inputs.accl, z.fy_f, z.fy_r, tyre->mu,
        fz_f, fz_r, tyre->s);

    // Force/moment balance in the rotating body frame
    float cd = cosf(z.delta), sd = sinf(z.delta);
    float vx_dot = ((af.fx_f * cd) - (af.fy_f * sd) + af.fx_r)/chas->m + vy*z.r;
    float vy_dot = ((af.fx_f * sd) + (af.fy_f * cd) + af.fy_r)/chas->m  - vx*z.r;
    dz.r = (chas->lf*(af.fy_f*cd + af.fx_f*sd) - chas->lr*af.fy_r) / chas->inertia;

    // Body-frame velocities back to (v, beta)
    dz.v = (vx * vx_dot + vy * vy_dot) / z.v;
    dz.beta = (vx*vy_dot - vy*vx_dot) / (z.v * z.v);
    dz.x = z.v*cosf(z.beta+z.yaw);
    dz.y = z.v*sinf(z.beta+z.yaw);
    dz.yaw = z.r;
    dz.delta = inputs.sv;

    return dz;
}

static State zak(State z, float a, State k){
    State out;
    out.x    = z.x    + a * k.x;
    out.y    = z.y    + a * k.y;
    out.delta= z.delta+ a * k.delta;
    out.v    = z.v    + a * k.v;
    out.yaw  = z.yaw  + a * k.yaw;
    out.r    = z.r    + a * k.r;
    out.beta = z.beta + a * k.beta;
    out.fy_f = z.fy_f + a * k.fy_f;
    out.fy_r = z.fy_r + a * k.fy_r;
    return out;
 }

static State rk4_step(State z, Inputs inputs, float dt, Chassis *chas, Limits *lim, Tyre *tyre){
    State k1 = dynamics(z, inputs, chas, lim, tyre);
    State k2 = dynamics(zak(z,dt/2.0f,k1), inputs, chas, lim, tyre);
    State k3 = dynamics(zak(z,dt/2.0f,k2), inputs, chas, lim, tyre);
    State k4 = dynamics(zak(z,dt,k3), inputs, chas, lim, tyre);
    State wsum = zak(zak(zak(k1, 2.0f, k2), 2.0f, k3), 1.0f, k4);
    State z_next = zak(z, dt/6.0f, wsum);

    return z_next;
}

// Actuator cascade: delayed steer/speed commands to steering rate and acceleration.
static Inputs get_inputs(Roboracer *env, float delayed_delta_cmd, float delayed_speed_cmd){
    float delayed_delta_cmd_sq = delayed_delta_cmd * delayed_delta_cmd;
    float delayed_delta_cmd_cb = delayed_delta_cmd_sq * delayed_delta_cmd;
    float delta_th_sq = env->limits.delta_th * env->limits.delta_th;
    float delta_target = delayed_delta_cmd_cb / ((delayed_delta_cmd_sq + delta_th_sq)*
        (1 + env->limits.steer_gamma * delayed_delta_cmd_sq));
    float delta_dot_des = (delta_target - env->state.delta) / env->limits.tau_delta;
    float delta_ddot = (delta_dot_des - env->act.delta_dot) / env->limits.tau_delta_dot;
    float sv =  env->act.delta_dot + DT_SIM * delta_ddot;
    if (sv > env->limits.sv_max){
        sv = env->limits.sv_max;
    } else if (sv < env->limits.sv_min){
        sv = env->limits.sv_min;
    }
    float a_des = (delayed_speed_cmd - env->state.v ) / env->limits.tau_v;
    if (a_des > env->limits.a_max){
        a_des = env->limits.a_max;
    }
    else if (a_des < -env->limits.a_max){
        a_des = -env->limits.a_max;
    }

    // first-order lag on acceleration
    float accl = env->act.accel + DT_SIM * ((a_des - env->act.accel) / env->limits.tau_a);
    if (accl > env->limits.a_max){
        accl = env->limits.a_max;
    } else if (accl < -env->limits.a_max){
        accl = -env->limits.a_max;
    }
    return make_inputs(sv, accl);
}

// Ring-buffer transport delay: push cmd, return the one n ticks old.
static float push_delay(float *buf, int *head, size_t n, float cmd){
    if (n == 0) return cmd;
    float delayed_cmd = buf[*head];
    buf[*head] = cmd;
    *head = (*head + 1) % n;
    return delayed_cmd;
}

static inline float normalise_yaw(float yaw){
    yaw = fmodf(yaw + (float)M_PI, 2.0f*(float)M_PI);
    if (yaw < 0.0f) yaw += 2.0f*(float)M_PI;
    yaw -= (float)M_PI;
    return yaw;
}

static void reset_actuator_buffers(Actuator *act){
     memset(act->delta_cmd, 0, sizeof(act->delta_cmd));
     memset(act->v_cmd,     0, sizeof(act->v_cmd));
     act->head_delta = 0;
     act->head_v     = 0;
     act->delta_dot  = 0.0f;
     act->accel      = 0.0f;
 }

static void unload_track_blob(Roboracer* env);
void c_close(Roboracer* env) {  // Unload the track blob.
    if (env->track.blob_base) unload_track_blob(env);
}

static void construct_observation(Roboracer *env, Observation *observation){
    int8_t k = 0;
    env->observations[k++] = observation->odom.s_norm; // 0
    env->observations[k++] = observation->odom.r_norm; // 1
    env->observations[k++] = observation->odom.v_norm; // 2
    env->observations[k++] = observation->odom.yaw_norm; // 3
    for (int b = 0; b < NUM_LIDAR_DOWNSAMPLE; b++){
        env->observations[k++] = observation->lidar[obs_beam_index((int)env->perception.n_beams, b)];
    }
    env->observations[k++] = observation->prev_delta; // 24
    env->observations[k++] = observation->prev_v; // 25
}

static inline float clampf(float x, float lo, float hi){ return x < lo ? lo : (x > hi ? hi : x); }

// Apply sensor noise only to the completed policy observation.
static void apply_obs_noise(Roboracer *env){
    float *o = env->observations;
    // odom: additive per-channel Gaussian. s_norm is arc-length in [0,1]; rest in [-1,1].
    o[0] = clampf(o[0] + env->noise.odom_std[0]*rng_normal(&env->rng), 0.0f, 1.0f);
    for (int i = 1; i < 4; i++)
        o[i] = clampf(o[i] + env->noise.odom_std[i]*rng_normal(&env->rng), -1.0f, 1.0f);
    // lidar: fixed per-episode bias (mount/calibration offset) + fresh per-beam stochastic.
    for (int b = 0; b < NUM_LIDAR_DOWNSAMPLE; b++)
        o[4+b] = clampf(o[4+b] + env->noise.lidar_bias_norm
                        + env->noise.lidar_stoch_norm*rng_normal(&env->rng), -1.0f, 1.0f);
}

// mmap the track blob (signed EDT + raceline) written by sim/track.py.
static int8_t load_track_blob(Roboracer *env, const char* path){
    int fd = open(path, O_RDONLY);
    if (fd == -1) {
        perror("track blob: open");
        return -1;
    }
    struct stat sb;
    if (fstat(fd, &sb) == -1){
        perror("track blob: fstat");
        close(fd);
        return -1;
    }
    env->track.blob_len = sb.st_size;
    uint8_t *base = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (base == MAP_FAILED){
        perror("mmap failed.");
        return -1;
    }
    if (memcmp(base, "RRBC", 4) != 0){
        fprintf(stderr, "track blob: bad magic in %s\n", path);
        munmap(base, sb.st_size);
        return -1;
    }

    env->track.blob_base = base;
    env->track.W = *(int32_t*)(base + 8);
    env->track.H = *(int32_t*)(base + 12);
    env->track.res = *(float*)(base + 16);
    env->track.origin_x = *(float*)(base + 20);
    env->track.origin_y = *(float*)(base + 24);
    env->track.n_rline = *(int32_t*)(base + 28);
    int32_t edt_offset   = *(int32_t*)(base + 32);
    int32_t rline_offset = *(int32_t*)(base + 36);
    env->track.edt   = (float*)(base + edt_offset);
    env->track.rline = (float*)(base + rline_offset);
    env->track.lap_length = rline_lap_length(&env->track);

    return 0;
}

static void unload_track_blob(Roboracer* env){
    munmap(env->track.blob_base, env->track.blob_len);
}

// Signed wall distance [m] at a world point, bilinear in the EDT grid; -1 off the map.
static float bilinear_sample_edt(Track *track, float wx, float wy){
    float gx = (wx - track->origin_x)/track->res;
    float gy = (wy - track->origin_y)/track->res;

    int ix = (int)floorf(gx);
    int iy = (int)floorf(gy);
    float a = gx-ix;
    float b = gy-iy;
    if (ix < 0 || iy < 0 || ix+1 >= track->W || iy+1 >=track->H) {
        return -1.0f;
    }
    float q11 = track->edt[iy * track->W + ix];
    float q21 = track->edt[iy * track->W + (ix+1)];
    float q12 = track->edt[(iy+1) * track->W + ix];
    float q22 = track->edt[(iy+1)* track->W + (ix+1)];

    float r1 = q11 * (1-a) + q21 * a;
    float r2 = q12 * (1-a) + q22 * a;
    float interp = r1 * (1 - b)+ r2 * b;

    return interp;
}

// Normalised raceline progress of the nearest sample within WINDOW_S of the cached index.
static float get_s_norm(State *state, Track *track, int *rl_ndx){
    int n = track->n_rline;

    float min_dist = INFINITY;
    int min_ndx = 0;
    for (int j = -WINDOW_S; j <= WINDOW_S; ++j) {
        int ndx = (*rl_ndx + j + n) % n;
        const float *row = rline_row(track, ndx);
        float dx = row[RL_X] - state->x;
        float dy = row[RL_Y] - state->y;
        float dist_sq = dx*dx + dy*dy;
        if (dist_sq<min_dist){
            min_dist = dist_sq;
            min_ndx = ndx;
        }
    }

    *rl_ndx = min_ndx;
    return rline_row(track, min_ndx)[RL_S_NORM];
}

// ---- Obstacles: static rectangles in the track frame ----

// Wrap a station into [0, lap_length).
static float wrap_station(float s, float lap_length){
    float wrapped = fmodf(s, lap_length);
    return wrapped < 0.0f ? wrapped + lap_length : wrapped;
}

// Raceline segment holding station s: the largest index i with s_i <= s.
// Binary search over the monotone s column.
static int rline_segment_at_s(const Track *t, float s){
    int lo = 0;
    int hi = t->n_rline - 1;
    while (lo < hi){
        int mid = (lo + hi + 1) / 2;
        if (rline_row(t, mid)[RL_S] <= s) lo = mid;
        else hi = mid - 1;
    }
    return lo;
}

// Racing-line point and track yaw at station s, interpolated inside its segment.
static void rline_pose_at_s(const Track *t, float s, float *x, float *y, float *yaw){
    int i = rline_segment_at_s(t, s);
    int j = (i + 1) % t->n_rline;
    const float *a = rline_row(t, i);
    const float *b = rline_row(t, j);
    float s_b = (j == 0) ? t->lap_length : b[RL_S];
    float f = (s - a[RL_S]) / (s_b - a[RL_S]);
    *x = a[RL_X] + f * (b[RL_X] - a[RL_X]);
    *y = a[RL_Y] + f * (b[RL_Y] - a[RL_Y]);
    *yaw = a[RL_YAW] + f * normalise_yaw(b[RL_YAW] - a[RL_YAW]);
}

// World pose from the track-frame position: raceline point at s plus n along
// the left normal (-sin yaw, cos yaw); long axis along the track.
static void place_obstacle_in_world(const Track *t, Obstacle *o){
    float x, y, yaw;
    rline_pose_at_s(t, o->s, &x, &y, &yaw);
    o->cx = x - o->n * sinf(yaw);
    o->cy = y + o->n * cosf(yaw);
    o->yaw = yaw;
}

// Return the distance from a ray origin to an oriented box, or infinity.
static float ray_box_intersect(
        float origin_x,
        float origin_y,
        float direction_x,
        float direction_y,
        const Obstacle* obstacle){
    // Transform ray origin into the obstacle's local frame (rotate by -yaw).
    float cyaw = cosf(obstacle->yaw), syaw = sinf(obstacle->yaw);
    float lx0 =  (origin_x - obstacle->cx)*cyaw + (origin_y - obstacle->cy)*syaw;
    float ly0 = -(origin_x - obstacle->cx)*syaw + (origin_y - obstacle->cy)*cyaw;
    float ldx =  direction_x*cyaw + direction_y*syaw;
    float ldy = -direction_x*syaw + direction_y*cyaw;

    // Slab method against [-hl, hl] x [-hw, hw].
    float hl = obstacle->half_length, hw = obstacle->half_width;
    float tmin = 0.0f, tmax = INFINITY;
    // X slab
    if (fabsf(ldx) < 1e-12f){
        if (lx0 < -hl || lx0 > hl) return INFINITY;
    } else {
        float t1 = (-hl - lx0) / ldx;
        float t2 = ( hl - lx0) / ldx;
        if (t1 > t2){ float tmp = t1; t1 = t2; t2 = tmp; }
        tmin = fmaxf(tmin, t1);
        tmax = fminf(tmax, t2);
        if (tmin > tmax) return INFINITY;
    }
    // Y slab
    if (fabsf(ldy) < 1e-12f){
        if (ly0 < -hw || ly0 > hw) return INFINITY;
    } else {
        float t1 = (-hw - ly0) / ldy;
        float t2 = ( hw - ly0) / ldy;
        if (t1 > t2){ float tmp = t1; t1 = t2; t2 = tmp; }
        tmin = fmaxf(tmin, t1);
        tmax = fminf(tmax, t2);
        if (tmin > tmax) return INFINITY;
    }
    // Origin inside the box → surface at t=0.
    if (tmin < 0.0f) return 0.0f;
    return tmin;
}

// Trace one LiDAR beam and return its normalized range.
static inline float trace_one_beam(float ox, float oy, float cpsi, float spsi,
                                   float bcos, float bsin, Roboracer *env){
    Track *track = &env->track;
    Perception *p = &env->perception;
    float range = p->max_range;
    float c = cpsi*bcos - spsi*bsin;
    float s = spsi*bcos + cpsi*bsin;
    float x = ox, y = oy;
    float d = bilinear_sample_edt(track, x, y);
    float total = d;
    int k = 0;
    while (d > EPS_LIDAR_HIT && total <= p->max_range && k < CAP_RAYTRACE_ITER){
        x += d*c;  y += d*s;
        d = bilinear_sample_edt(track, x, y);
        if (d <= -0.5f) break;  // off-map sentinel (-1.0) → genuine miss, range stays max
        if (d <= EPS_LIDAR_HIT){ range = fmaxf(0.0f, total + d); break; }  // hit, incl. slight wall penetration
        total += d;
        k++;
    }
    // Obstacles are intersected analytically; the nearer of wall and box wins.
    for (int i = 0; i < env->n_obstacles; i++){
        float t = ray_box_intersect(ox, oy, c, s, &env->obstacles[i]);
        if (t < range) range = t;
    }
    return 2 * (range/p->max_range) - 1;
}

// Trace only the policy's downsampled LiDAR beams, written at their full-scan indices.
static void raytrace_obs(Roboracer *env, float *lidar){
    State *state = &env->state;
    Chassis *chas = &env->chassis;
    Perception *p = &env->perception;
    float cpsi = cosf(state->yaw), spsi = sinf(state->yaw);
    float ox = state->x + chas->lidar_mount_offset * cpsi;
    float oy = state->y + chas->lidar_mount_offset * spsi;
    for (int b = 0; b < NUM_LIDAR_DOWNSAMPLE; b++){
        int i = obs_beam_index((int)p->n_beams, b);
        lidar[i] = trace_one_beam(ox, oy, cpsi, spsi, p->beam_cos[i], p->beam_sin[i], env);
    }
}

// Separating-axis test for two oriented rectangles. Touching counts as a
// collision, matching the inclusive static-wall check.
static int is_obstacle_overlap(
        const State* state,
        const Chassis* chassis,
        const Obstacle* obstacle){
    float ac = cosf(state->yaw), as = sinf(state->yaw);
    float bc = cosf(obstacle->yaw), bs = sinf(obstacle->yaw);
    float dx = obstacle->cx - state->x;
    float dy = obstacle->cy - state->y;
    float ego_half_length = chassis->ego_l / 2.0f;
    float ego_half_width = chassis->ego_w / 2.0f;
    const float ux[4] = { ac, -as, bc, -bs };
    const float uy[4] = { as,  ac, bs,  bc };
    for (int i = 0; i < 4; i++){
        float centre = fabsf(dx*ux[i] + dy*uy[i]);
        float ra = ego_half_length*fabsf(ac*ux[i] + as*uy[i])
                 + ego_half_width*fabsf(-as*ux[i] + ac*uy[i]);
        float rb = obstacle->half_length*fabsf(bc*ux[i] + bs*uy[i])
                 + obstacle->half_width*fabsf(-bs*ux[i] + bc*uy[i]);
        if (centre > ra + rb) return 0;
    }
    return 1;
}

// COLLISION on overlap with an obstacle or a wall, else NOT_DONE.
static int8_t check_done(Roboracer *env){
    State *state = &env->state;
    Track *track = &env->track;
    Chassis *chas = &env->chassis;
    float cyaw = cosf(state->yaw), syaw = sinf(state->yaw);
    float hl = chas->ego_l / 2.0f;   // half length (body +x = forward)
    float hw = chas->ego_w / 2.0f;   // half width  (body +y = left)

    // body-frame corners: FR, FL, RL, RR
    const float bx[4] = { +hl, +hl, -hl, -hl };
    const float by[4] = { -hw, +hw, +hw, -hw };

    for (int j = 0; j < env->n_obstacles; j++){
        if (is_obstacle_overlap(state, chas, &env->obstacles[j]))
            return COLLISION;
    }

    // Static EDT collision: inscribed-reject → 4-corner check.
    float p = bilinear_sample_edt(track, state->x, state->y);
    if (p > chas->r_circumscribed) return NOT_DONE;
    if (p <= chas->r_inscribed) return COLLISION;
    for (int i = 0; i < 4; i++) {
        float wx = state->x + bx[i]*cyaw - by[i]*syaw;
        float wy = state->y + bx[i]*syaw + by[i]*cyaw;
        float d  = bilinear_sample_edt(track, wx, wy);
        if (d <= 0.0f) return COLLISION;   // -1.0 off-map also trips this
    }
    return NOT_DONE;
}

// Lap bonus is (opt_time/lap_time)^2 times the completion weight when lap_done.
static float get_reward(Reward *reward, float ds, int8_t done, bool lap_done, float lap_time, float opt_time){
    float lap_bonus = 0.0f;
    if (lap_done) {
        float ratio = opt_time / lap_time;
        lap_bonus = reward->lap_completion_bonus * ratio * ratio;
    }
    return  reward->progress*fmaxf(0,ds)  +  (done==COLLISION ? reward->collision : 0)  +  lap_bonus;
}

int c_init(Roboracer *env, const char* blob_path){  // Load the track blob; 0 on success.
    env->current_lap = 0;
    return load_track_blob(env, blob_path);
}

void c_reset(Roboracer *env) {  // Spawn at the raceline start and write the first observation.
    const float *start = rline_row(&env->track, 0);
    env->state = (State){0};
    env->state.x   = start[RL_X];
    env->state.y   = start[RL_Y];
    env->state.yaw = start[RL_YAW];
    reset_actuator_buffers(&env->act);
    env->tick = 0;
    env->act.prev_delta_cmd = 0.0f; // denormalised cmd held during the compute-latency window
    env->act.prev_v_cmd = 0.0f;
    env->rl_ndx = 0;
    env->current_lap = 0;
    env->lap_start_tick = 0;
    env->last_lap_time = 0.0f;

    // Odom stream phase: randomise the initial offset under sim2real so episodes don't
    // all begin with odom/control perfectly aligned; deterministic (phase 0) otherwise.
    env->next_odom_tick = env->sim2real ? (int)(rng_uniform(&env->rng)*ODOM_NOM) : 0;

    // Tyre randomisation: draw one (mu, S, alpha_pk) row and rebuild the Pacejka curve.
    if (env->sim2real) {
        int row = (int)(xorshift32(&env->rng) % (unsigned)TYRE_BASIN_N);
        env->tyre.mu = TYRE_BASIN[row][0];
        set_tyre_curve(&env->tyre, TYRE_BASIN[row][1], TYRE_BASIN[row][2]);
    }

    // Draw one fixed LiDAR calibration bias for the episode.
    env->noise.lidar_bias_norm = env->sim2real
        ? (2.0f*rng_uniform(&env->rng) - 1.0f) * env->noise.lidar_bias_amp : 0.0f;

    // Prime the odom snapshot cache from true state (reset counts as the first arrival).
    env->odom_cache_s   = get_s_norm(&env->state, &env->track, &env->rl_ndx);
    env->odom_cache_v   = 0.0f;
    env->odom_cache_yaw = env->state.yaw;
    env->odom_cache_r   = 0.0f;

    // Build the observation from the cache (matches the in-step path in c_step).
    Observation observation = {0};
    observation.odom.s_norm   = env->odom_cache_s;
    observation.odom.r_norm   = env->odom_cache_r / env->limits.r_max;
    observation.odom.v_norm   = (env->odom_cache_v / env->limits.v_max)*2 - 1;
    observation.odom.yaw_norm = env->odom_cache_yaw / env->limits.yaw_max;
    observation.prev_v = -1.0;
    observation.prev_delta = 0.0f;
    raytrace_obs(env, observation.lidar);
    construct_observation(env, &observation);
    if (env->sim2real) apply_obs_noise(env);
    schedule_next_decision_timing(env);
}

void c_step(Roboracer *env) {  // Apply one action over the control period.
    // Clip policy actions before applying and storing them.
    float a_delta = fmaxf(-1.0f, fminf(1.0f, env->actions[0]));
    float a_v     = fmaxf(-1.0f, fminf(1.0f, env->actions[1]));

    float delta_cmd = a_delta * env->limits.s_max;
    float v_cmd = (a_v + 1)/2 * env->limits.v_max;

    env->terminals[0] = 0;
    env->rewards[0] = 0;

    Observation observation;
    observation.prev_delta = a_delta, observation.prev_v = a_v; // the next decision sees the action applied in this step

    int cp = env->time.control_period;
    int cl = env->time.pending_compute_latency;
    assert(cl < cp); // the hold must fit inside the control period

    // Progress and laps use the true state, not the odometry cache.
    float s_prev = rline_row(&env->track, env->rl_ndx)[RL_S_NORM];
    for (int i=0; i<cp; i+=(int)(DT_SIM*1000)){
        // Hold the previous command until the simulated computation completes.
        float u_delta = (i < cl) ? env->act.prev_delta_cmd : delta_cmd;
        float u_v     = (i < cl) ? env->act.prev_v_cmd     : v_cmd;
        float d_delta = push_delay(env->act.delta_cmd, &env->act.head_delta, env->act.n_delta, u_delta);
        float d_v= push_delay(env->act.v_cmd, &env->act.head_v, env->act.n_v, u_v);
        Inputs inputs = get_inputs(env, d_delta, d_v);
        env->act.delta_dot = inputs.sv;
        env->act.accel = inputs.accl;
        env->state = rk4_step(env->state, inputs, DT_SIM, &env->chassis,&env->limits, &env->tyre);
        env->state.yaw = normalise_yaw(env->state.yaw);

        float s_true = get_s_norm(&env->state, &env->track, &env->rl_ndx);
        float ds = s_true - s_prev;
        bool  lap_done = false;
        if (ds < -0.5f){  // crossed the start/finish seam (wrap correction always applies)
             ds += 1.0f;
             float elapsed = (env->tick - env->lap_start_tick) * DT_SIM; // seconds
             if (elapsed > 0.5f * env->optimal_lap_time){ // skip bounce-back wraps
                 lap_done = true;
                 env->current_lap++;
                 env->last_lap_time = elapsed;
                 env->lap_start_tick = env->tick;
             }
        }
        s_prev = s_true;

        // Odom arrivals: snapshot true state into the cache, then advance the (jittered)
        // deadline. while() drains multiple arrivals if a substep crosses several.
        while (env->tick >= env->next_odom_tick){
            env->odom_cache_s   = s_true;
            env->odom_cache_v   = env->state.v;
            env->odom_cache_yaw = env->state.yaw;
            env->odom_cache_r   = env->state.r;
            env->next_odom_tick += env->sim2real
                ? sample_norm_int(&env->rng, ODOM_NOM, ODOM_SD, ODOM_MIN, ODOM_MAX)
                : (int)ODOM_NOM;
        }

        int8_t done = check_done(env);
        if (done == NOT_DONE && lap_done && env->current_lap >= env->lap_cap){
             done = COMPLETE; // race finished on this lap wrap
        }
        env->rewards[0] += get_reward(&env->reward, ds, done, lap_done, env->last_lap_time, env->optimal_lap_time);
        int tcap = env->tick_cap > 0 ? env->tick_cap : MAX_TICKS;
        if ((done > 0) || (env->tick > tcap)){
            env->terminals[0] = 1;
            env->done_lap = env->current_lap;      // latch pre-reset episode outcome
            env->done_lap_time = env->last_lap_time;
            env->done_kind = done > 0 ? done : 0;  // 0 = tick-cap
            c_reset(env);
            return;
        }
        env->tick += (int)(DT_SIM*1000);
    }

    raytrace_obs(env, observation.lidar);

    // Odometry channels read the last arrival (up to one odometry period stale), as on the car.
    observation.odom.s_norm   = env->odom_cache_s;
    observation.odom.r_norm   = env->odom_cache_r / env->limits.r_max;
    observation.odom.v_norm   = (env->odom_cache_v / env->limits.v_max)*2 - 1;
    observation.odom.yaw_norm = env->odom_cache_yaw / env->limits.yaw_max;

    construct_observation(env, &observation);
    if (env->sim2real) apply_obs_noise(env);   // sensor noise: obs only, never dynamics/reward

    env->act.prev_delta_cmd = delta_cmd; // stash for the next step's compute-latency hold
    env->act.prev_v_cmd = v_cmd;
    schedule_next_decision_timing(env);
}

// Install an obstacle layout that persists across automatic episode resets.
void c_set_obstacles(Roboracer* env, const Obstacle* obstacles, int n){
    if (!env) return;
    if (n > MAX_OBSTACLES) n = MAX_OBSTACLES;
    if (n < 0) n = 0;
    for (int i = 0; i < n; i++){
        Obstacle o = obstacles[i];
        o.s = wrap_station(o.s, env->track.lap_length);
        place_obstacle_in_world(&env->track, &o);
        env->obstacles[i] = o;
    }
    env->n_obstacles = n;
    // Rebuild the already-written observation so the first action sees the obstacles.
    if (!env->observations || !env->track.rline) return;
    Observation observation = {0};
    observation.odom.s_norm   = env->odom_cache_s;
    observation.odom.r_norm   = env->odom_cache_r / env->limits.r_max;
    observation.odom.v_norm   = (env->odom_cache_v / env->limits.v_max)*2 - 1;
    observation.odom.yaw_norm = env->odom_cache_yaw / env->limits.yaw_max;
    observation.prev_delta = env->observations[24];
    observation.prev_v     = env->observations[25];
    raytrace_obs(env, observation.lidar);
    construct_observation(env, &observation);
    if (env->sim2real) apply_obs_noise(env);
}
