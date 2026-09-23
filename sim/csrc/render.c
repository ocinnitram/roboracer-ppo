/* Raylib window for one env; built separately as build/librender.so (make render). */

#include "roboracer.h"
#include "raylib.h"

#define WIN_W 1440
#define WIN_H 900
#define TOP_H 52       // top strip: layout, lap, last lap, outcome
#define HUD_H 56       // bottom strip: steer and speed bars
#define MARGIN 24      // pixel border around the map
#define WHEEL_LEN 0.13f
#define WHEEL_WID 0.052f
#define WHEEL_R 0.048f
#define HUD_AVG_TICKS 5

// Outcome shown in the top-right pill.
#define OUTCOME_RUNNING -1
#define OUTCOME_TIMEOUT 0
#define OUTCOME_CRASHED 1
#define OUTCOME_COMPLETED 2

typedef struct {
    Texture2D track_tex;
    bool tex_ready;
    float scale, off_x, off_y, map_w_m, map_h_m;
    float zoom, pan_x, pan_y;
    bool dragging;
    Vector2 drag_last;
    float hud_steer_buf[HUD_AVG_TICKS], hud_speed_buf[HUD_AVG_TICKS];
    int hud_avg_n, hud_avg_i, hud_avg_tick;
    float wheel_spin;
    Roboracer last;  // last running frame, held on screen after the episode ends
    bool has_last;
} Renderer;

static Renderer R = { .zoom = 1.0f, .hud_avg_tick = -1 };

// World metres (y up) to screen pixels (y down).
static inline Vector2 world_to_screen(const Track* t, float wx, float wy){
    return (Vector2){
        (wx - t->origin_x) * R.scale + R.off_x,
        (R.map_h_m - (wy - t->origin_y)) * R.scale + R.off_y
    };
}

// Fit the whole map into the content area, then apply zoom and pan.
static void layout_fit(const Track* t){
    R.map_w_m = t->W * t->res;
    R.map_h_m = t->H * t->res;
    float avail_w = (float)(WIN_W - 2 * MARGIN);
    float avail_h = (float)(WIN_H - TOP_H - HUD_H - 2 * MARGIN);
    float fit = fminf(avail_w / R.map_w_m, avail_h / R.map_h_m);
    R.scale = fit * R.zoom;
    float map_px_w = R.map_w_m * R.scale;
    float map_px_h = R.map_h_m * R.scale;
    R.off_x = MARGIN + avail_w * 0.5f + R.pan_x - map_px_w * 0.5f;
    R.off_y = TOP_H + MARGIN + avail_h * 0.5f + R.pan_y - map_px_h * 0.5f;
    // Centre the map when it fits; clamp the pan once zoomed in.
    if (map_px_w <= avail_w) R.off_x = MARGIN + (avail_w - map_px_w) * 0.5f;
    else R.off_x = fminf(fmaxf(R.off_x, MARGIN + avail_w - map_px_w), (float)MARGIN);
    if (map_px_h <= avail_h) R.off_y = TOP_H + MARGIN + (avail_h - map_px_h) * 0.5f;
    else R.off_y = fminf(fmaxf(R.off_y, TOP_H + MARGIN + avail_h - map_px_h), (float)(TOP_H + MARGIN));
}

static void oriented_box(const Track* t, float cx, float cy, float ang, float length, float width,
                         Color fill, Color outline, float outline_w){
    float ca = cosf(ang), sa = sinf(ang);
    float hl = 0.5f * length, hw = 0.5f * width;
    const float bx[4] = { +hl, +hl, -hl, -hl };
    const float by[4] = { -hw, +hw, +hw, -hw };
    Vector2 p[4];
    for (int i = 0; i < 4; i++)
        p[i] = world_to_screen(t, cx + bx[i] * ca - by[i] * sa, cy + bx[i] * sa + by[i] * ca);
    DrawTriangleFan(p, 4, fill);
    if (outline.a && outline_w > 0.0f)
        for (int i = 0; i < 4; i++) DrawLineEx(p[i], p[(i + 1) % 4], outline_w, outline);
}

static void draw_wheel(const Track* t, float cx, float cy, float ang, float spin){
    Color rubber = (Color){22, 22, 26, 255};
    Color rim = (Color){168, 170, 176, 255};
    Color spoke = (Color){214, 216, 220, 230};
    oriented_box(t, cx, cy, ang, WHEEL_LEN, WHEEL_WID, rubber, rim, 1.4f);
    float ca = cosf(ang), sa = sinf(ang);
    float half = 0.42f * WHEEL_LEN * cosf(spin);
    DrawLineEx(world_to_screen(t, cx - half * ca, cy - half * sa),
               world_to_screen(t, cx + half * ca, cy + half * sa), 2.0f, spoke);
    DrawCircleV(world_to_screen(t, cx, cy), 2.0f, rim);
}

// Bake walls and the drivable corridor (flood-filled from the raceline) into a texture.
static void build_track_texture(const Track* t){
    const Color outside = (Color){46, 58, 89, 255};
    const Color asphalt = (Color){58, 64, 74, 255};
    const Color wall = (Color){236, 240, 246, 255};
    const int n = t->W * t->H;
    Color* px = (Color*)malloc((size_t)n * sizeof(Color));
    unsigned char* seen = (unsigned char*)calloc((size_t)n, 1);
    int* q = (int*)malloc((size_t)n * sizeof(int));
    for (int i = 0; i < n; i++) px[i] = outside;

    int qt = 0;
    for (int i = 0; i < t->n_rline; i++){
        int ix = (int)floorf((rline_row(t, i)[RL_X] - t->origin_x) / t->res);
        int iy = (int)floorf((rline_row(t, i)[RL_Y] - t->origin_y) / t->res);
        if (ix < 0 || iy < 0 || ix >= t->W || iy >= t->H) continue;
        int id = iy * t->W + ix;
        if (seen[id] || t->edt[id] <= 0.0f) continue;
        seen[id] = 1;
        q[qt++] = id;
    }
    const int dx[4] = {1, -1, 0, 0};
    const int dy[4] = {0, 0, 1, -1};
    for (int qh = 0; qh < qt; qh++){
        int ix = q[qh] % t->W, iy = q[qh] / t->W;
        for (int k = 0; k < 4; k++){
            int xx = ix + dx[k], yy = iy + dy[k];
            if (xx < 0 || yy < 0 || xx >= t->W || yy >= t->H) continue;
            int nid = yy * t->W + xx;
            if (seen[nid] || t->edt[nid] <= 0.0f) continue;
            seen[nid] = 1;
            q[qt++] = nid;
        }
    }
    for (int iy = 0; iy < t->H; iy++){
        for (int ix = 0; ix < t->W; ix++){
            int id = iy * t->W + ix;
            int tex = (t->H - 1 - iy) * t->W + ix;
            if (t->edt[id] <= 0.0f) px[tex] = wall;
            else if (seen[id]) px[tex] = asphalt;
        }
    }
    free(seen);
    free(q);

    Image img = { .data = px, .width = t->W, .height = t->H, .mipmaps = 1,
                  .format = PIXELFORMAT_UNCOMPRESSED_R8G8B8A8 };
    R.track_tex = LoadTextureFromImage(img);
    SetTextureFilter(R.track_tex, TEXTURE_FILTER_POINT);
    free(px);
    R.tex_ready = true;
}

// Mouse wheel zooms about the cursor; left drag pans.
static void handle_mouse(const Track* t){
    float wheel = GetMouseWheelMove();
    if (wheel != 0.0f){
        Vector2 m = GetMousePosition();
        float wx = (m.x - R.off_x) / R.scale + t->origin_x;
        float wy = t->origin_y + R.map_h_m - (m.y - R.off_y) / R.scale;
        R.zoom = fminf(fmaxf(R.zoom * powf(1.2f, wheel), 0.5f), 60.0f);
        layout_fit(t);
        R.pan_x += m.x - ((wx - t->origin_x) * R.scale + R.off_x);
        R.pan_y += m.y - ((R.map_h_m - (wy - t->origin_y)) * R.scale + R.off_y);
        layout_fit(t);
    }
    if (IsMouseButtonPressed(MOUSE_BUTTON_LEFT)){
        R.dragging = true;
        R.drag_last = GetMousePosition();
    }
    if (IsMouseButtonReleased(MOUSE_BUTTON_LEFT)) R.dragging = false;
    if (R.dragging && IsMouseButtonDown(MOUSE_BUTTON_LEFT)){
        Vector2 m = GetMousePosition();
        R.pan_x += m.x - R.drag_last.x;
        R.pan_y += m.y - R.drag_last.y;
        R.drag_last = m;
        layout_fit(t);
    }
}

// Rear wheels, body, then front wheels steered by the plant delta; spokes roll with speed.
static void draw_car(const Roboracer* env){
    const Track* t = &env->track;
    float cyaw = cosf(env->state.yaw), syaw = sinf(env->state.yaw);
    float x = env->state.x, y = env->state.y;
    float lf = env->chassis.lf, lr = env->chassis.lr;
    float hl = env->chassis.ego_l * 0.5f, hw = env->chassis.ego_w * 0.5f;
    float track = hw + 0.012f;
    float axle_x[4] = { +lf, +lf, -lr, -lr };  // FR, FL, RR, RL
    float axle_y[4] = { -track, +track, -track, +track };
    float wang[4] = { env->state.yaw + env->state.delta, env->state.yaw + env->state.delta,
                      env->state.yaw, env->state.yaw };
    float wx[4], wy[4];
    for (int i = 0; i < 4; i++){
        wx[i] = x + axle_x[i] * cyaw - axle_y[i] * syaw;
        wy[i] = y + axle_x[i] * syaw + axle_y[i] * cyaw;
    }
    // Shadow: the body shifted 3 px down.
    Vector2 c[4];
    const float bx[4] = { +hl, +hl, -hl, -hl };
    const float by[4] = { -hw, +hw, +hw, -hw };
    for (int i = 0; i < 4; i++){
        c[i] = world_to_screen(t, x + bx[i] * cyaw - by[i] * syaw, y + bx[i] * syaw + by[i] * cyaw);
        c[i].y += 3.0f;
    }
    DrawTriangleFan(c, 4, (Color){0, 0, 0, 70});
    draw_wheel(t, wx[2], wy[2], wang[2], R.wheel_spin);
    draw_wheel(t, wx[3], wy[3], wang[3], R.wheel_spin);
    oriented_box(t, x, y, env->state.yaw, env->chassis.ego_l, env->chassis.ego_w,
                 (Color){227, 66, 52, 255}, (Color){90, 26, 20, 255}, 1.5f);
    draw_wheel(t, wx[0], wy[0], wang[0], R.wheel_spin);
    draw_wheel(t, wx[1], wy[1], wang[1], R.wheel_spin);
}

static void draw_top_strip(const Roboracer* env, const char* label, int outcome){
    Color mute = (Color){148, 162, 184, 255};
    Color ice = (Color){236, 242, 250, 255};
    DrawRectangle(0, 0, WIN_W, TOP_H, (Color){22, 28, 42, 255});
    DrawLine(0, TOP_H, WIN_W, TOP_H, (Color){58, 70, 96, 255});

    DrawText("RUN", 20, 8, 12, mute);
    DrawText(label, 20, 22, 26, ice);
    int x = 20 + MeasureText(label, 26) + 40;
    DrawText("LAP", x, 8, 12, mute);
    DrawText(TextFormat("%d / %d", env->current_lap, env->lap_cap), x, 22, 26, ice);
    DrawText("LAST", x + 140, 8, 12, mute);
    if (env->last_lap_time > 0.5f) DrawText(TextFormat("%.2f s", env->last_lap_time), x + 140, 22, 26, ice);
    else DrawText("--", x + 140, 22, 26, mute);

    const char* pill = "DRIVING";
    Color bg = (Color){40, 70, 120, 255}, fg = (Color){200, 220, 255, 255};
    if (outcome == OUTCOME_COMPLETED){
        pill = "COMPLETED"; bg = (Color){36, 110, 78, 255}; fg = (Color){196, 240, 214, 255};
    } else if (outcome == OUTCOME_CRASHED){
        pill = "CRASHED"; bg = (Color){150, 42, 48, 255}; fg = (Color){255, 214, 214, 255};
    } else if (outcome == OUTCOME_TIMEOUT){
        pill = "TIMEOUT"; bg = (Color){168, 108, 28, 255}; fg = (Color){255, 226, 160, 255};
    }
    int pw = 150, ph = 32;
    int px = WIN_W - pw - 18, py = (TOP_H - ph) / 2;
    DrawRectangleRounded((Rectangle){(float)px, (float)py, (float)pw, (float)ph}, 0.5f, 6, bg);
    DrawText(pill, px + (pw - MeasureText(pill, 20)) / 2, py + 6, 20, fg);
}

// Steer (centred) and speed bars, averaged over the last HUD_AVG_TICKS control steps.
static void draw_bottom_strip(const Roboracer* env){
    int top = WIN_H - HUD_H;
    Color mute = (Color){148, 162, 184, 255};
    Color ice = (Color){220, 228, 240, 255};
    Color groove = (Color){36, 44, 62, 255};
    DrawRectangle(0, top, WIN_W, HUD_H, (Color){22, 28, 42, 255});
    DrawLine(0, top, WIN_W, top, (Color){58, 70, 96, 255});
    int bx = 90, bw = WIN_W - 110, bh = 10;

    float smax = env->limits.s_max > 0.05f ? env->limits.s_max : 0.42f;
    float vmax = env->limits.v_max > 0.1f ? env->limits.v_max : 8.0f;
    if (env->tick != R.hud_avg_tick){
        R.hud_steer_buf[R.hud_avg_i] = fminf(fmaxf(env->state.delta / smax, -1.0f), 1.0f);
        R.hud_speed_buf[R.hud_avg_i] = fminf(fmaxf(env->state.v / vmax, 0.0f), 1.0f);
        R.hud_avg_i = (R.hud_avg_i + 1) % HUD_AVG_TICKS;
        if (R.hud_avg_n < HUD_AVG_TICKS) R.hud_avg_n++;
        R.hud_avg_tick = env->tick;
    }
    float steer = 0.0f, speed = 0.0f;
    for (int i = 0; i < R.hud_avg_n; i++){
        steer += R.hud_steer_buf[i];
        speed += R.hud_speed_buf[i];
    }
    int n = R.hud_avg_n > 0 ? R.hud_avg_n : 1;
    steer /= (float)n;
    speed /= (float)n;

    DrawText("STEER", 20, top + 8, 12, mute);
    int sty = top + 10;
    DrawRectangleRounded((Rectangle){(float)bx, (float)sty, (float)bw, (float)bh}, 0.5f, 4, groove);
    int mid = bx + bw / 2;
    int nx = mid + (int)(steer * (bw * 0.5f));
    int fill_x = steer >= 0.0f ? mid : nx;
    int fill_w = abs(nx - mid) < 2 ? 2 : abs(nx - mid);
    DrawRectangle(fill_x, sty, fill_w, bh, ice);
    DrawRectangle(mid - 1, sty - 2, 2, bh + 4, (Color){90, 104, 128, 255});

    DrawText("SPEED", 20, top + 32, 12, mute);
    DrawText(TextFormat("%.1f m/s", env->state.v), 20, top + 44, 10, ice);
    int spy = top + 34;
    DrawRectangleRounded((Rectangle){(float)bx, (float)spy, (float)bw, (float)bh}, 0.5f, 4, groove);
    if (speed > 0.01f)
        DrawRectangleRounded((Rectangle){(float)bx, (float)spy, bw * speed, (float)bh}, 0.5f, 4, ice);
}

// Draw env (or, once outcome >= 0, its last running frame). Returns 0 once closed, -1 if it cannot open.
int render_frame(const Roboracer* live, const char* label, int outcome){
    if (outcome == OUTCOME_RUNNING){
        if (R.has_last && live->tick < R.last.tick) {  // new episode
            R.wheel_spin = 0.0f;
            R.hud_avg_n = R.hud_avg_i = 0;
            R.hud_avg_tick = -1;
        }
        R.last = *live;
        R.has_last = true;
        R.wheel_spin = fmodf(R.wheel_spin + live->state.v * (live->time.control_period * DT_SIM) / WHEEL_R,
                             6.2831853f);
    }
    const Roboracer* env = R.has_last ? &R.last : live;
    const Track* t = &env->track;

    if (!IsWindowReady()){
        SetTraceLogLevel(LOG_WARNING);
        SetConfigFlags(FLAG_MSAA_4X_HINT);
        InitWindow(WIN_W, WIN_H, "roboracer-ppo");
        if (!IsWindowReady()) return -1;
        SetTargetFPS(60);
    }
    if (!R.tex_ready) build_track_texture(t);
    layout_fit(t);
    handle_mouse(t);
    if (WindowShouldClose()) return 0;

    BeginDrawing();
    ClearBackground((Color){46, 58, 89, 255});
    Rectangle src = {0, 0, (float)t->W, (float)t->H};
    Rectangle dst = {R.off_x, R.off_y, R.map_w_m * R.scale, R.map_h_m * R.scale};
    DrawTexturePro(R.track_tex, src, dst, (Vector2){0, 0}, 0.0f, WHITE);
    for (int i = 0; i < env->n_obstacles; i++){
        const Obstacle* o = &env->obstacles[i];
        oriented_box(t, o->cx, o->cy, o->yaw, 2.0f * o->half_length, 2.0f * o->half_width,
                     (Color){196, 140, 72, 255}, (Color){120, 80, 36, 255}, 2.0f);
    }
    draw_car(env);
    draw_top_strip(env, label, outcome);
    draw_bottom_strip(env);
    EndDrawing();
    return 1;
}

void render_close(void){
    if (R.tex_ready){
        UnloadTexture(R.track_tex);
        R.tex_ready = false;
    }
    if (IsWindowReady()) CloseWindow();
    R.has_last = false;
}
