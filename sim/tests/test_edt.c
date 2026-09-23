/* Track blob loader and EDT sampler against the numpy reference in fixtures_edt.h:
 *
 *   H1  load_track_blob()      : header (W,H,res,ox,oy,n_rline) round-trips
 *   H2  bilinear_sample_edt()  : matches numpy at every fixture point
 *   H3  on-track semantics     : every raceline point reads clearance > 0
 *                                (the y-flip / world->grid transform is right)
 *   H4  off-map sentinel       : out-of-bounds points return -1.0
 */
#include <stdio.h>
#include <math.h>

#include "roboracer.h"
#include "fixtures_edt.h"

#define ATOL 1e-3   /* abs floor: f32-vs-f64 noise, not a bug */
#define RTOL 1e-4

static int g_fail = 0;
static int g_checks = 0;

static void check(const char *test, const char *what, double got, double ref) {
    g_checks++;
    double err = fabs(got - ref);
    double tol = ATOL + RTOL * fabs(ref);
    if (err > tol) {
        g_fail++;
        printf("  \x1b[31mFAIL\x1b[0m %-4s %-24s got=% .6g  ref=% .6g  |err|=%.3g (tol %.3g)\n",
               test, what, got, ref, err, tol);
    }
}

int main(void) {
    printf("roboracer track / EDT faithfulness tests\n");
    printf("========================================\n");

    Roboracer env = {0};
    if (load_track_blob(&env, EDT_BLOB_PATH) != 0) {
        printf("\x1b[31mFATAL\x1b[0m load_track_blob failed for %s\n", EDT_BLOB_PATH);
        return 1;
    }

    /* H1: header round-trip */
    printf("\nH1  load_track_blob() header\n");
    int h0 = g_fail;
    check("H1", "W", env.track.W, EDT_W);
    check("H1", "H", env.track.H, EDT_H);
    check("H1", "res", env.track.res, EDT_RES);
    check("H1", "origin_x", env.track.origin_x, EDT_OX);
    check("H1", "origin_y", env.track.origin_y, EDT_OY);
    check("H1", "n_rline", env.track.n_rline, EDT_N_RLINE);
    printf("  %s\n", g_fail == h0 ? "\x1b[32mok\x1b[0m" : "\x1b[31mhad failures\x1b[0m");

    /* H2: bilinear matches numpy at every point */
    printf("\nH2  bilinear_sample_edt() vs numpy (%d points)\n", N_EDT_CASES);
    int h2 = g_fail;
    for (int i = 0; i < N_EDT_CASES; i++) {
        const EdtCase *c = &EDT_CASES[i];
        float got = bilinear_sample_edt(&env.track, c->wx, c->wy);
        check("H2", "edt", got, c->expected);
    }
    printf("  %s\n", g_fail == h2 ? "\x1b[32mok\x1b[0m" : "\x1b[31mhad failures\x1b[0m");

    /* H3: every on-track (raceline) point has positive clearance */
    printf("\nH3  on-track clearance > 0 (transform/flip sanity)\n");
    int h3 = g_fail, n_on = 0;
    float min_clear = INFINITY;
    for (int i = 0; i < N_EDT_CASES; i++) {
        const EdtCase *c = &EDT_CASES[i];
        if (!c->on_track) continue;
        n_on++;
        float d = bilinear_sample_edt(&env.track, c->wx, c->wy);
        if (d < min_clear) min_clear = d;
        if (d <= 0.0f) {
            g_fail++; g_checks++;
            printf("  \x1b[31mFAIL\x1b[0m raceline pt (% .3f,% .3f) reads %.3f m <= 0 "
                   "(in a wall!)\n", c->wx, c->wy, d);
        }
    }
    printf("  %d on-track pts, min clearance = %.3f m  %s\n", n_on, min_clear,
           g_fail == h3 ? "\x1b[32mok\x1b[0m" : "\x1b[31mhad failures\x1b[0m");

    /* H4: off-map returns the -1.0 sentinel */
    printf("\nH4  off-map sentinel == -1.0\n");
    int h4 = g_fail;
    for (int i = 0; i < N_EDT_CASES; i++) {
        const EdtCase *c = &EDT_CASES[i];
        if (c->expected != -1.0f) continue;     /* reference flagged it off-map */
        float got = bilinear_sample_edt(&env.track, c->wx, c->wy);
        check("H4", "sentinel", got, -1.0);
    }
    printf("  %s\n", g_fail == h4 ? "\x1b[32mok\x1b[0m" : "\x1b[31mhad failures\x1b[0m");

    unload_track_blob(&env);

    printf("\n----------------------------------------\n");
    if (g_fail == 0)
        printf("\x1b[32mALL PASSED\x1b[0m (%d checks)\n", g_checks);
    else
        printf("\x1b[31m%d / %d checks FAILED\x1b[0m\n", g_fail, g_checks);
    return g_fail ? 1 : 0;
}
