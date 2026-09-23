/* Vector-level rr_* ABI; the per-env ABI and the vecenv implementation live in binding.c. */
#include <string.h>
#include "vecenv.h"

// Dict stores key pointers, so keys are copied. NULL when the track blob fails to load.
void* rr_create(int num_envs, const char* blob_path, int n_kwargs, const char** keys, const double* values) {
    Dict* env_kwargs = create_dict(n_kwargs);
    for (int i = 0; i < n_kwargs; i++) {
        dict_set(env_kwargs, strdup(keys[i]), values[i]);
    }
    return create_static_vec(num_envs, blob_path, env_kwargs);
}

void rr_reset(void* vec) { static_vec_reset((StaticVec*)vec); }
void rr_step(void* vec)  { static_vec_step((StaticVec*)vec); }
void rr_close(void* vec) { static_vec_close((StaticVec*)vec); }

float* rr_observations(void* vec) { return ((StaticVec*)vec)->observations; }
float* rr_actions(void* vec)      { return ((StaticVec*)vec)->actions; }
float* rr_rewards(void* vec)      { return ((StaticVec*)vec)->rewards; }
float* rr_terminals(void* vec)    { return ((StaticVec*)vec)->terminals; }

int rr_obs_size(void) { return get_obs_size(); }
int rr_num_atns(void) { return get_num_atns(); }
