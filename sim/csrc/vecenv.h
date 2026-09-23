/* CPU vector of environments: flat zero-copy buffers plus reset/step across envs. */
/* Follows PufferLib's environment template (MIT, Copyright (c) 2022 PufferAI); see THIRD_PARTY_NOTICES.md. */
#ifndef ROBORACER_VECENV_H
#define ROBORACER_VECENV_H

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    const char* key;
    double value;
} DictItem;

typedef struct {
    DictItem* items;
    int size;
    int capacity;
} Dict;

typedef struct StaticVec {
    void* envs;
    int size;
    float* observations;
    float* actions;
    float* rewards;
    float* terminals;
} StaticVec;

static inline Dict* create_dict(int capacity) {
    Dict* dict = (Dict*)calloc(1, sizeof(Dict));
    assert(dict != NULL);
    dict->capacity = capacity;
    dict->items = (DictItem*)calloc((size_t)capacity, sizeof(DictItem));
    assert(dict->items != NULL);
    return dict;
}

static inline DictItem* dict_get_unsafe(Dict* dict, const char* key) {
    for (int i = 0; i < dict->size; i++) {
        if (strcmp(dict->items[i].key, key) == 0) {
            return &dict->items[i];
        }
    }
    return NULL;
}

static inline DictItem* dict_get(Dict* dict, const char* key) {
    DictItem* item = dict_get_unsafe(dict, key);
    if (item == NULL) {
        fprintf(stderr, "dict_get failed to find key: %s\n", key);
        abort();
    }
    return item;
}

static inline void dict_set(Dict* dict, const char* key, double value) {
    DictItem* item = dict_get_unsafe(dict, key);
    if (item != NULL) {
        item->value = value;
        return;
    }
    assert(dict->size < dict->capacity);
    item = &dict->items[dict->size++];
    item->key = key;
    item->value = value;
}

StaticVec* create_static_vec(int num_envs, const char* blob_path, Dict* env_kwargs);
void static_vec_reset(StaticVec* vec);
void static_vec_step(StaticVec* vec);
void static_vec_close(StaticVec* vec);
int get_obs_size(void);
int get_num_atns(void);

// binding.c defines Env, OBS_SIZE and NUM_ATNS before including this file; shim.c sees declarations only.
#ifdef OBS_SIZE

int my_init(Env* env, Dict* kwargs, const char* blob_path);

// NULL when an env fails to load the track blob.
StaticVec* create_static_vec(int num_envs, const char* blob_path, Dict* env_kwargs) {
    assert(num_envs > 0);
    StaticVec* vec = (StaticVec*)calloc(1, sizeof(StaticVec));
    assert(vec != NULL);
    vec->size = num_envs;
    vec->envs = calloc((size_t)num_envs, sizeof(Env));
    vec->observations = (float*)calloc((size_t)num_envs * OBS_SIZE, sizeof(float));
    vec->actions = (float*)calloc((size_t)num_envs * NUM_ATNS, sizeof(float));
    vec->rewards = (float*)calloc((size_t)num_envs, sizeof(float));
    vec->terminals = (float*)calloc((size_t)num_envs, sizeof(float));
    assert(vec->envs != NULL);
    assert(vec->observations != NULL);
    assert(vec->actions != NULL);
    assert(vec->rewards != NULL);
    assert(vec->terminals != NULL);

    Env* envs = (Env*)vec->envs;
    for (int i = 0; i < num_envs; i++) {
        envs[i].rng = (unsigned int)i;
        if (my_init(&envs[i], env_kwargs, blob_path) != 0) {
            static_vec_close(vec);
            return NULL;
        }
        envs[i].observations = vec->observations + (size_t)i * OBS_SIZE;
        envs[i].actions = vec->actions + (size_t)i * NUM_ATNS;
        envs[i].rewards = vec->rewards + i;
        envs[i].terminals = vec->terminals + i;
    }
    return vec;
}

void static_vec_reset(StaticVec* vec) {
    Env* envs = (Env*)vec->envs;
    memset(vec->rewards, 0, (size_t)vec->size * sizeof(float));
    memset(vec->terminals, 0, (size_t)vec->size * sizeof(float));
    for (int i = 0; i < vec->size; i++) {
        c_reset(&envs[i]);
    }
}

void static_vec_step(StaticVec* vec) {
    Env* envs = (Env*)vec->envs;
    memset(vec->rewards, 0, (size_t)vec->size * sizeof(float));
    memset(vec->terminals, 0, (size_t)vec->size * sizeof(float));
#pragma omp parallel for schedule(static)
    for (int i = 0; i < vec->size; i++) {
        c_step(&envs[i]);
    }
}

void static_vec_close(StaticVec* vec) {
    if (vec == NULL) {
        return;
    }
    Env* envs = (Env*)vec->envs;
    for (int i = 0; i < vec->size; i++) {
        c_close(&envs[i]);
    }
    free(vec->envs);
    free(vec->observations);
    free(vec->actions);
    free(vec->rewards);
    free(vec->terminals);
    free(vec);
}

int get_obs_size(void) {
    return OBS_SIZE;
}

int get_num_atns(void) {
    return NUM_ATNS;
}

#endif
#endif
