#pragma once

// Loading the llama 3.2 1B-Instruct snapshot off disk: the safetensors blob onto the GPU.
// Nothing here runs per step.

#include "config.h"

// Device pointers into the one safetensors blob, so every weight is reachable by name.
// TODO: right now I know the model structure since it's always llama 3.2 1B-Instruct, but
//       maybe it would be convenient to store dimensions somewhere for even easier access?
struct Weights
{
    __nv_bfloat16 *embed_tokens;
    __nv_bfloat16 *input_layernorm[N_LAYERS];
    __nv_bfloat16 *mlp_gate_proj[N_LAYERS];
    __nv_bfloat16 *mlp_up_proj[N_LAYERS];
    __nv_bfloat16 *mlp_down_proj[N_LAYERS];
    __nv_bfloat16 *post_attn_layernorms[N_LAYERS];
    __nv_bfloat16 *w_k[N_LAYERS];
    __nv_bfloat16 *w_o[N_LAYERS];
    __nv_bfloat16 *w_q[N_LAYERS];
    __nv_bfloat16 *w_v[N_LAYERS];
    __nv_bfloat16 *norm;
};

// Uploads model.safetensors and fills `weights` with pointers into it. Non-zero on failure.
int loadWeights(Weights &weights);
