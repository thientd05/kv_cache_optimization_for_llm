// tiny-vllm: continuous-batching inference server for llama 3.2 1B-Instruct with a paged
// KV cache. Prompts arrive on stdin, per-token progress and statistics go out on stdout as
// JSON lines; generated text is never sent, only counted.
//
// This file is startup plus the step loop; everything else lives next to it:
//   config.h        every tunable and model dimension
//   kernels.cuh     the CUDA kernels
//   model.cuh       weights loading
//   request_queue.h the stdin reader thread and its queue
//   engine.cuh      scheduler state, packed prefill and batched decode
//   utils.cuh       paths, allocation, device info

#include <string>
#include <cassert>
#include <chrono>
#include <cstdlib>
#include <deque>
#include <iostream>
#include <thread>
#include <vector>

#include "config.h"
#include "engine.cuh"
#include "kernels.cuh"
#include "model.cuh"
#include "request_queue.h"
#include "utils.cuh"

int main(int argc, char *argv[])
{
    cublasHandle_t cublas_handle;
    cublasStatus_t status = cublasCreate(&cublas_handle);
    if (status != CUBLAS_STATUS_SUCCESS)
    {
        std::cerr << "cuBLAS init failed, status: " << status << "\n";
        return 1;
    }

    if (argc > 1)
    {
        g_model_dir = argv[1];
    }
    else if (const char *env_model_dir = std::getenv("MODEL_DIR"))
    {
        g_model_dir = env_model_dir;
    }
    std::cerr << "Model directory: " << g_model_dir << "\n";

    if (const char *ignore_eos = std::getenv("IGNORE_EOS"))
    {
        g_ignore_eos = (ignore_eos[0] == '1');
    }

    // TEMPERATURE unset or 0 keeps greedy decoding, which every experiment before §6.3 needs:
    // the equivalence gate between the two builds is a checksum over the generated ids, and a
    // sampler would make that gate meaningless. A value > 0 turns on random sampling, and then
    // SAMPLING_SEED is what makes a run reproducible - the draw is a pure function of
    // (seed, request id, position), never of a running generator state.
    if (const char *temp = std::getenv("TEMPERATURE"))
    {
        g_temperature = (float)atof(temp);
        if (g_temperature < 0.0f)
        {
            g_temperature = 0.0f;
        }
    }
    // §7.3: which recovery path a preemption takes. Default recompute, so a run launched
    // without this variable behaves exactly as every run taken before the swap path existed.
    if (const char *policy = std::getenv("PREEMPTION_POLICY"))
    {
        if (std::string(policy) == "swap")
        {
            g_preemption_policy = PreemptionPolicy::Swap;
        }
        else if (std::string(policy) != "recompute")
        {
            std::cerr << "PREEMPTION_POLICY must be \"recompute\" or \"swap\", got \""
                      << policy << "\"\n";
            return 1;
        }
    }
    // §6.4: how many tokens every prompt in this workload is known to share. The engine reserves
    // the whole-page part of it once and maps it into every prompt that really carries it;
    // 0 (the default) switches the whole path off.
    if (const char *prefix = std::getenv("SHARED_PREFIX_LEN"))
    {
        g_shared_prefix_len = atoi(prefix);
        if (g_shared_prefix_len < 0 || g_shared_prefix_len > MAX_PROMPT_LEN)
        {
            std::cerr << "SHARED_PREFIX_LEN must be between 0 and " << MAX_PROMPT_LEN << "\n";
            return 1;
        }
    }
    if (const char *seed = std::getenv("SAMPLING_SEED"))
    {
        g_sampling_seed = (unsigned int)strtoul(seed, nullptr, 10);
    }
    if (g_temperature > 0.0f)
    {
        std::cerr << "Sampling: temperature " << g_temperature
                  << ", seed " << g_sampling_seed << " (greedy decoding is OFF)\n";
    }

    Weights weights{};
    if (loadWeights(weights) != 0)
    {
        return 1;
    }

    // RoPE cos/sin tables, precomputed once for every position up to MAX_SEQ_LEN
    initRopeFrequencies();

    // Order matters: the KV cache is sized from whatever VRAM is left, so every other
    // allocation has to be in place before it asks. See allocateKVCache().
    DeviceBuffers buffers = allocateDeviceBuffers();
    KVCacheState kv = allocateKVCache();

    // Prompts arrive on stdin as "<request_id> <max_tokens> <token> <token> ..."; the reader thread owns
    // the parsing and the scheduler only ever drains this queue.
    std::deque<Request> queue;
    std::thread in_thread(input_thread_func, std::ref(queue));
    in_thread.detach();

    SlotState slots;
    long long step = 0;

    reportEngineConfig();

    while (true) // exit condition irrelevant for now, since it's an inference server that's supposed to run foreveeer!!!
    {
        // ---- admit queued prompts into every free slot and prefill them all together ----
        // The old loop called prefill() once per slot, so a batch of 16 prompts meant 16
        // full passes over the model back to back and the 16th prompt waited for all of them.
        // Now they share a single packed pass, which is where the latency win comes from.
        // Sequences parked in host RAM come back first: they arrived before anything still
        // queued, and under FCFS that is their turn. No-op under the recompute policy.
        swapInSequences(slots, kv, buffers);

        std::vector<PrefillBatchItem> prefill_items = admitQueuedRequests(queue, slots, kv);
        if (!prefill_items.empty())
        {
            prefillBatch(prefill_items, buffers, weights, cublas_handle, slots, kv, queue);

            // decode timings measure decode only, so restart the clock for whoever survived
            // prefill and is about to start generating
            auto decode_begin = std::chrono::high_resolution_clock::now();
            for (const PrefillBatchItem &item : prefill_items)
            {
                if (!slots.is_slot_free[item.slot])
                {
                    slots.decode_start[item.slot] = decode_begin;
                }
            }
        }

        // ---- make sure every running sequence can still be served this step ----
        enforcePageBudget(slots, kv, queue);

        // Page accounting invariant (§4.4). Every physical page is either on the free list or
        // held by at least one block-table entry; nothing may be on both and nothing may be
        // lost. Sharing is what makes this worth asserting - a refcount dropped one time too
        // many returns a page that a sibling is still reading, and the result is another
        // sequence's tokens appearing in this one's output: no crash, no wrong token count,
        // just quietly wrong text. Cheap enough to leave on (one pass over the free list).
        assert(kvPageAccountingHolds(kv) && "a page was double-freed or leaked");

        if (decodeStep(buffers, weights, cublas_handle, slots, kv) == 0)
        {
            // nothing running, so wait for the reader thread instead of spinning on the queue
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
        else if (++step % STEP_TELEMETRY_EVERY == 0)
        {
            // one sample of "how many requests are batched, how full is the cache" per
            // handful of steps - the two series the serving benchmark plots over time
            emitStepTelemetry(slots, kv, queue);
        }
    }
}
