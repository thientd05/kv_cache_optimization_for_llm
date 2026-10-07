# Machine Configuration

Hardware/software profile of the development machine for this repo. Measured on
2026-10-07 with `nvidia-smi`, `cudaGetDeviceProperties`, `lscpu`. Kernel launch
geometry, VRAM budgets and comments in `paged_attention/src/` are tuned against
these numbers — re-verify this file before assuming anything about the GPU.

## GPU — NVIDIA GeForce GTX 1650 (laptop, GDDR6)

| Property | Value |
| --- | --- |
| Architecture | Turing, TU117 |
| Compute capability | **7.5** (`sm_75`) |
| Tensor cores | **none** (GTX 16-series has no tensor cores, no RT cores) |
| SM count | **14** |
| FP32 lanes | 896 (14 SM × 64) |
| SM clock | 1515 MHz base / 1785 MHz max boost |
| Peak FP32 | ≈ 2.7 TFLOP/s base, ≈ 3.2 TFLOP/s boost |
| VRAM total | 4096 MiB advertised / **3715 MiB visible** / 3654 MiB free at idle |
| `totalGlobalMem` | 3 895 132 160 B = **3.63 GiB** |
| Memory type / bus | GDDR6, **128-bit** |
| Memory clock | 6001 MHz effective (12 Gbps) |
| Theoretical bandwidth | **192 GB/s** |
| L2 cache | 1024 KiB |
| Power cap | 30 W (laptop TGP) |
| ECC | disabled (not supported) |

### Launch limits (these drive the kernel guards)

| Limit | Value |
| --- | --- |
| `maxThreadsPerBlock` | **1024** |
| `maxThreadsPerMultiProcessor` | 1024 |
| `maxBlocksPerMultiProcessor` | 16 |
| `maxThreadsDim` | 1024 × 1024 × 64 |
| `maxGridSize` | 2147483647 × 65535 × 65535 |
| `warpSize` | **32** (→ `WARP_FULL_MASK = 0xffffffff`) |
| Shared mem / block (default) | 48 KiB |
| Shared mem / block (opt-in) | 64 KiB |
| Shared mem / SM | 64 KiB |
| Registers / block & / SM | 65536 |
| Constant memory | 64 KiB |
| Concurrent kernels | yes |
| Async copy engines | 3 |
| Cooperative launch | yes |
| Managed memory / UVA | yes |

Note: `maxThreadsPerMultiProcessor` is 1024 on Turing TU117, so a single
1024-thread block already saturates an SM — occupancy is 1 block/SM at that size.

### bfloat16 on this GPU — important

The engine stores all weights and activations as `__nv_bfloat16`. On `sm_75`:

- **Native bf16 arithmetic is NOT available.** Intrinsics like `__hmul`/`__hadd`
  on `__nv_bfloat16` are compiled out below `sm_80`; using them fails to compile
  (the overload falls back to `__half` and errors). The kernels here only ever
  *convert* bf16 ↔ float (`(__nv_bfloat16)(x1 * c - x2 * s)` etc.), which is
  supported on every architecture — so this is fine, but do not introduce bf16
  intrinsic math.
- **`cublasGemmEx` / `cublasGemmStridedBatchedEx` with `CUDA_R_16BF` inputs and
  `CUBLAS_COMPUTE_32F` DO work** — verified numerically correct on this GPU.
  Because there are no tensor cores, cuBLAS runs these on FP32 CUDA cores, so
  expect FP32-class throughput (≈2.7 TFLOP/s ceiling), not tensor-core speeds.

## CPU / Host

| Property | Value |
| --- | --- |
| Model | AMD Ryzen 5 5600H with Radeon Graphics (Zen 3, Cezanne) |
| Cores / threads | 6 cores / 12 threads (2 per core) |
| Max / min clock | 4280 MHz / 412 MHz |
| L2 / L3 cache | 3 MiB (6 × 512 KiB) / 16 MiB |
| System RAM | **7.1 GiB total** (~2.9 GiB available), 4.0 GiB swap |
| Laptop | Acer Nitro AN515-45 |
| Root disk | NVMe `/dev/nvme0n1p2`, 98 GB, ~19 GB free |

System RAM is the tighter constraint for host-side work: `loadWeights` reads the
whole safetensors payload into a host `std::vector` (~2.3 GiB) before the
`cudaMemcpy`, which is a significant fraction of available RAM.

## Software stack

| Component | Version |
| --- | --- |
| OS | Ubuntu 24.04.4 LTS |
| Kernel | 7.0.0-34-generic |
| NVIDIA driver | 580.178.04 (supports CUDA up to 13.0) |
| CUDA toolkit (`nvcc`) | **12.0.140** at `/usr/bin/nvcc` |
| CUDA runtime linked | 12.0 |
| gcc | 13.3.0 |
| CMake / Ninja | 3.28.3 / 1.11.1 |
| Python | 3.12.3 (`.venv/`) |
| PyTorch | 2.6.0+cu124, CUDA available ✓ |

The driver (13.0-capable) is newer than the toolkit (12.0); that direction is
fine. Do not assume CUDA 12.4+ toolkit features in `.cu` code — `nvcc` is 12.0.

## Build configuration

`paged_attention/CMakeLists.txt` sets `CMAKE_CUDA_ARCHITECTURES 75`, which is
correct for this GPU. Verified the emitted flags are
`--generate-code=arch=compute_75,code=[compute_75,sm_75]` and the linked binary
contains a single `sm_75` cubin.

Gotcha: `build/CMakeCache.txt` may still show `CMAKE_CUDA_ARCHITECTURES:STRING=52`
— that is CMake's initial autodetect value in the cache; the `set()` in
`CMakeLists.txt` runs after `project()` and creates a normal variable that
shadows it. The actual compile flags are `sm_75`. Check `build/build.ninja`
rather than the cache if in doubt.

## Model

| Property | Value |
| --- | --- |
| Model | Llama-3.2-1B-Instruct (`models/Llama-3.2-1B-Instruct/`) |
| `model.safetensors` | 2 471 645 608 B = **2.30 GiB** (bf16) |
| Layers | 16 |
| Embedding dim | 2048 |
| MLP hidden dim | 8192 |
| Q heads / KV heads | 32 / 8 (GQA ratio 4) |
| Head dim | 64 |
| KV dim | 512 |
| Vocab | 128256 |

## VRAM budget (why the constants are what they are)

With the current constants — `MAX_SEQ_LEN 2048`, `MAX_PROMPT_LEN 512`,
`BATCH_SIZE 16`, `BLOCK_SIZE 16`, `KV_CACHE_SIZE_BYTES 1000 MiB`:

| Allocation | Size |
| --- | --- |
| `model_weights` | 2357.14 MiB |
| `kv_cache` | 1000.00 MiB |
| `embed_proj` (logits, 512 × 128256 bf16) | 125.25 MiB |
| `prefill_attn_scores` (512² × 32 bf16) | 16.00 MiB |
| `gate` + `up` (512 × 8192 bf16 each) | 16.00 MiB |
| 4 × `EMBEDDING_LENGTH` buffers + `input_embeddings` | 10.00 MiB |
| RoPE cos/sin tables | 1.00 MiB |
| `k/v_proj_temp_buf`, block table, misc | ~1.16 MiB |
| **Total** | **≈ 3526 MiB = 3.44 GiB** |
| Available | 3715 MiB = 3.63 GiB |
| **Headroom** | **≈ 188 MiB** |

This is why `KV_CACHE_SIZE_BYTES` is 1000 MiB rather than the 2 GiB the upstream
code used: 2 GiB does not fit in 3.63 GiB alongside 2.30 GiB of weights.
`embed_proj` is the third-largest allocation and scales with `MAX_PROMPT_LEN ×
VOCAB_SIZE` — it is the first thing to shrink if more headroom is needed.

### Paging arithmetic

- `BLOCK_BYTES` = `BLOCK_SIZE × KV_DIM × 2 (bf16) × 2 (K and V)` = 32768 B/page
- `NUM_BLOCKS` = 1000 MiB / 32768 = **32000** pages
- `MAX_BLOCKS_PER_SEQ` = 2048 / 16 = **128**
- KV footprint per token = 32768 B across all 16 layers (2048 B per layer)

Worth knowing: a fully-loaded batch (16 sequences × 16 layers × 128 pages) would
want 32768 pages but only 32000 exist, so the pool is ~2% oversubscribed at the
absolute worst case. Not reachable in practice since `MAX_PROMPT_LEN` is 512 and
`MAX_NEW_TOKENS_GENERATED` is 512, capping a sequence at ~1024 tokens = 64 pages
per layer (16 × 16 × 64 = 16384 pages, half the pool).

## Practical implications

1. **1024 threads/block is a hard ceiling.** The guards in `kernels.cu` that bail
   out above 1024 threads are real limits on this GPU, not placeholders. Kernels
   that want 2048 lanes (embedding dim) must have each thread do 2 elements —
   which is what `embeddingGatherKernel` and the RMS-norm kernels do.
2. **No tensor cores** — bf16 GEMMs are FP32-rate. Profiling numbers will look
   nothing like Ampere/Ada/Blackwell results.
3. **192 GB/s bandwidth with 14 SMs** means decode is firmly memory-bound;
   streaming 2.30 GiB of weights per token caps decode at ~78 tokens/s even at
   100% bandwidth efficiency.
4. **188 MiB of VRAM headroom.** Any new persistent allocation needs a matching
   reduction elsewhere. Run with no other GPU consumers — Xorg already takes a
   few MiB, and a browser can easily take hundreds.
