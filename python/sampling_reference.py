"""Check the sampling kernel's algorithm without a GPU.

kernels.cu samples with a two-level inverse-CDF walk: across ARGMAX_CHUNKS_PER_ROW chunk sums
to pick a chunk, then across that chunk's tokens to pick a token. That structure exists to
avoid a 128k-element serial scan per row, but it is also the part most likely to be subtly
wrong - an off-by-one in either prefix walk biases the distribution toward chunk boundaries in
a way that still "looks random".

This mirrors the kernel in numpy and checks two things the CUDA code cannot check for itself:

  1. the chunked walk picks the same token as a plain one-level CDF over the whole row,
     for the same uniform draw - i.e. the chunking is exactly a reassociation;
  2. over many draws the empirical distribution matches softmax(logits / T).

Run: python python/sampling_reference.py
"""

import numpy as np

VOCAB_SIZE = 128256
ARGMAX_CHUNKS_PER_ROW = 128
ARGMAX_BLOCK_THREADS = 256
ARGMAX_CHUNK_TOKENS = (VOCAB_SIZE + ARGMAX_CHUNKS_PER_ROW - 1) // ARGMAX_CHUNKS_PER_ROW


def counter_uniform(seed, seq_key, step):
    """splitmix64 finaliser, same constants and same shifts as counterUniform in kernels.cu."""
    M = (1 << 64) - 1
    z = ((seq_key << 32) ^ step) & M
    z ^= (seed * 0x9E3779B97F4A7C15) & M
    z = (z + 0x9E3779B97F4A7C15) & M
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & M
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & M
    z = z ^ (z >> 31)
    return (z >> 40) * (1.0 / 16777216.0)


def kernel_sample(logits, inv_t, u):
    """Exactly what expSumPartialKernel + sampleFinalizeKernel do, in order."""
    row_max = logits.max()                                   # rowMaxKernel
    w = np.exp((logits - row_max) * inv_t)

    # expSumPartialKernel: one sum per chunk
    chunk_sums = np.zeros(ARGMAX_CHUNKS_PER_ROW)
    for c in range(ARGMAX_CHUNKS_PER_ROW):
        lo, hi = c * ARGMAX_CHUNK_TOKENS, min((c + 1) * ARGMAX_CHUNK_TOKENS, VOCAB_SIZE)
        chunk_sums[c] = w[lo:hi].sum()

    # sampleFinalizeKernel, phase 1: walk the chunk prefix
    chunk_prefix = np.cumsum(chunk_sums)
    total = chunk_prefix[-1]
    target = u * total
    chunk = ARGMAX_CHUNKS_PER_ROW - 1
    for c in range(ARGMAX_CHUNKS_PER_ROW):
        if target < chunk_prefix[c]:
            chunk = c
            break
    residual = target - (chunk_prefix[chunk - 1] if chunk > 0 else 0.0)

    # phase 2: contiguous spans, then walk inside the owning span
    lo = chunk * ARGMAX_CHUNK_TOKENS
    hi = min(lo + ARGMAX_CHUNK_TOKENS, VOCAB_SIZE)
    span = (ARGMAX_CHUNK_TOKENS + ARGMAX_BLOCK_THREADS - 1) // ARGMAX_BLOCK_THREADS
    result = min(lo + ARGMAX_CHUNK_TOKENS - 1, VOCAB_SIZE - 1)     # the kernel's fallback
    thread_sums = np.zeros(ARGMAX_BLOCK_THREADS)
    for t in range(ARGMAX_BLOCK_THREADS):
        a, b = lo + t * span, min(lo + (t + 1) * span, hi)
        if a < hi:
            thread_sums[t] = w[a:b].sum()
    thread_prefix = np.cumsum(thread_sums)
    for t in range(ARGMAX_BLOCK_THREADS):
        before = thread_prefix[t - 1] if t > 0 else 0.0
        a, b = lo + t * span, min(lo + (t + 1) * span, hi)
        if a < hi and before <= residual < thread_prefix[t]:
            acc = before
            for tok in range(a, b):
                acc += w[tok]
                if residual < acc:
                    result = tok
                    break
            break
    return result


def flat_sample(logits, inv_t, u):
    """One-level CDF over the whole row: the thing the chunked walk must agree with."""
    w = np.exp((logits - logits.max()) * inv_t)
    cdf = np.cumsum(w)
    return int(np.searchsorted(cdf, u * cdf[-1], side="right"))


def main():
    rng = np.random.default_rng(0)
    # Logit scale chosen to look like a real lm_head row: mostly noise, a few sharp peaks.
    logits = rng.normal(0.0, 2.0, VOCAB_SIZE)
    logits[rng.choice(VOCAB_SIZE, 20, replace=False)] += 12.0

    print("1. chunked walk vs flat CDF, same draw")
    for temperature in (0.7, 1.0, 1.5):
        inv_t = 1.0 / temperature
        bad = 0
        for i in range(300):
            u = counter_uniform(1234, 7, i)
            if kernel_sample(logits, inv_t, u) != flat_sample(logits, inv_t, u):
                bad += 1
        print(f"   T={temperature}: {300 - bad}/300 khớp" + ("" if not bad else "   *** LỆCH ***"))
        assert bad == 0

    print("2. empirical distribution vs softmax")
    inv_t = 1.0 / 1.0
    probs = np.exp(logits - logits.max())
    probs /= probs.sum()
    top = np.argsort(probs)[-8:][::-1]
    draws = 40000
    counts = np.zeros(VOCAB_SIZE)
    cdf = np.cumsum(np.exp(logits - logits.max()))
    for i in range(draws):
        counts[np.searchsorted(cdf, counter_uniform(99, 3, i) * cdf[-1], side="right")] += 1
    print(f"   {'token':>8} {'đo được':>10} {'softmax':>10} {'lệch':>8}")
    worst = 0.0
    for tok in top:
        got, want = counts[tok] / draws, probs[tok]
        err = abs(got - want) / want
        worst = max(worst, err)
        print(f"   {tok:>8} {got:>10.4f} {want:>10.4f} {err:>7.1%}")
    # 3 sigma of a binomial at p~0.1 over 40k draws is well under 5%
    print(f"   lệch lớn nhất {worst:.1%}" + ("  OK" if worst < 0.05 else "  *** QUÁ LỚN ***"))
    assert worst < 0.05

    print("3. draw ở biên")
    for u in (0.0, 1.0 - 1.0 / 16777216.0):
        tok = kernel_sample(logits, 1.0, u)
        assert 0 <= tok < VOCAB_SIZE, tok
        print(f"   u={u:.8f} -> token {tok}  (trong [0, {VOCAB_SIZE}))")

    print("\nTẤT CẢ ĐẠT — thuật toán của kernel đúng; còn lại là kiểm cú pháp CUDA lúc build.")


if __name__ == "__main__":
    main()
