#pragma once

// Small host-side helpers that are not part of the inference pipeline: where the model
// files live, a loud cudaMalloc and the startup device dump.

#include <string>

// Directory the HF model snapshot was unpacked into. Set once at startup from argv[1] or
// the MODEL_DIR env var, defaults to DEFAULT_MODEL_DIR.
extern std::string g_model_dir;

std::string modelFilePath(const std::string &filename);

// cudaMalloc failing silently on a 4 GiB card leads to garbage results rather than a crash,
// and the prefill scratch is now large enough that it is worth being loud about.
void *allocDevice(size_t bytes, const char *what);

// Prints the device the engine will run on; returns non-zero when there is no GPU at all.
int checkGPUStatus();
