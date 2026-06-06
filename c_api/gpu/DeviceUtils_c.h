/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_DEVICE_UTILS_C_H
#define FAISS_DEVICE_UTILS_C_H

#include <cublas_v2.h>
#include <cuda_runtime_api.h>
#include "../faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Returns the number of available GPU devices
int faiss_get_num_gpus(int* p_output);

/// Starts the CUDA profiler (exposed via SWIG)
int faiss_gpu_profiler_start();

/// Stops the CUDA profiler (exposed via SWIG)
int faiss_gpu_profiler_stop();

/// Synchronizes the CPU against all devices (equivalent to
/// cudaDeviceSynchronize for each device)
int faiss_gpu_sync_all_devices();

/// Returns the free memory (in bytes) on the specified device
int faiss_gpu_free_memory(int device, size_t* p_free_bytes);

/// Probes the specified GPU device for basic health.
/// On success, writes 0 to *p_probe_result if healthy, or -1 if unhealthy.
/// Returns 0 on success, or a negative value on exception.
int faiss_probe_gpu(int device, int* p_probe_result);

#ifdef __cplusplus
}
#endif
#endif
