/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_UTILS_C_H
#define FAISS_UTILS_C_H

#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

const char* faiss_get_version();

/** Convert a real-valued vector to a binary vector.
 *
 * Each bit in the output is set to 1 if the corresponding input value is
 * greater than 0, and 0 otherwise.
 *
 * @param d      dimension of the input vector (must be a multiple of 8)
 * @param x_in   input float vector (float table of size d)
 * @param x_out  output binary vector (uint8_t table of size d / 8)
 */
void faiss_real_to_binary(size_t d, const float* x_in, uint8_t* x_out);

#ifdef __cplusplus
}
#endif

#endif
