/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "utils_c.h"
#include <faiss/Index.h>
#include <faiss/utils/utils.h>

const char* faiss_get_version() {
    return VERSION_STRING;
}

void faiss_real_to_binary(size_t d, const float* x_in, uint8_t* x_out) {
    faiss::real_to_binary(d, x_in, x_out);
}
