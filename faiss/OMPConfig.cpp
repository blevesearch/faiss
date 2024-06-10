/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-


#include <faiss/OMPConfig.h>

namespace faiss {

    unsigned int num_omp_threads = std::thread::hardware_concurrency();

    void set_num_omp_threads(unsigned int value) {
        num_omp_threads = value;
    }
}
