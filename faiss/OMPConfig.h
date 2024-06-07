/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef OMPCONFIG_H
#define OMPCONFIG_H

#include <thread>

namespace faiss {
	// Degree of OpenMP concurrency, determining the number of threads 
	// used in OpenMP parallel regions.
	//
	// This variable specifies the number of threads to be employed in 
	// parallel regions defined by OpenMP directives. If not set explicitly, 
	// the default value is the number of hardware threads available on 
	// the system. This allows for efficient utilization of available 
	// processing resources.
	//
	// The value of num_omp_threads can be modified to control the level 
	// of parallelism and optimize performance based on the specific 
	// requirements and characteristics of the workload.
	//
	// Usage:
	// - Setting num_omp_threads to a specific value before entering an 
	//   OpenMP parallel region will limit or expand the number of threads 
	//   accordingly.
	// - If num_omp_threads is not set, OpenMP will automatically use the 
	//   number of hardware threads as the default value.
	//
	// Example:
	// unsigned int num_omp_threads = 4; // Use 4 threads in OpenMP parallel regions.
	extern unsigned int num_omp_threads;
	// Setter function for num_omp_threads.
	//
	// Parameters:
	// - value: The desired number of threads to be used in OpenMP parallel regions.
	//
	// Usage:
	// set_num_omp_threads(8); // Set to use 8 threads in OpenMP parallel regions.
	void set_num_omp_threads(unsigned int value);
}
#endif // OMPCONFIG_H
