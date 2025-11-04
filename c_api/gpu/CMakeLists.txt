# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

foreach(lib faiss_c faiss_c_avx2 faiss_c_avx512 faiss_c_avx512_spr faiss_c_sve)
  if (TARGET ${lib})
    target_sources(${lib} PRIVATE
      DeviceUtils_c.cpp
      GpuAutoTune_c.cpp
      GpuClonerOptions_c.cpp
      GpuIndex_c.cpp
      GpuResources_c.cpp
      StandardGpuResources_c.cpp
    )
  endif()
endforeach()

file(GLOB FAISS_C_API_GPU_HEADERS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "*.h")
faiss_install_headers("${FAISS_C_API_GPU_HEADERS}" c_api/gpu)

if (FAISS_ENABLE_ROCM)
  foreach(lib faiss_c faiss_c_avx2 faiss_c_avx512 faiss_c_avx512_spr faiss_c_sve)
    if (TARGET ${lib})
      target_link_libraries(${lib} PUBLIC hip::host roc::hipblas)
    endif()
  endforeach()
else()
  find_package(CUDAToolkit REQUIRED)
  foreach(lib faiss_c faiss_c_avx2 faiss_c_avx512 faiss_c_avx512_spr faiss_c_sve)
    if (TARGET ${lib})
      target_link_libraries(${lib} PUBLIC CUDA::cudart CUDA::cublas $<$<BOOL:${FAISS_ENABLE_CUVS}>:cuvs::cuvs>)
    endif()
  endforeach()
endif()

add_executable(example_gpu_c EXCLUDE_FROM_ALL example_gpu_c.c)
target_link_libraries(example_gpu_c PRIVATE faiss_c)
