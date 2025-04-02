# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.17)

project(
  nvllm
  VERSION 0.1.0.0
  LANGUAGES CXX
)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules/set_ifndef.cmake)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_POSTION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


option(USE_STUBS "Build with stub implementations instead of real CUDA code" OFF)

if (USE_STUBS)
  add_definitions(-DUSE_STUBS)
  set(SOURCE_FILES
    src/nvllm_trt.cpp
    src/engine_stub/engine.cpp
)

add_library(tensorrt_llm SHARED src/engine_stub/tensorrt_llm.cpp)

else()

#SET(TRTLLM_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../submodules/tensorrt_llm" CACHE STRING "TRTLLM_SRC_DIR: /../../submodules/tensorrt_llm")
SET(TRTLLM_LIB_DIR "/usr/local/lib" CACHE STRING "TRTLLM_LIB_DIR: /usr/local/lib")

#include(${TRTLLM_SRC_DIR}/cpp/cmake/modules/find_library_create_target.cmake)

set(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib;/opt/hpcx/ompi/lib:/usr/local/cuda/lib64:/usr/local/tensorrt/targets/aarch64-linux-gnu/lib:/src/tensorrt_llm/cpp/build/tensorrt_llm/plugins")
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

include(FetchContent)

FetchContent_Declare(
  json
  GIT_REPOSITORY https://github.com/nlohmann/json.git
  GIT_TAG v3.11.2
)

FetchContent_Declare(
  spdlog
  GIT_REPOSITORY https://github.com/gabime/spdlog.git
  GIT_TAG v1.15.0
)

# Make nlohmann/json available
FetchContent_MakeAvailable(json)
FetchContent_MakeAvailable(spdlog)
set_property(TARGET spdlog PROPERTY POSITION_INDEPENDENT_CODE ON)


add_library(tensorrt_llm SHARED IMPORTED)
set_target_properties(
  tensorrt_llm
  PROPERTIES
    IMPORTED_LOCATION "${TRTLLM_LIB_DIR}/libtensorrt_llm.so"
)

add_library(nvinfer_plugin_tensorrt_llm SHARED IMPORTED)
set_target_properties(
  nvinfer_plugin_tensorrt_llm
  PROPERTIES
    IMPORTED_LOCATION "${TRTLLM_LIB_DIR}/libnvinfer_plugin_tensorrt_llm.so"
)


add_library(xxhash STATIC IMPORTED)
set_target_properties(
  xxhash
  PROPERTIES
    IMPORTED_LOCATION "/usr/lib/aarch64-linux-gnu/libxxhash.a"
)

set(SOURCE_FILES
    src/nvllm_trt.cpp
    src/engine_trt/engine.cpp
    src/engine_trt/request.cpp
    src/engine_trt/response.cpp
    src/engine_trt/config.cpp
    src/engine_trt/kv_event.cpp
    src/engine_trt/stats.cpp
    ${PROTO_SRCS} ${PROTO_HDRS}
    # ... other source files ...
)

endif()

function(set_library_target_properties target)

target_include_directories(
  ${target}
  PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/>
    $<INSTALL_INTERFACE:include/>
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src
          ${CMAKE_BINARY_DIR}
          /usr/local/cuda-12/targets/sbsa-linux/include
          /usr/local/tensorrt/include/
)


target_compile_features(${target} PRIVATE cxx_std_17)
set_target_properties(${target} PROPERTIES OUTPUT_NAME nvllm_trt)

target_compile_options(
  ${target}
  PRIVATE
    $<$<OR:$<CXX_COMPILER_ID:Clang>,$<CXX_COMPILER_ID:AppleClang>,$<CXX_COMPILER_ID:GNU>>:
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-type-limits>
    -Wno-deprecated-declarations
    $<$<CXX_COMPILER_ID:MSVC>:/Wall
    /D_WIN32_WINNT=0x0A00
    /EHsc>)

if (USE_STUBS)
else()
target_link_libraries(
  ${target}
  PRIVATE tensorrt_llm
          ${Protobuf_LIBRARIES}
          xxhash
          # ${MPI_LIBRARIES}
          # ${CUDA_LIBRARIES}
          # nvinfer
          nvinfer_plugin_tensorrt_llm
          nlohmann_json::nlohmann_json
          spdlog::spdlog
)
endif()

# target_link_options(${target} PRIVATE "-static")

target_link_libraries(${target} PUBLIC
)

endfunction()

add_library(nvllm_trt SHARED ${SOURCE_FILES})
set_library_target_properties(nvllm_trt)


include(CMakePackageConfigHelpers)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/nvllmConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/nvllmConfig.cmake
    INSTALL_DESTINATION lib/cmake/nvllm
)

write_basic_package_version_file(
    "nvllmConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion
)

# Installation rules
install(TARGETS nvllm_trt
    EXPORT nvllmConfig # This should match the name used in configure_package_config_file
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
    RUNTIME DESTINATION bin
    INCLUDES DESTINATION include
)

# Install the nvllmConfig.cmake and nvllmConfigVersion.cmake files
install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/nvllmConfig.cmake # Corrected the file name
    ${CMAKE_CURRENT_BINARY_DIR}/nvllmConfigVersion.cmake
    DESTINATION lib/cmake/nvllm
)

# # Install config.h
# install(FILES "${PROJECT_BINARY_DIR}/config.h"
#         DESTINATION include/nvidia/nvllm)

# Install header files
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/include/
        DESTINATION include)
