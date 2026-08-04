# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Build the _sycl_vmm pybind11 extension.
#
# Usage:
#   cmake -B build -DCMAKE_CXX_COMPILER=$(which icpx) .
#   cmake --build build

cmake_minimum_required(VERSION 3.20)
project(_sycl_vmm LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ============================================================================
# 1. Verify SYCL compiler
# ============================================================================

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fsycl" COMPILER_SUPPORTS_FSYCL)
if(NOT COMPILER_SUPPORTS_FSYCL)
    message(FATAL_ERROR
        "The CXX compiler (${CMAKE_CXX_COMPILER}) does not support -fsycl.\n"
        "Use Intel DPC++: cmake -DCMAKE_CXX_COMPILER=icpx ...")
endif()

# ============================================================================
# 2. Detect oneAPI version and probe SYCL headers
# ============================================================================

# All header probes need -fsycl so the compiler adds the SYCL include path.
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_FLAGS "-fsycl")

# --- VMM core: probe nightly path, then 2025.3 path ---
check_cxx_source_compiles("
    #include <sycl/ext/oneapi/experimental/virtual_mem/virtual_mem.hpp>
    int main() {}
" HAS_VMM_EXPERIMENTAL_PATH)

check_cxx_source_compiles("
    #include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
    int main() {}
" HAS_VMM_LEGACY_PATH)

if(HAS_VMM_EXPERIMENTAL_PATH)
    message(STATUS "_sycl_vmm: VMM core headers: nightly path (experimental/virtual_mem/)")
elseif(HAS_VMM_LEGACY_PATH)
    message(STATUS "_sycl_vmm: VMM core headers: 2025.3 path (virtual_mem/)")
else()
    message(FATAL_ERROR
        "SYCL VMM headers (virtual_mem.hpp) not found.\n"
        "These are required. Upgrade to oneAPI 2025.3+ or DPC++ nightly.")
endif()

# --- IPC physical memory ---
# Header location varies by oneAPI version:
#   - 2025.3: sycl/ext/oneapi/experimental/ipc_physical_memory.hpp
#   - 2026.0+: sycl/ext/oneapi/experimental/ipc_memory.hpp
check_cxx_source_compiles("
    #include <sycl/ext/oneapi/experimental/ipc_physical_memory.hpp>
    int main() {}
" _HAS_IPC_LEGACY)
check_cxx_source_compiles("
    #include <sycl/ext/oneapi/experimental/ipc_memory.hpp>
    int main() {}
" _HAS_IPC_NEW)
if(_HAS_IPC_LEGACY OR _HAS_IPC_NEW)
    set(HAS_SYCL_IPC_HEADER TRUE)
else()
    set(HAS_SYCL_IPC_HEADER FALSE)
endif()
message(STATUS "_sycl_vmm: IPC physical memory header:  ${HAS_SYCL_IPC_HEADER}")

# --- Host register ---
check_cxx_source_compiles("
    #include <sycl/ext/oneapi/experimental/register_host_memory.hpp>
    int main() {}
" HAS_SYCL_HOST_REGISTER_HEADER)
message(STATUS "_sycl_vmm: Host register header:        ${HAS_SYCL_HOST_REGISTER_HEADER}")

# --- Intel free_memory ---
check_cxx_source_compiles("
    #include <sycl/ext/intel/info/device.hpp>
    int main() {}
" HAS_SYCL_FREE_MEMORY_HEADER)
message(STATUS "_sycl_vmm: Intel free_memory header:    ${HAS_SYCL_FREE_MEMORY_HEADER}")

# Done with probes
unset(CMAKE_REQUIRED_FLAGS)

# ============================================================================
# 3. Find Python + pybind11
# ============================================================================

find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG QUIET)

if(NOT pybind11_FOUND)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE _pb11_rc
    )
    if(_pb11_rc EQUAL 0)
        list(APPEND CMAKE_PREFIX_PATH "${PYBIND11_CMAKE_DIR}")
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR
            "pybind11 not found. Install via: pip install pybind11")
    endif()
endif()

# ============================================================================
# 3b. Find Level-Zero (for L0 interop IPC path)
# ============================================================================

# Level-Zero is optional; if found, enables L0 interop IPC path.
find_path(LEVEL_ZERO_INCLUDE_DIR
    NAMES level_zero/ze_api.h
    PATHS /usr/include /usr/local/include
          ENV LEVEL_ZERO_DIR
    PATH_SUFFIXES include)

find_library(LEVEL_ZERO_LIBRARY
    NAMES ze_loader
    PATHS /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/local/lib
          ENV LEVEL_ZERO_DIR
    PATH_SUFFIXES lib lib64)

if(LEVEL_ZERO_INCLUDE_DIR AND LEVEL_ZERO_LIBRARY)
    set(HAS_LEVEL_ZERO TRUE)
    message(STATUS "_sycl_vmm: Level-Zero:                  YES (${LEVEL_ZERO_LIBRARY})")
else()
    set(HAS_LEVEL_ZERO FALSE)
    message(STATUS "_sycl_vmm: Level-Zero:                  NO (L0 interop IPC disabled)")
endif()

# ============================================================================
# 3c. Find PyTorch (libtorch) for tensor_from_device_ptr
# ============================================================================

# Use torch's cmake prefix to find Torch package
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _torch_rc)

if(_torch_rc EQUAL 0)
    list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX}")
    find_package(Torch REQUIRED)
    set(HAS_TORCH TRUE)
    message(STATUS "_sycl_vmm: PyTorch (libtorch):          YES")
else()
    set(HAS_TORCH FALSE)
    message(WARNING "_sycl_vmm: PyTorch not found. tensor_from_device_ptr will not be available.")
endif()

# ============================================================================
# 4. Build the extension
# ============================================================================

pybind11_add_module(_sycl_vmm sycl_vmm.cpp)

target_compile_options(_sycl_vmm PRIVATE -fsycl)
target_link_options(_sycl_vmm PRIVATE -fsycl)

# Link Level-Zero if available (enables L0 interop IPC)
if(HAS_LEVEL_ZERO)
    target_include_directories(_sycl_vmm PRIVATE ${LEVEL_ZERO_INCLUDE_DIR})
    target_link_libraries(_sycl_vmm PRIVATE ${LEVEL_ZERO_LIBRARY})
    target_compile_definitions(_sycl_vmm PRIVATE SYCL_VMM_HAS_LEVEL_ZERO=1)
endif()

# Link PyTorch (for tensor_from_device_ptr using at::from_blob)
if(HAS_TORCH)
    # Get torch lib directory for libtorch_python.so (has THPVariable_Wrap)
    execute_process(
        COMMAND ${Python3_EXECUTABLE} -c "import torch, os; print(os.path.join(os.path.dirname(torch.__file__), 'lib'))"
        OUTPUT_VARIABLE TORCH_LIB_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE)
    find_library(TORCH_PYTHON_LIBRARY torch_python PATHS "${TORCH_LIB_DIR}" NO_DEFAULT_PATH)
    target_include_directories(_sycl_vmm PRIVATE ${TORCH_INCLUDE_DIRS})
    target_link_libraries(_sycl_vmm PRIVATE ${TORCH_LIBRARIES} ${TORCH_PYTHON_LIBRARY})
    target_compile_definitions(_sycl_vmm PRIVATE SYCL_VMM_HAS_TORCH=1)
endif()

install(TARGETS _sycl_vmm
    LIBRARY DESTINATION gpu_memory_service/common/vmm/_sycl_vmm)

# ============================================================================
# Summary
# ============================================================================

# Helper: show YES/NO instead of TRUE/empty
macro(_yn var out)
    if(${var})
        set(${out} "YES")
    else()
        set(${out} "NO")
    endif()
endmacro()
_yn(HAS_SYCL_IPC_HEADER          _IPC_YN)
_yn(HAS_SYCL_HOST_REGISTER_HEADER _HREG_YN)
_yn(HAS_SYCL_FREE_MEMORY_HEADER   _FMEM_YN)
_yn(HAS_LEVEL_ZERO                 _L0_YN)

message(STATUS "")
message(STATUS "_sycl_vmm build summary:")
message(STATUS "  Compiler:        ${CMAKE_CXX_COMPILER}")
message(STATUS "  VMM core:        YES")
message(STATUS "  IPC phys mem:    ${_IPC_YN}")
message(STATUS "  Host register:   ${_HREG_YN}")
message(STATUS "  Free memory:     ${_FMEM_YN}")
message(STATUS "  Level-Zero:      ${_L0_YN}")
message(STATUS "")
