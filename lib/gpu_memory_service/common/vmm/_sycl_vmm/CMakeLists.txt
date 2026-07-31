# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

# Build the _sycl_vmm pybind11 extension.
#
# Usage:
#   cmake -B build -DCMAKE_CXX_COMPILER=icpx .
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
check_cxx_source_compiles("
    #include <sycl/ext/oneapi/experimental/ipc_physical_memory.hpp>
    int main() {}
" HAS_SYCL_IPC_HEADER)
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
# 4. Build the extension
# ============================================================================

pybind11_add_module(_sycl_vmm sycl_vmm.cpp)

target_compile_options(_sycl_vmm PRIVATE -fsycl)
target_link_options(_sycl_vmm PRIVATE -fsycl)

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

message(STATUS "")
message(STATUS "_sycl_vmm build summary:")
message(STATUS "  Compiler:        ${CMAKE_CXX_COMPILER}")
message(STATUS "  VMM core:        YES")
message(STATUS "  IPC phys mem:    ${_IPC_YN}")
message(STATUS "  Host register:   ${_HREG_YN}")
message(STATUS "  Free memory:     ${_FMEM_YN}")
message(STATUS "")
