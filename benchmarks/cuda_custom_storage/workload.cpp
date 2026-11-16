/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "workload.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace cuda_custom_storage_benchmark {
namespace {

class RestoredBytesMismatch : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

std::vector<unsigned char>
Pattern(size_t size, size_t logical_offset)
{
  std::vector<unsigned char> pattern(size);
  for (size_t index = 0; index < size; ++index) {
    const uint64_t position = logical_offset + index;
    pattern[index] = static_cast<unsigned char>(((position * 1315423911ULL) ^ (position >> 7U) ^ 0x5aU) & 0xffU);
  }
  return pattern;
}

void
FillDevicePattern(CUdeviceptr pointer, size_t size)
{
  for (size_t offset = 0; offset < size;) {
    const size_t length = std::min(kCopyChunkBytes, size - offset);
    const std::vector<unsigned char> pattern = Pattern(length, offset);
    CheckCUDA(cuMemcpyHtoD(pointer + offset, pattern.data(), length), "initialize application buffer");
    offset += length;
  }
}

void
VerifyDevicePattern(CUdeviceptr pointer, size_t size)
{
  std::vector<unsigned char> actual(std::min(kCopyChunkBytes, size));
  for (size_t offset = 0; offset < size;) {
    const size_t length = std::min(actual.size(), size - offset);
    const std::vector<unsigned char> expected = Pattern(length, offset);
    CheckCUDA(cuMemcpyDtoH(actual.data(), pointer + offset, length), "read restored application buffer");
    const auto mismatch =
        std::mismatch(actual.begin(), actual.begin() + static_cast<ptrdiff_t>(length), expected.begin());
    if (mismatch.first != actual.begin() + static_cast<ptrdiff_t>(length)) {
      const size_t mismatch_offset = offset + static_cast<size_t>(mismatch.first - actual.begin());
      throw RestoredBytesMismatch("restored application bytes differ at offset " + std::to_string(mismatch_offset));
    }
    offset += length;
  }
}

int
ReportResult(int ready_fd, WorkloadResult result) noexcept
{
  try {
    WriteAll(ready_fd, &result, sizeof(result));
  }
  catch (const std::exception& exception) {
    std::fprintf(stderr, "workload failed to report result: %s\n", exception.what());
  }
  return result == WorkloadResult::kPassed ? 0 : 1;
}

}  // namespace

int
RunWorkload(int ready_fd, int command_fd, const Options& options)
{
  try {
    CheckCUDA(cuInit(0), "workload cuInit");
    CUdevice device = 0;
    CUcontext context = nullptr;
    CheckCUDA(cuDeviceGet(&device, options.device), "workload cuDeviceGet");
    CheckCUDA(cuDevicePrimaryCtxRetain(&context, device), "workload cuDevicePrimaryCtxRetain");
    CheckCUDA(cuCtxSetCurrent(context), "workload cuCtxSetCurrent");

    CUdeviceptr application = 0;
    CUdeviceptr scratch = 0;
    CheckCUDA(cuMemAlloc(&application, options.bytes), "workload application allocation");
    CheckCUDA(cuMemAlloc(&scratch, 4096), "workload scratch allocation");
    FillDevicePattern(application, options.bytes);

    const ReadyMessage ready{application, options.bytes};
    WriteAll(ready_fd, &ready, sizeof(ready));
    WorkloadCommand command{};
    ReadAll(command_fd, &command, sizeof(command));
    if (command != WorkloadCommand::kVerify) {
      throw std::runtime_error("workload received an invalid command");
    }

    try {
      VerifyDevicePattern(application, options.bytes);
    }
    catch (const RestoredBytesMismatch& exception) {
      std::fprintf(stderr, "workload restored-byte mismatch: %s\n", exception.what());
      return ReportResult(ready_fd, WorkloadResult::kRestoredBytesMismatch);
    }

    try {
      CheckCUDA(cuMemsetD8(scratch, 0x5a, 4096), "post-restore CUDA operation");
      std::array<unsigned char, 4096> scratch_bytes{};
      CheckCUDA(
          cuMemcpyDtoH(scratch_bytes.data(), scratch, scratch_bytes.size()), "verify post-restore CUDA operation");
      if (!std::all_of(scratch_bytes.begin(), scratch_bytes.end(), [](unsigned char value) { return value == 0x5a; })) {
        throw std::runtime_error("post-restore CUDA operation produced incorrect data");
      }
    }
    catch (const std::exception& exception) {
      std::fprintf(stderr, "workload post-restore CUDA failure: %s\n", exception.what());
      return ReportResult(ready_fd, WorkloadResult::kPostRestoreCudaFailure);
    }

    try {
      CheckCUDA(cuMemFree(scratch), "workload scratch cleanup");
      CheckCUDA(cuMemFree(application), "workload application cleanup");
      CheckCUDA(cuDevicePrimaryCtxRelease(device), "workload primary context cleanup");
    }
    catch (const std::exception& exception) {
      std::fprintf(stderr, "workload cleanup failure: %s\n", exception.what());
      return ReportResult(ready_fd, WorkloadResult::kCleanupFailure);
    }
    return ReportResult(ready_fd, WorkloadResult::kPassed);
  }
  catch (const std::exception& exception) {
    std::fprintf(stderr, "workload failed: %s\n", exception.what());
    return ReportResult(ready_fd, WorkloadResult::kUnexpectedFailure);
  }
}

}  // namespace cuda_custom_storage_benchmark
