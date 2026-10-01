/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <stdexcept>

#include "roundtrip_common.hpp"

namespace cuda_custom_storage_benchmark {

class RestoreRejected : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class ExtentSizeMismatch : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

void ValidateExtentFile(const fs::path& path, size_t expected_size);
uintmax_t ExtentFileSize(const fs::path& path);
void CorruptExtentSameSize(const fs::path& path, size_t size);

Artifact CheckpointProcess(
    pid_t pid, const fs::path& directory, uint64_t application_ptr, size_t application_bytes, int requested_device,
    const std::string& device_uuid, CUcontext context);
void RestoreProcess(pid_t pid, const fs::path& directory, const Artifact& artifact, CUcontext expected_context);

}  // namespace cuda_custom_storage_benchmark
