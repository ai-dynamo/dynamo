/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <string>
#include <string_view>

#include "benchmark_cache.h"
#include "transfer_config.h"

namespace cuda_checkpoint_benchmark {

struct BenchmarkStorageOptions {
  bool restore = true;
  bool existing_file = false;
  bool warmups_set = false;
  size_t warmups = 1;
  bool cache_profile_set = false;
  CacheProfile cache_profile = CacheProfile::kBufferedWarm;
};

struct ClientCacheState {
  std::string_view residency;
  bool preconditioned_by_invocation;
};

size_t EffectiveWarmups(bool existing_file, bool warmups_set, size_t warmups);
bool ValidateBenchmarkStorageOptions(const BenchmarkStorageOptions& options, std::string* error);
ClientCacheState ClientCacheStateBeforeMeasuredIteration(
    bool restore, bool existing_file, size_t effective_warmups, size_t measured_iteration, CacheProfile cache_profile);
std::string ClientCacheStateJSON(const ClientCacheState& state);
void FillPattern(unsigned char* data, size_t size, size_t logical_offset);
bool ValidateExistingStorageLayout(const cuda_checkpoint_transfer::StorageLayout& layout, std::string* error);

}  // namespace cuda_checkpoint_benchmark
