/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "transfer_config.h"

namespace cuda_checkpoint_benchmark {

enum class CacheProfile {
  kBufferedWarm,
  kClientEvicted,
};

struct CacheEvictionResult {
  size_t files_requested = 0;
  size_t files_advised = 0;
  std::vector<std::string> errors;
};

bool ParseCacheProfile(std::string_view value, CacheProfile* profile);
std::string_view CacheProfileName(CacheProfile profile);
CacheEvictionResult EvictClientFileCache(const cuda_checkpoint_transfer::StorageLayout& layout);

}  // namespace cuda_checkpoint_benchmark
