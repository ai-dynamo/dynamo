/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark_cache.h"

#include <fcntl.h>
#include <unistd.h>

#include <cerrno>
#include <cstring>
#include <limits>
#include <string>

namespace cuda_checkpoint_benchmark {

bool
ParseCacheProfile(std::string_view value, CacheProfile* profile)
{
  if (profile == nullptr) {
    return false;
  }
  if (value == "buffered-warm") {
    *profile = CacheProfile::kBufferedWarm;
    return true;
  }
  if (value == "client-evicted") {
    *profile = CacheProfile::kClientEvicted;
    return true;
  }
  return false;
}

std::string_view
CacheProfileName(CacheProfile profile)
{
  switch (profile) {
    case CacheProfile::kBufferedWarm:
      return "buffered-warm";
    case CacheProfile::kClientEvicted:
      return "client-evicted";
  }
  return "unknown";
}

CacheEvictionResult
EvictClientFileCache(const cuda_checkpoint_transfer::StorageLayout& layout)
{
  CacheEvictionResult result;
  result.files_requested = layout.files.size();
  for (const auto& file : layout.files) {
    if (file.size > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
      result.errors.push_back(file.path.string() + ": file is too large for POSIX offsets");
      continue;
    }
    const int descriptor = open(file.path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor < 0) {
      result.errors.push_back(file.path.string() + ": open failed: " + std::strerror(errno));
      continue;
    }
    const int advice_status =
        posix_fadvise(descriptor, 0, static_cast<off_t>(file.size), POSIX_FADV_DONTNEED);
    const int close_status = close(descriptor);
    if (advice_status != 0) {
      result.errors.push_back(file.path.string() + ": POSIX_FADV_DONTNEED failed: " + std::strerror(advice_status));
      continue;
    }
    if (close_status != 0) {
      result.errors.push_back(file.path.string() + ": close failed: " + std::strerror(errno));
      continue;
    }
    ++result.files_advised;
  }
  return result;
}

}  // namespace cuda_checkpoint_benchmark
