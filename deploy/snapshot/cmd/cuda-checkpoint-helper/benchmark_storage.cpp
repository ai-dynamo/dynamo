/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark_storage.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>

namespace cuda_checkpoint_benchmark {

namespace {

uint64_t
PatternWord(uint64_t index)
{
  uint64_t value = index + 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

}  // namespace

size_t
EffectiveWarmups(bool existing_file, bool warmups_set, size_t warmups)
{
  return existing_file && !warmups_set ? 0 : warmups;
}

bool
ValidateBenchmarkStorageOptions(const BenchmarkStorageOptions& options, std::string* error)
{
  if (error == nullptr) {
    return false;
  }
  if (options.existing_file && !options.restore) {
    *error = "--existing-file is valid only with --operation restore or read";
    return false;
  }
  if (!options.restore && options.cache_profile_set) {
    *error = "--cache-profile is valid only with --operation restore or read";
    return false;
  }
  return true;
}

ClientCacheState
ClientCacheStateBeforeMeasuredIteration(
    bool restore, bool existing_file, size_t effective_warmups, size_t measured_iteration, CacheProfile cache_profile)
{
  if (!restore) {
    return {"not-applicable", false};
  }
  if (cache_profile == CacheProfile::kClientEvicted) {
    return {"unknown-after-advice", false};
  }
  const bool preconditioned = !existing_file || effective_warmups > 0 || measured_iteration > 0;
  return {preconditioned ? "buffered-warm" : "unknown-not-preconditioned", preconditioned};
}

std::string
ClientCacheStateJSON(const ClientCacheState& state)
{
  return "\"client_cache_preconditioned_by_invocation\":" +
         std::string(state.preconditioned_by_invocation ? "true" : "false") +
         ",\"client_cache_residency\":" + cuda_checkpoint_transfer::JsonEscape(state.residency);
}

void
FillPattern(unsigned char* data, size_t size, size_t logical_offset)
{
  size_t done = 0;
  while (done < size) {
    const size_t absolute = logical_offset + done;
    const uint64_t word = PatternWord(absolute / sizeof(uint64_t));
    const size_t byte_in_word = absolute % sizeof(uint64_t);
    const size_t length = std::min(sizeof(uint64_t) - byte_in_word, size - done);
    for (size_t index = 0; index < length; ++index) {
      data[done + index] = static_cast<unsigned char>(word >> (8 * (byte_in_word + index)));
    }
    done += length;
  }
}

bool
ValidateExistingStorageLayout(const cuda_checkpoint_transfer::StorageLayout& layout, std::string* error)
{
  if (error == nullptr) {
    return false;
  }
  for (const auto& file : layout.files) {
    if (file.size > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
      *error = "benchmark file is too large for POSIX offsets: " + file.path.string();
      return false;
    }
    const int descriptor = open(file.path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
    if (descriptor < 0) {
      *error = "open existing benchmark file failed: " + file.path.string() + ": " + std::strerror(errno);
      return false;
    }
    struct stat file_stat {};
    const int stat_result = fstat(descriptor, &file_stat);
    const int stat_error = errno;
    (void)close(descriptor);
    if (stat_result != 0) {
      *error = "stat existing benchmark file failed: " + file.path.string() + ": " + std::strerror(stat_error);
      return false;
    }
    if (!S_ISREG(file_stat.st_mode) || file_stat.st_size < 0 || static_cast<size_t>(file_stat.st_size) != file.size) {
      *error = "existing benchmark file is not regular or has the wrong size: " + file.path.string() + ": expected " +
               std::to_string(file.size) + " bytes";
      return false;
    }
  }
  return true;
}

}  // namespace cuda_checkpoint_benchmark
