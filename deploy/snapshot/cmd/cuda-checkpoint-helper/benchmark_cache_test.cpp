/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark_cache.h"

#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <filesystem>
#include <iostream>
#include <string>

namespace benchmark = cuda_checkpoint_benchmark;
namespace transfer = cuda_checkpoint_transfer;

namespace {

bool
Check(bool condition, const std::string& message)
{
  if (!condition) {
    std::cerr << message << "\n";
  }
  return condition;
}

bool
TestProfileParsing()
{
  benchmark::CacheProfile profile = benchmark::CacheProfile::kBufferedWarm;
  return Check(
             benchmark::ParseCacheProfile("buffered-warm", &profile) &&
                 profile == benchmark::CacheProfile::kBufferedWarm,
             "buffered-warm profile was not parsed") &&
         Check(
             benchmark::ParseCacheProfile("client-evicted", &profile) &&
                 profile == benchmark::CacheProfile::kClientEvicted,
             "client-evicted profile was not parsed") &&
         Check(!benchmark::ParseCacheProfile("cold", &profile), "ambiguous cold profile was accepted") &&
         Check(!benchmark::ParseCacheProfile("buffered-warm", nullptr), "null profile output was accepted") &&
         Check(
             benchmark::CacheProfileName(benchmark::CacheProfile::kBufferedWarm) == "buffered-warm",
             "buffered-warm profile name is wrong") &&
         Check(
             benchmark::CacheProfileName(benchmark::CacheProfile::kClientEvicted) == "client-evicted",
             "client-evicted profile name is wrong");
}

bool
TestClientEvictionAdvice()
{
  std::array<char, 4096> data{};
  std::string path = "/tmp/cuda-checkpoint-benchmark-cache-test-XXXXXX";
  const int descriptor = mkstemp(path.data());
  if (!Check(descriptor >= 0, "mkstemp failed")) {
    return false;
  }
  const bool prepared =
      Check(write(descriptor, data.data(), data.size()) == static_cast<ssize_t>(data.size()), "test write failed") &&
      Check(fsync(descriptor) == 0, "test fsync failed") && Check(close(descriptor) == 0, "test close failed");
  if (!prepared) {
    (void)close(descriptor);
    (void)unlink(path.c_str());
    return false;
  }

  const transfer::StorageLayout layout{{{path, data.size()}}, {{0, data.size(), 0, 0}}};
  const benchmark::CacheEvictionResult result = benchmark::EvictClientFileCache(layout);
  (void)unlink(path.c_str());
  return Check(result.files_requested == 1, "eviction did not report the requested file") &&
         Check(result.files_advised == 1, "kernel did not accept eviction advice for the test file") &&
         Check(result.errors.empty(), "successful eviction advice reported an error");
}

bool
TestClientEvictionFailureIsReported()
{
  const transfer::StorageLayout layout{
      {{"/tmp/cuda-checkpoint-benchmark-cache-test-missing", 4096}},
      {{0, 4096, 0, 0}},
  };
  const benchmark::CacheEvictionResult result = benchmark::EvictClientFileCache(layout);
  return Check(result.files_requested == 1, "failed eviction did not report the requested file") &&
         Check(result.files_advised == 0, "failed eviction reported accepted advice") &&
         Check(result.errors.size() == 1, "failed eviction did not preserve its error");
}

}  // namespace

int
main()
{
  if (!TestProfileParsing() || !TestClientEvictionAdvice() || !TestClientEvictionFailureIsReported()) {
    return 1;
  }
  std::cout << "CUDA benchmark cache-profile tests passed\n";
  return 0;
}
