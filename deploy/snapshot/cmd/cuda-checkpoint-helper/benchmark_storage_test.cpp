/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "benchmark_storage.h"

#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <cstring>
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
CreateSizedFile(const std::filesystem::path& path, size_t size)
{
  const int descriptor = open(path.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600);
  if (descriptor < 0) {
    return false;
  }
  const bool success = ftruncate(descriptor, static_cast<off_t>(size)) == 0;
  (void)close(descriptor);
  return success;
}

bool
TestOptionSemantics()
{
  std::string error;
  return Check(benchmark::EffectiveWarmups(false, false, 1) == 1, "normal default warmups changed") &&
         Check(benchmark::EffectiveWarmups(true, false, 1) == 0, "existing-file default warmups are not zero") &&
         Check(benchmark::EffectiveWarmups(true, true, 2) == 2, "explicit existing-file warmups were discarded") &&
         Check(benchmark::ValidateBenchmarkStorageOptions({true, true, false, 0, false, {}}, &error), error) &&
         Check(benchmark::ValidateBenchmarkStorageOptions({true, false, false, 1, true, {}}, &error), error) &&
         Check(benchmark::ValidateBenchmarkStorageOptions({false, false, false, 1, false, {}}, &error), error) &&
         Check(
             !benchmark::ValidateBenchmarkStorageOptions({false, true, false, 1, false, {}}, &error),
             "checkpoint accepted --existing-file") &&
         Check(
             !benchmark::ValidateBenchmarkStorageOptions(
                 {false, false, false, 1, true, benchmark::CacheProfile::kClientEvicted}, &error),
             "checkpoint accepted explicit restore-only cache profile");
}

bool
CheckCacheState(
    bool restore, bool existing_file, size_t warmups, size_t iteration, benchmark::CacheProfile profile,
    std::string_view want_residency, bool want_preconditioned, const std::string& message)
{
  const benchmark::ClientCacheState state =
      benchmark::ClientCacheStateBeforeMeasuredIteration(restore, existing_file, warmups, iteration, profile);
  return Check(state.residency == want_residency && state.preconditioned_by_invocation == want_preconditioned, message);
}

bool
TestPerIterationCacheStatePolicy()
{
  return CheckCacheState(
             true, true, 0, 0, benchmark::CacheProfile::kBufferedWarm, "unknown-not-preconditioned", false,
             "unread existing file was reported as preconditioned") &&
         CheckCacheState(
             true, true, 0, 1, benchmark::CacheProfile::kBufferedWarm, "buffered-warm", true,
             "first measured restore did not establish iteration 1 cache state") &&
         CheckCacheState(
             true, true, 0, 2, benchmark::CacheProfile::kBufferedWarm, "buffered-warm", true,
             "prior measured restores did not preserve buffered-warm state") &&
         CheckCacheState(
             true, true, 1, 0, benchmark::CacheProfile::kBufferedWarm, "buffered-warm", true,
             "explicit restore warmup did not establish buffered-warm residency") &&
         CheckCacheState(
             true, false, 0, 0, benchmark::CacheProfile::kBufferedWarm, "buffered-warm", true,
             "invocation-prepared storage was not reported as buffered-warm") &&
         CheckCacheState(
             true, true, 0, 0, benchmark::CacheProfile::kClientEvicted, "unknown-after-advice", false,
             "client-evicted iteration 0 cache state is wrong") &&
         CheckCacheState(
             true, true, 2, 3, benchmark::CacheProfile::kClientEvicted, "unknown-after-advice", false,
             "client-evicted later iteration was reported as preconditioned") &&
         CheckCacheState(
             false, false, 1, 2, benchmark::CacheProfile::kBufferedWarm, "not-applicable", false,
             "checkpoint cache state is applicable");
}

bool
TestPerIterationCacheStateJSON()
{
  constexpr std::array<std::string_view, 3> kExpected{
      "\"client_cache_preconditioned_by_invocation\":false,"
      "\"client_cache_residency\":\"unknown-not-preconditioned\"",
      "\"client_cache_preconditioned_by_invocation\":true,"
      "\"client_cache_residency\":\"buffered-warm\"",
      "\"client_cache_preconditioned_by_invocation\":true,"
      "\"client_cache_residency\":\"buffered-warm\"",
  };
  for (size_t iteration = 0; iteration < kExpected.size(); ++iteration) {
    const benchmark::ClientCacheState state = benchmark::ClientCacheStateBeforeMeasuredIteration(
        true, true, 0, iteration, benchmark::CacheProfile::kBufferedWarm);
    if (!Check(
            benchmark::ClientCacheStateJSON(state) == kExpected[iteration],
            "existing-file JSON cache attribution is wrong at measured iteration " + std::to_string(iteration))) {
      return false;
    }
  }
  return true;
}

bool
TestPatternVectorsAndShardContinuity()
{
  constexpr std::array<unsigned char, 24> kExpected{
      0xaf, 0xcd, 0x1d, 0x7b, 0x39, 0xa8, 0x20, 0xe2, 0xc1, 0x5c, 0x02, 0x89,
      0xec, 0x2d, 0x0a, 0x91, 0xce, 0x56, 0x97, 0x1c, 0xde, 0x35, 0x58, 0x97,
  };
  std::array<unsigned char, kExpected.size()> complete{};
  benchmark::FillPattern(complete.data(), complete.size(), 0);
  if (!Check(complete == kExpected, "fixed little-endian pattern vector mismatch")) {
    return false;
  }

  std::array<unsigned char, 7> first_shard{};
  std::array<unsigned char, 17> second_shard{};
  benchmark::FillPattern(first_shard.data(), first_shard.size(), 0);
  benchmark::FillPattern(second_shard.data(), second_shard.size(), first_shard.size());
  return Check(
      std::memcmp(first_shard.data(), complete.data(), first_shard.size()) == 0 &&
          std::memcmp(second_shard.data(), complete.data() + first_shard.size(), second_shard.size()) == 0,
      "second shard restarted instead of continuing at its logical offset");
}

bool
TestExistingLayoutValidation()
{
  char directory_template[] = "/tmp/cuda-benchmark-storage-test-XXXXXX";
  const char* directory = mkdtemp(directory_template);
  if (!Check(directory != nullptr, "mkdtemp failed")) {
    return false;
  }
  const std::filesystem::path base = std::filesystem::path(directory) / "extent";
  transfer::StorageLayout layout;
  std::string error;
  bool success =
      Check(transfer::BuildContiguousStorageLayout(base, 9, 2, &layout, &error), error) &&
      Check(!benchmark::ValidateExistingStorageLayout(layout, &error), "missing shards were accepted") &&
      Check(CreateSizedFile(layout.files[0].path, layout.files[0].size), "failed to create first shard") &&
      Check(CreateSizedFile(layout.files[1].path, layout.files[1].size + 1), "failed to create wrong-size shard") &&
      Check(!benchmark::ValidateExistingStorageLayout(layout, &error), "wrong-size shard was accepted") &&
      Check(CreateSizedFile(layout.files[1].path, layout.files[1].size), "failed to resize second shard") &&
      Check(benchmark::ValidateExistingStorageLayout(layout, &error), error);

  const std::filesystem::path target = std::filesystem::path(directory) / "target";
  const std::filesystem::path link = std::filesystem::path(directory) / "link";
  success = Check(CreateSizedFile(target, 9), "failed to create symlink target") && success;
  if (symlink(target.c_str(), link.c_str()) != 0) {
    success = Check(false, "failed to create symlink") && success;
  } else {
    const transfer::StorageLayout symlink_layout{{{link, 9}}, {{0, 9, 0, 0}}};
    success =
        Check(!benchmark::ValidateExistingStorageLayout(symlink_layout, &error), "symlink was accepted") && success;
  }
  std::filesystem::remove_all(directory);
  return success;
}

}  // namespace

int
main()
{
  return TestOptionSemantics() && TestPerIterationCacheStatePolicy() && TestPerIterationCacheStateJSON() &&
                 TestPatternVectorsAndShardContinuity() && TestExistingLayoutValidation()
             ? 0
             : 1;
}
