/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "storage_manifest.h"

#include <unistd.h>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace storage = cuda_checkpoint_storage;

namespace {

constexpr const char* kSourceA = "GPU-00000000-0000-0000-0000-00000000000a";
constexpr const char* kSourceB = "GPU-00000000-0000-0000-0000-00000000000b";
constexpr const char* kSourceFallback = "GPU-00000000-0000-0000-0000-00000000000c";
constexpr const char* kDestinationA = "GPU-10000000-0000-0000-0000-00000000000a";
constexpr const char* kDestinationB = "GPU-10000000-0000-0000-0000-00000000000b";
constexpr const char* kDestinationFallback = "GPU-10000000-0000-0000-0000-00000000000c";

bool
Check(bool condition, const std::string& message)
{
  if (!condition) {
    std::cerr << message << "\n";
  }
  return condition;
}

bool
TestEqualSizeNonOrderPreservingMap()
{
  const std::vector<storage::ManifestExtent> extents{
      {kSourceA, 4096, storage::DeviceFilename(0)},
      {kSourceB, 4096, storage::DeviceFilename(1)},
  };
  // CUDA returns destination B first. Equal sizes must not permit an
  // index-based A/B swap.
  const std::vector<storage::DeviceExtent> destinations{
      {kDestinationB, 4096},
      {kDestinationA, 4096},
  };
  const std::vector<storage::DevicePair> pairs{
      {kSourceA, kDestinationA},
      {kSourceB, kDestinationB},
      // This GPU is assigned to the container but is not exported by this
      // process. Its explicit fallback pair must not invalidate the subset.
      {kSourceFallback, kDestinationFallback},
  };

  std::vector<storage::TransferJob> jobs;
  std::string error;
  return Check(storage::BuildTransferJobs(extents, destinations, pairs, &jobs, &error), error) &&
         Check(jobs.size() == 2, "expected two transfer jobs") &&
         Check(
             jobs[0].device_index == 0 && jobs[0].extent_index == 1,
             "destination B was not matched to source B's deterministic file") &&
         Check(
             jobs[1].device_index == 1 && jobs[1].extent_index == 0,
             "destination A was not matched to source A's deterministic file");
}

bool
TestEmptyV2Manifest()
{
  char path[] = "/tmp/cuda-storage-manifest-test-XXXXXX";
  const char* directory = mkdtemp(path);
  if (!Check(directory != nullptr, "mkdtemp failed")) {
    return false;
  }

  std::string error;
  std::vector<storage::ManifestExtent> loaded;
  std::vector<storage::TransferJob> jobs;
  const bool result = Check(storage::WriteManifest(directory, {}, &error), error) &&
                      Check(storage::ReadManifest(directory, &loaded, &error), error) &&
                      Check(loaded.empty(), "empty v2 manifest did not round-trip") &&
                      Check(storage::BuildTransferJobs(loaded, {}, {}, &jobs, &error), error) &&
                      Check(jobs.empty(), "zero-device restore produced transfer jobs");
  std::error_code ignored;
  std::filesystem::remove_all(directory, ignored);
  return result;
}

bool
TestNonemptyV2ManifestRoundTrip()
{
  char path[] = "/tmp/cuda-storage-manifest-roundtrip-test-XXXXXX";
  const char* directory = mkdtemp(path);
  if (!Check(directory != nullptr, "mkdtemp failed")) {
    return false;
  }
  const std::vector<storage::ManifestExtent> extents{
      {kSourceA, 4096, storage::DeviceFilename(0)},
      {kSourceB, 8192, storage::DeviceFilename(1)},
  };

  std::string error;
  std::vector<storage::ManifestExtent> loaded;
  const bool result = Check(storage::WriteManifest(directory, extents, &error), error) &&
                      Check(storage::ReadManifest(directory, &loaded, &error), error) &&
                      Check(
                          loaded.size() == 2 && loaded[0].source_uuid == kSourceA && loaded[0].size == 4096 &&
                              loaded[0].filename == "device-0000.bin" && loaded[1].source_uuid == kSourceB &&
                              loaded[1].size == 8192 && loaded[1].filename == "device-0001.bin",
                          "nonempty v2 manifest did not preserve UUID, size, and filename");
  std::error_code ignored;
  std::filesystem::remove_all(directory, ignored);
  return result;
}

bool
TestV1Rejected()
{
  char path[] = "/tmp/cuda-storage-manifest-v1-test-XXXXXX";
  const char* directory = mkdtemp(path);
  if (!Check(directory != nullptr, "mkdtemp failed")) {
    return false;
  }
  {
    std::ofstream output(std::filesystem::path(directory) / storage::kManifestName);
    output << "version 1\n"
              "device_count 1\n"
              "device 0 4096 device-0000.bin\n";
  }

  std::vector<storage::ManifestExtent> extents;
  std::string error;
  const bool result =
      Check(!storage::ReadManifest(directory, &extents, &error), "unsafe v1 manifest was accepted") &&
      Check(error.find("version 1") != std::string::npos, "v1 rejection did not identify the unsafe version");
  std::error_code ignored;
  std::filesystem::remove_all(directory, ignored);
  return result;
}

bool
TestUnconsumedExtentRejected()
{
  const std::vector<storage::ManifestExtent> extents{
      {kSourceA, 4096, storage::DeviceFilename(0)},
      {kSourceB, 4096, storage::DeviceFilename(1)},
  };
  const std::vector<storage::DeviceExtent> destinations{{kSourceA, 4096}};
  std::vector<storage::TransferJob> jobs;
  std::string error;
  return Check(
      !storage::BuildTransferJobs(extents, destinations, {}, &jobs, &error),
      "restore accepted an unconsumed saved extent");
}

bool
TestUnsafeMappingsRejected()
{
  const std::vector<storage::ManifestExtent> extents{
      {kSourceA, 4096, storage::DeviceFilename(0)},
      {kSourceB, 4096, storage::DeviceFilename(1)},
  };
  std::vector<storage::TransferJob> jobs;
  std::string error;

  if (!Check(
          !storage::BuildTransferJobs(
              extents, {{kDestinationA, 4096}, {kDestinationB, 4096}}, {{kSourceA, kDestinationA}}, &jobs, &error),
          "restore accepted a destination missing from the device map")) {
    return false;
  }
  if (!Check(
          !storage::BuildTransferJobs(
              extents, {{kDestinationA, 4096}, {kDestinationB, 4096}},
              {{kSourceA, kDestinationA}, {kSourceB, kDestinationA}}, &jobs, &error),
          "restore accepted an ambiguous destination UUID")) {
    return false;
  }
  if (!Check(
          !storage::BuildTransferJobs(
              extents, {{kDestinationA, 4096}, {kDestinationB, 8192}},
              {{kSourceA, kDestinationA}, {kSourceB, kDestinationB}}, &jobs, &error),
          "restore accepted a UUID-matched extent with the wrong size")) {
    return false;
  }
  if (!Check(
          !storage::BuildTransferJobs(
              {{kSourceA, 4096, storage::DeviceFilename(0)}, {kSourceA, 4096, storage::DeviceFilename(1)}},
              {{kSourceA, 4096}, {kSourceB, 4096}}, {}, &jobs, &error),
          "restore accepted duplicate saved source UUIDs")) {
    return false;
  }
  return Check(
      !storage::BuildTransferJobs(extents, {{kSourceA, 4096}, {kSourceA, 4096}}, {}, &jobs, &error),
      "restore accepted duplicate destination UUIDs");
}

bool
TestDuplicateCheckpointUUIDRejected()
{
  std::vector<storage::ManifestExtent> extents;
  std::string error;
  return Check(
      !storage::BuildCheckpointManifest({{kSourceA, 4096}, {kSourceA, 4096}}, &extents, &error),
      "checkpoint accepted duplicate source UUIDs");
}

}  // namespace

int
main()
{
  if (!TestEqualSizeNonOrderPreservingMap() || !TestEmptyV2Manifest() || !TestNonemptyV2ManifestRoundTrip() ||
      !TestV1Rejected() || !TestUnconsumedExtentRejected() || !TestUnsafeMappingsRejected() ||
      !TestDuplicateCheckpointUUIDRejected()) {
    return 1;
  }
  std::cout << "cuda checkpoint storage manifest tests passed\n";
  return 0;
}
