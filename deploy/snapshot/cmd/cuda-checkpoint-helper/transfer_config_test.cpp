/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "transfer_config.h"

#include <iostream>
#include <limits>
#include <string>
#include <vector>

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
TestOptionParsingAndBounds()
{
  size_t parsed = 0;
  std::string error;
  return Check(transfer::ParseSize("67108864", &parsed) && parsed == 67108864, "valid size was not parsed") &&
         Check(!transfer::ParseSize("-1", &parsed), "negative size was accepted") &&
         Check(!transfer::ParseSize("64MiB", &parsed), "size with trailing data was accepted") &&
         Check(
             transfer::ValidateTransferOptions({transfer::kDefaultBufferCount, transfer::kDefaultChunkBytes}, &error),
             error) &&
         Check(!transfer::ValidateTransferOptions({0, transfer::kDefaultChunkBytes}, &error), "zero slots accepted") &&
         Check(
             !transfer::ValidateTransferOptions({1, transfer::kMinimumChunkBytes + 1}, &error),
             "unaligned chunk accepted") &&
         Check(
             !transfer::ValidateTransferOptions({transfer::kMaximumBufferCount, transfer::kMaximumChunkBytes}, &error),
             "more than 1 GiB per device was accepted");
}

bool
TestPinnedMemoryCalculation()
{
  size_t bytes = 0;
  std::string error;
  return Check(transfer::CalculatePinnedBytes(8, {1, transfer::kDefaultChunkBytes}, &bytes, &error), error) &&
         Check(bytes == 512ULL * 1024ULL * 1024ULL, "default eight-device pinned bytes are wrong") &&
         Check(
             !transfer::CalculatePinnedBytes(
                 3, {transfer::kMaximumBufferCount, 128ULL * 1024ULL * 1024ULL}, &bytes, &error),
             "operation pinned-memory cap was not enforced") &&
         Check(
             !transfer::CalculatePinnedBytes(
                 std::numeric_limits<size_t>::max(), {1, transfer::kDefaultChunkBytes}, &bytes, &error),
             "pinned-memory overflow was accepted");
}

bool
TestChunkRingAndShardedLayout()
{
  transfer::StorageLayout storage;
  std::vector<transfer::TransferChunk> chunks;
  std::string error;
  constexpr size_t kMiB = 1024ULL * 1024ULL;
  if (!Check(transfer::BuildContiguousStorageLayout("/tmp/extent", 130 * kMiB, 2, &storage, &error), error) ||
      !Check(storage.files.size() == 2 && storage.ranges.size() == 2, "expected two files and ranges") ||
      !Check(
          storage.files[0].size == 65 * kMiB && storage.files[0].path == "/tmp/extent.part-0000" &&
              storage.files[1].size == 65 * kMiB && storage.files[1].path == "/tmp/extent.part-0001" &&
              storage.ranges[0].logical_offset == 0 && storage.ranges[0].size == 65 * kMiB &&
              storage.ranges[0].file_index == 0 && storage.ranges[1].logical_offset == 65 * kMiB &&
              storage.ranges[1].size == 65 * kMiB && storage.ranges[1].file_index == 1,
          "contiguous storage range mapping is wrong") ||
      !Check(transfer::BuildTransferChunks(130 * kMiB, storage, {2, 64 * kMiB}, &chunks, &error), error) ||
      !Check(chunks.size() == 4, "expected shard boundaries to split the four chunks")) {
    return false;
  }

  return Check(
             chunks[0].logical_offset == 0 && chunks[0].size == 64 * kMiB && chunks[0].file_index == 0 &&
                 chunks[0].file_offset == 0 && chunks[0].slot_index == 0,
             "first chunk is wrong") &&
         Check(
             chunks[1].logical_offset == 64 * kMiB && chunks[1].size == kMiB && chunks[1].file_index == 0 &&
                 chunks[1].file_offset == 64 * kMiB && chunks[1].slot_index == 1,
             "first shard tail is wrong") &&
         Check(
             chunks[2].logical_offset == 65 * kMiB && chunks[2].size == 64 * kMiB && chunks[2].file_index == 1 &&
                 chunks[2].file_offset == 0 && chunks[2].slot_index == 0,
             "second shard first chunk is wrong") &&
         Check(
             chunks[3].logical_offset == 129 * kMiB && chunks[3].size == kMiB && chunks[3].file_index == 1 &&
                 chunks[3].file_offset == 64 * kMiB && chunks[3].slot_index == 1,
             "second shard tail is wrong");
}

bool
TestLayoutGapsAndOverflowRejected()
{
  std::vector<transfer::TransferChunk> chunks;
  std::string error;
  const transfer::StorageLayout gap_layout{
      {{"/tmp/a", 4096}, {"/tmp/b", 4095}},
      {{0, 4096, 0, 0}, {4097, 4095, 1, 0}},
  };
  const transfer::StorageLayout overflow_layout{
      {{"/tmp/a", std::numeric_limits<size_t>::max()}},
      {{0, std::numeric_limits<size_t>::max(), 0, 1}},
  };
  const size_t excessive_size = (transfer::kMaximumTransferChunkCount + 1) * transfer::kMinimumChunkBytes;
  const transfer::StorageLayout excessive_layout{
      {{"/tmp/a", excessive_size}},
      {{0, excessive_size, 0, 0}},
  };
  const transfer::StorageLayout duplicate_file_layout{
      {{"/tmp/a", 4096}, {"/tmp/../tmp/a", 4096}},
      {{0, 4096, 0, 0}, {4096, 4096, 1, 0}},
  };
  return Check(
             !transfer::BuildTransferChunks(8192, gap_layout, {1, transfer::kMinimumChunkBytes}, &chunks, &error),
             "logical layout gap was accepted") &&
         Check(
             !transfer::BuildTransferChunks(
                 std::numeric_limits<size_t>::max(), overflow_layout, {1, transfer::kMinimumChunkBytes}, &chunks,
                 &error),
             "file offset overflow was accepted") &&
         Check(
             !transfer::BuildTransferChunks(
                 excessive_size, excessive_layout, {1, transfer::kMinimumChunkBytes}, &chunks, &error),
             "excessive transfer chunk count was accepted") &&
         Check(
             !transfer::BuildTransferChunks(
                 8192, duplicate_file_layout, {1, transfer::kMinimumChunkBytes}, &chunks, &error),
             "duplicate physical file path was accepted");
}

bool
TestJsonEscaping()
{
  return Check(
      transfer::JsonEscape("file\\name\n\"value\"") == "\"file\\\\name\\n\\\"value\\\"\"", "JSON escaping is wrong");
}

}  // namespace

int
main()
{
  if (!TestOptionParsingAndBounds() || !TestPinnedMemoryCalculation() || !TestChunkRingAndShardedLayout() ||
      !TestLayoutGapsAndOverflowRejected() || !TestJsonEscaping()) {
    return 1;
  }
  std::cout << "CUDA transfer configuration tests passed\n";
  return 0;
}
