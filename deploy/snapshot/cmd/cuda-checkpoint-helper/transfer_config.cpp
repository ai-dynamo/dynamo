/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "transfer_config.h"

#include <algorithm>
#include <charconv>
#include <cstdio>
#include <limits>
#include <set>
#include <utility>

namespace cuda_checkpoint_transfer {
namespace {

bool
CheckedAdd(size_t left, size_t right, size_t* result)
{
  if (right > std::numeric_limits<size_t>::max() - left) {
    return false;
  }
  *result = left + right;
  return true;
}

bool
CheckedMultiply(size_t left, size_t right, size_t* result)
{
  if (left != 0 && right > std::numeric_limits<size_t>::max() / left) {
    return false;
  }
  *result = left * right;
  return true;
}

}  // namespace

bool
ParseSize(std::string_view value, size_t* parsed)
{
  if (parsed == nullptr || value.empty()) {
    return false;
  }
  size_t result = 0;
  const auto conversion = std::from_chars(value.data(), value.data() + value.size(), result);
  if (conversion.ec != std::errc{} || conversion.ptr != value.data() + value.size()) {
    return false;
  }
  *parsed = result;
  return true;
}

bool
ValidateTransferOptions(const TransferOptions& options, std::string* error)
{
  if (error == nullptr) {
    return false;
  }
  if (options.buffer_count == 0 || options.buffer_count > kMaximumBufferCount) {
    *error = "transfer buffer count must be between 1 and " + std::to_string(kMaximumBufferCount);
    return false;
  }
  if (options.chunk_bytes < kMinimumChunkBytes || options.chunk_bytes > kMaximumChunkBytes ||
      options.chunk_bytes % kBufferAlignment != 0) {
    *error = "transfer chunk bytes must be a 4096-byte multiple between " + std::to_string(kMinimumChunkBytes) +
             " and " + std::to_string(kMaximumChunkBytes);
    return false;
  }
  size_t pinned_bytes = 0;
  if (!CheckedMultiply(options.buffer_count, options.chunk_bytes, &pinned_bytes) ||
      pinned_bytes > kMaximumPinnedBytesPerDevice) {
    *error = "transfer buffers exceed the 1 GiB per-device pinned-memory limit";
    return false;
  }
  return true;
}

bool
CalculatePinnedBytes(size_t device_count, const TransferOptions& options, size_t* bytes, std::string* error)
{
  if (bytes == nullptr || error == nullptr || !ValidateTransferOptions(options, error)) {
    return false;
  }
  size_t per_device = 0;
  if (!CheckedMultiply(options.buffer_count, options.chunk_bytes, &per_device) ||
      !CheckedMultiply(device_count, per_device, bytes)) {
    *error = "pinned-memory size calculation overflow";
    return false;
  }
  if (*bytes > kMaximumPinnedBytesPerOperation) {
    *error = "transfer buffers exceed the 2 GiB per-operation pinned-memory limit";
    return false;
  }
  return true;
}

bool
BuildTransferChunks(
    size_t extent_size, const StorageLayout& storage, const TransferOptions& options,
    std::vector<TransferChunk>* chunks, std::string* error)
{
  if (chunks == nullptr || error == nullptr || !ValidateTransferOptions(options, error)) {
    return false;
  }
  if (extent_size == 0 || storage.files.empty() || storage.ranges.empty()) {
    *error = "transfer extent and storage layout must be nonempty";
    return false;
  }

  std::set<std::filesystem::path> paths;
  for (const auto& file : storage.files) {
    if (file.path.empty() || !file.path.is_absolute() || file.size == 0) {
      *error = "storage files must have absolute paths and nonzero sizes";
      return false;
    }
    if (!paths.insert(file.path.lexically_normal()).second) {
      *error = "storage file paths must be unique";
      return false;
    }
  }

  chunks->clear();
  size_t expected_logical_offset = 0;
  size_t chunk_index = 0;
  std::vector<std::vector<std::pair<size_t, size_t>>> physical_ranges(storage.files.size());
  for (const auto& range : storage.ranges) {
    if (range.size == 0 || range.logical_offset != expected_logical_offset ||
        range.file_index >= storage.files.size()) {
      *error = "storage ranges must be nonempty and exactly cover the logical extent in order";
      return false;
    }
    size_t logical_end = 0;
    size_t file_end = 0;
    if (!CheckedAdd(range.logical_offset, range.size, &logical_end) ||
        !CheckedAdd(range.file_offset, range.size, &file_end)) {
      *error = "storage range offset calculation overflow";
      return false;
    }
    if (logical_end > extent_size || file_end > storage.files[range.file_index].size) {
      *error = "storage range exceeds its logical extent or physical file";
      return false;
    }
    physical_ranges[range.file_index].push_back({range.file_offset, file_end});

    const size_t range_chunk_count =
        range.size / options.chunk_bytes + static_cast<size_t>(range.size % options.chunk_bytes != 0);
    if (range_chunk_count > kMaximumTransferChunkCount - chunks->size()) {
      *error = "transfer layout has too many chunks";
      return false;
    }
    for (size_t range_offset = 0; range_offset < range.size;) {
      const size_t length = std::min(options.chunk_bytes, range.size - range_offset);
      chunks->push_back(
          {range.logical_offset + range_offset, length, range.file_index, range.file_offset + range_offset,
           chunk_index % options.buffer_count});
      range_offset += length;
      ++chunk_index;
    }
    expected_logical_offset = logical_end;
  }
  if (expected_logical_offset != extent_size) {
    *error = "storage ranges do not cover the complete logical extent";
    return false;
  }
  for (auto& ranges : physical_ranges) {
    std::sort(ranges.begin(), ranges.end());
  }
  for (size_t file_index = 0; file_index < physical_ranges.size(); ++file_index) {
    const auto& ranges = physical_ranges[file_index];
    if (ranges.empty() || ranges.front().first != 0) {
      *error = "storage ranges must exactly cover every physical file";
      return false;
    }
    for (size_t index = 1; index < ranges.size(); ++index) {
      if (ranges[index].first != ranges[index - 1].second) {
        *error = "storage ranges must not overlap or leave gaps within a physical file";
        return false;
      }
    }
    if (ranges.back().second != storage.files[file_index].size) {
      *error = "storage ranges must exactly cover every physical file";
      return false;
    }
  }
  return true;
}

bool
BuildContiguousStorageLayout(
    const std::filesystem::path& base_path, size_t extent_size, size_t file_count, StorageLayout* storage,
    std::string* error)
{
  if (storage == nullptr || error == nullptr) {
    return false;
  }
  if (base_path.empty() || extent_size == 0 || file_count == 0 || file_count > 64 || file_count > extent_size) {
    *error = "file count must be between 1 and 64 and cannot exceed the transfer byte count";
    return false;
  }

  storage->files.clear();
  storage->ranges.clear();
  storage->files.reserve(file_count);
  storage->ranges.reserve(file_count);
  const size_t base_size = extent_size / file_count;
  const size_t remainder = extent_size % file_count;
  size_t logical_offset = 0;
  for (size_t index = 0; index < file_count; ++index) {
    const size_t size = base_size + (index < remainder ? 1 : 0);
    std::filesystem::path path = base_path;
    if (file_count > 1) {
      char suffix[32];
      std::snprintf(suffix, sizeof(suffix), ".part-%04zu", index);
      path += suffix;
    }
    storage->files.push_back({std::move(path), size});
    storage->ranges.push_back({logical_offset, size, index, 0});
    logical_offset += size;
  }
  return true;
}

std::string
JsonEscape(std::string_view value)
{
  static constexpr char kHex[] = "0123456789abcdef";
  std::string result;
  result.reserve(value.size() + 2);
  result.push_back('"');
  for (const unsigned char byte : value) {
    switch (byte) {
      case '"':
        result += "\\\"";
        break;
      case '\\':
        result += "\\\\";
        break;
      case '\b':
        result += "\\b";
        break;
      case '\f':
        result += "\\f";
        break;
      case '\n':
        result += "\\n";
        break;
      case '\r':
        result += "\\r";
        break;
      case '\t':
        result += "\\t";
        break;
      default:
        if (byte < 0x20) {
          result += "\\u00";
          result.push_back(kHex[byte >> 4]);
          result.push_back(kHex[byte & 0x0f]);
        } else {
          result.push_back(static_cast<char>(byte));
        }
    }
  }
  result.push_back('"');
  return result;
}

}  // namespace cuda_checkpoint_transfer
