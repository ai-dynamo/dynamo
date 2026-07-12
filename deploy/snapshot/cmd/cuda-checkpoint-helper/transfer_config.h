/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <string_view>
#include <vector>

namespace cuda_checkpoint_transfer {

enum class TransferOperation {
  kCheckpoint,
  kRestore,
};

constexpr size_t kDefaultBufferCount = 1;
constexpr size_t kDefaultChunkBytes = 64ULL * 1024ULL * 1024ULL;
constexpr size_t kMinimumChunkBytes = 1ULL * 1024ULL * 1024ULL;
constexpr size_t kMaximumChunkBytes = 256ULL * 1024ULL * 1024ULL;
constexpr size_t kMaximumBufferCount = 8;
constexpr size_t kMaximumPinnedBytesPerDevice = 1ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr size_t kMaximumPinnedBytesPerOperation = 2ULL * 1024ULL * 1024ULL * 1024ULL;
constexpr size_t kMaximumTransferChunkCount = 1024ULL * 1024ULL;
constexpr size_t kBufferAlignment = 4096;

struct TransferOptions {
  size_t buffer_count = kDefaultBufferCount;
  size_t chunk_bytes = kDefaultChunkBytes;
};

struct StorageFile {
  std::filesystem::path path;
  size_t size = 0;
};

struct StorageRange {
  size_t logical_offset = 0;
  size_t size = 0;
  size_t file_index = 0;
  size_t file_offset = 0;
};

struct StorageLayout {
  std::vector<StorageFile> files;
  std::vector<StorageRange> ranges;
};

struct TransferChunk {
  size_t logical_offset;
  size_t size;
  size_t file_index;
  size_t file_offset;
  size_t slot_index;
};

bool ParseSize(std::string_view value, size_t* parsed);
bool ValidateTransferOptions(const TransferOptions& options, std::string* error);
bool CalculatePinnedBytes(size_t device_count, const TransferOptions& options, size_t* bytes, std::string* error);
int StorageFileOpenFlags(TransferOperation operation);
bool BuildTransferChunks(
    size_t extent_size, const StorageLayout& storage, const TransferOptions& options,
    std::vector<TransferChunk>* chunks, std::string* error);
bool BuildContiguousStorageLayout(
    const std::filesystem::path& base_path, size_t extent_size, size_t file_count, StorageLayout* storage,
    std::string* error);
std::string JsonEscape(std::string_view value);

}  // namespace cuda_checkpoint_transfer
