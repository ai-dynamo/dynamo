/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "storage_manifest.h"

#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

namespace cuda_checkpoint_storage {
namespace {

constexpr size_t kMaximumDeviceCount = 1024;

class FileDescriptor {
 public:
  explicit FileDescriptor(int fd = -1) : fd_(fd) {}
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;

  ~FileDescriptor()
  {
    if (fd_ >= 0) {
      (void)close(fd_);
    }
  }

  int get() const { return fd_; }

  bool Close()
  {
    if (fd_ < 0) {
      return true;
    }
    const int fd = fd_;
    fd_ = -1;
    return close(fd) == 0;
  }

 private:
  int fd_;
};

int
HexValue(char value)
{
  if (value >= '0' && value <= '9') {
    return value - '0';
  }
  if (value >= 'a' && value <= 'f') {
    return value - 'a' + 10;
  }
  if (value >= 'A' && value <= 'F') {
    return value - 'A' + 10;
  }
  return -1;
}

bool
NormalizeExtent(const ManifestExtent& extent, size_t index, ManifestExtent* normalized, std::string* error)
{
  if (!CanonicalizeGPUUUID(extent.source_uuid, &normalized->source_uuid)) {
    *error = "invalid source GPU UUID in helper manifest";
    return false;
  }
  if (extent.size == 0) {
    *error = "zero-sized extent in helper manifest";
    return false;
  }
  normalized->size = extent.size;
  normalized->filename = extent.filename;
  if (normalized->filename != DeviceFilename(index)) {
    *error = "invalid deterministic extent filename in helper manifest";
    return false;
  }
  return true;
}

bool
NormalizeManifest(
    const std::vector<ManifestExtent>& extents, std::vector<ManifestExtent>* normalized, std::string* error)
{
  if (extents.size() > kMaximumDeviceCount) {
    *error = "helper manifest has too many device extents";
    return false;
  }
  normalized->clear();
  normalized->reserve(extents.size());
  std::unordered_set<std::string> source_uuids;
  for (size_t index = 0; index < extents.size(); ++index) {
    ManifestExtent extent;
    if (!NormalizeExtent(extents[index], index, &extent, error)) {
      return false;
    }
    if (!source_uuids.insert(extent.source_uuid).second) {
      *error = "duplicate source GPU UUID in helper manifest";
      return false;
    }
    normalized->push_back(std::move(extent));
  }
  return true;
}

void
RemoveTemporaryManifest(const std::filesystem::path& directory)
{
  std::error_code ignored;
  std::filesystem::remove(directory / "manifest.txt.tmp", ignored);
}

}  // namespace

bool
ParseGPUUUID(std::string_view value, std::array<unsigned char, 16>* bytes_out)
{
  if (bytes_out == nullptr) {
    return false;
  }
  if (value.size() == 40) {
    if (value.substr(0, 4) != "GPU-") {
      return false;
    }
    value.remove_prefix(4);
  }
  if (value.size() != 36 || value[8] != '-' || value[13] != '-' || value[18] != '-' || value[23] != '-') {
    return false;
  }

  size_t input_index = 0;
  for (size_t byte_index = 0; byte_index < bytes_out->size(); ++byte_index) {
    if (input_index == 8 || input_index == 13 || input_index == 18 || input_index == 23) {
      ++input_index;
    }
    const int high = HexValue(value[input_index]);
    const int low = HexValue(value[input_index + 1]);
    if (high < 0 || low < 0) {
      return false;
    }
    (*bytes_out)[byte_index] = static_cast<unsigned char>((high << 4) | low);
    input_index += 2;
  }
  return input_index == value.size();
}

std::string
FormatGPUUUID(const std::array<unsigned char, 16>& bytes)
{
  static constexpr char kHex[] = "0123456789abcdef";
  std::string result = "GPU-";
  result.reserve(40);
  for (size_t index = 0; index < bytes.size(); ++index) {
    if (index == 4 || index == 6 || index == 8 || index == 10) {
      result.push_back('-');
    }
    result.push_back(kHex[bytes[index] >> 4]);
    result.push_back(kHex[bytes[index] & 0x0f]);
  }
  return result;
}

bool
CanonicalizeGPUUUID(std::string_view value, std::string* canonical_out)
{
  if (canonical_out == nullptr) {
    return false;
  }
  std::array<unsigned char, 16> bytes{};
  if (!ParseGPUUUID(value, &bytes)) {
    return false;
  }
  *canonical_out = FormatGPUUUID(bytes);
  return true;
}

std::string
DeviceFilename(size_t index)
{
  char filename[32];
  std::snprintf(filename, sizeof(filename), "device-%04zu.bin", index);
  return filename;
}

bool
BuildCheckpointManifest(
    const std::vector<DeviceExtent>& devices, std::vector<ManifestExtent>* extents, std::string* error)
{
  if (extents == nullptr || error == nullptr) {
    return false;
  }
  if (devices.size() > kMaximumDeviceCount) {
    *error = "CUDA returned too many custom storage devices";
    return false;
  }

  extents->clear();
  extents->reserve(devices.size());
  std::unordered_set<std::string> source_uuids;
  for (size_t index = 0; index < devices.size(); ++index) {
    std::string source_uuid;
    if (!CanonicalizeGPUUUID(devices[index].uuid, &source_uuid)) {
      *error = "CUDA returned an invalid source GPU UUID";
      return false;
    }
    if (!source_uuids.insert(source_uuid).second) {
      *error = "CUDA returned duplicate source GPU UUIDs";
      return false;
    }
    if (devices[index].size == 0) {
      *error = "CUDA returned a zero-sized custom storage extent";
      return false;
    }
    extents->push_back({std::move(source_uuid), devices[index].size, DeviceFilename(index)});
  }
  return true;
}

bool
BuildTransferJobs(
    const std::vector<ManifestExtent>& extents, const std::vector<DeviceExtent>& devices,
    const std::vector<DevicePair>& device_pairs, std::vector<TransferJob>* jobs, std::string* error)
{
  if (jobs == nullptr || error == nullptr) {
    return false;
  }

  std::vector<ManifestExtent> normalized_extents;
  if (!NormalizeManifest(extents, &normalized_extents, error)) {
    return false;
  }

  std::unordered_map<std::string, size_t> extent_by_source;
  for (size_t index = 0; index < normalized_extents.size(); ++index) {
    extent_by_source.emplace(normalized_extents[index].source_uuid, index);
  }

  std::unordered_map<std::string, std::string> source_to_destination;
  std::unordered_map<std::string, std::string> destination_to_source;
  for (const auto& pair : device_pairs) {
    std::string source;
    std::string destination;
    if (!CanonicalizeGPUUUID(pair.source_uuid, &source) || !CanonicalizeGPUUUID(pair.destination_uuid, &destination)) {
      *error = "invalid GPU UUID in CUDA device map";
      return false;
    }
    if (!source_to_destination.emplace(source, destination).second) {
      *error = "duplicate source GPU UUID in CUDA device map";
      return false;
    }
    if (!destination_to_source.emplace(destination, source).second) {
      *error = "ambiguous destination GPU UUID in CUDA device map";
      return false;
    }
  }

  jobs->clear();
  jobs->reserve(devices.size());
  std::unordered_set<std::string> destination_uuids;
  std::unordered_set<size_t> consumed_extents;
  for (size_t device_index = 0; device_index < devices.size(); ++device_index) {
    std::string destination;
    if (!CanonicalizeGPUUUID(devices[device_index].uuid, &destination)) {
      *error = "CUDA returned an invalid destination GPU UUID";
      return false;
    }
    if (!destination_uuids.insert(destination).second) {
      *error = "CUDA returned duplicate destination GPU UUIDs";
      return false;
    }
    if (devices[device_index].size == 0) {
      *error = "CUDA returned a zero-sized custom storage extent";
      return false;
    }

    std::string source = destination;
    if (!device_pairs.empty()) {
      const auto source_it = destination_to_source.find(destination);
      if (source_it == destination_to_source.end()) {
        *error = "destination GPU UUID is missing from the CUDA device map";
        return false;
      }
      source = source_it->second;
    }

    const auto extent_it = extent_by_source.find(source);
    if (extent_it == extent_by_source.end()) {
      *error = "no saved extent matches the source GPU UUID";
      return false;
    }
    const size_t extent_index = extent_it->second;
    if (!consumed_extents.insert(extent_index).second) {
      *error = "saved source GPU extent was matched more than once";
      return false;
    }
    if (normalized_extents[extent_index].size != devices[device_index].size) {
      *error = "saved extent size does not match the source GPU UUID";
      return false;
    }
    jobs->push_back({device_index, extent_index});
  }

  if (consumed_extents.size() != normalized_extents.size()) {
    *error = "one or more saved GPU extents were not consumed";
    return false;
  }
  return true;
}

bool
WriteManifest(const std::filesystem::path& directory, const std::vector<ManifestExtent>& extents, std::string* error)
{
  std::vector<ManifestExtent> normalized;
  if (!NormalizeManifest(extents, &normalized, error)) {
    return false;
  }

  const auto temporary = directory / "manifest.txt.tmp";
  const auto final = directory / kManifestName;
  RemoveTemporaryManifest(directory);
  {
    std::ofstream output(temporary, std::ios::out | std::ios::trunc);
    if (!output) {
      *error = "open helper manifest";
      return false;
    }
    output << "version 2\n";
    output << "device_count " << normalized.size() << "\n";
    for (size_t index = 0; index < normalized.size(); ++index) {
      output << "device " << index << " " << normalized[index].source_uuid << " " << normalized[index].size << " "
             << normalized[index].filename << "\n";
    }
    output.flush();
    if (!output) {
      RemoveTemporaryManifest(directory);
      *error = "write helper manifest";
      return false;
    }
  }
  if (chmod(temporary.c_str(), 0600) != 0) {
    RemoveTemporaryManifest(directory);
    *error = "set helper manifest permissions";
    return false;
  }

  FileDescriptor fd(open(temporary.c_str(), O_RDONLY | O_CLOEXEC));
  if (fd.get() < 0) {
    RemoveTemporaryManifest(directory);
    *error = "open helper manifest for fsync";
    return false;
  }
  if (fsync(fd.get()) != 0 || !fd.Close()) {
    RemoveTemporaryManifest(directory);
    *error = "fsync helper manifest";
    return false;
  }
  if (rename(temporary.c_str(), final.c_str()) != 0) {
    RemoveTemporaryManifest(directory);
    *error = "commit helper manifest";
    return false;
  }

  FileDescriptor directory_fd(open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
  if (directory_fd.get() < 0) {
    std::error_code ignored;
    std::filesystem::remove(final, ignored);
    *error = "open helper directory for fsync";
    return false;
  }
  if (fsync(directory_fd.get()) != 0 || !directory_fd.Close()) {
    std::error_code ignored;
    std::filesystem::remove(final, ignored);
    *error = "fsync helper directory";
    return false;
  }
  return true;
}

bool
ReadManifest(const std::filesystem::path& directory, std::vector<ManifestExtent>* extents, std::string* error)
{
  if (extents == nullptr || error == nullptr) {
    return false;
  }
  const auto manifest_path = directory / kManifestName;
  struct stat manifest_stat {};
  if (lstat(manifest_path.c_str(), &manifest_stat) != 0 || !S_ISREG(manifest_stat.st_mode)) {
    *error = "helper manifest is missing or not a regular file";
    return false;
  }

  std::ifstream input(manifest_path);
  std::string key;
  unsigned int version = 0;
  if (!(input >> key >> version) || key != "version") {
    *error = "invalid helper manifest version header";
    return false;
  }
  if (version == 1) {
    *error = "unsafe helper manifest version 1 is not supported";
    return false;
  }
  if (version != 2) {
    *error = "unsupported helper manifest version";
    return false;
  }

  size_t device_count = 0;
  if (!(input >> key >> device_count) || key != "device_count" || device_count > kMaximumDeviceCount) {
    *error = "invalid helper manifest device count";
    return false;
  }

  std::vector<ManifestExtent> parsed;
  parsed.reserve(device_count);
  for (size_t expected_index = 0; expected_index < device_count; ++expected_index) {
    size_t index = 0;
    ManifestExtent extent;
    if (!(input >> key >> index >> extent.source_uuid >> extent.size >> extent.filename) || key != "device" ||
        index != expected_index) {
      *error = "invalid helper manifest device entry";
      return false;
    }
    std::string canonical;
    if (!CanonicalizeGPUUUID(extent.source_uuid, &canonical) || canonical != extent.source_uuid) {
      *error = "helper manifest source GPU UUID is not canonical";
      return false;
    }
    parsed.push_back(std::move(extent));
  }
  std::string trailing;
  if (input >> trailing) {
    *error = "unexpected helper manifest data";
    return false;
  }

  std::vector<ManifestExtent> normalized;
  if (!NormalizeManifest(parsed, &normalized, error)) {
    return false;
  }
  *extents = std::move(normalized);
  return true;
}

bool
ValidateExtentFiles(
    const std::filesystem::path& directory, const std::vector<ManifestExtent>& extents, std::string* error)
{
  std::vector<ManifestExtent> normalized;
  if (!NormalizeManifest(extents, &normalized, error)) {
    return false;
  }
  for (const auto& extent : normalized) {
    struct stat extent_stat {};
    const auto path = directory / extent.filename;
    if (lstat(path.c_str(), &extent_stat) != 0 || !S_ISREG(extent_stat.st_mode) || extent_stat.st_size < 0 ||
        static_cast<size_t>(extent_stat.st_size) != extent.size) {
      *error = "extent file is missing, invalid, or has the wrong size";
      return false;
    }
  }
  return true;
}

bool
RemoveManifest(const std::filesystem::path& directory, std::string* error)
{
  bool removed = false;
  for (const char* name : {kManifestName, "manifest.txt.tmp"}) {
    if (unlink((directory / name).c_str()) == 0) {
      removed = true;
    } else if (errno != ENOENT) {
      *error = "remove stale helper manifest: " + std::string(std::strerror(errno));
      return false;
    }
  }
  if (removed) {
    FileDescriptor directory_fd(open(directory.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    if (directory_fd.get() < 0 || fsync(directory_fd.get()) != 0 || !directory_fd.Close()) {
      *error = "fsync helper directory after removing manifest";
      return false;
    }
  }
  return true;
}

}  // namespace cuda_checkpoint_storage
