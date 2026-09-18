/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "storage.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <system_error>
#include <vector>

#include "custom_storage_operation.hpp"

namespace cuda_custom_storage_benchmark {
namespace {

enum class CopyDirection {
  kDeviceToFile,
  kFileToDevice,
};

class PinnedBuffer {
 public:
  explicit PinnedBuffer(size_t size) : size_(size)
  {
    CheckCUDA(cuMemHostAlloc(&data_, size_, CU_MEMHOSTALLOC_PORTABLE), "cuMemHostAlloc");
  }
  PinnedBuffer(const PinnedBuffer&) = delete;
  PinnedBuffer& operator=(const PinnedBuffer&) = delete;
  ~PinnedBuffer()
  {
    if (data_ != nullptr) {
      WarnCUDA(cuMemFreeHost(data_), "cuMemFreeHost");
    }
  }
  void* data() const { return data_; }
  size_t size() const { return size_; }

 private:
  void* data_ = nullptr;
  size_t size_ = 0;
};

class ProcessUnlockGuard {
 public:
  ProcessUnlockGuard(pid_t pid, bool is_armed) : pid_(pid), is_armed_(is_armed) {}
  ProcessUnlockGuard(const ProcessUnlockGuard&) = delete;
  ProcessUnlockGuard& operator=(const ProcessUnlockGuard&) = delete;
  ~ProcessUnlockGuard()
  {
    if (is_armed_) {
      CUcheckpointUnlockArgs args{};
      WarnCUDA(cuCheckpointProcessUnlock(pid_, &args), "cuCheckpointProcessUnlock after restore failure");
    }
  }

  void Unlock()
  {
    CUcheckpointUnlockArgs args{};
    const CUresult status = cuCheckpointProcessUnlock(pid_, &args);
    if (status == CUDA_SUCCESS) {
      is_armed_ = false;
    }
    CheckCUDA(status, "cuCheckpointProcessUnlock");
  }

 private:
  pid_t pid_;
  bool is_armed_;
};

void
PWriteAll(int fd, const void* data, size_t size, size_t offset)
{
  const auto* bytes = static_cast<const unsigned char*>(data);
  size_t done = 0;
  while (done < size) {
    const ssize_t result = pwrite(fd, bytes + done, size - done, static_cast<off_t>(offset + done));
    if (result < 0) {
      if (errno == EINTR) {
        continue;
      }
      ThrowErrno("pwrite");
    }
    if (result == 0) {
      throw std::runtime_error("pwrite returned zero bytes");
    }
    done += static_cast<size_t>(result);
  }
}

void
PReadAll(int fd, void* data, size_t size, size_t offset)
{
  auto* bytes = static_cast<unsigned char*>(data);
  size_t done = 0;
  while (done < size) {
    const ssize_t result = pread(fd, bytes + done, size - done, static_cast<off_t>(offset + done));
    if (result < 0) {
      if (errno == EINTR) {
        continue;
      }
      ThrowErrno("pread");
    }
    if (result == 0) {
      throw std::runtime_error("artifact extent ended before its declared size");
    }
    done += static_cast<size_t>(result);
  }
}

void
CopyExtent(
    const CUcheckpointCustomStoragePerDeviceData& device_data, CUcontext context, const fs::path& path,
    CopyDirection direction)
{
  CheckCUDA(cuCtxSetCurrent(context), "cuCtxSetCurrent");

  const size_t buffer_size = std::min(kCopyChunkBytes, device_data.size);
  PinnedBuffer pinned(buffer_size);
  const bool is_checkpoint = direction == CopyDirection::kDeviceToFile;
  const int flags = is_checkpoint ? O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC : O_RDONLY | O_CLOEXEC | O_NOFOLLOW;
  ScopedFd file(open(path.c_str(), flags, 0600));
  if (file.get() < 0) {
    ThrowErrno("open artifact extent");
  }
  if (is_checkpoint && ftruncate(file.get(), static_cast<off_t>(device_data.size)) != 0) {
    ThrowErrno("size artifact extent");
  }

  for (size_t offset = 0; offset < device_data.size;) {
    const size_t length = std::min(pinned.size(), device_data.size - offset);
    if (is_checkpoint) {
      CheckCUDA(
          cuMemcpyDtoHAsync(pinned.data(), device_data.devPtr + offset, length, device_data.stream),
          "cuMemcpyDtoHAsync");
      CheckCUDA(cuStreamSynchronize(device_data.stream), "checkpoint stream synchronization");
      PWriteAll(file.get(), pinned.data(), length, offset);
    } else {
      PReadAll(file.get(), pinned.data(), length, offset);
      CheckCUDA(
          cuMemcpyHtoDAsync(device_data.devPtr + offset, pinned.data(), length, device_data.stream),
          "cuMemcpyHtoDAsync");
      CheckCUDA(cuStreamSynchronize(device_data.stream), "restore stream synchronization");
    }
    offset += length;
  }
  if (is_checkpoint && fsync(file.get()) != 0) {
    ThrowErrno("fsync artifact extent");
  }
}

void
ValidateStandaloneStorageInfo(const CUcheckpointCustomStorageInfo* info)
{
  if (info == nullptr || info->deviceCount != 1 || info->perDeviceData == nullptr ||
      info->perDeviceData[0].devPtr == 0 || info->perDeviceData[0].size == 0) {
    throw std::runtime_error("standalone proof requires exactly one valid CustomStorage device extent");
  }
}

void
ValidateExtentDevice(const CUcheckpointCustomStoragePerDeviceData& device_data, int expected_device)
{
  CUdevice actual_device = 0;
  CheckCUDA(cuStreamGetDevice(device_data.stream, &actual_device), "cuStreamGetDevice");
  if (actual_device != expected_device) {
    throw std::runtime_error("restore returned a device extent for a different CUDA device");
  }
}

[[noreturn]] void
ThrowRestoreRejected(CUresult status, std::string_view operation)
{
  throw RestoreRejected(std::string(operation) + " failed: " + CudaError(status));
}

}  // namespace

uintmax_t
ExtentFileSize(const fs::path& path)
{
  std::error_code error;
  const uintmax_t size = fs::file_size(path, error);
  if (error) {
    throw std::runtime_error("failed to stat checkpoint file: " + error.message());
  }
  return size;
}

void
ValidateExtentFile(const fs::path& path, size_t expected_size)
{
  if (ExtentFileSize(path) != expected_size) {
    throw ExtentSizeMismatch("checkpoint file has the wrong size");
  }
}

void
CorruptExtentSameSize(const fs::path& path, size_t size)
{
  ScopedFd file(open(path.c_str(), O_WRONLY | O_CLOEXEC | O_NOFOLLOW));
  if (file.get() < 0) {
    ThrowErrno("open artifact extent for corruption");
  }
  std::vector<unsigned char> zeros(std::min(kCopyChunkBytes, size), 0);
  for (size_t offset = 0; offset < size;) {
    const size_t length = std::min(zeros.size(), size - offset);
    PWriteAll(file.get(), zeros.data(), length, offset);
    offset += length;
  }
  if (fsync(file.get()) != 0) {
    ThrowErrno("fsync corrupted artifact extent");
  }
}

Artifact
CheckpointProcess(
    pid_t pid, const fs::path& directory, uint64_t application_ptr, size_t application_bytes, int requested_device,
    const std::string& device_uuid, CUcontext context)
{
  CUcheckpointLockArgs lock_args{};
  lock_args.timeoutMs = kLockTimeoutMs;
  CheckCUDA(cuCheckpointProcessLock(pid, &lock_args), "cuCheckpointProcessLock");
  ExpectProcessState(pid, CU_PROCESS_STATE_LOCKED, "after_lock");

  CustomStorageOperation operation;
  CheckCUDA(operation.BeginCheckpoint(pid), "cuCheckpointProcessCheckpoint(CustomStorage)");
  const CUcheckpointCustomStorageInfo* info = operation.storage_info();
  ValidateStandaloneStorageInfo(info);

  Artifact artifact{
      .application_ptr = application_ptr,
      .application_bytes = application_bytes,
      .extent =
          {
              .device = requested_device,
              .device_uuid = device_uuid,
              .checkpoint_ptr = info->perDeviceData[0].devPtr,
              .stream = reinterpret_cast<uintptr_t>(info->perDeviceData[0].stream),
              .size = info->perDeviceData[0].size,
          },
  };
  const fs::path checkpoint_file = directory / kExtentFilename;
  try {
    CopyExtent(info->perDeviceData[0], context, checkpoint_file, CopyDirection::kDeviceToFile);
    CheckCUDA(operation.Complete(), "cuCheckpointOperationComplete(checkpoint)");
    ExpectProcessState(pid, CU_PROCESS_STATE_CHECKPOINTED, "after_checkpoint_completion");
  }
  catch (...) {
    std::error_code ignored;
    (void)fs::remove(checkpoint_file, ignored);
    throw;
  }
  return artifact;
}

void
RestoreProcess(pid_t pid, const fs::path& directory, const Artifact& artifact, CUcontext expected_context)
{
  CustomStorageOperation operation;
  const CUresult begin_status = operation.BeginRestore(pid);
  ProcessUnlockGuard unlock_guard(pid, operation.has_started());
  if (begin_status != CUDA_SUCCESS) {
    ThrowRestoreRejected(begin_status, "cuCheckpointProcessRestore(CustomStorage)");
  }
  const CUcheckpointCustomStorageInfo* info = operation.storage_info();
  ValidateStandaloneStorageInfo(info);
  if (info->perDeviceData[0].size != artifact.extent.size) {
    throw std::runtime_error("restore returned a different device extent size");
  }
  ValidateExtentDevice(info->perDeviceData[0], artifact.extent.device);
  CopyExtent(info->perDeviceData[0], expected_context, directory / kExtentFilename, CopyDirection::kFileToDevice);
  const CUresult complete_status = operation.Complete();
  if (complete_status != CUDA_SUCCESS) {
    ThrowRestoreRejected(complete_status, "cuCheckpointOperationComplete(restore)");
  }
  ExpectProcessState(pid, CU_PROCESS_STATE_LOCKED, "after_restore_completion");
  unlock_guard.Unlock();
  ExpectProcessState(pid, CU_PROCESS_STATE_RUNNING, "after_unlock");
}

}  // namespace cuda_custom_storage_benchmark
