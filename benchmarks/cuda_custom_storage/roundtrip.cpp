/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "cuda_checkpoint_compat.h"

namespace {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
namespace compat = cuda_checkpoint_compat;

constexpr size_t kDefaultBytes = 64ULL * 1024ULL * 1024ULL;
constexpr size_t kCopyChunkBytes = 16ULL * 1024ULL * 1024ULL;
constexpr unsigned int kLockTimeoutMs = 30'000;
constexpr unsigned int kWatchdogSeconds = 120;
constexpr std::string_view kManifestMagic = "cuda-custom-storage-roundtrip-v2";

volatile sig_atomic_t g_owned_child_pid = 0;

enum class CorruptionMode {
  kNone,
  kTruncate,
  kSameSize,
};

struct Options {
  fs::path artifact_dir;
  size_t bytes = kDefaultBytes;
  int device = 0;
  CorruptionMode corruption = CorruptionMode::kNone;
};

struct ReadyMessage {
  uint64_t device_ptr;
  uint64_t bytes;
};

struct Extent {
  size_t index = 0;
  int device = 0;
  std::string device_uuid;
  uint64_t checkpoint_ptr = 0;
  uint64_t stream = 0;
  size_t size = 0;
  std::string filename;
};

struct Artifact {
  uint64_t application_ptr = 0;
  size_t application_bytes = 0;
  std::vector<Extent> extents;
};

struct TransferMetrics {
  size_t bytes = 0;
  double cuda_seconds = 0.0;
  double storage_seconds = 0.0;
  double durability_seconds = 0.0;
};

struct OperationMetrics {
  TransferMetrics transfer;
  double cuda_api_seconds = 0.0;
  double completion_seconds = 0.0;
  double total_seconds = 0.0;
};

struct CheckpointResult {
  Artifact artifact;
  OperationMetrics metrics;
};

[[noreturn]] void
ThrowErrno(std::string_view operation)
{
  throw std::runtime_error(std::string(operation) + ": " + std::strerror(errno));
}

std::string
CudaError(CUresult status)
{
  const char* name = nullptr;
  const char* message = nullptr;
  (void)cuGetErrorName(status, &name);
  (void)cuGetErrorString(status, &message);
  return std::string(name == nullptr ? "CUDA_ERROR_UNKNOWN" : name) + ": " +
         (message == nullptr ? "unknown CUDA error" : message);
}

void
CheckCUDA(CUresult status, std::string_view operation)
{
  if (status != CUDA_SUCCESS) {
    throw std::runtime_error(std::string(operation) + " failed: " + CudaError(status));
  }
}

void
WarnCUDA(CUresult status, const char* operation) noexcept
{
  if (status == CUDA_SUCCESS) {
    return;
  }
  const char* name = nullptr;
  (void)cuGetErrorName(status, &name);
  std::fprintf(
      stderr, "warning: %s failed during cleanup: %s (%d)\n", operation, name == nullptr ? "CUDA_ERROR_UNKNOWN" : name,
      static_cast<int>(status));
}

double
SecondsSince(Clock::time_point start)
{
  return std::chrono::duration<double>(Clock::now() - start).count();
}

std::string
DeviceUUID(CUdevice device)
{
  CUuuid uuid{};
  CheckCUDA(cuDeviceGetUuid(&uuid, device), "cuDeviceGetUuid");
  static constexpr char hex[] = "0123456789abcdef";
  std::string formatted = "GPU-";
  formatted.reserve(40);
  for (size_t index = 0; index < sizeof(uuid.bytes); ++index) {
    if (index == 4 || index == 6 || index == 8 || index == 10) {
      formatted.push_back('-');
    }
    const auto byte = static_cast<unsigned char>(uuid.bytes[index]);
    formatted.push_back(hex[byte >> 4U]);
    formatted.push_back(hex[byte & 0x0fU]);
  }
  return formatted;
}

const char*
ProcessStateName(CUprocessState state)
{
  switch (state) {
    case CU_PROCESS_STATE_RUNNING:
      return "RUNNING";
    case CU_PROCESS_STATE_LOCKED:
      return "LOCKED";
    case CU_PROCESS_STATE_CHECKPOINTED:
      return "CHECKPOINTED";
    case CU_PROCESS_STATE_FAILED:
      return "FAILED";
    default:
      return "UNKNOWN";
  }
}

void
ExpectProcessState(pid_t pid, CUprocessState expected, std::string_view stage)
{
  CUprocessState actual{};
  CheckCUDA(cuCheckpointProcessGetState(pid, &actual), "cuCheckpointProcessGetState");
  if (actual != expected) {
    throw std::runtime_error(
        std::string(stage) + " left process in " + ProcessStateName(actual) + "; expected " +
        ProcessStateName(expected));
  }
  std::cout << "state stage=" << stage << " value=" << ProcessStateName(actual) << '\n';
}

[[noreturn]] void
WatchdogExpired(int)
{
  static constexpr char message[] = "cuda-custom-storage-roundtrip: 120s watchdog expired\n";
  (void)write(STDERR_FILENO, message, sizeof(message) - 1);
  const pid_t child = static_cast<pid_t>(g_owned_child_pid);
  if (child > 0) {
    (void)kill(child, SIGKILL);
  }
  _exit(124);
}

class Watchdog {
 public:
  Watchdog()
  {
    struct sigaction action {};
    action.sa_handler = WatchdogExpired;
    sigemptyset(&action.sa_mask);
    action.sa_flags = 0;
    if (sigaction(SIGALRM, &action, &old_action_) != 0) {
      ThrowErrno("install watchdog signal handler");
    }
    installed_ = true;
  }
  Watchdog(const Watchdog&) = delete;
  Watchdog& operator=(const Watchdog&) = delete;
  ~Watchdog()
  {
    (void)alarm(0);
    g_owned_child_pid = 0;
    if (installed_) {
      (void)sigaction(SIGALRM, &old_action_, nullptr);
    }
  }
  void Arm(pid_t child)
  {
    g_owned_child_pid = static_cast<sig_atomic_t>(child);
    (void)alarm(kWatchdogSeconds);
  }

 private:
  struct sigaction old_action_ {};
  bool installed_ = false;
};

int
RemainingMilliseconds(Clock::time_point deadline)
{
  const auto remaining = std::chrono::duration_cast<std::chrono::milliseconds>(deadline - Clock::now()).count();
  if (remaining <= 0) {
    return 0;
  }
  return static_cast<int>(std::min<int64_t>(remaining, std::numeric_limits<int>::max()));
}

void
WaitReadable(int fd, Clock::time_point deadline, std::string_view operation)
{
  struct pollfd descriptor {
    .fd = fd, .events = POLLIN, .revents = 0,
  };
  while (true) {
    const int timeout_ms = RemainingMilliseconds(deadline);
    if (timeout_ms == 0) {
      throw std::runtime_error(std::string(operation) + " timed out");
    }
    const int result = poll(&descriptor, 1, timeout_ms);
    if (result > 0) {
      if ((descriptor.revents & (POLLIN | POLLHUP)) != 0) {
        return;
      }
      throw std::runtime_error(std::string(operation) + " failed: pipe reported an error");
    }
    if (result == 0) {
      throw std::runtime_error(std::string(operation) + " timed out");
    }
    if (errno != EINTR) {
      ThrowErrno(operation);
    }
  }
}

void
WriteAll(int fd, const void* data, size_t size)
{
  const auto* bytes = static_cast<const unsigned char*>(data);
  size_t done = 0;
  while (done < size) {
    const ssize_t result = write(fd, bytes + done, size - done);
    if (result < 0) {
      if (errno == EINTR) {
        continue;
      }
      ThrowErrno("write");
    }
    if (result == 0) {
      throw std::runtime_error("write returned zero bytes");
    }
    done += static_cast<size_t>(result);
  }
}

void
ReadAll(int fd, void* data, size_t size, Clock::time_point deadline = Clock::time_point::max())
{
  auto* bytes = static_cast<unsigned char*>(data);
  size_t done = 0;
  while (done < size) {
    if (deadline != Clock::time_point::max()) {
      WaitReadable(fd, deadline, "pipe read");
    }
    const ssize_t result = read(fd, bytes + done, size - done);
    if (result < 0) {
      if (errno == EINTR) {
        continue;
      }
      ThrowErrno("read");
    }
    if (result == 0) {
      throw std::runtime_error("unexpected EOF");
    }
    done += static_cast<size_t>(result);
  }
}

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

std::vector<unsigned char>
Pattern(size_t size, size_t logical_offset)
{
  std::vector<unsigned char> pattern(size);
  for (size_t index = 0; index < size; ++index) {
    const uint64_t position = logical_offset + index;
    pattern[index] = static_cast<unsigned char>(((position * 1315423911ULL) ^ (position >> 7U) ^ 0x5aU) & 0xffU);
  }
  return pattern;
}

class ScopedFd {
 public:
  explicit ScopedFd(int fd = -1) : fd_(fd) {}
  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;
  ScopedFd(ScopedFd&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  ~ScopedFd()
  {
    if (fd_ >= 0) {
      (void)close(fd_);
    }
  }
  int get() const { return fd_; }

 private:
  int fd_;
};

class ChildProcess {
 public:
  explicit ChildProcess(pid_t pid) : pid_(pid) {}
  ChildProcess(const ChildProcess&) = delete;
  ChildProcess& operator=(const ChildProcess&) = delete;
  ~ChildProcess() { KillAndWait(); }

  pid_t pid() const { return pid_; }

  int Wait(Clock::time_point deadline)
  {
    while (true) {
      sigset_t old_set;
      BlockWatchdog(&old_set);
      int status = 0;
      const pid_t result = waitpid(pid_, &status, WNOHANG);
      if (result == pid_) {
        InvalidateWhileBlocked();
        RestoreSignals(&old_set);
        return status;
      }
      const int saved_errno = errno;
      RestoreSignals(&old_set);
      errno = saved_errno;
      if (result < 0 && errno != EINTR) {
        ThrowErrno("waitpid");
      }
      if (RemainingMilliseconds(deadline) == 0) {
        throw std::runtime_error("waiting for workload exit timed out");
      }
      struct timespec pause {
        .tv_sec = 0, .tv_nsec = 10'000'000,
      };
      (void)nanosleep(&pause, nullptr);
    }
  }

  void KillAndWait() noexcept
  {
    if (pid_ <= 0) {
      return;
    }
    if (kill(pid_, SIGKILL) != 0 && errno != ESRCH) {
      std::fprintf(stderr, "warning: failed to kill workload pid %d: %s\n", pid_, std::strerror(errno));
    }
    while (true) {
      sigset_t old_set;
      BlockWatchdog(&old_set);
      const pid_t result = waitpid(pid_, nullptr, WNOHANG);
      const int saved_errno = errno;
      if (result == pid_ || (result < 0 && saved_errno == ECHILD)) {
        InvalidateWhileBlocked();
        RestoreSignals(&old_set);
        return;
      }
      RestoreSignals(&old_set);
      if (result < 0 && saved_errno != EINTR) {
        std::fprintf(stderr, "warning: failed to reap workload pid %d: %s\n", pid_, std::strerror(saved_errno));
        BlockWatchdog(&old_set);
        InvalidateWhileBlocked();
        RestoreSignals(&old_set);
        return;
      }
      struct timespec pause {
        .tv_sec = 0, .tv_nsec = 10'000'000,
      };
      (void)nanosleep(&pause, nullptr);
    }
  }

 private:
  static void BlockWatchdog(sigset_t* old_set) noexcept
  {
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGALRM);
    (void)sigprocmask(SIG_BLOCK, &set, old_set);
  }

  static void RestoreSignals(const sigset_t* old_set) noexcept { (void)sigprocmask(SIG_SETMASK, old_set, nullptr); }

  void InvalidateWhileBlocked() noexcept
  {
    if (g_owned_child_pid == static_cast<sig_atomic_t>(pid_)) {
      g_owned_child_pid = 0;
    }
    pid_ = -1;
  }

  pid_t pid_ = -1;
};

void
CreatePipe(int descriptors[2])
{
  if (pipe(descriptors) != 0) {
    ThrowErrno("pipe");
  }
  for (int index = 0; index < 2; ++index) {
    const int flags = fcntl(descriptors[index], F_GETFD);
    if (flags < 0 || fcntl(descriptors[index], F_SETFD, flags | FD_CLOEXEC) != 0) {
      const int saved_errno = errno;
      (void)close(descriptors[0]);
      (void)close(descriptors[1]);
      errno = saved_errno;
      ThrowErrno("mark pipe close-on-exec");
    }
  }
}

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

class RetainedPrimaryContexts {
 public:
  explicit RetainedPrimaryContexts(int requested_device)
  {
    CheckCUDA(cuDeviceGet(&device_, requested_device), "cuDeviceGet");
    CheckCUDA(cuDevicePrimaryCtxRetain(&context_, device_), "cuDevicePrimaryCtxRetain");
  }
  RetainedPrimaryContexts(const RetainedPrimaryContexts&) = delete;
  RetainedPrimaryContexts& operator=(const RetainedPrimaryContexts&) = delete;
  ~RetainedPrimaryContexts()
  {
    if (context_ != nullptr) {
      WarnCUDA(cuDevicePrimaryCtxRelease(device_), "cuDevicePrimaryCtxRelease");
    }
  }
  CUcontext context() const { return context_; }
  CUdevice device() const { return device_; }

 private:
  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
};

void
WriteManifestTemporary(const fs::path& directory, const Artifact& artifact)
{
  const fs::path temporary = directory / "manifest.tmp";
  std::ofstream output(temporary, std::ios::out | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to create temporary manifest");
  }
  output << kManifestMagic << '\n';
  output << artifact.application_ptr << ' ' << artifact.application_bytes << ' ' << artifact.extents.size() << '\n';
  for (const Extent& extent : artifact.extents) {
    output << extent.index << ' ' << extent.device << ' ' << extent.device_uuid << ' ' << extent.checkpoint_ptr << ' '
           << extent.stream << ' ' << extent.size << ' ' << extent.filename << '\n';
  }
  output.flush();
  if (!output) {
    throw std::runtime_error("failed to write manifest");
  }
  output.close();
}

void
PublishManifest(const fs::path& directory)
{
  fs::rename(directory / "manifest.tmp", directory / "manifest.txt");
}

Artifact
ReadManifest(const fs::path& directory)
{
  std::ifstream input(directory / "manifest.txt");
  if (!input) {
    throw std::runtime_error("manifest.txt is missing");
  }
  std::string magic;
  std::getline(input, magic);
  if (magic != kManifestMagic) {
    throw std::runtime_error("artifact manifest has an unsupported format");
  }
  Artifact artifact;
  size_t extent_count = 0;
  if (!(input >> artifact.application_ptr >> artifact.application_bytes >> extent_count) || extent_count != 1) {
    throw std::runtime_error("artifact manifest header is invalid");
  }
  artifact.extents.reserve(extent_count);
  for (size_t index = 0; index < extent_count; ++index) {
    Extent extent;
    if (!(input >> extent.index >> extent.device >> extent.device_uuid >> extent.checkpoint_ptr >> extent.stream >>
          extent.size >> extent.filename) ||
        extent.index != index || extent.size == 0 || extent.filename != "extent-" + std::to_string(index) + ".bin") {
      throw std::runtime_error("artifact manifest extent is invalid");
    }
    const fs::path extent_path = directory / extent.filename;
    std::error_code error;
    const uintmax_t file_size = fs::file_size(extent_path, error);
    if (error || file_size != extent.size) {
      throw std::runtime_error("artifact extent has the wrong size: " + extent.filename);
    }
    artifact.extents.push_back(std::move(extent));
  }
  std::string trailing;
  if (input >> trailing) {
    throw std::runtime_error("artifact manifest has unexpected trailing data");
  }
  return artifact;
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

TransferMetrics
CopyExtent(const compat::PerDeviceData& device_data, CUcontext context, const fs::path& path, bool checkpoint)
{
  TransferMetrics metrics{.bytes = device_data.size};
  CheckCUDA(cuCtxSetCurrent(context), "cuCtxSetCurrent");

  const size_t buffer_size = std::min(kCopyChunkBytes, device_data.size);
  PinnedBuffer pinned(buffer_size);
  const int flags = checkpoint ? O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC : O_RDONLY | O_CLOEXEC | O_NOFOLLOW;
  ScopedFd file(open(path.c_str(), flags, 0600));
  if (file.get() < 0) {
    ThrowErrno("open artifact extent");
  }
  if (checkpoint && ftruncate(file.get(), static_cast<off_t>(device_data.size)) != 0) {
    ThrowErrno("size artifact extent");
  }

  for (size_t offset = 0; offset < device_data.size;) {
    const size_t length = std::min(pinned.size(), device_data.size - offset);
    if (checkpoint) {
      const auto cuda_start = Clock::now();
      CheckCUDA(
          cuMemcpyDtoHAsync(pinned.data(), device_data.devPtr + offset, length, device_data.stream),
          "cuMemcpyDtoHAsync");
      CheckCUDA(cuStreamSynchronize(device_data.stream), "checkpoint stream synchronization");
      metrics.cuda_seconds += SecondsSince(cuda_start);
      const auto storage_start = Clock::now();
      PWriteAll(file.get(), pinned.data(), length, offset);
      metrics.storage_seconds += SecondsSince(storage_start);
    } else {
      const auto storage_start = Clock::now();
      PReadAll(file.get(), pinned.data(), length, offset);
      metrics.storage_seconds += SecondsSince(storage_start);
      const auto cuda_start = Clock::now();
      CheckCUDA(
          cuMemcpyHtoDAsync(device_data.devPtr + offset, pinned.data(), length, device_data.stream),
          "cuMemcpyHtoDAsync");
      CheckCUDA(cuStreamSynchronize(device_data.stream), "restore stream synchronization");
      metrics.cuda_seconds += SecondsSince(cuda_start);
    }
    offset += length;
  }
  if (checkpoint) {
    const auto durability_start = Clock::now();
    if (fsync(file.get()) != 0) {
      ThrowErrno("fsync artifact extent");
    }
    metrics.durability_seconds = SecondsSince(durability_start);
  }
  return metrics;
}

void
Accumulate(TransferMetrics* total, const TransferMetrics& extent)
{
  total->bytes += extent.bytes;
  total->cuda_seconds += extent.cuda_seconds;
  total->storage_seconds += extent.storage_seconds;
  total->durability_seconds += extent.durability_seconds;
}

void
ValidateStorageInfo(const compat::StorageInfo* info)
{
  if (info == nullptr || info->handle == nullptr || info->deviceCount != 1 || info->perDeviceData == nullptr) {
    throw std::runtime_error("standalone proof requires exactly one valid CustomStorage device extent");
  }
  for (unsigned int index = 0; index < info->deviceCount; ++index) {
    if (info->perDeviceData[index].devPtr == 0 || info->perDeviceData[index].size == 0) {
      throw std::runtime_error("CUDA returned an invalid CustomStorage device extent");
    }
  }
}

CheckpointResult
CheckpointProcess(
    pid_t pid, const fs::path& directory, uint64_t application_ptr, size_t application_bytes, int requested_device,
    const std::string& device_uuid, CUcontext context, compat::OperationCompleteFn operation_complete)
{
  const auto total_start = Clock::now();
  CUcheckpointLockArgs lock_args{};
  lock_args.timeoutMs = kLockTimeoutMs;
  CheckCUDA(cuCheckpointProcessLock(pid, &lock_args), "cuCheckpointProcessLock");
  ExpectProcessState(pid, CU_PROCESS_STATE_LOCKED, "after_lock");

  compat::StorageInfo* info = nullptr;
  compat::CheckpointArgs checkpoint_args{};
  checkpoint_args.customStorageInfo_out = &info;
  const auto cuda_api_start = Clock::now();
  CheckCUDA(cuCheckpointProcessCheckpoint(pid, compat::NativeArgs(&checkpoint_args)), "cuCheckpointProcessCheckpoint");
  const double cuda_api_seconds = SecondsSince(cuda_api_start);
  ValidateStorageInfo(info);

  Artifact artifact{
      .application_ptr = application_ptr,
      .application_bytes = application_bytes,
  };
  TransferMetrics transfer;
  artifact.extents.reserve(info->deviceCount);
  for (unsigned int index = 0; index < info->deviceCount; ++index) {
    Extent extent{
        .index = index,
        .device = requested_device,
        .device_uuid = device_uuid,
        .checkpoint_ptr = info->perDeviceData[index].devPtr,
        .stream = reinterpret_cast<uintptr_t>(info->perDeviceData[index].stream),
        .size = info->perDeviceData[index].size,
        .filename = "extent-" + std::to_string(index) + ".bin",
    };
    Accumulate(&transfer, CopyExtent(info->perDeviceData[index], context, directory / extent.filename, true));
    artifact.extents.push_back(std::move(extent));
  }
  WriteManifestTemporary(directory, artifact);
  double completion_seconds = 0.0;
  try {
    const auto completion_start = Clock::now();
    CheckCUDA(operation_complete(info->handle), "cuCheckpointOperationComplete(checkpoint)");
    completion_seconds = SecondsSince(completion_start);
    ExpectProcessState(pid, CU_PROCESS_STATE_CHECKPOINTED, "after_checkpoint_completion");
    PublishManifest(directory);
  }
  catch (...) {
    std::error_code ignored;
    (void)fs::remove(directory / "manifest.tmp", ignored);
    (void)fs::remove(directory / "manifest.txt", ignored);
    throw;
  }
  return {
      .artifact = std::move(artifact),
      .metrics =
          {
              .transfer = transfer,
              .cuda_api_seconds = cuda_api_seconds,
              .completion_seconds = completion_seconds,
              .total_seconds = SecondsSince(total_start),
          },
  };
}

OperationMetrics
RestoreProcess(
    pid_t pid, const fs::path& directory, const Artifact& artifact, CUcontext context,
    compat::OperationCompleteFn operation_complete)
{
  const auto total_start = Clock::now();
  compat::StorageInfo* info = nullptr;
  compat::RestoreArgs restore_args{};
  restore_args.customStorageInfo_out = &info;
  const auto cuda_api_start = Clock::now();
  CheckCUDA(cuCheckpointProcessRestore(pid, compat::NativeArgs(&restore_args)), "cuCheckpointProcessRestore");
  const double cuda_api_seconds = SecondsSince(cuda_api_start);
  ValidateStorageInfo(info);
  if (info->deviceCount != artifact.extents.size()) {
    throw std::runtime_error("restore returned a different device extent count");
  }
  TransferMetrics transfer;
  for (unsigned int index = 0; index < info->deviceCount; ++index) {
    if (info->perDeviceData[index].size != artifact.extents[index].size) {
      throw std::runtime_error("restore returned a different device extent size");
    }
    Accumulate(
        &transfer,
        CopyExtent(info->perDeviceData[index], context, directory / artifact.extents[index].filename, false));
  }
  const auto completion_start = Clock::now();
  CheckCUDA(operation_complete(info->handle), "cuCheckpointOperationComplete(restore)");
  const double completion_seconds = SecondsSince(completion_start);
  ExpectProcessState(pid, CU_PROCESS_STATE_LOCKED, "after_restore_completion");
  CUcheckpointUnlockArgs unlock_args{};
  CheckCUDA(cuCheckpointProcessUnlock(pid, &unlock_args), "cuCheckpointProcessUnlock");
  ExpectProcessState(pid, CU_PROCESS_STATE_RUNNING, "after_unlock");
  return {
      .transfer = transfer,
      .cuda_api_seconds = cuda_api_seconds,
      .completion_seconds = completion_seconds,
      .total_seconds = SecondsSince(total_start),
  };
}

void
FillDevicePattern(CUdeviceptr pointer, size_t size)
{
  for (size_t offset = 0; offset < size;) {
    const size_t length = std::min(kCopyChunkBytes, size - offset);
    const std::vector<unsigned char> pattern = Pattern(length, offset);
    CheckCUDA(cuMemcpyHtoD(pointer + offset, pattern.data(), length), "initialize application buffer");
    offset += length;
  }
}

void
VerifyDevicePattern(CUdeviceptr pointer, size_t size)
{
  std::vector<unsigned char> actual(std::min(kCopyChunkBytes, size));
  for (size_t offset = 0; offset < size;) {
    const size_t length = std::min(actual.size(), size - offset);
    const std::vector<unsigned char> expected = Pattern(length, offset);
    CheckCUDA(cuMemcpyDtoH(actual.data(), pointer + offset, length), "read restored application buffer");
    const auto mismatch =
        std::mismatch(actual.begin(), actual.begin() + static_cast<ptrdiff_t>(length), expected.begin());
    if (mismatch.first != actual.begin() + static_cast<ptrdiff_t>(length)) {
      const size_t mismatch_offset = offset + static_cast<size_t>(mismatch.first - actual.begin());
      throw std::runtime_error("restored application bytes differ at offset " + std::to_string(mismatch_offset));
    }
    offset += length;
  }
}

int
RunWorkload(int ready_fd, int command_fd, const Options& options)
{
  try {
    CheckCUDA(cuInit(0), "workload cuInit");
    CUdevice device = 0;
    CUcontext context = nullptr;
    CheckCUDA(cuDeviceGet(&device, options.device), "workload cuDeviceGet");
    CheckCUDA(cuDevicePrimaryCtxRetain(&context, device), "workload cuDevicePrimaryCtxRetain");
    CheckCUDA(cuCtxSetCurrent(context), "workload cuCtxSetCurrent");

    CUdeviceptr application = 0;
    CUdeviceptr scratch = 0;
    CheckCUDA(cuMemAlloc(&application, options.bytes), "workload application allocation");
    CheckCUDA(cuMemAlloc(&scratch, 4096), "workload scratch allocation");
    FillDevicePattern(application, options.bytes);

    const ReadyMessage ready{application, options.bytes};
    WriteAll(ready_fd, &ready, sizeof(ready));
    char command = 0;
    ReadAll(command_fd, &command, sizeof(command));
    if (command != 'V') {
      throw std::runtime_error("workload received an invalid command");
    }

    VerifyDevicePattern(application, options.bytes);
    CheckCUDA(cuMemsetD8(scratch, 0x5a, 4096), "post-restore CUDA operation");
    std::array<unsigned char, 4096> scratch_bytes{};
    CheckCUDA(cuMemcpyDtoH(scratch_bytes.data(), scratch, scratch_bytes.size()), "verify post-restore CUDA operation");
    if (!std::all_of(scratch_bytes.begin(), scratch_bytes.end(), [](unsigned char value) { return value == 0x5a; })) {
      throw std::runtime_error("post-restore CUDA operation produced incorrect data");
    }

    CheckCUDA(cuMemFree(scratch), "workload scratch cleanup");
    CheckCUDA(cuMemFree(application), "workload application cleanup");
    CheckCUDA(cuDevicePrimaryCtxRelease(device), "workload primary context cleanup");
    const char result = 'P';
    WriteAll(ready_fd, &result, sizeof(result));
    return 0;
  }
  catch (const std::exception& exception) {
    std::fprintf(stderr, "workload failed: %s\n", exception.what());
    const char result = 'F';
    try {
      WriteAll(ready_fd, &result, sizeof(result));
    }
    catch (...) {
    }
    return 1;
  }
}

bool
ParseUnsigned(std::string_view value, size_t* result)
{
  if (value.empty()) {
    return false;
  }
  for (const char character : value) {
    if (character < '0' || character > '9') {
      return false;
    }
  }
  const std::string input(value);
  char* end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(input.c_str(), &end, 10);
  if (errno != 0 || end == nullptr || *end != '\0' || parsed > std::numeric_limits<size_t>::max()) {
    return false;
  }
  *result = static_cast<size_t>(parsed);
  return true;
}

Options
ParseOptions(int argc, char** argv)
{
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--artifact-dir" && ++index < argc) {
      options.artifact_dir = argv[index];
    } else if (
        argument == "--bytes" && ++index < argc && ParseUnsigned(argv[index], &options.bytes) && options.bytes > 0) {
    } else if (argument == "--device" && ++index < argc) {
      size_t device = 0;
      if (!ParseUnsigned(argv[index], &device) || device > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("--device must be a non-negative CUDA device ordinal");
      }
      options.device = static_cast<int>(device);
    } else if (argument == "--truncate-before-restore" && options.corruption == CorruptionMode::kNone) {
      options.corruption = CorruptionMode::kTruncate;
    } else if (argument == "--corrupt-before-restore" && options.corruption == CorruptionMode::kNone) {
      options.corruption = CorruptionMode::kSameSize;
    } else {
      throw std::runtime_error(
          "usage: cuda-custom-storage-roundtrip --artifact-dir <absolute-empty-directory> "
          "[--bytes N] [--device N] [--truncate-before-restore | --corrupt-before-restore]");
    }
  }
  if (options.artifact_dir.empty() || !options.artifact_dir.is_absolute()) {
    throw std::runtime_error("--artifact-dir must name an absolute directory");
  }
  std::error_code error;
  if (!fs::create_directory(options.artifact_dir, error) || error) {
    throw std::runtime_error("artifact directory must not already exist and its parent must be writable");
  }
  fs::permissions(options.artifact_dir, fs::perms::owner_all, fs::perm_options::replace, error);
  if (error) {
    throw std::runtime_error("failed to restrict artifact directory permissions");
  }
  return options;
}

int
Run(const Options& options)
{
  int child_to_parent[2] = {-1, -1};
  int parent_to_child[2] = {-1, -1};
  CreatePipe(child_to_parent);
  try {
    CreatePipe(parent_to_child);
  }
  catch (...) {
    (void)close(child_to_parent[0]);
    (void)close(child_to_parent[1]);
    throw;
  }

  Watchdog watchdog;
  const pid_t child_pid = fork();
  if (child_pid < 0) {
    ThrowErrno("fork");
  }
  if (child_pid == 0) {
    (void)close(child_to_parent[0]);
    (void)close(parent_to_child[1]);
    const int status = RunWorkload(child_to_parent[1], parent_to_child[0], options);
    _exit(status);
  }
  ChildProcess child(child_pid);
  watchdog.Arm(child.pid());
  const Clock::time_point deadline = Clock::now() + std::chrono::seconds(kWatchdogSeconds);

  ScopedFd ready_read(child_to_parent[0]);
  ScopedFd command_write(parent_to_child[1]);
  (void)close(child_to_parent[1]);
  (void)close(parent_to_child[0]);

  ReadyMessage ready{};
  ReadAll(ready_read.get(), &ready, sizeof(ready), deadline);
  CheckCUDA(cuInit(0), "controller cuInit");
  RetainedPrimaryContexts retained_context(options.device);
  const std::string device_uuid = DeviceUUID(retained_context.device());
  const compat::OperationCompleteFn operation_complete = compat::ResolveOperationComplete();
  if (operation_complete == nullptr) {
    throw std::runtime_error("CUDA driver does not expose CUDA 13.4 CustomStorage");
  }
  ExpectProcessState(child.pid(), CU_PROCESS_STATE_RUNNING, "before_lock");

  const CheckpointResult checkpoint = CheckpointProcess(
      child.pid(), options.artifact_dir, ready.device_ptr, ready.bytes, options.device, device_uuid,
      retained_context.context(), operation_complete);
  const Artifact artifact = ReadManifest(options.artifact_dir);
  if (artifact.application_ptr != ready.device_ptr || artifact.application_bytes != ready.bytes ||
      artifact.extents.front().device != options.device || artifact.extents.front().device_uuid != device_uuid) {
    throw std::runtime_error("manifest application or device identity changed");
  }

  size_t storage_bytes = 0;
  for (const Extent& extent : checkpoint.artifact.extents) {
    storage_bytes += extent.size;
    std::cout << "extent index=" << extent.index << " device=" << extent.device << " device_uuid=" << extent.device_uuid
              << " checkpoint_ptr=0x" << std::hex << extent.checkpoint_ptr << " stream=0x" << extent.stream << std::dec
              << " bytes=" << extent.size << " file=" << extent.filename << '\n';
  }
  if (checkpoint.metrics.transfer.bytes != storage_bytes) {
    throw std::runtime_error("checkpoint transfer did not cover every storage byte");
  }

  const fs::path first_extent = options.artifact_dir / artifact.extents.front().filename;
  if (options.corruption == CorruptionMode::kTruncate) {
    fs::resize_file(first_extent, artifact.extents.front().size - 1);
    try {
      (void)ReadManifest(options.artifact_dir);
      throw std::runtime_error("truncated artifact was incorrectly accepted");
    }
    catch (const std::runtime_error& error) {
      if (std::string(error.what()).find("wrong size") == std::string::npos) {
        throw;
      }
    }
    child.KillAndWait();
    std::cout << "corruption_check=passed mode=truncated detection=manifest_size_validation\n";
    return 0;
  }
  if (options.corruption == CorruptionMode::kSameSize) {
    CorruptExtentSameSize(first_extent, artifact.extents.front().size);
    (void)ReadManifest(options.artifact_dir);
  }

  OperationMetrics restore;
  try {
    restore =
        RestoreProcess(child.pid(), options.artifact_dir, artifact, retained_context.context(), operation_complete);
  }
  catch (const std::exception& error) {
    if (options.corruption == CorruptionMode::kSameSize) {
      child.KillAndWait();
      std::cout << "corruption_check=passed mode=same_size detection=cuda_restore_error error=" << error.what() << '\n';
      return 0;
    }
    throw;
  }
  if (restore.transfer.bytes != storage_bytes) {
    throw std::runtime_error("restore transfer did not cover every storage byte");
  }

  const char verify = 'V';
  WriteAll(command_write.get(), &verify, sizeof(verify));
  char result = 0;
  ReadAll(ready_read.get(), &result, sizeof(result), deadline);
  const int child_status = child.Wait(deadline);
  const bool workload_passed = result == 'P' && WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0;
  if (options.corruption == CorruptionMode::kSameSize) {
    if (workload_passed) {
      throw std::runtime_error("same-size corruption did not affect restored application state");
    }
    if (result != 'F' || !WIFEXITED(child_status) || WEXITSTATUS(child_status) == 0) {
      throw std::runtime_error("corrupted restore did not produce an explicit workload verification failure");
    }
    std::cout << "corruption_check=passed mode=same_size detection=workload_verification\n";
    return 0;
  }
  if (!workload_passed) {
    throw std::runtime_error("workload did not verify restored bytes and post-restore CUDA execution");
  }

  std::cout << std::fixed << std::setprecision(6) << "roundtrip=passed application_ptr=0x" << std::hex
            << ready.device_ptr << std::dec << " application_bytes=" << ready.bytes
            << " storage_bytes=" << storage_bytes << " checkpoint_total_seconds=" << checkpoint.metrics.total_seconds
            << " checkpoint_cuda_api_seconds=" << checkpoint.metrics.cuda_api_seconds
            << " checkpoint_d2h_seconds=" << checkpoint.metrics.transfer.cuda_seconds
            << " checkpoint_storage_write_seconds=" << checkpoint.metrics.transfer.storage_seconds
            << " checkpoint_fsync_seconds=" << checkpoint.metrics.transfer.durability_seconds
            << " checkpoint_complete_seconds=" << checkpoint.metrics.completion_seconds
            << " restore_total_seconds=" << restore.total_seconds
            << " restore_cuda_api_seconds=" << restore.cuda_api_seconds
            << " restore_storage_read_seconds=" << restore.transfer.storage_seconds
            << " restore_h2d_seconds=" << restore.transfer.cuda_seconds
            << " restore_complete_seconds=" << restore.completion_seconds << '\n';
  return 0;
}

}  // namespace

int
main(int argc, char** argv)
{
  try {
    const Options options = ParseOptions(argc, argv);
    return Run(options);
  }
  catch (const std::exception& exception) {
    std::fprintf(stderr, "cuda-custom-storage-roundtrip failed: %s\n", exception.what());
    return 1;
  }
}
