/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cuda.h>
#include <signal.h>
#include <sys/types.h>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

namespace cuda_custom_storage_benchmark {

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

inline constexpr size_t kDefaultBytes = 64ULL * 1024ULL * 1024ULL;
inline constexpr size_t kCopyChunkBytes = 16ULL * 1024ULL * 1024ULL;
inline constexpr unsigned int kLockTimeoutMs = 30'000;
inline constexpr unsigned int kWatchdogSeconds = 120;
inline constexpr std::string_view kExtentFilename = "checkpoint.bin";

enum class CorruptionMode {
  kNone,
  kTruncate,
  kSameSize,
};

enum class WorkloadResult : uint8_t {
  kPassed,
  kRestoredBytesMismatch,
  kPostRestoreCudaFailure,
  kCleanupFailure,
  kUnexpectedFailure,
};

static_assert(sizeof(WorkloadResult) == 1);

enum class WorkloadCommand : uint8_t {
  kVerify = 1,
};

static_assert(sizeof(WorkloadCommand) == 1);

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
  int device = 0;
  std::string device_uuid;
  uint64_t checkpoint_ptr = 0;
  uint64_t stream = 0;
  size_t size = 0;
};

struct Artifact {
  uint64_t application_ptr = 0;
  size_t application_bytes = 0;
  Extent extent;
};

[[noreturn]] void ThrowErrno(std::string_view operation);
std::string CudaError(CUresult status);
void CheckCUDA(CUresult status, std::string_view operation);
void WarnCUDA(CUresult status, const char* operation) noexcept;
std::string DeviceUUID(CUdevice device);
void ExpectProcessState(pid_t pid, CUprocessState expected, std::string_view stage);
void IgnoreBrokenPipeSignal();

int RemainingMilliseconds(Clock::time_point deadline);
void WaitReadable(int fd, Clock::time_point deadline, std::string_view operation);
void WriteAll(int fd, const void* data, size_t size);
void ReadAll(int fd, void* data, size_t size, Clock::time_point deadline = Clock::time_point::max());
void CreatePipe(int descriptors[2]);

class ScopedFd {
 public:
  explicit ScopedFd(int fd = -1);
  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;
  ScopedFd(ScopedFd&& other) noexcept;
  ~ScopedFd();

  int get() const;

 private:
  int fd_;
};

class ChildProcess {
 public:
  explicit ChildProcess(pid_t pid);
  ChildProcess(const ChildProcess&) = delete;
  ChildProcess& operator=(const ChildProcess&) = delete;
  ~ChildProcess();

  pid_t pid() const;
  int Wait(Clock::time_point deadline);
  void KillAndWait() noexcept;

 private:
  static void BlockWatchdog(sigset_t* old_set) noexcept;
  static void RestoreSignals(const sigset_t* old_set) noexcept;
  void InvalidateWhileBlocked() noexcept;

  pid_t pid_ = -1;
};

class Watchdog {
 public:
  Watchdog();
  Watchdog(const Watchdog&) = delete;
  Watchdog& operator=(const Watchdog&) = delete;
  ~Watchdog();

  void Arm(pid_t child);

 private:
  struct sigaction old_action_ {};
  bool is_installed_ = false;
};

class RetainedPrimaryContext {
 public:
  explicit RetainedPrimaryContext(int requested_device);
  RetainedPrimaryContext(const RetainedPrimaryContext&) = delete;
  RetainedPrimaryContext& operator=(const RetainedPrimaryContext&) = delete;
  ~RetainedPrimaryContext();

  CUcontext context() const;
  CUdevice device() const;

 private:
  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
};

}  // namespace cuda_custom_storage_benchmark
