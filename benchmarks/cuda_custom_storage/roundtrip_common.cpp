/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "roundtrip_common.hpp"

#include <fcntl.h>
#include <poll.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>

namespace cuda_custom_storage_benchmark {
namespace {

volatile sig_atomic_t g_owned_child_pid = 0;

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

[[noreturn]] void
WatchdogExpired(int)
{
  static constexpr char message[] = "cuda-custom-storage-roundtrip: 120s watchdog expired\n";
  const ssize_t ignored = write(STDERR_FILENO, message, sizeof(message) - 1);
  (void)ignored;
  const pid_t child = static_cast<pid_t>(g_owned_child_pid);
  if (child > 0) {
    (void)kill(child, SIGKILL);
  }
  _exit(124);
}

}  // namespace

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

void
IgnoreBrokenPipeSignal()
{
  struct sigaction action {};
  action.sa_handler = SIG_IGN;
  sigemptyset(&action.sa_mask);
  action.sa_flags = 0;
  if (sigaction(SIGPIPE, &action, nullptr) != 0) {
    ThrowErrno("ignore SIGPIPE");
  }
}

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
ReadAll(int fd, void* data, size_t size, Clock::time_point deadline)
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

ScopedFd::ScopedFd(int fd) : fd_(fd) {}

ScopedFd::ScopedFd(ScopedFd&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}

ScopedFd::~ScopedFd()
{
  if (fd_ >= 0) {
    (void)close(fd_);
  }
}

int
ScopedFd::get() const
{
  return fd_;
}

ChildProcess::ChildProcess(pid_t pid) : pid_(pid) {}

ChildProcess::~ChildProcess()
{
  KillAndWait();
}

pid_t
ChildProcess::pid() const
{
  return pid_;
}

int
ChildProcess::Wait(Clock::time_point deadline)
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

void
ChildProcess::KillAndWait() noexcept
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

void
ChildProcess::BlockWatchdog(sigset_t* old_set) noexcept
{
  sigset_t set;
  sigemptyset(&set);
  sigaddset(&set, SIGALRM);
  (void)sigprocmask(SIG_BLOCK, &set, old_set);
}

void
ChildProcess::RestoreSignals(const sigset_t* old_set) noexcept
{
  (void)sigprocmask(SIG_SETMASK, old_set, nullptr);
}

void
ChildProcess::InvalidateWhileBlocked() noexcept
{
  if (g_owned_child_pid == static_cast<sig_atomic_t>(pid_)) {
    g_owned_child_pid = 0;
  }
  pid_ = -1;
}

Watchdog::Watchdog()
{
  struct sigaction action {};
  action.sa_handler = WatchdogExpired;
  sigemptyset(&action.sa_mask);
  action.sa_flags = 0;
  if (sigaction(SIGALRM, &action, &old_action_) != 0) {
    ThrowErrno("install watchdog signal handler");
  }
  is_installed_ = true;
}

Watchdog::~Watchdog()
{
  (void)alarm(0);
  g_owned_child_pid = 0;
  if (is_installed_) {
    (void)sigaction(SIGALRM, &old_action_, nullptr);
  }
}

void
Watchdog::Arm(pid_t child)
{
  g_owned_child_pid = static_cast<sig_atomic_t>(child);
  (void)alarm(kWatchdogSeconds);
}

RetainedPrimaryContext::RetainedPrimaryContext(int requested_device)
{
  CheckCUDA(cuDeviceGet(&device_, requested_device), "cuDeviceGet");
  CheckCUDA(cuDevicePrimaryCtxRetain(&context_, device_), "cuDevicePrimaryCtxRetain");
}

RetainedPrimaryContext::~RetainedPrimaryContext()
{
  if (context_ != nullptr) {
    WarnCUDA(cuDevicePrimaryCtxRelease(device_), "cuDevicePrimaryCtxRelease");
  }
}

CUcontext
RetainedPrimaryContext::context() const
{
  return context_;
}

CUdevice
RetainedPrimaryContext::device() const
{
  return device_;
}

}  // namespace cuda_custom_storage_benchmark
