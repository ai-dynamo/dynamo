/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
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
namespace compat = cuda_checkpoint_compat;
using Clock = std::chrono::steady_clock;

constexpr size_t kDefaultBytes = 64ULL * 1024ULL * 1024ULL;
constexpr size_t kCopyChunkBytes = 16ULL * 1024ULL * 1024ULL;
constexpr unsigned int kLockTimeoutMs = 30'000;
constexpr unsigned int kWatchdogSeconds = 120;
constexpr std::string_view kManifestMagic = "gms-custom-storage-roundtrip-v1";

volatile sig_atomic_t g_server_pid = 0;

struct Options {
  fs::path artifact_dir;
  fs::path socket_path;
  fs::path repo_root;
  std::string python = "python3";
  size_t bytes = kDefaultBytes;
  int device = 0;
};

struct Extent {
  int device = 0;
  std::string device_uuid;
  uint64_t checkpoint_ptr = 0;
  uint64_t stream = 0;
  size_t size = 0;
  std::string filename = "extent-0.bin";
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
  if (status != CUDA_SUCCESS) {
    std::fprintf(stderr, "warning: %s failed during cleanup: %s\n", operation, CudaError(status).c_str());
  }
}

bool
ParseUnsigned(std::string_view value, size_t* result)
{
  if (value.empty() || !std::all_of(value.begin(), value.end(), [](char c) { return c >= '0' && c <= '9'; })) {
    return false;
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
    } else if (argument == "--socket-path" && ++index < argc) {
      options.socket_path = argv[index];
    } else if (argument == "--repo-root" && ++index < argc) {
      options.repo_root = argv[index];
    } else if (argument == "--python" && ++index < argc) {
      options.python = argv[index];
    } else if (argument == "--bytes" && ++index < argc) {
      if (!ParseUnsigned(argv[index], &options.bytes) || options.bytes == 0) {
        throw std::runtime_error("--bytes must be a positive decimal integer");
      }
    } else if (argument == "--device" && ++index < argc) {
      size_t device = 0;
      if (!ParseUnsigned(argv[index], &device) || device > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("--device must be a non-negative CUDA device ordinal");
      }
      options.device = static_cast<int>(device);
    } else {
      throw std::runtime_error(
          "usage: gms-custom-storage-roundtrip --artifact-dir PATH --socket-path PATH --repo-root PATH "
          "[--python PATH] [--bytes N] [--device N]");
    }
  }
  if (!options.artifact_dir.is_absolute() || !options.socket_path.is_absolute() || !options.repo_root.is_absolute()) {
    throw std::runtime_error("--artifact-dir, --socket-path, and --repo-root must be absolute paths");
  }
  if (!fs::is_regular_file(options.repo_root / "lib/gpu_memory_service/__init__.py")) {
    throw std::runtime_error("--repo-root does not contain the GMS Python package");
  }
  std::error_code error;
  if (!fs::create_directory(options.artifact_dir, error) || error) {
    throw std::runtime_error("--artifact-dir must not exist and its parent must be writable");
  }
  fs::permissions(options.artifact_dir, fs::perms::owner_all, fs::perm_options::replace, error);
  if (error) {
    throw std::runtime_error("failed to restrict artifact directory permissions");
  }
  if (fs::exists(options.socket_path)) {
    throw std::runtime_error("--socket-path already exists");
  }
  return options;
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
        std::string(stage) + " left GMS in " + ProcessStateName(actual) + "; expected " + ProcessStateName(expected));
  }
  std::cout << "state stage=" << stage << " value=" << ProcessStateName(actual) << '\n' << std::flush;
}

[[noreturn]] void
WatchdogExpired(int)
{
  static constexpr char message[] = "gms-custom-storage-roundtrip: 120s watchdog expired\n";
  const ssize_t ignored = write(STDERR_FILENO, message, sizeof(message) - 1);
  (void)ignored;
  if (g_server_pid > 0) {
    (void)kill(static_cast<pid_t>(g_server_pid), SIGKILL);
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
    if (sigaction(SIGALRM, &action, &old_action_) != 0) {
      ThrowErrno("install watchdog");
    }
  }
  ~Watchdog()
  {
    (void)alarm(0);
    g_server_pid = 0;
    (void)sigaction(SIGALRM, &old_action_, nullptr);
  }
  void Arm(pid_t server)
  {
    g_server_pid = static_cast<sig_atomic_t>(server);
    (void)alarm(kWatchdogSeconds);
  }

 private:
  struct sigaction old_action_ {};
};

class ChildProcess {
 public:
  explicit ChildProcess(pid_t pid = -1) : pid_(pid) {}
  ChildProcess(const ChildProcess&) = delete;
  ChildProcess& operator=(const ChildProcess&) = delete;
  ChildProcess(ChildProcess&& other) noexcept : pid_(std::exchange(other.pid_, -1)) {}
  ~ChildProcess() { KillAndWait(); }
  pid_t pid() const { return pid_; }

  void MarkReaped() noexcept { Invalidate(); }

  void KillAndWait() noexcept
  {
    if (pid_ <= 0) {
      return;
    }
    if (kill(pid_, SIGTERM) != 0 && errno != ESRCH) {
      std::fprintf(stderr, "warning: failed to terminate pid %d: %s\n", pid_, std::strerror(errno));
    }
    for (int attempt = 0; attempt < 100; ++attempt) {
      sigset_t old_set;
      BlockWatchdog(&old_set);
      const pid_t result = waitpid(pid_, nullptr, WNOHANG);
      const int saved_errno = errno;
      if (result == pid_ || (result < 0 && saved_errno == ECHILD)) {
        Invalidate();
        RestoreSignals(&old_set);
        return;
      }
      RestoreSignals(&old_set);
      struct timespec pause {
        .tv_sec = 0, .tv_nsec = 10'000'000
      };
      (void)nanosleep(&pause, nullptr);
    }
    (void)kill(pid_, SIGKILL);
    sigset_t old_set;
    BlockWatchdog(&old_set);
    while (waitpid(pid_, nullptr, 0) < 0 && errno == EINTR) {
    }
    Invalidate();
    RestoreSignals(&old_set);
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

  void Invalidate() noexcept
  {
    if (g_server_pid == static_cast<sig_atomic_t>(pid_)) {
      g_server_pid = 0;
    }
    pid_ = -1;
  }
  pid_t pid_ = -1;
};

std::vector<char*>
ExecArguments(std::vector<std::string>* arguments)
{
  std::vector<char*> result;
  result.reserve(arguments->size() + 1);
  for (std::string& argument : *arguments) {
    result.push_back(argument.data());
  }
  result.push_back(nullptr);
  return result;
}

ChildProcess
Spawn(std::vector<std::string> arguments)
{
  const pid_t pid = fork();
  if (pid < 0) {
    ThrowErrno("fork");
  }
  if (pid == 0) {
    std::vector<char*> argv = ExecArguments(&arguments);
    execvp(argv[0], argv.data());
    std::fprintf(stderr, "exec %s failed: %s\n", argv[0], std::strerror(errno));
    _exit(127);
  }
  return ChildProcess(pid);
}

void
RunCommand(std::vector<std::string> arguments)
{
  ChildProcess child = Spawn(std::move(arguments));
  int status = 0;
  while (waitpid(child.pid(), &status, 0) < 0) {
    if (errno != EINTR) {
      ThrowErrno("waitpid client");
    }
  }
  child.MarkReaped();
  if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
    throw std::runtime_error("GMS client subprocess failed");
  }
}

void
WaitForSocket(const fs::path& socket_path, pid_t server_pid)
{
  const auto deadline = Clock::now() + std::chrono::seconds(30);
  while (Clock::now() < deadline) {
    struct stat status {};
    if (lstat(socket_path.c_str(), &status) == 0 && S_ISSOCK(status.st_mode)) {
      return;
    }
    if (kill(server_pid, 0) != 0 && errno == ESRCH) {
      throw std::runtime_error("GMS server exited before opening its socket");
    }
    struct timespec pause {
      .tv_sec = 0, .tv_nsec = 100'000'000
    };
    (void)nanosleep(&pause, nullptr);
  }
  throw std::runtime_error("timed out waiting for the GMS socket");
}

class ScopedFd {
 public:
  explicit ScopedFd(int fd) : fd_(fd) {}
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

class PinnedBuffer {
 public:
  explicit PinnedBuffer(size_t size) : size_(size)
  {
    CheckCUDA(cuMemHostAlloc(&data_, size_, CU_MEMHOSTALLOC_PORTABLE), "cuMemHostAlloc");
  }
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

void
PWriteAll(int fd, const void* data, size_t size, size_t offset)
{
  const auto* bytes = static_cast<const unsigned char*>(data);
  for (size_t done = 0; done < size;) {
    const ssize_t result = pwrite(fd, bytes + done, size - done, static_cast<off_t>(offset + done));
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result <= 0) {
      ThrowErrno("pwrite extent");
    }
    done += static_cast<size_t>(result);
  }
}

void
PReadAll(int fd, void* data, size_t size, size_t offset)
{
  auto* bytes = static_cast<unsigned char*>(data);
  for (size_t done = 0; done < size;) {
    const ssize_t result = pread(fd, bytes + done, size - done, static_cast<off_t>(offset + done));
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result <= 0) {
      throw std::runtime_error("checkpoint extent ended before its declared size");
    }
    done += static_cast<size_t>(result);
  }
}

void
CopyExtent(const compat::PerDeviceData& data, CUcontext context, const fs::path& path, bool checkpoint)
{
  CheckCUDA(cuCtxSetCurrent(context), "cuCtxSetCurrent");
  PinnedBuffer buffer(std::min(kCopyChunkBytes, data.size));
  const int flags = checkpoint ? O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC : O_RDONLY | O_CLOEXEC | O_NOFOLLOW;
  ScopedFd file(open(path.c_str(), flags, 0600));
  if (file.get() < 0) {
    ThrowErrno("open checkpoint extent");
  }
  if (checkpoint && ftruncate(file.get(), static_cast<off_t>(data.size)) != 0) {
    ThrowErrno("size checkpoint extent");
  }
  for (size_t offset = 0; offset < data.size;) {
    const size_t length = std::min(buffer.size(), data.size - offset);
    if (checkpoint) {
      CheckCUDA(cuMemcpyDtoHAsync(buffer.data(), data.devPtr + offset, length, data.stream), "cuMemcpyDtoHAsync");
      CheckCUDA(cuStreamSynchronize(data.stream), "checkpoint stream synchronization");
      PWriteAll(file.get(), buffer.data(), length, offset);
    } else {
      PReadAll(file.get(), buffer.data(), length, offset);
      CheckCUDA(cuMemcpyHtoDAsync(data.devPtr + offset, buffer.data(), length, data.stream), "cuMemcpyHtoDAsync");
      CheckCUDA(cuStreamSynchronize(data.stream), "restore stream synchronization");
    }
    offset += length;
  }
  if (checkpoint && fsync(file.get()) != 0) {
    ThrowErrno("fsync checkpoint extent");
  }
}

void
ValidateStorageInfo(const compat::StorageInfo* info)
{
  if (info == nullptr || info->handle == nullptr || info->deviceCount != 1 || info->perDeviceData == nullptr ||
      info->perDeviceData[0].devPtr == 0 || info->perDeviceData[0].size == 0) {
    throw std::runtime_error("GMS proof requires exactly one valid CustomStorage device extent");
  }
}

std::string
DeviceUUID(CUdevice device)
{
  CUuuid uuid{};
  CheckCUDA(cuDeviceGetUuid(&uuid, device), "cuDeviceGetUuid");
  static constexpr char hex[] = "0123456789abcdef";
  std::string formatted = "GPU-";
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
WriteManifest(const fs::path& directory, const Extent& extent)
{
  std::ofstream output(directory / "manifest.tmp", std::ios::out | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("failed to create temporary manifest");
  }
  output << kManifestMagic << '\n';
  output << extent.device << ' ' << extent.device_uuid << ' ' << extent.checkpoint_ptr << ' ' << extent.stream << ' '
         << extent.size << ' ' << extent.filename << '\n';
  output.flush();
  if (!output) {
    throw std::runtime_error("failed to write manifest");
  }
}

Extent
Checkpoint(
    pid_t server_pid, const Options& options, CUcontext context, CUdevice device, compat::OperationCompleteFn complete)
{
  CUcheckpointLockArgs lock_args{};
  lock_args.timeoutMs = kLockTimeoutMs;
  CheckCUDA(cuCheckpointProcessLock(server_pid, &lock_args), "cuCheckpointProcessLock(GMS)");
  ExpectProcessState(server_pid, CU_PROCESS_STATE_LOCKED, "after_lock");

  compat::StorageInfo* info = nullptr;
  compat::CheckpointArgs checkpoint_args{};
  checkpoint_args.customStorageInfo_out = &info;
  CheckCUDA(
      cuCheckpointProcessCheckpoint(server_pid, compat::NativeArgs(&checkpoint_args)),
      "cuCheckpointProcessCheckpoint(GMS)");
  ValidateStorageInfo(info);

  const auto& data = info->perDeviceData[0];
  Extent extent{
      .device = options.device,
      .device_uuid = DeviceUUID(device),
      .checkpoint_ptr = data.devPtr,
      .stream = reinterpret_cast<uintptr_t>(data.stream),
      .size = data.size,
  };
  CopyExtent(data, context, options.artifact_dir / extent.filename, true);
  WriteManifest(options.artifact_dir, extent);
  try {
    CheckCUDA(complete(info->handle), "cuCheckpointOperationComplete(checkpoint)");
    ExpectProcessState(server_pid, CU_PROCESS_STATE_CHECKPOINTED, "after_checkpoint_completion");
    fs::rename(options.artifact_dir / "manifest.tmp", options.artifact_dir / "manifest.txt");
  }
  catch (...) {
    std::error_code ignored;
    (void)fs::remove(options.artifact_dir / "manifest.tmp", ignored);
    throw;
  }
  return extent;
}

void
Restore(
    pid_t server_pid, const Options& options, const Extent& checkpoint_extent, CUcontext context,
    compat::OperationCompleteFn complete)
{
  compat::StorageInfo* info = nullptr;
  compat::RestoreArgs restore_args{};
  restore_args.customStorageInfo_out = &info;
  CheckCUDA(
      cuCheckpointProcessRestore(server_pid, compat::NativeArgs(&restore_args)), "cuCheckpointProcessRestore(GMS)");
  ValidateStorageInfo(info);
  if (info->perDeviceData[0].size != checkpoint_extent.size) {
    throw std::runtime_error("restore returned a different CustomStorage extent size");
  }
  CopyExtent(info->perDeviceData[0], context, options.artifact_dir / checkpoint_extent.filename, false);
  CheckCUDA(complete(info->handle), "cuCheckpointOperationComplete(restore)");
  ExpectProcessState(server_pid, CU_PROCESS_STATE_LOCKED, "after_restore_completion");
  CUcheckpointUnlockArgs unlock_args{};
  CheckCUDA(cuCheckpointProcessUnlock(server_pid, &unlock_args), "cuCheckpointProcessUnlock(GMS)");
  ExpectProcessState(server_pid, CU_PROCESS_STATE_RUNNING, "after_unlock");
}

std::vector<std::string>
ClientCommand(const Options& options, std::string mode)
{
  return {
      options.python,
      (options.repo_root / "lib/gpu_memory_service/experiments/cuda_custom_storage/gms_client.py").string(),
      std::move(mode),
      "--socket-path",
      options.socket_path.string(),
      "--device",
      std::to_string(options.device),
      "--bytes",
      std::to_string(options.bytes),
  };
}

int
Run(const Options& options)
{
  const std::string package_path = (options.repo_root / "lib").string();
  const char* existing_pythonpath = std::getenv("PYTHONPATH");
  const std::string pythonpath = package_path + ":" + options.repo_root.string() +
                                 (existing_pythonpath == nullptr ? "" : ":" + std::string(existing_pythonpath));
  if (setenv("PYTHONPATH", pythonpath.c_str(), 1) != 0) {
    ThrowErrno("set PYTHONPATH");
  }

  Watchdog watchdog;
  ChildProcess server = Spawn({
      options.python,
      "-m",
      "gpu_memory_service",
      "--device",
      std::to_string(options.device),
      "--tag",
      "weights",
      "--socket-path",
      options.socket_path.string(),
  });
  watchdog.Arm(server.pid());
  WaitForSocket(options.socket_path, server.pid());

  RunCommand(ClientCommand(options, "write"));

  CheckCUDA(cuInit(0), "controller cuInit");
  CUdevice device = 0;
  CUcontext context = nullptr;
  CheckCUDA(cuDeviceGet(&device, options.device), "controller cuDeviceGet");
  CheckCUDA(cuDevicePrimaryCtxRetain(&context, device), "controller cuDevicePrimaryCtxRetain");
  try {
    CheckCUDA(cuCtxSetCurrent(context), "controller cuCtxSetCurrent");
    const compat::OperationCompleteFn complete = compat::ResolveOperationComplete();
    if (complete == nullptr) {
      throw std::runtime_error("CUDA driver does not expose CUDA 13.4 CustomStorage");
    }
    ExpectProcessState(server.pid(), CU_PROCESS_STATE_RUNNING, "before_lock");
    const Extent extent = Checkpoint(server.pid(), options, context, device, complete);
    Restore(server.pid(), options, extent, context, complete);
    RunCommand(ClientCommand(options, "verify"));
    std::cout << "extent device=" << extent.device << " device_uuid=" << extent.device_uuid << " checkpoint_ptr=0x"
              << std::hex << extent.checkpoint_ptr << " stream=0x" << extent.stream << std::dec
              << " bytes=" << extent.size << " file=" << extent.filename << '\n';
    std::cout << "gms_custom_storage_roundtrip=passed application_bytes=" << options.bytes
              << " storage_bytes=" << extent.size << '\n';
  }
  catch (...) {
    WarnCUDA(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
    throw;
  }
  CheckCUDA(cuDevicePrimaryCtxRelease(device), "cuDevicePrimaryCtxRelease");
  server.KillAndWait();
  std::error_code ignored;
  (void)fs::remove(options.socket_path, ignored);
  return 0;
}

}  // namespace

int
main(int argc, char** argv)
{
  try {
    return Run(ParseOptions(argc, argv));
  }
  catch (const std::exception& exception) {
    std::fprintf(stderr, "gms-custom-storage-roundtrip failed: %s\n", exception.what());
    return 1;
  }
}
