/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <nixl.h>
#include <nixl_descriptors.h>
#include <nixl_params.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>

#include "cuda_checkpoint_compat.h"
#include "storage_manifest.h"

namespace {

constexpr size_t kChunkSize = 64ULL * 1024ULL * 1024ULL;
constexpr size_t kPageSize = 4096;

namespace storage = cuda_checkpoint_storage;

class RetainedContexts {
 public:
  CUresult RetainAll()
  {
    int count = 0;
    CUresult status = cuDeviceGetCount(&count);
    if (status != CUDA_SUCCESS) {
      return status;
    }
    contexts_.reserve(count);
    for (int ordinal = 0; ordinal < count; ++ordinal) {
      CUdevice device = 0;
      status = cuDeviceGet(&device, ordinal);
      if (status != CUDA_SUCCESS) {
        return status;
      }
      CUcontext context = nullptr;
      status = cuDevicePrimaryCtxRetain(&context, device);
      if (status != CUDA_SUCCESS) {
        return status;
      }
      contexts_.push_back({device, context});
    }
    return CUDA_SUCCESS;
  }

  size_t size() const { return contexts_.size(); }

  CUresult ContextAndDeviceForStream(CUstream stream, CUcontext* context_out, CUdevice* device_out) const
  {
    CUcontext stream_context = nullptr;
    CUresult status = cuStreamGetCtx(stream, &stream_context);
    if (status != CUDA_SUCCESS) {
      return status;
    }
    for (const auto& retained : contexts_) {
      if (retained.context == stream_context) {
        *context_out = retained.context;
        *device_out = retained.device;
        return CUDA_SUCCESS;
      }
    }
    return CUDA_ERROR_INVALID_CONTEXT;
  }

  ~RetainedContexts()
  {
    for (auto it = contexts_.rbegin(); it != contexts_.rend(); ++it) {
      (void)cuDevicePrimaryCtxRelease(it->device);
    }
  }

 private:
  struct Entry {
    CUdevice device;
    CUcontext context;
  };
  std::vector<Entry> contexts_;
};

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

class RegisteredBuffer {
 public:
  CUresult Allocate()
  {
    if (posix_memalign(&data_, kPageSize, kChunkSize) != 0) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    CUresult status = cuMemHostRegister(data_, kChunkSize, 0);
    if (status != CUDA_SUCCESS) {
      free(data_);
      data_ = nullptr;
    }
    return status;
  }

  ~RegisteredBuffer()
  {
    if (data_ != nullptr) {
      (void)cuMemHostUnregister(data_);
      free(data_);
    }
  }

  void* data() const { return data_; }

 private:
  void* data_ = nullptr;
};

int
PrintUsage(FILE* stream)
{
  return std::fprintf(
             stream,
             "Usage:\n"
             "  cuda-checkpoint-helper --get-state --pid <pid>\n"
             "  cuda-checkpoint-helper --get-restore-tid --pid <pid>\n"
             "  cuda-checkpoint-helper --action lock|checkpoint|restore|unlock --pid <pid> "
             "[--timeout <ms>] [--device-map <uuids>] "
             "[--storage-mode posix --storage-dir <absolute-directory>]\n") < 0
             ? 1
             : 0;
}

void
PrintCudaError(CUresult status)
{
  const char* name = nullptr;
  const char* message = nullptr;
  (void)cuGetErrorName(status, &name);
  (void)cuGetErrorString(status, &message);
  std::fprintf(
      stderr, "%s: %s\n", name == nullptr ? "CUDA_ERROR_UNKNOWN" : name,
      message == nullptr ? "unknown CUDA error" : message);
}

bool
ParsePID(const char* value, int* pid_out)
{
  char* end = nullptr;
  long pid = std::strtol(value, &end, 10);
  if (value[0] == '\0' || end == nullptr || *end != '\0' || pid <= 0 || pid > INT_MAX) {
    return false;
  }
  *pid_out = static_cast<int>(pid);
  return true;
}

bool
ParseTimeout(const char* value, unsigned int* timeout_out)
{
  char* end = nullptr;
  unsigned long timeout = std::strtoul(value, &end, 10);
  if (value[0] == '\0' || end == nullptr || *end != '\0' || timeout > UINT_MAX) {
    return false;
  }
  *timeout_out = static_cast<unsigned int>(timeout);
  return true;
}

bool
ParseUUID(const char* value, CUuuid* uuid_out)
{
  if (value == nullptr || uuid_out == nullptr) {
    return false;
  }
  std::array<unsigned char, 16> bytes{};
  if (!storage::ParseGPUUUID(value, &bytes)) {
    return false;
  }
  static_assert(sizeof(uuid_out->bytes) == bytes.size());
  std::memcpy(uuid_out->bytes, bytes.data(), bytes.size());
  return true;
}

bool
ParseDeviceMap(
    const std::string& device_map, std::vector<CUcheckpointGpuPair>* pairs,
    std::vector<storage::DevicePair>* storage_pairs = nullptr)
{
  if (device_map.empty()) {
    return true;
  }
  std::unordered_set<std::string> source_uuids;
  std::unordered_set<std::string> destination_uuids;
  std::istringstream input(device_map);
  std::string pair;
  while (std::getline(input, pair, ',')) {
    size_t separator = pair.find('=');
    if (separator == std::string::npos || pair.find('=', separator + 1) != std::string::npos) {
      return false;
    }
    CUcheckpointGpuPair parsed{};
    const std::string source_input = pair.substr(0, separator);
    const std::string destination_input = pair.substr(separator + 1);
    std::string source;
    std::string destination;
    if (!ParseUUID(source_input.c_str(), &parsed.oldUuid) || !ParseUUID(destination_input.c_str(), &parsed.newUuid) ||
        !storage::CanonicalizeGPUUUID(source_input, &source) ||
        !storage::CanonicalizeGPUUUID(destination_input, &destination) || !source_uuids.insert(source).second ||
        !destination_uuids.insert(destination).second) {
      return false;
    }
    pairs->push_back(parsed);
    if (storage_pairs != nullptr) {
      storage_pairs->push_back({std::move(source), std::move(destination)});
    }
  }
  return !pairs->empty();
}

CUresult
DeviceUUID(CUdevice device, std::string* uuid_out)
{
  CUuuid uuid{};
  CUresult status = cuDeviceGetUuid(&uuid, device);
  if (status != CUDA_SUCCESS) {
    return status;
  }
  std::array<unsigned char, 16> bytes{};
  static_assert(sizeof(uuid.bytes) == bytes.size());
  std::memcpy(bytes.data(), uuid.bytes, bytes.size());
  *uuid_out = storage::FormatGPUUUID(bytes);
  return CUDA_SUCCESS;
}

const char*
ProcessStateString(CUprocessState state)
{
  switch (state) {
    case CU_PROCESS_STATE_RUNNING:
      return "running";
    case CU_PROCESS_STATE_LOCKED:
      return "locked";
    case CU_PROCESS_STATE_CHECKPOINTED:
      return "checkpointed";
    case CU_PROCESS_STATE_FAILED:
      return "failed";
    default:
      return "unknown";
  }
}

bool
NixlTransfer(
    nixlAgent* agent, const std::string& agent_name, nixl_xfer_op_t operation, void* buffer, int file_fd,
    size_t file_offset, size_t length, std::string* error)
{
  nixl_xfer_dlist_t dram(DRAM_SEG);
  nixl_xfer_dlist_t file(FILE_SEG);
  dram.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(buffer), length, 0));
  file.addDesc(nixlBlobDesc(file_offset, length, file_fd));
  nixlXferReqH* request = nullptr;
  nixl_status_t status = agent->createXferReq(operation, dram, file, agent_name, request);
  if (status != NIXL_SUCCESS) {
    *error = "NIXL createXferReq failed with status " + std::to_string(status);
    return false;
  }
  status = agent->postXferReq(request);
  while (status == NIXL_IN_PROG) {
    status = agent->getXferStatus(request);
    if (status == NIXL_IN_PROG) {
      std::this_thread::yield();
    }
  }
  nixl_status_t release_status = agent->releaseXferReq(request);
  if (status != NIXL_SUCCESS) {
    *error = "NIXL transfer failed with status " + std::to_string(status);
    if (release_status != NIXL_SUCCESS) {
      *error += "; releaseXferReq also failed with status " + std::to_string(release_status);
    }
    return false;
  }
  if (release_status != NIXL_SUCCESS) {
    *error = "NIXL releaseXferReq failed with status " + std::to_string(release_status);
    return false;
  }
  return true;
}

void
AppendError(std::string* error, const std::string& detail)
{
  if (!error->empty()) {
    *error += "; ";
  }
  *error += detail;
}

bool
TransferExtent(
    const cuda_checkpoint_compat::PerDeviceData& device_data, CUcontext context, const std::filesystem::path& path,
    bool checkpoint, std::string* error)
{
  CUresult cuda_status = cuCtxSetCurrent(context);
  if (cuda_status != CUDA_SUCCESS) {
    *error = "set retained primary context";
    return false;
  }

  RegisteredBuffer buffer;
  cuda_status = buffer.Allocate();
  if (cuda_status != CUDA_SUCCESS) {
    *error = "allocate CUDA-registered transfer buffer";
    return false;
  }

  int flags = O_RDWR | O_CLOEXEC;
  if (checkpoint) {
    flags |= O_CREAT | O_TRUNC;
  }
  if (device_data.size > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
    *error = "extent is too large for POSIX file offsets";
    return false;
  }
  FileDescriptor file_fd(open(path.c_str(), flags | O_NOFOLLOW, 0600));
  if (file_fd.get() < 0) {
    *error = "open extent file";
    return false;
  }
  if (checkpoint && fchmod(file_fd.get(), 0600) != 0) {
    *error = "set extent file permissions";
    return false;
  }
  struct stat file_stat {};
  if ((!checkpoint && (fstat(file_fd.get(), &file_stat) != 0 || !S_ISREG(file_stat.st_mode) || file_stat.st_size < 0 ||
                       static_cast<size_t>(file_stat.st_size) != device_data.size)) ||
      (checkpoint && ftruncate(file_fd.get(), static_cast<off_t>(device_data.size)) != 0)) {
    *error = checkpoint ? "size extent file" : "extent file size mismatch";
    return false;
  }

  const std::string agent_name = "cuda-custom-storage-" + path.filename().string();
  nixlAgentConfig config;
  config.useProgThread = true;
  nixlAgent agent(agent_name, config);
  nixl_b_params_t params;
  params["use_aio"] = "true";
  nixlBackendH* backend = nullptr;
  nixl_status_t nixl_status = agent.createBackend("POSIX", params, backend);
  if (nixl_status != NIXL_SUCCESS) {
    *error = "create NIXL POSIX backend failed with status " + std::to_string(nixl_status);
    return false;
  }

  nixl_reg_dlist_t dram_registration(DRAM_SEG);
  nixl_reg_dlist_t file_registration(FILE_SEG);
  dram_registration.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(buffer.data()), kChunkSize, 0));
  file_registration.addDesc(nixlBlobDesc(0, device_data.size, file_fd.get()));
  nixl_status = agent.registerMem(dram_registration);
  if (nixl_status != NIXL_SUCCESS) {
    *error = "register NIXL DRAM failed with status " + std::to_string(nixl_status);
    return false;
  }
  nixl_status = agent.registerMem(file_registration);
  if (nixl_status != NIXL_SUCCESS) {
    const nixl_status_t deregister_status = agent.deregisterMem(dram_registration);
    *error = "register NIXL file failed with status " + std::to_string(nixl_status);
    if (deregister_status != NIXL_SUCCESS) {
      AppendError(error, "NIXL DRAM cleanup failed with status " + std::to_string(deregister_status));
    }
    return false;
  }

  bool success = true;
  for (size_t offset = 0; offset < device_data.size && success; offset += kChunkSize) {
    const size_t length = std::min(kChunkSize, device_data.size - offset);
    if (checkpoint) {
      cuda_status = cuMemcpyDtoHAsync(buffer.data(), device_data.devPtr + offset, length, device_data.stream);
      if (cuda_status == CUDA_SUCCESS) {
        cuda_status = cuStreamSynchronize(device_data.stream);
      }
      if (cuda_status != CUDA_SUCCESS) {
        *error = "CUDA device-to-host copy failed with status " + std::to_string(cuda_status);
        success = false;
      } else {
        success = NixlTransfer(&agent, agent_name, NIXL_WRITE, buffer.data(), file_fd.get(), offset, length, error);
      }
    } else {
      success = NixlTransfer(&agent, agent_name, NIXL_READ, buffer.data(), file_fd.get(), offset, length, error);
      if (success) {
        cuda_status = cuMemcpyHtoDAsync(device_data.devPtr + offset, buffer.data(), length, device_data.stream);
        if (cuda_status == CUDA_SUCCESS) {
          cuda_status = cuStreamSynchronize(device_data.stream);
        }
        success = cuda_status == CUDA_SUCCESS;
        if (!success) {
          *error = "CUDA host-to-device copy failed with status " + std::to_string(cuda_status);
        }
      }
    }
  }
  if (success && checkpoint && fsync(file_fd.get()) != 0) {
    success = false;
    *error = "fsync extent file";
  }
  const nixl_status_t file_deregister_status = agent.deregisterMem(file_registration);
  const nixl_status_t dram_deregister_status = agent.deregisterMem(dram_registration);
  if (file_deregister_status != NIXL_SUCCESS) {
    success = false;
    AppendError(error, "NIXL file deregistration failed with status " + std::to_string(file_deregister_status));
  }
  if (dram_deregister_status != NIXL_SUCCESS) {
    success = false;
    AppendError(error, "NIXL DRAM deregistration failed with status " + std::to_string(dram_deregister_status));
  }
  if (!file_fd.Close()) {
    success = false;
    AppendError(error, "close extent file failed: " + std::string(std::strerror(errno)));
  }
  return success;
}

cuda_checkpoint_compat::OperationCompleteFn
ResolveOperationComplete()
{
  dlerror();
  void* symbol = dlsym(RTLD_DEFAULT, "cuCheckpointOperationComplete");
  return dlerror() == nullptr ? reinterpret_cast<cuda_checkpoint_compat::OperationCompleteFn>(symbol) : nullptr;
}

CUresult
DoCustomStorage(int pid, bool checkpoint, const std::string& device_map, const std::filesystem::path& storage_dir)
{
  cuda_checkpoint_compat::OperationCompleteFn operation_complete = ResolveOperationComplete();
  if (operation_complete == nullptr) {
    std::fprintf(stderr, "CUDA custom storage unavailable: cuCheckpointOperationComplete symbol not found\n");
    return CUDA_ERROR_NOT_SUPPORTED;
  }
  if (!storage_dir.is_absolute()) {
    std::fprintf(stderr, "custom storage directory must be absolute\n");
    return CUDA_ERROR_INVALID_VALUE;
  }
  if (checkpoint) {
    std::error_code filesystem_error;
    std::filesystem::create_directories(storage_dir, filesystem_error);
    struct stat directory_stat {};
    if (filesystem_error || lstat(storage_dir.c_str(), &directory_stat) != 0 || !S_ISDIR(directory_stat.st_mode) ||
        chmod(storage_dir.c_str(), 0700) != 0) {
      std::fprintf(stderr, "failed to create custom storage directory\n");
      return CUDA_ERROR_OPERATING_SYSTEM;
    }
    std::string remove_error;
    if (!storage::RemoveManifest(storage_dir, &remove_error)) {
      std::fprintf(stderr, "failed to clear stale custom storage manifest: %s\n", remove_error.c_str());
      return CUDA_ERROR_OPERATING_SYSTEM;
    }
  } else {
    struct stat directory_stat {};
    if (lstat(storage_dir.c_str(), &directory_stat) != 0 || !S_ISDIR(directory_stat.st_mode) ||
        (directory_stat.st_mode & 0022) != 0) {
      std::fprintf(stderr, "custom storage directory is missing or invalid\n");
      return CUDA_ERROR_INVALID_VALUE;
    }
  }

  CUresult status = cuInit(0);
  if (status != CUDA_SUCCESS) {
    return status;
  }
  RetainedContexts retained_contexts;
  status = retained_contexts.RetainAll();
  if (status != CUDA_SUCCESS) {
    return status;
  }

  std::vector<storage::ManifestExtent> manifest;
  std::string manifest_error;
  if (!checkpoint && (!storage::ReadManifest(storage_dir, &manifest, &manifest_error) ||
                      !storage::ValidateExtentFiles(storage_dir, manifest, &manifest_error))) {
    std::fprintf(stderr, "custom storage manifest validation failed: %s\n", manifest_error.c_str());
    return CUDA_ERROR_INVALID_VALUE;
  }

  cuda_checkpoint_compat::StorageInfo* info = nullptr;
  std::vector<CUcheckpointGpuPair> gpu_pairs;
  std::vector<storage::DevicePair> storage_pairs;
  if (checkpoint) {
    cuda_checkpoint_compat::CheckpointArgs args{};
    args.customStorageInfo_out = &info;
    status = cuCheckpointProcessCheckpoint(pid, cuda_checkpoint_compat::NativeArgs(&args));
  } else {
    if (!ParseDeviceMap(device_map, &gpu_pairs, &storage_pairs)) {
      return CUDA_ERROR_INVALID_VALUE;
    }
    cuda_checkpoint_compat::RestoreArgs args{};
    args.gpuPairs = gpu_pairs.empty() ? nullptr : gpu_pairs.data();
    args.gpuPairsCount = gpu_pairs.size();
    args.customStorageInfo_out = &info;
    status = cuCheckpointProcessRestore(pid, cuda_checkpoint_compat::NativeArgs(&args));
  }
  if (status != CUDA_SUCCESS) {
    return status;
  }
  if (info == nullptr || info->handle == nullptr || info->deviceCount > retained_contexts.size() ||
      (info->deviceCount > 0 && info->perDeviceData == nullptr)) {
    std::fprintf(stderr, "CUDA returned invalid custom storage information\n");
    return CUDA_ERROR_INVALID_VALUE;
  }

  std::vector<CUcontext> contexts(info->deviceCount);
  std::vector<CUdevice> devices(info->deviceCount);
  std::vector<storage::DeviceExtent> device_extents;
  device_extents.reserve(info->deviceCount);
  for (unsigned int index = 0; index < info->deviceCount; ++index) {
    status = retained_contexts.ContextAndDeviceForStream(
        info->perDeviceData[index].stream, &contexts[index], &devices[index]);
    if (status != CUDA_SUCCESS) {
      return status;
    }
    std::string uuid;
    status = DeviceUUID(devices[index], &uuid);
    if (status != CUDA_SUCCESS) {
      return status;
    }
    device_extents.push_back({std::move(uuid), info->perDeviceData[index].size});
  }

  if (checkpoint && !storage::BuildCheckpointManifest(device_extents, &manifest, &manifest_error)) {
    std::fprintf(stderr, "invalid checkpoint custom storage mapping: %s\n", manifest_error.c_str());
    return CUDA_ERROR_INVALID_VALUE;
  }
  std::vector<storage::TransferJob> transfer_jobs;
  if (!storage::BuildTransferJobs(
          manifest, device_extents, checkpoint ? std::vector<storage::DevicePair>{} : storage_pairs, &transfer_jobs,
          &manifest_error)) {
    std::fprintf(stderr, "invalid restore custom storage mapping: %s\n", manifest_error.c_str());
    return CUDA_ERROR_INVALID_VALUE;
  }

  const auto start = std::chrono::steady_clock::now();
  std::vector<std::thread> workers;
  std::vector<unsigned char> worker_success(transfer_jobs.size(), 0);
  std::vector<std::string> worker_errors(transfer_jobs.size());
  std::string worker_start_error;
  try {
    workers.reserve(transfer_jobs.size());
    for (size_t job_index = 0; job_index < transfer_jobs.size(); ++job_index) {
      workers.emplace_back([&, job_index] {
        const auto& job = transfer_jobs[job_index];
        try {
          worker_success[job_index] = TransferExtent(
              info->perDeviceData[job.device_index], contexts[job.device_index],
              storage_dir / manifest[job.extent_index].filename, checkpoint, &worker_errors[job_index]);
        }
        catch (const std::exception& exception) {
          worker_errors[job_index] = exception.what();
        }
        catch (...) {
          worker_errors[job_index] = "unknown worker exception";
        }
      });
    }
  }
  catch (const std::exception& exception) {
    worker_start_error = exception.what();
  }
  catch (...) {
    worker_start_error = "unknown thread creation exception";
  }
  for (auto& worker : workers) {
    worker.join();
  }
  if (!worker_start_error.empty()) {
    std::fprintf(stderr, "failed to start custom storage worker: %s\n", worker_start_error.c_str());
    return CUDA_ERROR_OPERATING_SYSTEM;
  }
  for (size_t job_index = 0; job_index < transfer_jobs.size(); ++job_index) {
    if (!worker_success[job_index]) {
      std::fprintf(
          stderr, "custom storage transfer failed for device index %zu: %s\n", transfer_jobs[job_index].device_index,
          worker_errors[job_index].c_str());
      return CUDA_ERROR_OPERATING_SYSTEM;
    }
  }

  if (checkpoint) {
    if (!storage::ValidateExtentFiles(storage_dir, manifest, &manifest_error)) {
      std::fprintf(stderr, "custom storage extent validation failed: %s\n", manifest_error.c_str());
      return CUDA_ERROR_OPERATING_SYSTEM;
    }
    if (!storage::WriteManifest(storage_dir, manifest, &manifest_error)) {
      std::fprintf(stderr, "custom storage manifest write failed: %s\n", manifest_error.c_str());
      return CUDA_ERROR_OPERATING_SYSTEM;
    }
  }

  size_t total_bytes = 0;
  for (const auto& extent : manifest) {
    total_bytes += extent.size;
  }

  // This is the sole acknowledgment point; CUDA exposes no public abort for failures above.
  status = operation_complete(info->handle);
  if (status != CUDA_SUCCESS) {
    if (checkpoint && !storage::RemoveManifest(storage_dir, &manifest_error)) {
      std::fprintf(
          stderr, "failed to remove custom storage manifest after CUDA completion failure: %s\n",
          manifest_error.c_str());
    }
    return status;
  }

  const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  const double gib_per_second =
      seconds == 0.0 ? 0.0 : static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0) / seconds;
  std::fprintf(
      stdout,
      "cuda_custom_storage action=%s devices=%zu bytes=%zu duration_seconds=%.6f "
      "throughput_gib_per_second=%.3f\n",
      checkpoint ? "checkpoint" : "restore", manifest.size(), total_bytes, seconds, gib_per_second);
  return CUDA_SUCCESS;
}

CUresult
DoLegacyRestore(int pid, const std::string& device_map)
{
  std::vector<CUcheckpointGpuPair> pairs;
  if (!ParseDeviceMap(device_map, &pairs)) {
    return CUDA_ERROR_INVALID_VALUE;
  }
  CUcheckpointRestoreArgs args{};
  args.gpuPairs = pairs.empty() ? nullptr : pairs.data();
  args.gpuPairsCount = pairs.size();
  return cuCheckpointProcessRestore(pid, &args);
}

}  // namespace

int
main(int argc, char** argv)
{
  std::string action;
  std::string device_map;
  std::string storage_mode = "legacy";
  std::string storage_dir;
  int pid = 0;
  bool have_pid = false;
  bool get_state = false;
  bool get_restore_tid = false;
  unsigned int timeout_ms = 0;

  if (argc == 1) {
    return PrintUsage(stderr);
  }
  for (int index = 1; index < argc; ++index) {
    std::string argument = argv[index];
    if (argument == "--get-state") {
      get_state = true;
    } else if (argument == "--get-restore-tid") {
      get_restore_tid = true;
    } else if (argument == "--action" && ++index < argc) {
      action = argv[index];
    } else if ((argument == "--pid" || argument == "-p") && ++index < argc && ParsePID(argv[index], &pid)) {
      have_pid = true;
    } else if (
        (argument == "--timeout" || argument == "-t") && ++index < argc && ParseTimeout(argv[index], &timeout_ms)) {
    } else if ((argument == "--device-map" || argument == "-d") && ++index < argc) {
      device_map = argv[index];
    } else if (argument == "--storage-mode" && ++index < argc) {
      storage_mode = argv[index];
    } else if (argument == "--storage-dir" && ++index < argc) {
      storage_dir = argv[index];
    } else if (argument == "--help" || argument == "-h") {
      return PrintUsage(stdout);
    } else {
      return PrintUsage(stderr);
    }
  }

  if (static_cast<int>(get_state) + static_cast<int>(get_restore_tid) + static_cast<int>(!action.empty()) != 1 ||
      !have_pid || (storage_mode != "legacy" && storage_mode != "posix") ||
      ((storage_mode == "posix") != !storage_dir.empty())) {
    return PrintUsage(stderr);
  }

  CUresult status = CUDA_SUCCESS;
  if (get_state) {
    if (timeout_ms != 0 || !device_map.empty() || storage_mode != "legacy") {
      return PrintUsage(stderr);
    }
    CUprocessState state;
    status = cuCheckpointProcessGetState(pid, &state);
    if (status == CUDA_SUCCESS) {
      return std::fprintf(stdout, "%s\n", ProcessStateString(state)) < 0 ? 1 : 0;
    }
  } else if (get_restore_tid) {
    if (timeout_ms != 0 || !device_map.empty() || storage_mode != "legacy") {
      return PrintUsage(stderr);
    }
    int tid = 0;
    status = cuCheckpointProcessGetRestoreThreadId(pid, &tid);
    if (status == CUDA_SUCCESS) {
      return std::fprintf(stdout, "%d\n", tid) < 0 ? 1 : 0;
    }
  } else if (action == "lock") {
    if (!device_map.empty() || storage_mode != "legacy") {
      return PrintUsage(stderr);
    }
    CUcheckpointLockArgs args{};
    args.timeoutMs = timeout_ms;
    status = cuCheckpointProcessLock(pid, &args);
  } else if (action == "checkpoint") {
    if (timeout_ms != 0 || !device_map.empty()) {
      return PrintUsage(stderr);
    }
    if (storage_mode == "posix") {
      status = DoCustomStorage(pid, true, "", storage_dir);
    } else {
      CUcheckpointCheckpointArgs args{};
      status = cuCheckpointProcessCheckpoint(pid, &args);
    }
  } else if (action == "restore") {
    if (timeout_ms != 0) {
      return PrintUsage(stderr);
    }
    status = storage_mode == "posix" ? DoCustomStorage(pid, false, device_map, storage_dir)
                                     : DoLegacyRestore(pid, device_map);
  } else if (action == "unlock") {
    if (timeout_ms != 0 || !device_map.empty() || storage_mode != "legacy") {
      return PrintUsage(stderr);
    }
    CUcheckpointUnlockArgs args{};
    status = cuCheckpointProcessUnlock(pid, &args);
  } else {
    return PrintUsage(stderr);
  }

  if (status != CUDA_SUCCESS) {
    PrintCudaError(status);
    return 1;
  }
  return 0;
}
