/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cctype>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <limits>
#include <sstream>
#include <string>
#include <system_error>
#include <thread>
#include <unordered_set>
#include <vector>

#include "cuda_checkpoint_compat.h"
#include "daemon_protocol.h"
#include "storage_manifest.h"
#include "transfer_config.h"
#include "transfer_engine.h"

namespace {

namespace storage = cuda_checkpoint_storage;
namespace transfer = cuda_checkpoint_transfer;
using Clock = std::chrono::steady_clock;
namespace daemon_protocol = cuda_checkpoint_daemon;

class ScopedFd {
 public:
  explicit ScopedFd(int fd) : fd_(fd) {}
  ScopedFd(const ScopedFd&) = delete;
  ScopedFd& operator=(const ScopedFd&) = delete;
  ~ScopedFd() noexcept
  {
    if (fd_ >= 0) {
      close(fd_);
    }
  }

  int get() const { return fd_; }

 private:
  int fd_;
};

class DaemonThreadShutdown {
 public:
  DaemonThreadShutdown(
      daemon_protocol::ShutdownSignalOwner* signal_owner,
      daemon_protocol::ShutdownSignalOwner::ShutdownResult* result)
      : signal_owner_(signal_owner), result_(result)
  {
  }
  DaemonThreadShutdown(const DaemonThreadShutdown&) = delete;
  DaemonThreadShutdown& operator=(const DaemonThreadShutdown&) = delete;

  ~DaemonThreadShutdown() noexcept
  {
    *result_ = signal_owner_->StopAndJoinNoThrow();
  }

 private:
  daemon_protocol::ShutdownSignalOwner* signal_owner_;
  daemon_protocol::ShutdownSignalOwner::ShutdownResult* result_;
};

bool
ParsePositiveSeconds(const char* value, uint64_t* seconds_out)
{
  char* end = nullptr;
  errno = 0;
  const unsigned long long seconds = std::strtoull(value, &end, 10);
  if (value[0] == '\0' || end == nullptr || *end != '\0' || errno != 0 || seconds == 0 ||
      seconds > static_cast<unsigned long long>(std::numeric_limits<int64_t>::max())) {
    return false;
  }
  *seconds_out = seconds;
  return true;
}

double
SecondsSince(Clock::time_point start)
{
  return std::chrono::duration<double>(Clock::now() - start).count();
}

double
SecondsBetween(Clock::time_point start, Clock::time_point end)
{
  return std::chrono::duration<double>(end - start).count();
}

class RetainedContexts {
 public:
  RetainedContexts() = default;
  RetainedContexts(const RetainedContexts&) = delete;
  RetainedContexts& operator=(const RetainedContexts&) = delete;

  CUresult RetainAll(int* device_count, double* enumeration_seconds, double* retain_seconds)
  {
    const auto enumeration_start = Clock::now();
    int count = 0;
    CUresult status = cuDeviceGetCount(&count);
    *enumeration_seconds = SecondsSince(enumeration_start);
    *device_count = count;
    if (status != CUDA_SUCCESS) {
      return status;
    }
    contexts_.reserve(count);
    for (int ordinal = 0; ordinal < count; ++ordinal) {
      const auto retain_start = Clock::now();
      CUdevice device = 0;
      status = cuDeviceGet(&device, ordinal);
      if (status != CUDA_SUCCESS) {
        *retain_seconds += SecondsSince(retain_start);
        return status;
      }
      CUcontext context = nullptr;
      status = cuDevicePrimaryCtxRetain(&context, device);
      *retain_seconds += SecondsSince(retain_start);
      if (status != CUDA_SUCCESS) {
        return status;
      }
      contexts_.push_back({device, context});
    }
    return CUDA_SUCCESS;
  }

  CUresult ReleaseAll()
  {
    CUresult first_error = CUDA_SUCCESS;
    while (!contexts_.empty()) {
      const CUresult status = cuDevicePrimaryCtxRelease(contexts_.back().device);
      if (first_error == CUDA_SUCCESS && status != CUDA_SUCCESS) {
        first_error = status;
      }
      contexts_.pop_back();
    }
    return first_error;
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
    (void)ReleaseAll();
  }

 private:
  struct Entry {
    CUdevice device;
    CUcontext context;
  };
  std::vector<Entry> contexts_;
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
             "[--storage-mode posix --storage-dir <absolute-directory> "
             "--transfer-buffer-count <count> --transfer-chunk-bytes <bytes>]\n"
             "  cuda-checkpoint-helper --daemon --socket <absolute-socket-path> "
             "[--max-operation-seconds <seconds>]\n"
             "  cuda-checkpoint-helper --health --socket <absolute-socket-path>\n"
             "\n"
             "POSIX transfer defaults: --transfer-buffer-count %zu "
             "--transfer-chunk-bytes %zu\n",
             transfer::kDefaultBufferCount, transfer::kDefaultChunkBytes) < 0
             ? 1
             : 0;
}

int
PrintUsageError()
{
  (void)PrintUsage(stderr);
  return 1;
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

cuda_checkpoint_compat::OperationCompleteFn
ResolveOperationComplete()
{
  dlerror();
  void* symbol = dlsym(RTLD_DEFAULT, "cuCheckpointOperationComplete");
  return dlerror() == nullptr ? reinterpret_cast<cuda_checkpoint_compat::OperationCompleteFn>(symbol) : nullptr;
}

struct CustomStorageResult {
  CUresult status = CUDA_SUCCESS;
  daemon_protocol::OperationState operation;
};

CustomStorageResult
DoCustomStorage(
    int pid, bool checkpoint, const std::string& device_map, const std::filesystem::path& storage_dir,
    const transfer::TransferOptions& transfer_options, Clock::time_point helper_main_start,
    RetainedContexts* daemon_contexts = nullptr, const daemon_protocol::Request* daemon_request = nullptr)
{
  const auto custom_storage_start = Clock::now();
  const auto symbol_resolution_start = Clock::now();
  cuda_checkpoint_compat::OperationCompleteFn operation_complete = ResolveOperationComplete();
  const double symbol_resolution_seconds = SecondsSince(symbol_resolution_start);
  if (operation_complete == nullptr) {
    std::fprintf(stderr, "CUDA custom storage unavailable: cuCheckpointOperationComplete symbol not found\n");
    return {CUDA_ERROR_NOT_SUPPORTED, {}};
  }
  const auto storage_directory_start = Clock::now();
  if (!storage_dir.is_absolute()) {
    std::fprintf(stderr, "custom storage directory must be absolute\n");
    return {CUDA_ERROR_INVALID_VALUE, {}};
  }
  if (checkpoint) {
    std::error_code filesystem_error;
    std::filesystem::create_directories(storage_dir, filesystem_error);
    struct stat directory_stat {};
    if (filesystem_error || lstat(storage_dir.c_str(), &directory_stat) != 0 || !S_ISDIR(directory_stat.st_mode) ||
        chmod(storage_dir.c_str(), 0700) != 0) {
      std::fprintf(stderr, "failed to create custom storage directory\n");
      return {CUDA_ERROR_OPERATING_SYSTEM, {}};
    }
    std::string remove_error;
    if (!storage::RemoveManifest(storage_dir, &remove_error)) {
      std::fprintf(stderr, "failed to clear stale custom storage manifest: %s\n", remove_error.c_str());
      return {CUDA_ERROR_OPERATING_SYSTEM, {}};
    }
  } else {
    struct stat directory_stat {};
    if (lstat(storage_dir.c_str(), &directory_stat) != 0 || !S_ISDIR(directory_stat.st_mode) ||
        (directory_stat.st_mode & 0022) != 0) {
      std::fprintf(stderr, "custom storage directory is missing or invalid\n");
      return {CUDA_ERROR_INVALID_VALUE, {}};
    }
  }
  const double storage_directory_validation_seconds = SecondsSince(storage_directory_start);

  double cuda_init_seconds = 0.0;
  int cuda_device_count = 0;
  double device_enumeration_seconds = 0.0;
  double primary_context_retain_seconds = 0.0;
  RetainedContexts operation_contexts;
  RetainedContexts* retained_contexts = daemon_contexts;
  CUresult status = CUDA_SUCCESS;
  if (retained_contexts == nullptr) {
    const auto cuda_init_start = Clock::now();
    status = cuInit(0);
    cuda_init_seconds = SecondsSince(cuda_init_start);
    if (status != CUDA_SUCCESS) {
      return {status, {}};
    }
    retained_contexts = &operation_contexts;
    status = retained_contexts->RetainAll(
        &cuda_device_count, &device_enumeration_seconds, &primary_context_retain_seconds);
    if (status != CUDA_SUCCESS) {
      return {status, {}};
    }
  } else {
    cuda_device_count = retained_contexts->size();
  }

  const auto manifest_validation_start = Clock::now();
  std::vector<storage::ManifestExtent> manifest;
  std::string manifest_error;
  if (!checkpoint && (!storage::ReadManifest(storage_dir, &manifest, &manifest_error) ||
                      !storage::ValidateExtentFiles(storage_dir, manifest, &manifest_error))) {
    std::fprintf(stderr, "custom storage manifest validation failed: %s\n", manifest_error.c_str());
    return {CUDA_ERROR_INVALID_VALUE, {}};
  }
  const double manifest_validation_seconds = SecondsSince(manifest_validation_start);

  cuda_checkpoint_compat::StorageInfo* info = nullptr;
  std::vector<CUcheckpointGpuPair> gpu_pairs;
  std::vector<storage::DevicePair> storage_pairs;
  const auto device_map_preparation_start = Clock::now();
  if (!checkpoint && !ParseDeviceMap(device_map, &gpu_pairs, &storage_pairs)) {
    return {CUDA_ERROR_INVALID_VALUE, {}};
  }
  const double device_map_preparation_seconds = SecondsSince(device_map_preparation_start);
  if (daemon_request != nullptr) {
    std::string identity_error;
    if (!daemon_protocol::ValidateProcessIdentity(*daemon_request, "/host/proc", &identity_error)) {
      std::fprintf(stderr, "process identity changed before CUDA operation: %s\n", identity_error.c_str());
      return {CUDA_ERROR_INVALID_VALUE, {}};
    }
  }
  const auto cuda_process_api_start = Clock::now();
  if (checkpoint) {
    cuda_checkpoint_compat::CheckpointArgs args{};
    args.customStorageInfo_out = &info;
    status = cuCheckpointProcessCheckpoint(pid, cuda_checkpoint_compat::NativeArgs(&args));
  } else {
    cuda_checkpoint_compat::RestoreArgs args{};
    args.gpuPairs = gpu_pairs.empty() ? nullptr : gpu_pairs.data();
    args.gpuPairsCount = gpu_pairs.size();
    args.customStorageInfo_out = &info;
    status = cuCheckpointProcessRestore(pid, cuda_checkpoint_compat::NativeArgs(&args));
  }
  const double cuda_process_api_seconds = SecondsSince(cuda_process_api_start);
  if (status != CUDA_SUCCESS) {
    return {status, {}};
  }
  daemon_protocol::OperationState operation{.handle_returned = true};
  const auto post_handle_failure = [&operation](CUresult failure) {
    const CUresult status = static_cast<CUresult>(daemon_protocol::FinishHandledOperation(
        false, failure, [] { return static_cast<int32_t>(CUDA_SUCCESS); }, &operation));
    return CustomStorageResult{.status = status, .operation = operation};
  };
  const auto metadata_job_construction_start = Clock::now();
  if (info == nullptr || info->handle == nullptr || info->deviceCount > retained_contexts->size() ||
      (info->deviceCount > 0 && info->perDeviceData == nullptr)) {
    std::fprintf(stderr, "CUDA returned invalid custom storage information\n");
    return post_handle_failure(CUDA_ERROR_INVALID_VALUE);
  }

  size_t pinned_bytes = 0;
  std::string transfer_config_error;
  if (!transfer::CalculatePinnedBytes(info->deviceCount, transfer_options, &pinned_bytes, &transfer_config_error)) {
    std::fprintf(stderr, "custom storage transfer configuration invalid: %s\n", transfer_config_error.c_str());
    return post_handle_failure(CUDA_ERROR_INVALID_VALUE);
  }

  std::vector<CUcontext> contexts(info->deviceCount);
  std::vector<CUdevice> devices(info->deviceCount);
  std::vector<storage::DeviceExtent> device_extents;
  device_extents.reserve(info->deviceCount);
  for (unsigned int index = 0; index < info->deviceCount; ++index) {
    status = retained_contexts->ContextAndDeviceForStream(
        info->perDeviceData[index].stream, &contexts[index], &devices[index]);
    if (status != CUDA_SUCCESS) {
      return post_handle_failure(status);
    }
    std::string uuid;
    status = DeviceUUID(devices[index], &uuid);
    if (status != CUDA_SUCCESS) {
      return post_handle_failure(status);
    }
    device_extents.push_back({std::move(uuid), info->perDeviceData[index].size});
  }

  if (checkpoint && !storage::BuildCheckpointManifest(device_extents, &manifest, &manifest_error)) {
    std::fprintf(stderr, "invalid checkpoint custom storage mapping: %s\n", manifest_error.c_str());
    return post_handle_failure(CUDA_ERROR_INVALID_VALUE);
  }
  std::vector<storage::TransferJob> transfer_jobs;
  if (!storage::BuildTransferJobs(
          manifest, device_extents, checkpoint ? std::vector<storage::DevicePair>{} : storage_pairs, &transfer_jobs,
          &manifest_error)) {
    std::fprintf(stderr, "invalid restore custom storage mapping: %s\n", manifest_error.c_str());
    return post_handle_failure(CUDA_ERROR_INVALID_VALUE);
  }

  size_t total_bytes = 0;
  for (const auto& extent : manifest) {
    if (extent.size > std::numeric_limits<size_t>::max() - total_bytes) {
      std::fprintf(stderr, "custom storage byte count overflow\n");
      return post_handle_failure(CUDA_ERROR_INVALID_VALUE);
    }
    total_bytes += extent.size;
  }
  const double metadata_job_construction_seconds = SecondsSince(metadata_job_construction_start);

  const auto start = Clock::now();
  const auto worker_orchestration_start = Clock::now();
  std::vector<std::thread> workers;
  std::vector<unsigned char> worker_success(transfer_jobs.size(), 0);
  std::vector<std::string> worker_errors(transfer_jobs.size());
  std::vector<transfer::TransferMetrics> worker_metrics(transfer_jobs.size());
  transfer::TransferCancellation cancellation;
  std::string worker_start_error;
  try {
    workers.reserve(transfer_jobs.size());
    for (size_t job_index = 0; job_index < transfer_jobs.size(); ++job_index) {
      workers.emplace_back([&, job_index] {
        const auto& job = transfer_jobs[job_index];
        try {
          const auto& device_data = info->perDeviceData[job.device_index];
          const transfer::StorageLayout layout{
              {{storage_dir / manifest[job.extent_index].filename, device_data.size}},
              {{0, device_data.size, 0, 0}},
          };
          worker_success[job_index] = transfer::TransferExtent(
              device_data.devPtr, device_data.size, device_data.stream, contexts[job.device_index], layout,
              checkpoint ? transfer::TransferOperation::kCheckpoint : transfer::TransferOperation::kRestore,
              transfer_options, &cancellation, &worker_metrics[job_index], &worker_errors[job_index]);
        }
        catch (const std::exception& exception) {
          cancellation.Cancel();
          worker_errors[job_index] = exception.what();
        }
        catch (...) {
          cancellation.Cancel();
          worker_errors[job_index] = "unknown worker exception";
        }
      });
    }
  }
  catch (const std::exception& exception) {
    cancellation.Cancel();
    worker_start_error = exception.what();
  }
  catch (...) {
    cancellation.Cancel();
    worker_start_error = "unknown thread creation exception";
  }
  for (auto& worker : workers) {
    worker.join();
  }
  const double worker_orchestration_seconds = SecondsSince(worker_orchestration_start);
  if (!worker_start_error.empty()) {
    std::fprintf(stderr, "failed to start custom storage worker: %s\n", worker_start_error.c_str());
    return post_handle_failure(CUDA_ERROR_OPERATING_SYSTEM);
  }
  for (size_t job_index = 0; job_index < transfer_jobs.size(); ++job_index) {
    if (!worker_success[job_index]) {
      std::fprintf(
          stderr, "custom storage transfer failed for device index %zu: %s\n", transfer_jobs[job_index].device_index,
          worker_errors[job_index].c_str());
      return post_handle_failure(CUDA_ERROR_OPERATING_SYSTEM);
    }
  }

  size_t transferred_bytes = 0;
  double setup_service_seconds = 0.0;
  double pipeline_service_seconds = 0.0;
  double storage_service_seconds = 0.0;
  double cuda_wait_service_seconds = 0.0;
  double fsync_service_seconds = 0.0;
  double cleanup_service_seconds = 0.0;
  for (const auto& metrics : worker_metrics) {
    if (metrics.bytes > std::numeric_limits<size_t>::max() - transferred_bytes) {
      std::fprintf(stderr, "custom storage transferred byte count overflow\n");
      return post_handle_failure(CUDA_ERROR_OPERATING_SYSTEM);
    }
    transferred_bytes += metrics.bytes;
    setup_service_seconds += metrics.setup_seconds;
    pipeline_service_seconds += metrics.pipeline_seconds;
    storage_service_seconds += metrics.storage_seconds;
    cuda_wait_service_seconds += metrics.cuda_wait_seconds;
    fsync_service_seconds += metrics.fsync_seconds;
    cleanup_service_seconds += metrics.cleanup_seconds;
  }
  if (transferred_bytes != total_bytes) {
    std::fprintf(
        stderr, "custom storage transfer coverage mismatch: transferred=%zu expected=%zu\n", transferred_bytes,
        total_bytes);
    return post_handle_failure(CUDA_ERROR_OPERATING_SYSTEM);
  }

  const auto post_transfer_validation_start = Clock::now();
  if (checkpoint) {
    if (!storage::ValidateExtentFiles(storage_dir, manifest, &manifest_error)) {
      std::fprintf(stderr, "custom storage extent validation failed: %s\n", manifest_error.c_str());
      return post_handle_failure(CUDA_ERROR_OPERATING_SYSTEM);
    }
    if (!storage::WriteManifest(storage_dir, manifest, &manifest_error)) {
      std::fprintf(stderr, "custom storage manifest write failed: %s\n", manifest_error.c_str());
      return post_handle_failure(CUDA_ERROR_OPERATING_SYSTEM);
    }
  }
  const double post_transfer_validation_seconds = SecondsSince(post_transfer_validation_start);

  // This is the sole acknowledgment point; CUDA exposes no public abort for failures above.
  const auto operation_complete_start = Clock::now();
  status = static_cast<CUresult>(daemon_protocol::FinishHandledOperation(
      true, CUDA_SUCCESS,
      [operation_complete, info] { return static_cast<int32_t>(operation_complete(info->handle)); },
      &operation));
  const double cuda_operation_complete_seconds = SecondsSince(operation_complete_start);
  if (status != CUDA_SUCCESS) {
    if (checkpoint && !storage::RemoveManifest(storage_dir, &manifest_error)) {
      std::fprintf(
          stderr, "failed to remove custom storage manifest after CUDA completion failure: %s\n",
          manifest_error.c_str());
    }
    return post_handle_failure(status);
  }
  // Preserve the original transfer interval: worker setup through CUDA acknowledgment.
  const double seconds = SecondsSince(start);
  const double gib_per_second =
      seconds == 0.0 ? 0.0 : static_cast<double>(total_bytes) / (1024.0 * 1024.0 * 1024.0) / seconds;
  const auto primary_context_release_start = Clock::now();
  const CUresult primary_context_release_status =
      daemon_contexts == nullptr ? retained_contexts->ReleaseAll() : CUDA_SUCCESS;
  const auto telemetry_end = Clock::now();
  const double primary_context_release_seconds = SecondsBetween(primary_context_release_start, telemetry_end);
  const double custom_storage_total_seconds = SecondsBetween(custom_storage_start, telemetry_end);
  const double helper_main_to_telemetry_seconds = SecondsBetween(helper_main_start, telemetry_end);
  std::fprintf(
      stdout,
      "{\"event\":\"cuda_custom_storage_transfer\",\"schema_version\":1,"
      "\"operation\":\"%s\",\"devices\":%zu,\"bytes\":%zu,\"duration_seconds\":%.6f,"
      "\"effective_gib_per_second\":%.6f,\"transfer_buffer_count\":%zu,"
      "\"transfer_chunk_bytes\":%zu,\"pinned_bytes\":%zu,\"setup_service_seconds\":%.6f,"
      "\"pipeline_service_seconds\":%.6f,\"storage_service_seconds\":%.6f,"
      "\"cuda_wait_service_seconds\":%.6f,\"fsync_service_seconds\":%.6f,"
      "\"cleanup_service_seconds\":%.6f,"
      "\"timing_scope\":\"monotonic_wall;totals_contain_subphases;"
      "service_seconds_are_cross_worker_sums_and_may_overlap\","
      "\"helper_main_to_telemetry_seconds\":%.6f,\"custom_storage_total_seconds\":%.6f,"
      "\"symbol_resolution_seconds\":%.6f,\"storage_directory_validation_seconds\":%.6f,"
      "\"cuda_init_seconds\":%.6f,\"cuda_device_count\":%d,\"device_enumeration_seconds\":%.6f,"
      "\"primary_context_retain_seconds\":%.6f,\"manifest_validation_seconds\":%.6f,"
      "\"device_map_preparation_seconds\":%.6f,\"cuda_process_api_seconds\":%.6f,"
      "\"metadata_job_construction_seconds\":%.6f,\"worker_orchestration_seconds\":%.6f,"
      "\"post_transfer_validation_seconds\":%.6f,\"cuda_operation_complete_seconds\":%.6f,"
      "\"primary_context_release_seconds\":%.6f,\"primary_context_release_state\":\"%s\","
      "\"primary_context_release_success\":%s,\"primary_context_release_status\":%d,"
      "\"context_lifecycle\":\"%s\"}\n",
      checkpoint ? "checkpoint" : "restore", manifest.size(), total_bytes, seconds, gib_per_second,
      transfer_options.buffer_count, transfer_options.chunk_bytes, pinned_bytes, setup_service_seconds,
      pipeline_service_seconds, storage_service_seconds, cuda_wait_service_seconds, fsync_service_seconds,
      cleanup_service_seconds, helper_main_to_telemetry_seconds, custom_storage_total_seconds,
      symbol_resolution_seconds, storage_directory_validation_seconds, cuda_init_seconds, cuda_device_count,
      device_enumeration_seconds, primary_context_retain_seconds, manifest_validation_seconds,
      device_map_preparation_seconds, cuda_process_api_seconds, metadata_job_construction_seconds,
      worker_orchestration_seconds, post_transfer_validation_seconds, cuda_operation_complete_seconds,
      primary_context_release_seconds, daemon_contexts == nullptr ? "completed" : "deferred",
      daemon_contexts == nullptr ? (primary_context_release_status == CUDA_SUCCESS ? "true" : "false") : "null",
      daemon_contexts == nullptr ? static_cast<int>(primary_context_release_status) : -1,
      daemon_contexts == nullptr ? "operation" : "daemon");
  if (primary_context_release_status != CUDA_SUCCESS) {
    std::fprintf(
        stderr, "warning: retained CUDA primary context release failed with status %d after operation acknowledgment\n",
        static_cast<int>(primary_context_release_status));
  }
  if (primary_context_release_status != CUDA_SUCCESS) {
    return {primary_context_release_status, operation};
  }
  return {CUDA_SUCCESS, operation};
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

bool
ReadCapturedFile(FILE* file, std::string* output)
{
  if (std::fflush(file) != 0 || std::fseek(file, 0, SEEK_END) != 0) {
    return false;
  }
  const long size = std::ftell(file);
  if (size < 0 || static_cast<size_t>(size) > daemon_protocol::kMaxResponseSize ||
      std::fseek(file, 0, SEEK_SET) != 0) {
    return false;
  }
  output->resize(size);
  return size == 0 || std::fread(output->data(), 1, size, file) == static_cast<size_t>(size);
}

daemon_protocol::Response
RunDaemonOperation(const daemon_protocol::Request& request, RetainedContexts* contexts)
{
  daemon_protocol::Response response;
  FILE* output_file = std::tmpfile();
  FILE* error_file = std::tmpfile();
  if (output_file == nullptr || error_file == nullptr) {
    if (output_file != nullptr) {
      std::fclose(output_file);
    }
    if (error_file != nullptr) {
      std::fclose(error_file);
    }
    response.cuda_status = CUDA_ERROR_OPERATING_SYSTEM;
    response.flags = daemon_protocol::kResponseFatal;
    response.error = "failed to create daemon output capture";
    return response;
  }
  const int saved_stdout = dup(STDOUT_FILENO);
  const int saved_stderr = dup(STDERR_FILENO);
  if (saved_stdout < 0 || saved_stderr < 0 || dup2(fileno(output_file), STDOUT_FILENO) < 0 ||
      dup2(fileno(error_file), STDERR_FILENO) < 0) {
    response.cuda_status = CUDA_ERROR_OPERATING_SYSTEM;
    response.flags = daemon_protocol::kResponseFatal;
    response.error = "failed to redirect daemon operation output";
  } else {
    if (request.action == daemon_protocol::Action::kLock ||
        request.action == daemon_protocol::Action::kUnlock) {
      std::string identity_error;
      if (!daemon_protocol::ValidateProcessIdentity(request, "/host/proc", &identity_error)) {
        response.cuda_status = CUDA_ERROR_INVALID_VALUE;
        std::fprintf(
            stderr, "process identity changed immediately before CUDA %s: %s\n",
            daemon_protocol::ActionName(request.action), identity_error.c_str());
      } else if (request.action == daemon_protocol::Action::kLock) {
        response.cuda_status = cuCheckpointProcessLock(request.pid, 0);
      } else {
        CUcheckpointUnlockArgs args{};
        response.cuda_status = cuCheckpointProcessUnlock(request.pid, &args);
      }
    } else {
      transfer::TransferOptions options{
          .buffer_count = request.transfer_buffer_count,
          .chunk_bytes = static_cast<size_t>(request.transfer_chunk_bytes),
      };
      std::string validation_error;
      if (!transfer::ValidateTransferOptions(options, &validation_error)) {
        response.cuda_status = CUDA_ERROR_INVALID_VALUE;
        std::fprintf(stderr, "invalid transfer configuration: %s\n", validation_error.c_str());
      } else {
        const CustomStorageResult result = DoCustomStorage(
            request.pid, request.action == daemon_protocol::Action::kCheckpoint, request.device_map,
            request.storage_dir, options, Clock::now(), contexts, &request);
        response.cuda_status = result.status;
        if (result.operation.fatal()) {
          response.flags |= daemon_protocol::kResponseFatal;
        }
      }
    }
    if (response.cuda_status != CUDA_SUCCESS) {
      PrintCudaError(static_cast<CUresult>(response.cuda_status));
    }
  }
  (void)std::fflush(stdout);
  (void)std::fflush(stderr);
  bool output_restore_failed = false;
  if (saved_stdout >= 0) {
    output_restore_failed = dup2(saved_stdout, STDOUT_FILENO) < 0;
    close(saved_stdout);
  }
  if (saved_stderr >= 0) {
    output_restore_failed = dup2(saved_stderr, STDERR_FILENO) < 0 || output_restore_failed;
    close(saved_stderr);
  }
  if (output_restore_failed) {
    response.cuda_status = CUDA_ERROR_OPERATING_SYSTEM;
    response.flags |= daemon_protocol::kResponseFatal;
    response.error = "failed to restore daemon output descriptors";
  }
  if (!ReadCapturedFile(output_file, &response.output) || !ReadCapturedFile(error_file, &response.error)) {
    response.cuda_status = CUDA_ERROR_OPERATING_SYSTEM;
    response.flags |= daemon_protocol::kResponseFatal;
    response.output.clear();
    response.error = "daemon operation output exceeded protocol limit";
  }
  std::fclose(output_file);
  std::fclose(error_file);
  return response;
}

bool
ValidSocketPath(const std::string& path)
{
  const std::filesystem::path socket_path(path);
  const std::string filename = socket_path.filename();
  const bool clean_filename = !filename.empty() && std::all_of(filename.begin(), filename.end(), [](unsigned char c) {
    return std::isalnum(c) || c == '.' || c == '_' || c == '-';
  });
  return !path.empty() && path.front() == '/' &&
         path.size() + sizeof(".health") <= sizeof(sockaddr_un::sun_path) &&
         socket_path.lexically_normal() == socket_path &&
         socket_path.parent_path() == std::filesystem::path("/run/cuda-checkpoint-helper") &&
         clean_filename;
}

int
RunHealthClient(const std::string& socket_path)
{
  sockaddr_un address{};
  const std::string health_socket_path = socket_path + ".health";
  if (!ValidSocketPath(socket_path) ||
      health_socket_path.size() >= sizeof(address.sun_path)) {
    std::fprintf(stderr, "invalid daemon socket path\n");
    return 1;
  }
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, health_socket_path.c_str(), health_socket_path.size() + 1);
  const int socket_fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  if (socket_fd < 0 || connect(socket_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
    if (socket_fd >= 0) {
      close(socket_fd);
    }
    return 1;
  }
  daemon_protocol::Request request;
  std::vector<unsigned char> packet;
  std::string error;
  if (!daemon_protocol::EncodeRequest(request, &packet, &error) ||
      send(socket_fd, packet.data(), packet.size(), MSG_NOSIGNAL) != static_cast<ssize_t>(packet.size())) {
    close(socket_fd);
    return 1;
  }
  packet.resize(daemon_protocol::kMaxResponseSize + 1);
  const ssize_t received = recv(socket_fd, packet.data(), packet.size(), MSG_TRUNC);
  close(socket_fd);
  daemon_protocol::Response response;
  if (received <= 0 || static_cast<size_t>(received) > daemon_protocol::kMaxResponseSize ||
      !daemon_protocol::ParseResponse(packet.data(), received, &response, &error) ||
      response.cuda_status != CUDA_SUCCESS ||
      (response.flags & daemon_protocol::kResponseCapabilityDeferredCUDA) == 0) {
    return 1;
  }
  return 0;
}

void
RunHealthServer(
    daemon_protocol::OwnedUnixSocket* socket, int shutdown_fd,
    const daemon_protocol::OperationHealth* health)
{
  std::vector<unsigned char> packet(daemon_protocol::kMaxRequestSize + 1);
  for (;;) {
    if (daemon_protocol::PollForInputOrStop(socket->fd(), shutdown_fd) == 0) {
      break;
    }
    const int accepted_fd = accept4(socket->fd(), nullptr, nullptr, SOCK_CLOEXEC);
    if (accepted_fd < 0) {
      if (errno == EINTR) {
        continue;
      }
      break;
    }
    ScopedFd client_fd(accepted_fd);
    if (daemon_protocol::PollForInputOrStop(client_fd.get(), shutdown_fd) == 0) {
      break;
    }
    const ssize_t received = recv(client_fd.get(), packet.data(), packet.size(), MSG_TRUNC);
    daemon_protocol::Request request;
    daemon_protocol::Response response;
    std::string error;
    if (received <= 0 || static_cast<size_t>(received) > daemon_protocol::kMaxRequestSize ||
        !daemon_protocol::ParseRequest(packet.data(), received, &request, &error)) {
      response.cuda_status = CUDA_ERROR_INVALID_VALUE;
      response.error = received <= 0 ? "failed to receive health request" : error;
    } else if (request.action != daemon_protocol::Action::kHealth) {
      response.cuda_status = CUDA_ERROR_INVALID_VALUE;
      response.error = "health socket accepts only health requests";
    } else {
      response = daemon_protocol::HealthResponse(*health);
    }
    std::vector<unsigned char> encoded;
    if (daemon_protocol::EncodeResponse(response, &encoded, &error)) {
      (void)send(client_fd.get(), encoded.data(), encoded.size(), MSG_NOSIGNAL);
    }
  }
}

int
RunDaemon(const std::string& socket_path, uint64_t max_operation_seconds)
{
  // Construct contexts first so signal ownership is stopped before contexts are
  // released on every return path.
  RetainedContexts contexts;
  daemon_protocol::ShutdownSignalOwner signal_owner;
  std::string setup_error;
  if (!signal_owner.Start(&setup_error)) {
    std::fprintf(stderr, "daemon shutdown setup failed: %s\n", setup_error.c_str());
    return 1;
  }

  const std::filesystem::path path(socket_path);
  if (!ValidSocketPath(socket_path)) {
    std::fprintf(stderr, "invalid daemon socket path\n");
    return 1;
  }
  std::error_code filesystem_error;
  std::filesystem::create_directories(path.parent_path(), filesystem_error);
  if (filesystem_error || chmod(path.parent_path().c_str(), 0700) != 0) {
    std::fprintf(stderr, "failed to create private daemon socket directory\n");
    return 1;
  }

  const auto init_start = Clock::now();
  CUresult status = cuInit(0);
  const double init_seconds = SecondsSince(init_start);
  if (status != CUDA_SUCCESS) {
    PrintCudaError(status);
    return 1;
  }
  int device_count = 0;
  double enumeration_seconds = 0.0;
  double retain_seconds = 0.0;
  status = contexts.RetainAll(&device_count, &enumeration_seconds, &retain_seconds);
  if (status != CUDA_SUCCESS) {
    PrintCudaError(status);
    return 1;
  }

  daemon_protocol::OwnedUnixSocket operation_socket;
  daemon_protocol::OwnedUnixSocket health_socket;
  std::string socket_error;
  if (!operation_socket.Bind(socket_path, 16, &socket_error)) {
    std::fprintf(stderr, "daemon operation socket setup failed: %s\n", socket_error.c_str());
    return 1;
  }
  if (!health_socket.Bind(socket_path + ".health", 4, &socket_error)) {
    std::fprintf(stderr, "daemon health socket setup failed: %s\n", socket_error.c_str());
    return 1;
  }
  daemon_protocol::OperationHealth operation_health{
      std::chrono::seconds(max_operation_seconds)};
  operation_health.MarkReady();
  daemon_protocol::ShutdownSignalOwner::ShutdownResult shutdown_result;
  daemon_protocol::ShutdownSignalOwner::ShutdownResult health_shutdown_result;
  std::atomic<bool> health_thread_failed{false};
  bool daemon_fatal = false;
  {
    // The guard is destroyed before the jthread: it stops and joins the signal
    // owner, which wakes the health server, and then jthread joins the server.
    std::jthread health_thread;
    try {
      health_thread = std::jthread([&]() noexcept {
        try {
          RunHealthServer(
              &health_socket, signal_owner.health_stop_fd(), &operation_health);
        }
        catch (...) {
          health_shutdown_result = signal_owner.RequestShutdownNoThrow();
          health_thread_failed.store(true, std::memory_order_release);
        }
      });
    }
    catch (const std::system_error& exception) {
      std::fprintf(stderr, "create daemon health thread failed: %s\n", exception.what());
      return 1;
    }
    DaemonThreadShutdown shutdown_threads(&signal_owner, &shutdown_result);
    try {
      std::fprintf(
          stdout,
          "{\"event\":\"cuda_checkpoint_daemon_ready\",\"schema_version\":1,\"cuda_init_seconds\":%.6f,"
          "\"cuda_device_count\":%d,\"device_enumeration_seconds\":%.6f,"
          "\"primary_context_retain_seconds\":%.6f}\n",
          init_seconds, device_count, enumeration_seconds, retain_seconds);
      std::fflush(stdout);

      std::vector<unsigned char> packet(daemon_protocol::kMaxRequestSize + 1);
      while (!signal_owner.ShutdownRequested() && !daemon_fatal) {
        if (daemon_protocol::PollForInputOrStop(
                operation_socket.fd(), signal_owner.operation_stop_fd()) == 0) {
          break;
        }
        const int accepted_fd =
            accept4(operation_socket.fd(), nullptr, nullptr, SOCK_CLOEXEC);
        if (accepted_fd < 0) {
          if (errno == EINTR) {
            continue;
          }
          if (signal_owner.ShutdownRequested()) {
            break;
          }
          std::perror("accept");
          break;
        }
        ScopedFd client_fd(accepted_fd);
        if (daemon_protocol::PollForInputOrStop(
                client_fd.get(), signal_owner.operation_stop_fd()) == 0) {
          break;
        }
        const ssize_t received =
            recv(client_fd.get(), packet.data(), packet.size(), MSG_TRUNC);
        daemon_protocol::Response response;
        daemon_protocol::Request request;
        std::string protocol_error;
        if (received <= 0 || static_cast<size_t>(received) > daemon_protocol::kMaxRequestSize ||
            !daemon_protocol::ParseRequest(packet.data(), received, &request, &protocol_error)) {
          response.cuda_status = CUDA_ERROR_INVALID_VALUE;
          response.error = received <= 0 ? "failed to receive request" : protocol_error;
        } else if (request.action == daemon_protocol::Action::kHealth) {
          response.cuda_status = CUDA_ERROR_INVALID_VALUE;
          response.error = "health requests must use the health socket";
        } else {
          const auto rpc_start = Clock::now();
          operation_health.Begin(request.action, request.pid);
          daemon_fatal = !daemon_protocol::ExecuteValidated(
              request, "/host/proc",
              [&contexts](const daemon_protocol::Request& validated) {
                return RunDaemonOperation(validated, &contexts);
              },
              &response);
          operation_health.End();
          std::fprintf(
              stdout,
              "{\"event\":\"cuda_checkpoint_daemon_operation\",\"schema_version\":1,\"action\":\"%s\","
              "\"pid\":%u,\"cuda_status\":%d,\"fatal\":%s,\"rpc_service_seconds\":%.6f}\n",
              daemon_protocol::ActionName(request.action), request.pid,
              response.cuda_status, (response.flags & daemon_protocol::kResponseFatal) != 0 ? "true" : "false",
              SecondsSince(rpc_start));
          std::fflush(stdout);
        }
        std::vector<unsigned char> encoded;
        if (!daemon_protocol::EncodeResponse(response, &encoded, &protocol_error)) {
          daemon_protocol::Response bounded{
              .cuda_status = CUDA_ERROR_OPERATING_SYSTEM,
              .flags = response.flags & daemon_protocol::kResponseFatal,
              .output = "",
              .error = "daemon response exceeded protocol limit",
          };
          (void)daemon_protocol::EncodeResponse(bounded, &encoded, &protocol_error);
        }
        (void)send(client_fd.get(), encoded.data(), encoded.size(), MSG_NOSIGNAL);
      }
    }
    catch (const std::exception& exception) {
      std::fprintf(stderr, "daemon processing failed: %s\n", exception.what());
      daemon_fatal = true;
    }
    catch (...) {
      std::fprintf(stderr, "daemon processing failed: unknown exception\n");
      daemon_fatal = true;
    }
  }
  if (health_thread_failed.load(std::memory_order_acquire)) {
    std::fprintf(stderr, "daemon health thread failed\n");
    daemon_fatal = true;
    if (!health_shutdown_result.ok()) {
      shutdown_result = health_shutdown_result;
    }
  }
  if (!shutdown_result.ok()) {
    std::fprintf(
        stderr, "daemon shutdown failed: %s: %s\n",
        shutdown_result.operation, std::strerror(shutdown_result.error_code));
    daemon_fatal = true;
  }
  signal_owner.Close();
  health_socket.Close();
  operation_socket.Close();
  const auto release_start = Clock::now();
  status = contexts.ReleaseAll();
  std::fprintf(
      stdout,
      "{\"event\":\"cuda_checkpoint_daemon_stopped\",\"schema_version\":1,"
      "\"primary_context_release_seconds\":%.6f,\"primary_context_release_status\":%d}\n",
      SecondsSince(release_start), static_cast<int>(status));
  return status == CUDA_SUCCESS && !daemon_fatal ? 0 : 1;
}

}  // namespace

int
main(int argc, char** argv)
{
  const auto helper_main_start = Clock::now();
  std::string action;
  std::string device_map;
  std::string storage_mode = "legacy";
  std::string storage_dir;
  int pid = 0;
  bool have_pid = false;
  bool get_state = false;
  bool get_restore_tid = false;
  bool daemon = false;
  bool health = false;
  std::string socket_path;
  uint64_t max_operation_seconds = 6 * 60 * 60;
  unsigned int timeout_ms = 0;
  transfer::TransferOptions transfer_options;
  bool transfer_options_set = false;

  if (argc == 1) {
    return PrintUsageError();
  }
  for (int index = 1; index < argc; ++index) {
    std::string argument = argv[index];
    if (argument == "--get-state") {
      get_state = true;
    } else if (argument == "--daemon") {
      daemon = true;
    } else if (argument == "--health") {
      health = true;
    } else if (argument == "--socket" && ++index < argc) {
      socket_path = argv[index];
    } else if (
        argument == "--max-operation-seconds" && ++index < argc &&
        ParsePositiveSeconds(argv[index], &max_operation_seconds)) {
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
    } else if (
        argument == "--transfer-buffer-count" && ++index < argc &&
        transfer::ParseSize(argv[index], &transfer_options.buffer_count)) {
      transfer_options_set = true;
    } else if (
        argument == "--transfer-chunk-bytes" && ++index < argc &&
        transfer::ParseSize(argv[index], &transfer_options.chunk_bytes)) {
      transfer_options_set = true;
    } else if (argument == "--help" || argument == "-h") {
      return PrintUsage(stdout);
    } else {
      return PrintUsageError();
    }
  }

  if (daemon || health) {
    if (static_cast<int>(daemon) + static_cast<int>(health) != 1 || socket_path.empty() || have_pid || get_state ||
        get_restore_tid || !action.empty() || storage_mode != "legacy" || !storage_dir.empty() ||
        !device_map.empty() || timeout_ms != 0 || transfer_options_set ||
        (health && max_operation_seconds != 6 * 60 * 60)) {
      return PrintUsageError();
    }
    return daemon ? RunDaemon(socket_path, max_operation_seconds) : RunHealthClient(socket_path);
  }

  if (static_cast<int>(get_state) + static_cast<int>(get_restore_tid) + static_cast<int>(!action.empty()) != 1 ||
      !have_pid || (storage_mode != "legacy" && storage_mode != "posix") ||
      ((storage_mode == "posix") != !storage_dir.empty())) {
    return PrintUsageError();
  }
  std::string transfer_error;
  if (!transfer::ValidateTransferOptions(transfer_options, &transfer_error)) {
    std::fprintf(stderr, "invalid transfer configuration: %s\n", transfer_error.c_str());
    return 1;
  }
  if (transfer_options_set && storage_mode != "posix") {
    return PrintUsageError();
  }

  CUresult status = CUDA_SUCCESS;
  if (get_state) {
    if (timeout_ms != 0 || !device_map.empty() || storage_mode != "legacy") {
      return PrintUsageError();
    }
    CUprocessState state;
    status = cuCheckpointProcessGetState(pid, &state);
    if (status == CUDA_SUCCESS) {
      return std::fprintf(stdout, "%s\n", ProcessStateString(state)) < 0 ? 1 : 0;
    }
  } else if (get_restore_tid) {
    if (timeout_ms != 0 || !device_map.empty() || storage_mode != "legacy") {
      return PrintUsageError();
    }
    int tid = 0;
    status = cuCheckpointProcessGetRestoreThreadId(pid, &tid);
    if (status == CUDA_SUCCESS) {
      return std::fprintf(stdout, "%d\n", tid) < 0 ? 1 : 0;
    }
  } else if (action == "lock") {
    if (!device_map.empty() || storage_mode != "legacy") {
      return PrintUsageError();
    }
    CUcheckpointLockArgs args{};
    args.timeoutMs = timeout_ms;
    status = cuCheckpointProcessLock(pid, &args);
  } else if (action == "checkpoint") {
    if (timeout_ms != 0 || !device_map.empty()) {
      return PrintUsageError();
    }
    if (storage_mode == "posix") {
      status = DoCustomStorage(pid, true, "", storage_dir, transfer_options, helper_main_start).status;
    } else {
      CUcheckpointCheckpointArgs args{};
      status = cuCheckpointProcessCheckpoint(pid, &args);
    }
  } else if (action == "restore") {
    if (timeout_ms != 0) {
      return PrintUsageError();
    }
    status = storage_mode == "posix"
                 ? DoCustomStorage(pid, false, device_map, storage_dir, transfer_options, helper_main_start).status
                                     : DoLegacyRestore(pid, device_map);
  } else if (action == "unlock") {
    if (timeout_ms != 0 || !device_map.empty() || storage_mode != "legacy") {
      return PrintUsageError();
    }
    CUcheckpointUnlockArgs args{};
    status = cuCheckpointProcessUnlock(pid, &args);
  } else {
    return PrintUsageError();
  }

  if (status != CUDA_SUCCESS) {
    PrintCudaError(status);
    return 1;
  }
  return 0;
}
