/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "transfer_engine.h"

#include <fcntl.h>
#include <nixl.h>
#include <nixl_descriptors.h>
#include <nixl_params.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace cuda_checkpoint_transfer {
namespace {

using Clock = std::chrono::steady_clock;

double
ElapsedSeconds(Clock::time_point start)
{
  return std::chrono::duration<double>(Clock::now() - start).count();
}

void
AppendError(std::string* error, const std::string& detail)
{
  if (!error->empty()) {
    *error += "; ";
  }
  *error += detail;
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

class FileDescriptor {
 public:
  explicit FileDescriptor(int fd = -1) : fd_(fd) {}
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  FileDescriptor(FileDescriptor&& other) noexcept : fd_(std::exchange(other.fd_, -1)) {}
  FileDescriptor& operator=(FileDescriptor&&) = delete;

  ~FileDescriptor()
  {
    if (fd_ >= 0) {
      (void)close(fd_);
    }
  }

  int get() const { return fd_; }

  bool Close(std::string* error)
  {
    if (fd_ < 0) {
      return true;
    }
    const int fd = std::exchange(fd_, -1);
    if (close(fd) == 0) {
      return true;
    }
    AppendError(error, "close storage file failed: " + std::string(std::strerror(errno)));
    return false;
  }

 private:
  int fd_;
};

class TransferSlot {
 public:
  TransferSlot() = default;
  TransferSlot(const TransferSlot&) = delete;
  TransferSlot& operator=(const TransferSlot&) = delete;

  ~TransferSlot()
  {
    if (cuda_may_access_) {
      return;
    }
    if (event_ != nullptr) {
      (void)cuEventDestroy(event_);
    }
    if (data_ != nullptr) {
      if (cuMemHostUnregister(data_) == CUDA_SUCCESS) {
        free(data_);
      }
    }
  }

  CUresult Allocate(size_t size)
  {
    if (posix_memalign(&data_, kBufferAlignment, size) != 0) {
      return CUDA_ERROR_OUT_OF_MEMORY;
    }
    CUresult status = cuMemHostRegister(data_, size, 0);
    if (status != CUDA_SUCCESS) {
      free(data_);
      data_ = nullptr;
      return status;
    }
    status = cuEventCreate(&event_, CU_EVENT_DISABLE_TIMING);
    if (status != CUDA_SUCCESS) {
      if (cuMemHostUnregister(data_) == CUDA_SUCCESS) {
        free(data_);
        data_ = nullptr;
      }
    }
    return status;
  }

  void* data() const { return data_; }
  CUevent event() const { return event_; }
  bool pending() const { return pending_; }
  void set_pending(bool pending) { pending_ = pending; }
  void set_cuda_may_access(bool cuda_may_access) { cuda_may_access_ = cuda_may_access; }

  void MarkCUDAComplete()
  {
    pending_ = false;
    cuda_may_access_ = false;
  }

  bool Close(std::string* error)
  {
    if (cuda_may_access_) {
      AppendError(error, "CUDA transfer buffer retained because stream drain did not complete");
      return false;
    }
    bool success = true;
    if (event_ != nullptr) {
      const CUresult status = cuEventDestroy(event_);
      event_ = nullptr;
      if (status != CUDA_SUCCESS) {
        success = false;
        AppendError(error, "destroy CUDA transfer event failed: " + CudaError(status));
      }
    }
    if (data_ != nullptr) {
      const CUresult status = cuMemHostUnregister(data_);
      if (status == CUDA_SUCCESS) {
        free(data_);
        data_ = nullptr;
      } else {
        success = false;
        AppendError(error, "unregister CUDA transfer buffer failed: " + CudaError(status));
      }
    }
    return success;
  }

 private:
  void* data_ = nullptr;
  CUevent event_ = nullptr;
  bool pending_ = false;
  bool cuda_may_access_ = false;
};

class StreamDrainGuard {
 public:
  StreamDrainGuard(CUstream stream, std::vector<std::unique_ptr<TransferSlot>>* slots) : stream_(stream), slots_(slots)
  {
  }
  StreamDrainGuard(const StreamDrainGuard&) = delete;
  StreamDrainGuard& operator=(const StreamDrainGuard&) = delete;

  ~StreamDrainGuard()
  {
    if (armed_ && cuStreamSynchronize(stream_) == CUDA_SUCCESS) {
      for (auto& slot : *slots_) {
        slot->MarkCUDAComplete();
      }
    }
  }

  void Disarm() { armed_ = false; }

 private:
  CUstream stream_;
  std::vector<std::unique_ptr<TransferSlot>>* slots_;
  bool armed_ = true;
};

class NixlRegistrationGuard {
 public:
  NixlRegistrationGuard(nixlAgent* agent, nixl_reg_dlist_t* dram_registration, nixl_reg_dlist_t* file_registration)
      : agent_(agent), dram_registration_(dram_registration), file_registration_(file_registration)
  {
  }
  NixlRegistrationGuard(const NixlRegistrationGuard&) = delete;
  NixlRegistrationGuard& operator=(const NixlRegistrationGuard&) = delete;

  ~NixlRegistrationGuard()
  {
    if (armed_) {
      (void)agent_->deregisterMem(*file_registration_);
      (void)agent_->deregisterMem(*dram_registration_);
    }
  }

  void Disarm() { armed_ = false; }

 private:
  nixlAgent* agent_;
  nixl_reg_dlist_t* dram_registration_;
  nixl_reg_dlist_t* file_registration_;
  bool armed_ = true;
};

bool
OpenStorageFiles(
    const StorageLayout& storage, TransferOperation operation, std::vector<FileDescriptor>* files, std::string* error)
{
  files->clear();
  files->reserve(storage.files.size());
  for (size_t index = 0; index < storage.files.size(); ++index) {
    const auto& file = storage.files[index];
    if (file.size > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
      *error = "storage file " + std::to_string(index) + " is too large for POSIX offsets";
      return false;
    }
    FileDescriptor descriptor(open(file.path.c_str(), StorageFileOpenFlags(operation), 0600));
    if (descriptor.get() < 0) {
      *error = "open storage file " + std::to_string(index) + " failed: " + std::strerror(errno);
      return false;
    }
    if (operation == TransferOperation::kCheckpoint) {
      if (fchmod(descriptor.get(), 0600) != 0 || ftruncate(descriptor.get(), static_cast<off_t>(file.size)) != 0) {
        *error = "prepare checkpoint storage file " + std::to_string(index) + " failed: " + std::strerror(errno);
        return false;
      }
    } else {
      struct stat file_stat {};
      if (fstat(descriptor.get(), &file_stat) != 0 || !S_ISREG(file_stat.st_mode) || file_stat.st_size < 0 ||
          static_cast<size_t>(file_stat.st_size) != file.size) {
        *error = "restore storage file " + std::to_string(index) + " is not regular or has the wrong size";
        return false;
      }
    }
    files->push_back(std::move(descriptor));
  }
  return true;
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
  const nixl_status_t release_status = agent->releaseXferReq(request);
  if (status != NIXL_SUCCESS) {
    *error = "NIXL transfer failed with status " + std::to_string(status);
    if (release_status != NIXL_SUCCESS) {
      AppendError(error, "releaseXferReq also failed with status " + std::to_string(release_status));
    }
    return false;
  }
  if (release_status != NIXL_SUCCESS) {
    *error = "NIXL releaseXferReq failed with status " + std::to_string(release_status);
    return false;
  }
  return true;
}

bool
WaitForSlot(TransferSlot* slot, TransferMetrics* metrics, std::string* error)
{
  if (!slot->pending()) {
    return true;
  }
  const auto start = Clock::now();
  const CUresult status = cuEventSynchronize(slot->event());
  metrics->cuda_wait_seconds += ElapsedSeconds(start);
  if (status != CUDA_SUCCESS) {
    *error = "CUDA event synchronization failed: " + CudaError(status);
    return false;
  }
  slot->MarkCUDAComplete();
  return true;
}

bool
EnqueueCopy(
    TransferOperation operation, const TransferChunk& chunk, TransferSlot* slot, CUdeviceptr device_ptr,
    CUstream stream, bool* cuda_work_posted, std::string* error)
{
  CUresult status = CUDA_SUCCESS;
  if (operation == TransferOperation::kCheckpoint) {
    status = cuMemcpyDtoHAsync(slot->data(), device_ptr + chunk.logical_offset, chunk.size, stream);
  } else {
    status = cuMemcpyHtoDAsync(device_ptr + chunk.logical_offset, slot->data(), chunk.size, stream);
  }
  if (status != CUDA_SUCCESS) {
    *error = "CUDA asynchronous copy failed at logical offset " + std::to_string(chunk.logical_offset) + ": " +
             CudaError(status);
    return false;
  }
  *cuda_work_posted = true;
  slot->set_cuda_may_access(true);
  status = cuEventRecord(slot->event(), stream);
  if (status != CUDA_SUCCESS) {
    *error =
        "CUDA event record failed at logical offset " + std::to_string(chunk.logical_offset) + ": " + CudaError(status);
    return false;
  }
  slot->set_pending(true);
  return true;
}

bool
DrainCUDA(
    std::vector<std::unique_ptr<TransferSlot>>* slots, CUstream stream, bool force_stream_sync, bool cuda_work_posted,
    TransferMetrics* metrics, std::string* error)
{
  bool success = true;
  for (auto& slot : *slots) {
    std::string wait_error;
    if (!WaitForSlot(slot.get(), metrics, &wait_error)) {
      success = false;
      AppendError(error, wait_error);
    }
  }
  if (cuda_work_posted && (force_stream_sync || !success)) {
    const auto start = Clock::now();
    const CUresult status = cuStreamSynchronize(stream);
    metrics->cuda_wait_seconds += ElapsedSeconds(start);
    if (status != CUDA_SUCCESS) {
      success = false;
      AppendError(error, "CUDA stream drain failed: " + CudaError(status));
    } else {
      for (auto& slot : *slots) {
        slot->MarkCUDAComplete();
      }
    }
  }
  return success;
}

bool
TransferPipeline(
    const std::vector<TransferChunk>& chunks, const std::vector<FileDescriptor>& files,
    std::vector<std::unique_ptr<TransferSlot>>* slots, nixlAgent* agent, const std::string& agent_name,
    CUdeviceptr device_ptr, CUstream stream, TransferOperation operation, TransferMetrics* metrics,
    TransferCancellation* cancellation, std::string* error)
{
  bool success = true;
  bool cuda_work_posted = false;
  if (operation == TransferOperation::kRestore) {
    for (const auto& chunk : chunks) {
      if (cancellation != nullptr && cancellation->IsCancelled()) {
        *error = "transfer canceled after another extent failed";
        success = false;
        break;
      }
      TransferSlot* slot = (*slots)[chunk.slot_index].get();
      if (!WaitForSlot(slot, metrics, error)) {
        success = false;
        if (cancellation != nullptr) {
          cancellation->Cancel();
        }
        break;
      }
      if (cancellation != nullptr && cancellation->IsCancelled()) {
        *error = "transfer canceled after another extent failed";
        success = false;
        break;
      }
      const auto storage_start = Clock::now();
      std::string transfer_error;
      const bool transferred = NixlTransfer(
          agent, agent_name, NIXL_READ, slot->data(), files[chunk.file_index].get(), chunk.file_offset, chunk.size,
          &transfer_error);
      const double storage_seconds = ElapsedSeconds(storage_start);
      metrics->storage_seconds += storage_seconds;
      metrics->files[chunk.file_index].storage_seconds += storage_seconds;
      if (!transferred) {
        *error =
            "storage read failed at logical offset " + std::to_string(chunk.logical_offset) + ": " + transfer_error;
        success = false;
        if (cancellation != nullptr) {
          cancellation->Cancel();
        }
        break;
      }
      metrics->files[chunk.file_index].bytes += chunk.size;
      if (cancellation != nullptr && cancellation->IsCancelled()) {
        *error = "transfer canceled after another extent failed";
        success = false;
        break;
      }
      if (!EnqueueCopy(operation, chunk, slot, device_ptr, stream, &cuda_work_posted, error)) {
        success = false;
        if (cancellation != nullptr) {
          cancellation->Cancel();
        }
        break;
      }
    }
  } else {
    size_t next_chunk = 0;
    const size_t initial_count = std::min(chunks.size(), slots->size());
    for (; next_chunk < initial_count; ++next_chunk) {
      if (cancellation != nullptr && cancellation->IsCancelled()) {
        *error = "transfer canceled after another extent failed";
        success = false;
        break;
      }
      const auto& chunk = chunks[next_chunk];
      if (!EnqueueCopy(
              operation, chunk, (*slots)[chunk.slot_index].get(), device_ptr, stream, &cuda_work_posted, error)) {
        success = false;
        if (cancellation != nullptr) {
          cancellation->Cancel();
        }
        break;
      }
    }
    size_t completed = 0;
    for (; success && completed < next_chunk; ++completed) {
      if (cancellation != nullptr && cancellation->IsCancelled()) {
        *error = "transfer canceled after another extent failed";
        success = false;
        break;
      }
      const auto& chunk = chunks[completed];
      TransferSlot* slot = (*slots)[chunk.slot_index].get();
      if (!WaitForSlot(slot, metrics, error)) {
        success = false;
        if (cancellation != nullptr) {
          cancellation->Cancel();
        }
        break;
      }
      if (cancellation != nullptr && cancellation->IsCancelled()) {
        *error = "transfer canceled after another extent failed";
        success = false;
        break;
      }
      const auto storage_start = Clock::now();
      std::string transfer_error;
      const bool transferred = NixlTransfer(
          agent, agent_name, NIXL_WRITE, slot->data(), files[chunk.file_index].get(), chunk.file_offset, chunk.size,
          &transfer_error);
      const double storage_seconds = ElapsedSeconds(storage_start);
      metrics->storage_seconds += storage_seconds;
      metrics->files[chunk.file_index].storage_seconds += storage_seconds;
      if (!transferred) {
        *error =
            "storage write failed at logical offset " + std::to_string(chunk.logical_offset) + ": " + transfer_error;
        success = false;
        if (cancellation != nullptr) {
          cancellation->Cancel();
        }
        break;
      }
      metrics->files[chunk.file_index].bytes += chunk.size;
      if (next_chunk < chunks.size()) {
        const auto& next = chunks[next_chunk];
        if (next.slot_index != chunk.slot_index) {
          *error = "internal transfer ring scheduling error";
          success = false;
          if (cancellation != nullptr) {
            cancellation->Cancel();
          }
          break;
        }
        if (cancellation != nullptr && cancellation->IsCancelled()) {
          *error = "transfer canceled after another extent failed";
          success = false;
          break;
        }
        if (!EnqueueCopy(operation, next, slot, device_ptr, stream, &cuda_work_posted, error)) {
          success = false;
          if (cancellation != nullptr) {
            cancellation->Cancel();
          }
          break;
        }
        ++next_chunk;
      }
    }
    if (success && completed != chunks.size()) {
      *error = "internal checkpoint transfer coverage error";
      success = false;
      if (cancellation != nullptr) {
        cancellation->Cancel();
      }
    }
  }

  if (!DrainCUDA(slots, stream, !success, cuda_work_posted, metrics, error)) {
    success = false;
    if (cancellation != nullptr) {
      cancellation->Cancel();
    }
  }
  return success;
}

std::string
MakeAgentName()
{
  static std::atomic<unsigned long long> sequence{0};
  return "cuda-custom-storage-" + std::to_string(getpid()) + "-" + std::to_string(sequence.fetch_add(1));
}

}  // namespace

bool
TransferExtent(
    CUdeviceptr device_ptr, size_t extent_size, CUstream stream, CUcontext context, const StorageLayout& storage,
    TransferOperation operation, const TransferOptions& options, TransferCancellation* cancellation,
    TransferMetrics* metrics, std::string* error)
{
  if (metrics == nullptr || error == nullptr) {
    return false;
  }
  *metrics = {};
  metrics->files.resize(storage.files.size());
  error->clear();
  const auto total_start = Clock::now();

  std::vector<TransferChunk> chunks;
  if (!BuildTransferChunks(extent_size, storage, options, &chunks, error)) {
    if (cancellation != nullptr) {
      cancellation->Cancel();
    }
    metrics->total_seconds = ElapsedSeconds(total_start);
    return false;
  }
  if (extent_size > std::numeric_limits<CUdeviceptr>::max() - device_ptr) {
    *error = "CUDA device extent address calculation overflow";
    if (cancellation != nullptr) {
      cancellation->Cancel();
    }
    metrics->total_seconds = ElapsedSeconds(total_start);
    return false;
  }
  if (device_ptr == 0 || context == nullptr) {
    *error = "CUDA device pointer and context must be valid";
    if (cancellation != nullptr) {
      cancellation->Cancel();
    }
    metrics->total_seconds = ElapsedSeconds(total_start);
    return false;
  }
  if (cancellation != nullptr && cancellation->IsCancelled()) {
    *error = "transfer canceled before setup";
    metrics->total_seconds = ElapsedSeconds(total_start);
    return false;
  }

  CUresult cuda_status = cuCtxSetCurrent(context);
  if (cuda_status != CUDA_SUCCESS) {
    *error = "set CUDA context failed: " + CudaError(cuda_status);
    if (cancellation != nullptr) {
      cancellation->Cancel();
    }
    metrics->total_seconds = ElapsedSeconds(total_start);
    return false;
  }

  const auto setup_start = Clock::now();
  std::vector<std::unique_ptr<TransferSlot>> slots;
  slots.reserve(options.buffer_count);
  for (size_t index = 0; index < options.buffer_count; ++index) {
    auto slot = std::make_unique<TransferSlot>();
    cuda_status = slot->Allocate(options.chunk_bytes);
    if (cuda_status != CUDA_SUCCESS) {
      *error = "allocate CUDA-registered transfer slot " + std::to_string(index) + " failed: " + CudaError(cuda_status);
      if (cancellation != nullptr) {
        cancellation->Cancel();
      }
      metrics->setup_seconds = ElapsedSeconds(setup_start);
      metrics->total_seconds = ElapsedSeconds(total_start);
      return false;
    }
    slots.push_back(std::move(slot));
  }

  std::vector<FileDescriptor> files;
  if (!OpenStorageFiles(storage, operation, &files, error)) {
    if (cancellation != nullptr) {
      cancellation->Cancel();
    }
    metrics->setup_seconds = ElapsedSeconds(setup_start);
    metrics->total_seconds = ElapsedSeconds(total_start);
    return false;
  }

  const std::string agent_name = MakeAgentName();
  nixlAgentConfig config;
  config.useProgThread = true;
  auto agent = std::make_unique<nixlAgent>(agent_name, config);
  nixl_b_params_t params;
  params["use_aio"] = "true";
  nixlBackendH* backend = nullptr;
  nixl_status_t nixl_status = agent->createBackend("POSIX", params, backend);
  if (nixl_status != NIXL_SUCCESS) {
    *error = "create NIXL POSIX backend failed with status " + std::to_string(nixl_status);
    if (cancellation != nullptr) {
      cancellation->Cancel();
    }
    metrics->setup_seconds = ElapsedSeconds(setup_start);
    metrics->total_seconds = ElapsedSeconds(total_start);
    return false;
  }

  nixl_reg_dlist_t dram_registration(DRAM_SEG);
  nixl_reg_dlist_t file_registration(FILE_SEG);
  for (const auto& slot : slots) {
    dram_registration.addDesc(nixlBlobDesc(reinterpret_cast<uintptr_t>(slot->data()), options.chunk_bytes, 0));
  }
  for (size_t index = 0; index < storage.files.size(); ++index) {
    file_registration.addDesc(nixlBlobDesc(0, storage.files[index].size, files[index].get()));
  }

  bool dram_registered = false;
  bool file_registered = false;
  nixl_status = agent->registerMem(dram_registration);
  if (nixl_status == NIXL_SUCCESS) {
    dram_registered = true;
    nixl_status = agent->registerMem(file_registration);
    file_registered = nixl_status == NIXL_SUCCESS;
  }
  if (!dram_registered || !file_registered) {
    *error = std::string("register NIXL ") + (dram_registered ? "file" : "DRAM") + " memory failed with status " +
             std::to_string(nixl_status);
    if (dram_registered) {
      const nixl_status_t cleanup_status = agent->deregisterMem(dram_registration);
      if (cleanup_status != NIXL_SUCCESS) {
        AppendError(error, "NIXL DRAM cleanup failed with status " + std::to_string(cleanup_status));
      }
    }
    metrics->setup_seconds = ElapsedSeconds(setup_start);
    metrics->total_seconds = ElapsedSeconds(total_start);
    return false;
  }
  metrics->setup_seconds = ElapsedSeconds(setup_start);
  NixlRegistrationGuard registration_guard(agent.get(), &dram_registration, &file_registration);

  const auto pipeline_start = Clock::now();
  bool success = false;
  {
    StreamDrainGuard stream_drain_guard(stream, &slots);
    success = TransferPipeline(
        chunks, files, &slots, agent.get(), agent_name, device_ptr, stream, operation, metrics, cancellation, error);
    if (success) {
      stream_drain_guard.Disarm();
    }
  }
  metrics->pipeline_seconds = ElapsedSeconds(pipeline_start);

  if (success && cancellation != nullptr && cancellation->IsCancelled()) {
    success = false;
    *error = "transfer canceled after another extent failed";
  }
  if (success && operation == TransferOperation::kCheckpoint) {
    for (size_t index = 0; index < files.size(); ++index) {
      const auto fsync_start = Clock::now();
      const int result = fsync(files[index].get());
      const double fsync_seconds = ElapsedSeconds(fsync_start);
      metrics->fsync_seconds += fsync_seconds;
      metrics->files[index].fsync_seconds += fsync_seconds;
      if (result != 0) {
        success = false;
        if (cancellation != nullptr) {
          cancellation->Cancel();
        }
        *error = "fsync storage file " + std::to_string(index) + " failed: " + std::strerror(errno);
        break;
      }
    }
  }

  const auto cleanup_start = Clock::now();
  const nixl_status_t file_deregister_status = agent->deregisterMem(file_registration);
  const nixl_status_t dram_deregister_status = agent->deregisterMem(dram_registration);
  registration_guard.Disarm();
  if (file_deregister_status != NIXL_SUCCESS) {
    success = false;
    if (cancellation != nullptr) {
      cancellation->Cancel();
    }
    AppendError(error, "NIXL file deregistration failed with status " + std::to_string(file_deregister_status));
  }
  if (dram_deregister_status != NIXL_SUCCESS) {
    success = false;
    if (cancellation != nullptr) {
      cancellation->Cancel();
    }
    AppendError(error, "NIXL DRAM deregistration failed with status " + std::to_string(dram_deregister_status));
  }
  agent.reset();
  for (auto& file : files) {
    if (!file.Close(error)) {
      success = false;
      if (cancellation != nullptr) {
        cancellation->Cancel();
      }
    }
  }
  for (auto& slot : slots) {
    if (!slot->Close(error)) {
      success = false;
      if (cancellation != nullptr) {
        cancellation->Cancel();
      }
    }
  }
  slots.clear();
  metrics->cleanup_seconds = ElapsedSeconds(cleanup_start);
  metrics->total_seconds = ElapsedSeconds(total_start);
  if (success) {
    metrics->bytes = extent_size;
  }
  return success;
}

}  // namespace cuda_checkpoint_transfer
