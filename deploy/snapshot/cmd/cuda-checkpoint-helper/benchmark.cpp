/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <cuda.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "benchmark_cache.h"
#include "transfer_config.h"
#include "transfer_engine.h"

namespace {

namespace benchmark = cuda_checkpoint_benchmark;
namespace transfer = cuda_checkpoint_transfer;

constexpr size_t kPatternBufferBytes = 16ULL * 1024ULL * 1024ULL;
constexpr size_t kMaximumIterations = 1000;

struct Options {
  transfer::TransferOperation operation = transfer::TransferOperation::kRestore;
  std::string operation_name = "restore";
  std::filesystem::path file;
  size_t bytes = 0;
  int device = 0;
  transfer::TransferOptions transfer;
  size_t file_count = 1;
  size_t warmups = 1;
  size_t iterations = 3;
  benchmark::CacheProfile cache_profile = benchmark::CacheProfile::kBufferedWarm;
  bool operation_set = false;
};

int
PrintUsage(FILE* stream)
{
  return std::fprintf(
             stream,
             "Usage:\n"
             "  cuda-nixl-posix-benchmark --operation checkpoint|restore "
             "--file <absolute-path> --bytes <bytes> [options]\n"
             "\n"
             "Options:\n"
             "  --device <ordinal>                 CUDA device ordinal (default: 0)\n"
             "  --transfer-buffer-count <count>    Pinned pipeline slots (default: %zu)\n"
             "  --transfer-chunk-bytes <bytes>     Bytes per slot (default: %zu)\n"
             "  --file-count <count>               Contiguous physical shards (default: 1)\n"
             "  --warmups <count>                  Untimed warmup runs (default: 1)\n"
             "  --iterations <count>               Measured runs (default: 3)\n"
             "  --cache-profile <profile>          Restore cache profile: buffered-warm or\n"
             "                                     client-evicted (default: buffered-warm)\n"
             "  --help                             Show this help\n"
             "\n"
             "Aliases: write=checkpoint, read=restore. Each measured run writes one JSON object "
             "to stdout; the human summary is written to stderr. client-evicted uses best-effort "
             "per-file POSIX_FADV_DONTNEED; server/storage caches may remain warm.\n",
             transfer::kDefaultBufferCount, transfer::kDefaultChunkBytes) < 0
             ? 1
             : 0;
}

bool
ParseInt(std::string_view value, int* parsed)
{
  size_t unsigned_value = 0;
  if (!transfer::ParseSize(value, &unsigned_value) ||
      unsigned_value > static_cast<size_t>(std::numeric_limits<int>::max())) {
    return false;
  }
  *parsed = static_cast<int>(unsigned_value);
  return true;
}

bool
ParseOptions(int argc, char** argv, Options* options, bool* help, std::string* error)
{
  *help = false;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--operation" && ++index < argc) {
      const std::string operation = argv[index];
      if (operation == "checkpoint" || operation == "write") {
        options->operation = transfer::TransferOperation::kCheckpoint;
        options->operation_name = "checkpoint";
        options->operation_set = true;
      } else if (operation == "restore" || operation == "read") {
        options->operation = transfer::TransferOperation::kRestore;
        options->operation_name = "restore";
        options->operation_set = true;
      } else {
        *error = "--operation must be checkpoint, write, restore, or read";
        return false;
      }
    } else if (argument == "--file" && ++index < argc) {
      options->file = argv[index];
    } else if (argument == "--bytes" && ++index < argc && transfer::ParseSize(argv[index], &options->bytes)) {
    } else if (argument == "--device" && ++index < argc && ParseInt(argv[index], &options->device)) {
    } else if (
        argument == "--transfer-buffer-count" && ++index < argc &&
        transfer::ParseSize(argv[index], &options->transfer.buffer_count)) {
    } else if (
        argument == "--transfer-chunk-bytes" && ++index < argc &&
        transfer::ParseSize(argv[index], &options->transfer.chunk_bytes)) {
    } else if (argument == "--file-count" && ++index < argc && transfer::ParseSize(argv[index], &options->file_count)) {
    } else if (argument == "--warmups" && ++index < argc && transfer::ParseSize(argv[index], &options->warmups)) {
    } else if (argument == "--iterations" && ++index < argc && transfer::ParseSize(argv[index], &options->iterations)) {
    } else if (
        argument == "--cache-profile" && ++index < argc &&
        benchmark::ParseCacheProfile(argv[index], &options->cache_profile)) {
    } else if (argument == "--help" || argument == "-h") {
      *help = true;
      return true;
    } else {
      *error = "invalid or incomplete argument: " + argument;
      return false;
    }
  }

  if (!options->operation_set) {
    *error = "--operation is required";
    return false;
  }
  if (options->file.empty() || !options->file.is_absolute()) {
    *error = "--file must be an absolute path";
    return false;
  }
  if (options->bytes == 0) {
    *error = "--bytes must be greater than zero";
    return false;
  }
  if (options->iterations == 0 || options->iterations > kMaximumIterations || options->warmups > kMaximumIterations) {
    *error = "--iterations must be between 1 and 1000 and --warmups cannot exceed 1000";
    return false;
  }
  return transfer::ValidateTransferOptions(options->transfer, error);
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

class CUDAResources {
 public:
  CUDAResources() = default;
  CUDAResources(const CUDAResources&) = delete;
  CUDAResources& operator=(const CUDAResources&) = delete;

  void Allocate(int ordinal, size_t bytes)
  {
    CheckCUDA(cuInit(0), "cuInit");
    CheckCUDA(cuDeviceGet(&device_, ordinal), "cuDeviceGet");
    CheckCUDA(cuDevicePrimaryCtxRetain(&context_, device_), "cuDevicePrimaryCtxRetain");
    CheckCUDA(cuCtxSetCurrent(context_), "cuCtxSetCurrent");
    CheckCUDA(cuMemAlloc(&device_ptr_, bytes), "cuMemAlloc");
    CheckCUDA(cuStreamCreate(&stream_, CU_STREAM_NON_BLOCKING), "cuStreamCreate");
  }

  ~CUDAResources()
  {
    if (context_ != nullptr) {
      (void)cuCtxSetCurrent(context_);
    }
    if (stream_ != nullptr) {
      (void)cuStreamDestroy(stream_);
    }
    if (device_ptr_ != 0) {
      (void)cuMemFree(device_ptr_);
    }
    if (context_ != nullptr) {
      (void)cuDevicePrimaryCtxRelease(device_);
    }
  }

  CUdeviceptr device_ptr() const { return device_ptr_; }
  CUstream stream() const { return stream_; }
  CUcontext context() const { return context_; }

 private:
  CUdevice device_ = 0;
  CUcontext context_ = nullptr;
  CUdeviceptr device_ptr_ = 0;
  CUstream stream_ = nullptr;
};

uint64_t
PatternWord(uint64_t index)
{
  uint64_t value = index + 0x9e3779b97f4a7c15ULL;
  value = (value ^ (value >> 30)) * 0xbf58476d1ce4e5b9ULL;
  value = (value ^ (value >> 27)) * 0x94d049bb133111ebULL;
  return value ^ (value >> 31);
}

void
FillPattern(unsigned char* data, size_t size, size_t logical_offset)
{
  size_t done = 0;
  while (done < size) {
    const size_t absolute = logical_offset + done;
    const uint64_t word = PatternWord(absolute / sizeof(uint64_t));
    const size_t byte_in_word = absolute % sizeof(uint64_t);
    const size_t length = std::min(sizeof(uint64_t) - byte_in_word, size - done);
    std::memcpy(data + done, reinterpret_cast<const unsigned char*>(&word) + byte_in_word, length);
    done += length;
  }
}

void
InitializeDevicePattern(const CUDAResources& cuda, size_t bytes)
{
  std::vector<unsigned char> pattern(std::min(bytes, kPatternBufferBytes));
  for (size_t offset = 0; offset < bytes;) {
    const size_t length = std::min(pattern.size(), bytes - offset);
    FillPattern(pattern.data(), length, offset);
    CheckCUDA(cuMemcpyHtoD(cuda.device_ptr() + offset, pattern.data(), length), "initialize device pattern");
    offset += length;
  }
}

void
ClearDevice(const CUDAResources& cuda, size_t bytes)
{
  CheckCUDA(cuMemsetD8Async(cuda.device_ptr(), 0xa5, bytes, cuda.stream()), "clear benchmark device memory");
  CheckCUDA(cuStreamSynchronize(cuda.stream()), "wait for benchmark device clear");
}

void
VerifyDevicePattern(const CUDAResources& cuda, size_t bytes)
{
  std::vector<unsigned char> actual(std::min(bytes, kPatternBufferBytes));
  std::vector<unsigned char> expected(actual.size());
  for (size_t offset = 0; offset < bytes;) {
    const size_t length = std::min(actual.size(), bytes - offset);
    CheckCUDA(cuMemcpyDtoH(actual.data(), cuda.device_ptr() + offset, length), "read device verification data");
    FillPattern(expected.data(), length, offset);
    const auto mismatch =
        std::mismatch(actual.begin(), actual.begin() + static_cast<std::ptrdiff_t>(length), expected.begin());
    if (mismatch.first != actual.begin() + static_cast<std::ptrdiff_t>(length)) {
      const size_t mismatch_offset = offset + static_cast<size_t>(mismatch.first - actual.begin());
      throw std::runtime_error("device pattern mismatch at logical offset " + std::to_string(mismatch_offset));
    }
    offset += length;
  }
}

class FileDescriptor {
 public:
  explicit FileDescriptor(int descriptor) : descriptor_(descriptor) {}
  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  FileDescriptor(FileDescriptor&& other) noexcept : descriptor_(std::exchange(other.descriptor_, -1)) {}
  ~FileDescriptor()
  {
    if (descriptor_ >= 0) {
      (void)close(descriptor_);
    }
  }
  int get() const { return descriptor_; }

 private:
  int descriptor_;
};

void
WriteAllAt(int descriptor, const unsigned char* data, size_t size, size_t offset)
{
  size_t done = 0;
  while (done < size) {
    const ssize_t result = pwrite(descriptor, data + done, size - done, static_cast<off_t>(offset + done));
    if (result < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error("prepare pattern write failed: " + std::string(std::strerror(errno)));
    }
    if (result == 0) {
      throw std::runtime_error("prepare pattern write returned zero bytes");
    }
    done += static_cast<size_t>(result);
  }
}

void
ReadAllAt(int descriptor, unsigned char* data, size_t size, size_t offset)
{
  size_t done = 0;
  while (done < size) {
    const ssize_t result = pread(descriptor, data + done, size - done, static_cast<off_t>(offset + done));
    if (result < 0) {
      if (errno == EINTR) {
        continue;
      }
      throw std::runtime_error("pattern verification read failed: " + std::string(std::strerror(errno)));
    }
    if (result == 0) {
      throw std::runtime_error("pattern verification encountered an unexpected EOF");
    }
    done += static_cast<size_t>(result);
  }
}

std::vector<FileDescriptor>
OpenLayoutFiles(const transfer::StorageLayout& layout, bool create)
{
  std::vector<FileDescriptor> descriptors;
  descriptors.reserve(layout.files.size());
  for (const auto& file : layout.files) {
    if (file.size > static_cast<size_t>(std::numeric_limits<off_t>::max())) {
      throw std::runtime_error("benchmark file is too large for POSIX offsets: " + file.path.string());
    }
    int flags = O_CLOEXEC | O_NOFOLLOW | (create ? O_RDWR | O_CREAT | O_TRUNC : O_RDONLY);
    const int descriptor = open(file.path.c_str(), flags, 0600);
    if (descriptor < 0) {
      throw std::runtime_error("open benchmark file failed: " + file.path.string() + ": " + std::strerror(errno));
    }
    FileDescriptor owned(descriptor);
    if (create && (fchmod(descriptor, 0600) != 0 || ftruncate(descriptor, static_cast<off_t>(file.size)) != 0)) {
      throw std::runtime_error("prepare benchmark file failed: " + file.path.string() + ": " + std::strerror(errno));
    }
    if (!create) {
      struct stat file_stat {};
      if (fstat(descriptor, &file_stat) != 0) {
        throw std::runtime_error("stat benchmark file failed: " + file.path.string() + ": " + std::strerror(errno));
      }
      if (!S_ISREG(file_stat.st_mode) || file_stat.st_size < 0 ||
          static_cast<size_t>(file_stat.st_size) != file.size) {
        throw std::runtime_error(
            "benchmark file is not regular or has the wrong size: " + file.path.string() + ": expected " +
            std::to_string(file.size) + " bytes");
      }
    }
    descriptors.push_back(std::move(owned));
  }
  return descriptors;
}

void
PrepareStoragePattern(const transfer::StorageLayout& layout)
{
  std::vector<FileDescriptor> files = OpenLayoutFiles(layout, true);
  std::vector<unsigned char> pattern(kPatternBufferBytes);
  for (const auto& range : layout.ranges) {
    for (size_t done = 0; done < range.size;) {
      const size_t length = std::min(pattern.size(), range.size - done);
      FillPattern(pattern.data(), length, range.logical_offset + done);
      WriteAllAt(files[range.file_index].get(), pattern.data(), length, range.file_offset + done);
      done += length;
    }
  }
  for (const auto& file : files) {
    if (fsync(file.get()) != 0) {
      throw std::runtime_error("fsync prepared benchmark file failed: " + std::string(std::strerror(errno)));
    }
  }
}

void
VerifyStoragePattern(const transfer::StorageLayout& layout)
{
  std::vector<FileDescriptor> files = OpenLayoutFiles(layout, false);
  std::vector<unsigned char> actual(kPatternBufferBytes);
  std::vector<unsigned char> expected(kPatternBufferBytes);
  for (const auto& range : layout.ranges) {
    for (size_t done = 0; done < range.size;) {
      const size_t length = std::min(actual.size(), range.size - done);
      ReadAllAt(files[range.file_index].get(), actual.data(), length, range.file_offset + done);
      FillPattern(expected.data(), length, range.logical_offset + done);
      const auto mismatch =
          std::mismatch(actual.begin(), actual.begin() + static_cast<std::ptrdiff_t>(length), expected.begin());
      if (mismatch.first != actual.begin() + static_cast<std::ptrdiff_t>(length)) {
        const size_t mismatch_offset =
            range.logical_offset + done + static_cast<size_t>(mismatch.first - actual.begin());
        throw std::runtime_error("storage pattern mismatch at logical offset " + std::to_string(mismatch_offset));
      }
      done += length;
    }
  }
}

void
ValidateStorageFiles(const transfer::StorageLayout& layout)
{
  (void)OpenLayoutFiles(layout, false);
}

double
GiBPerSecond(size_t bytes, double seconds)
{
  return seconds > 0.0 ? static_cast<double>(bytes) / (1024.0 * 1024.0 * 1024.0) / seconds : 0.0;
}

std::string
ResultJSON(
    const Options& options, size_t iteration, size_t pinned_bytes, const transfer::StorageLayout& layout,
    const transfer::TransferMetrics& metrics, const benchmark::CacheEvictionResult& eviction)
{
  const bool restore = options.operation == transfer::TransferOperation::kRestore;
  const bool eviction_requested = restore && options.cache_profile == benchmark::CacheProfile::kClientEvicted;
  const std::string_view cache_profile =
      restore ? benchmark::CacheProfileName(options.cache_profile) : std::string_view("not-applicable");
  const std::string_view client_cache_residency =
      !restore ? std::string_view("not-applicable")
               : (eviction_requested ? std::string_view("unknown-after-advice")
                                     : std::string_view("buffered-warm"));
  const std::string_view server_cache_residency =
      restore ? std::string_view("unknown-may-be-warm") : std::string_view("not-applicable");
  const std::string_view eviction_status =
      !eviction_requested
          ? std::string_view("not-requested")
          : (eviction.files_advised == eviction.files_requested
                 ? std::string_view("advice-accepted")
                 : (eviction.files_advised == 0 ? std::string_view("failed") : std::string_view("partial")));
  std::ostringstream output;
  output << std::fixed << std::setprecision(6) << "{\"event\":\"cuda_nixl_posix_benchmark\",\"schema_version\":1"
         << ",\"success\":true,\"operation\":" << transfer::JsonEscape(options.operation_name)
         << ",\"iteration\":" << iteration << ",\"bytes\":" << metrics.bytes << ",\"device\":" << options.device
         << ",\"transfer_buffer_count\":" << options.transfer.buffer_count
         << ",\"transfer_chunk_bytes\":" << options.transfer.chunk_bytes << ",\"pinned_bytes\":" << pinned_bytes
         << ",\"file_count\":" << layout.files.size() << ",\"max_storage_requests_in_flight\":1"
         << ",\"timing_scope\":\"transfer_engine_setup_through_cleanup\""
         << ",\"pattern_setup_timed\":false,\"verification_timed\":false"
         << ",\"cache_profile\":" << transfer::JsonEscape(cache_profile)
         << ",\"client_cache_eviction_requested\":" << (eviction_requested ? "true" : "false")
         << ",\"client_cache_eviction_best_effort\":" << (eviction_requested ? "true" : "false")
         << ",\"client_cache_eviction_status\":" << transfer::JsonEscape(eviction_status)
         << ",\"client_cache_eviction_files_requested\":" << eviction.files_requested
         << ",\"client_cache_eviction_files_advised\":" << eviction.files_advised
         << ",\"client_cache_eviction_errors\":[";
  for (size_t index = 0; index < eviction.errors.size(); ++index) {
    if (index != 0) {
      output << ",";
    }
    output << transfer::JsonEscape(eviction.errors[index]);
  }
  output << "]"
         << ",\"client_cache_residency\":" << transfer::JsonEscape(client_cache_residency)
         << ",\"server_cache_residency\":" << transfer::JsonEscape(server_cache_residency)
         << ",\"duration_seconds\":" << metrics.total_seconds
         << ",\"effective_gib_per_second\":" << GiBPerSecond(metrics.bytes, metrics.total_seconds)
         << ",\"pipeline_effective_gib_per_second\":" << GiBPerSecond(metrics.bytes, metrics.pipeline_seconds)
         << ",\"storage_service_gib_per_second\":" << GiBPerSecond(metrics.bytes, metrics.storage_seconds)
         << ",\"setup_seconds\":" << metrics.setup_seconds << ",\"pipeline_seconds\":" << metrics.pipeline_seconds
         << ",\"storage_service_seconds\":" << metrics.storage_seconds
         << ",\"cuda_wait_seconds\":" << metrics.cuda_wait_seconds << ",\"fsync_seconds\":" << metrics.fsync_seconds
         << ",\"cleanup_seconds\":" << metrics.cleanup_seconds << ",\"verified\":true,\"files\":[";
  for (size_t index = 0; index < layout.files.size(); ++index) {
    if (index != 0) {
      output << ",";
    }
    output << "{\"index\":" << index << ",\"path\":" << transfer::JsonEscape(layout.files[index].path.string())
           << ",\"configured_bytes\":" << layout.files[index].size
           << ",\"transferred_bytes\":" << metrics.files[index].bytes
           << ",\"storage_service_seconds\":" << metrics.files[index].storage_seconds
           << ",\"storage_service_gib_per_second\":"
           << GiBPerSecond(metrics.files[index].bytes, metrics.files[index].storage_seconds)
           << ",\"fsync_seconds\":" << metrics.files[index].fsync_seconds << "}";
  }
  output << "]}";
  return output.str();
}

std::string
FailureJSON(
    const Options& options, size_t iteration, const benchmark::CacheEvictionResult& eviction, std::string_view error)
{
  const bool restore = options.operation == transfer::TransferOperation::kRestore;
  const bool eviction_requested = restore && options.cache_profile == benchmark::CacheProfile::kClientEvicted;
  const std::string_view cache_profile =
      restore ? benchmark::CacheProfileName(options.cache_profile) : std::string_view("not-applicable");
  const std::string_view eviction_status =
      !eviction_requested
          ? std::string_view("not-requested")
          : (eviction.files_advised == eviction.files_requested
                 ? std::string_view("advice-accepted")
                 : (eviction.files_advised == 0 ? std::string_view("failed") : std::string_view("partial")));
  std::ostringstream output;
  output << "{\"event\":\"cuda_nixl_posix_benchmark\",\"schema_version\":1"
         << ",\"success\":false,\"operation\":" << transfer::JsonEscape(options.operation_name)
         << ",\"iteration\":" << iteration << ",\"cache_profile\":" << transfer::JsonEscape(cache_profile)
         << ",\"client_cache_eviction_requested\":" << (eviction_requested ? "true" : "false")
         << ",\"client_cache_eviction_best_effort\":" << (eviction_requested ? "true" : "false")
         << ",\"client_cache_eviction_status\":" << transfer::JsonEscape(eviction_status)
         << ",\"client_cache_eviction_files_requested\":" << eviction.files_requested
         << ",\"client_cache_eviction_files_advised\":" << eviction.files_advised
         << ",\"client_cache_eviction_errors\":[";
  for (size_t index = 0; index < eviction.errors.size(); ++index) {
    if (index != 0) {
      output << ",";
    }
    output << transfer::JsonEscape(eviction.errors[index]);
  }
  output << "],\"error\":" << transfer::JsonEscape(error) << "}";
  return output.str();
}

benchmark::CacheEvictionResult
ApplyCacheProfile(const Options& options, const transfer::StorageLayout& layout)
{
  if (options.operation != transfer::TransferOperation::kRestore ||
      options.cache_profile != benchmark::CacheProfile::kClientEvicted) {
    return {};
  }
  benchmark::CacheEvictionResult result = benchmark::EvictClientFileCache(layout);
  for (const auto& error : result.errors) {
    std::fprintf(stderr, "warning: client cache eviction advice: %s\n", error.c_str());
  }
  return result;
}

transfer::TransferMetrics
RunTransfer(const Options& options, const transfer::StorageLayout& layout, const CUDAResources& cuda, bool verify)
{
  if (options.operation == transfer::TransferOperation::kRestore) {
    ClearDevice(cuda, options.bytes);
  }

  transfer::TransferMetrics metrics;
  transfer::TransferCancellation cancellation;
  std::string error;
  if (!transfer::TransferExtent(
          cuda.device_ptr(), options.bytes, cuda.stream(), cuda.context(), layout, options.operation, options.transfer,
          &cancellation, &metrics, &error)) {
    throw std::runtime_error("transfer engine failed: " + error);
  }

  if (verify) {
    if (options.operation == transfer::TransferOperation::kCheckpoint) {
      VerifyStoragePattern(layout);
    } else {
      VerifyDevicePattern(cuda, options.bytes);
    }
  }
  return metrics;
}

int
RunBenchmark(const Options& options)
{
  transfer::StorageLayout layout;
  std::string error;
  if (!transfer::BuildContiguousStorageLayout(options.file, options.bytes, options.file_count, &layout, &error)) {
    throw std::runtime_error("invalid storage layout: " + error);
  }
  size_t pinned_bytes = 0;
  if (!transfer::CalculatePinnedBytes(1, options.transfer, &pinned_bytes, &error)) {
    throw std::runtime_error("invalid pinned-memory configuration: " + error);
  }

  CUDAResources cuda;
  cuda.Allocate(options.device, options.bytes);
  if (options.operation == transfer::TransferOperation::kCheckpoint) {
    InitializeDevicePattern(cuda, options.bytes);
  } else {
    PrepareStoragePattern(layout);
    ValidateStorageFiles(layout);
  }

  std::fprintf(
      stderr,
      "Benchmarking %s: bytes=%zu device=%d slots=%zu chunk_bytes=%zu files=%zu "
      "warmups=%zu iterations=%zu cache_profile=%.*s (one synchronous storage request in flight)\n",
      options.operation_name.c_str(), options.bytes, options.device, options.transfer.buffer_count,
      options.transfer.chunk_bytes, options.file_count, options.warmups, options.iterations,
      static_cast<int>(benchmark::CacheProfileName(options.cache_profile).size()),
      benchmark::CacheProfileName(options.cache_profile).data());

  for (size_t index = 0; index < options.warmups; ++index) {
    (void)ApplyCacheProfile(options, layout);
    (void)RunTransfer(options, layout, cuda, true);
  }

  double total_seconds = 0.0;
  double total_throughput = 0.0;
  for (size_t index = 0; index < options.iterations; ++index) {
    const benchmark::CacheEvictionResult eviction = ApplyCacheProfile(options, layout);
    try {
      const transfer::TransferMetrics metrics = RunTransfer(options, layout, cuda, true);
      std::cout << ResultJSON(options, index, pinned_bytes, layout, metrics, eviction) << std::endl;
      total_seconds += metrics.total_seconds;
      total_throughput += GiBPerSecond(metrics.bytes, metrics.total_seconds);
    }
    catch (const std::exception& exception) {
      std::cout << FailureJSON(options, index, eviction, exception.what()) << std::endl;
      throw;
    }
  }
  std::fprintf(
      stderr, "%s summary: runs=%zu mean_duration_seconds=%.6f mean_effective_gib_per_second=%.6f\n",
      options.operation_name.c_str(), options.iterations, total_seconds / options.iterations,
      total_throughput / options.iterations);
  return 0;
}

}  // namespace

int
main(int argc, char** argv)
{
  Options options;
  bool help = false;
  std::string error;
  if (!ParseOptions(argc, argv, &options, &help, &error)) {
    std::fprintf(stderr, "%s\n", error.c_str());
    (void)PrintUsage(stderr);
    return 1;
  }
  if (help) {
    return PrintUsage(stdout);
  }
  try {
    return RunBenchmark(options);
  }
  catch (const std::exception& exception) {
    std::fprintf(stderr, "benchmark failed: %s\n", exception.what());
    return 1;
  }
}
