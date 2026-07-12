/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

namespace cuda_checkpoint_daemon {

constexpr uint32_t kMagic = 0x50484344;  // "DCHP" in little-endian.
constexpr uint16_t kVersion = 3;
constexpr size_t kRequestHeaderSize = 48;
constexpr size_t kResponseHeaderSize = 24;
constexpr size_t kMaxRequestSize = 64 * 1024;
constexpr size_t kMaxResponseSize = 128 * 1024;
constexpr size_t kMaxCgroupSize = 4096;

constexpr uint32_t kResponseFatal = 1U << 0;
constexpr uint32_t kResponseCapabilityDeferredCUDA = 1U << 1;

enum class Action : uint16_t {
  kHealth = 0,
  kCheckpoint = 1,
  kRestore = 2,
  kLock = 3,
  kUnlock = 4,
};

struct Request {
  Action action = Action::kHealth;
  uint32_t pid = 0;
  uint32_t transfer_buffer_count = 0;
  uint64_t transfer_chunk_bytes = 0;
  uint64_t expected_start_time_ticks = 0;
  std::string device_map;
  std::string storage_dir;
  std::string expected_cgroup;
};

struct Response {
  int32_t cuda_status = 0;
  uint32_t flags = 0;
  std::string output;
  std::string error;
};

struct OperationState {
  bool handle_returned = false;
  bool completion_succeeded = false;

  bool fatal() const { return handle_returned && !completion_succeeded; }
};

struct HealthSnapshot {
  bool ready = false;
  bool busy = false;
  bool healthy = false;
  Action action = Action::kHealth;
  uint32_t pid = 0;
  uint64_t elapsed_seconds = 0;
  uint64_t seconds_since_progress = 0;
  uint64_t deadline_seconds = 0;
};

class OperationHealth {
 public:
  explicit OperationHealth(std::chrono::seconds max_operation_duration);

  void MarkReady();
  void Begin(Action action, uint32_t pid);
  void Progress();
  void End();
  HealthSnapshot Snapshot() const;

 private:
  using Clock = std::chrono::steady_clock;

  const std::chrono::seconds max_operation_duration_;
  mutable std::mutex mutex_;
  bool ready_ = false;
  bool busy_ = false;
  Action action_ = Action::kHealth;
  uint32_t pid_ = 0;
  Clock::time_point started_{};
  Clock::time_point last_progress_{};
};

class OwnedUnixSocket {
 public:
  OwnedUnixSocket() = default;
  OwnedUnixSocket(const OwnedUnixSocket&) = delete;
  OwnedUnixSocket& operator=(const OwnedUnixSocket&) = delete;
  ~OwnedUnixSocket();

  bool Bind(const std::string& path, int backlog, std::string* error);
  void Close();
  int fd() const { return fd_; }

 private:
  void UnlinkIfOwned();

  int fd_ = -1;
  int lock_fd_ = -1;
  std::string path_;
  uint64_t device_ = 0;
  uint64_t inode_ = 0;
  bool bound_ = false;
};

using OperationExecutor = std::function<Response(const Request&)>;
using CompletionExecutor = std::function<int32_t()>;

bool ParseRequest(const unsigned char* data, size_t size, Request* request, std::string* error);
bool EncodeRequest(const Request& request, std::vector<unsigned char>* data, std::string* error);
bool ParseResponse(const unsigned char* data, size_t size, Response* response, std::string* error);
bool EncodeResponse(const Response& response, std::vector<unsigned char>* data, std::string* error);
bool ValidateProcessIdentity(
    const Request& request, const std::string& proc_root, std::string* error);
bool ExecuteValidated(
    const Request& request, const std::string& proc_root, const OperationExecutor& executor,
    Response* response);
bool ResponseAllowsServerContinue(const Response& response);
int32_t FinishHandledOperation(
    bool post_handle_succeeded, int32_t failure_status,
    const CompletionExecutor& completion, OperationState* state);
int PollForInputOrStop(int input_fd, int stop_fd);
const char* ActionName(Action action);
Response HealthResponse(const OperationHealth& health);

}  // namespace cuda_checkpoint_daemon
