/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "daemon_protocol.h"

#include <climits>
#include <cerrno>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fcntl.h>
#include <fstream>
#include <limits>
#include <poll.h>
#include <sstream>
#include <sys/file.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>
#include <utility>

namespace cuda_checkpoint_daemon {
namespace {

uint16_t ReadU16(const unsigned char* data)
{
  return static_cast<uint16_t>(data[0]) | static_cast<uint16_t>(data[1]) << 8;
}

uint32_t ReadU32(const unsigned char* data)
{
  return static_cast<uint32_t>(data[0]) | static_cast<uint32_t>(data[1]) << 8 |
         static_cast<uint32_t>(data[2]) << 16 | static_cast<uint32_t>(data[3]) << 24;
}

uint64_t ReadU64(const unsigned char* data)
{
  return static_cast<uint64_t>(ReadU32(data)) | static_cast<uint64_t>(ReadU32(data + 4)) << 32;
}

void WriteU16(std::vector<unsigned char>* data, size_t offset, uint16_t value)
{
  (*data)[offset] = value & 0xff;
  (*data)[offset + 1] = value >> 8;
}

void WriteU32(std::vector<unsigned char>* data, size_t offset, uint32_t value)
{
  for (size_t index = 0; index < 4; ++index) {
    (*data)[offset + index] = value >> (index * 8);
  }
}

void WriteU64(std::vector<unsigned char>* data, size_t offset, uint64_t value)
{
  WriteU32(data, offset, value & 0xffffffffU);
  WriteU32(data, offset + 4, value >> 32);
}

bool ContainsNul(const std::string& value)
{
  return value.find('\0') != std::string::npos;
}

bool MakeSocketAddress(const std::string& path, sockaddr_un* address)
{
  if (path.empty() || path.front() != '/' || path.size() >= sizeof(address->sun_path)) {
    return false;
  }
  *address = {};
  address->sun_family = AF_UNIX;
  std::memcpy(address->sun_path, path.c_str(), path.size() + 1);
  return true;
}

bool SocketConfirmedStale(const sockaddr_un& address)
{
  const int probe_fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  if (probe_fd < 0) {
    return false;
  }
  const int result = connect(
      probe_fd, reinterpret_cast<const sockaddr*>(&address), sizeof(address));
  const int connect_error = errno;
  close(probe_fd);
  return result != 0 && connect_error == ECONNREFUSED;
}

uint64_t DurationSeconds(
    std::chrono::steady_clock::time_point start,
    std::chrono::steady_clock::time_point end)
{
  return std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
}

}  // namespace

bool
ParseRequest(const unsigned char* data, size_t size, Request* request, std::string* error)
{
  if (data == nullptr || request == nullptr || size < kRequestHeaderSize || size > kMaxRequestSize) {
    *error = "invalid request size";
    return false;
  }
  if (ReadU32(data) != kMagic || ReadU16(data + 4) != kVersion || ReadU16(data + 6) != kRequestHeaderSize) {
    *error = "invalid request protocol header";
    return false;
  }
  const auto action = static_cast<Action>(ReadU16(data + 8));
  if (ReadU16(data + 10) != 0) {
    *error = "request reserved field is nonzero";
    return false;
  }
  if (action != Action::kHealth && action != Action::kCheckpoint && action != Action::kRestore &&
      action != Action::kLock && action != Action::kUnlock) {
    *error = "invalid request action";
    return false;
  }
  const uint32_t device_map_size = ReadU32(data + 28);
  const uint32_t storage_dir_size = ReadU32(data + 32);
  const uint32_t cgroup_size = ReadU32(data + 36);
  if (ReadU32(data + 44) != 0 || device_map_size > kMaxRequestSize || storage_dir_size > kMaxRequestSize ||
      cgroup_size > kMaxCgroupSize ||
      static_cast<size_t>(device_map_size) + storage_dir_size + cgroup_size != size - kRequestHeaderSize) {
    *error = "invalid request payload lengths";
    return false;
  }
  Request parsed;
  parsed.action = action;
  parsed.pid = ReadU32(data + 12);
  parsed.transfer_buffer_count = ReadU32(data + 16);
  parsed.transfer_chunk_bytes = ReadU64(data + 20);
  parsed.expected_start_time_ticks = ReadU64(data + 40);
  const char* payload = reinterpret_cast<const char*>(data + kRequestHeaderSize);
  parsed.device_map.assign(payload, device_map_size);
  parsed.storage_dir.assign(payload + device_map_size, storage_dir_size);
  parsed.expected_cgroup.assign(payload + device_map_size + storage_dir_size, cgroup_size);
  if (ContainsNul(parsed.device_map) || ContainsNul(parsed.storage_dir) || ContainsNul(parsed.expected_cgroup)) {
    *error = "request strings contain NUL";
    return false;
  }
  if (action == Action::kHealth) {
    if (parsed.pid != 0 || parsed.transfer_buffer_count != 0 || parsed.transfer_chunk_bytes != 0 ||
        parsed.expected_start_time_ticks != 0 || !parsed.device_map.empty() || !parsed.storage_dir.empty() ||
        !parsed.expected_cgroup.empty()) {
      *error = "health request has arguments";
      return false;
    }
  } else if (parsed.pid == 0 || parsed.pid > INT_MAX || parsed.expected_start_time_ticks == 0 ||
             parsed.expected_cgroup.empty()) {
    *error = "invalid operation arguments";
    return false;
  } else if (action == Action::kLock || action == Action::kUnlock) {
    if (parsed.transfer_buffer_count != 0 || parsed.transfer_chunk_bytes != 0 ||
        !parsed.device_map.empty() || !parsed.storage_dir.empty()) {
      *error = "lock/unlock request has transfer arguments";
      return false;
    }
  } else if (
      parsed.transfer_buffer_count == 0 || parsed.transfer_chunk_bytes == 0 ||
      parsed.storage_dir.empty() || parsed.storage_dir.front() != '/' ||
      (action == Action::kCheckpoint && !parsed.device_map.empty())) {
    *error = "invalid transfer operation arguments";
    return false;
  }
  *request = std::move(parsed);
  return true;
}

bool
EncodeRequest(const Request& request, std::vector<unsigned char>* data, std::string* error)
{
  if (request.device_map.size() > std::numeric_limits<uint32_t>::max() ||
      request.storage_dir.size() > std::numeric_limits<uint32_t>::max() ||
      request.expected_cgroup.size() > kMaxCgroupSize ||
      kRequestHeaderSize + request.device_map.size() + request.storage_dir.size() + request.expected_cgroup.size() >
          kMaxRequestSize) {
    *error = "request is too large";
    return false;
  }
  data->assign(
      kRequestHeaderSize + request.device_map.size() + request.storage_dir.size() + request.expected_cgroup.size(), 0);
  WriteU32(data, 0, kMagic);
  WriteU16(data, 4, kVersion);
  WriteU16(data, 6, kRequestHeaderSize);
  WriteU16(data, 8, static_cast<uint16_t>(request.action));
  WriteU32(data, 12, request.pid);
  WriteU32(data, 16, request.transfer_buffer_count);
  WriteU64(data, 20, request.transfer_chunk_bytes);
  WriteU32(data, 28, request.device_map.size());
  WriteU32(data, 32, request.storage_dir.size());
  WriteU32(data, 36, request.expected_cgroup.size());
  WriteU64(data, 40, request.expected_start_time_ticks);
  std::memcpy(data->data() + kRequestHeaderSize, request.device_map.data(), request.device_map.size());
  std::memcpy(
      data->data() + kRequestHeaderSize + request.device_map.size(), request.storage_dir.data(),
      request.storage_dir.size());
  std::memcpy(
      data->data() + kRequestHeaderSize + request.device_map.size() + request.storage_dir.size(),
      request.expected_cgroup.data(), request.expected_cgroup.size());
  return true;
}

bool
ParseResponse(const unsigned char* data, size_t size, Response* response, std::string* error)
{
  if (data == nullptr || response == nullptr || size < kResponseHeaderSize || size > kMaxResponseSize) {
    *error = "invalid response size";
    return false;
  }
  if (ReadU32(data) != kMagic || ReadU16(data + 4) != kVersion || ReadU16(data + 6) != kResponseHeaderSize) {
    *error = "invalid response protocol header";
    return false;
  }
  const uint32_t flags = ReadU32(data + 12);
  if ((flags & ~(kResponseFatal | kResponseCapabilityDeferredCUDA)) != 0) {
    *error = "invalid response flags";
    return false;
  }
  const uint32_t output_size = ReadU32(data + 16);
  const uint32_t error_size = ReadU32(data + 20);
  if (static_cast<size_t>(output_size) + error_size != size - kResponseHeaderSize) {
    *error = "invalid response payload lengths";
    return false;
  }
  response->cuda_status = static_cast<int32_t>(ReadU32(data + 8));
  response->flags = flags;
  const char* payload = reinterpret_cast<const char*>(data + kResponseHeaderSize);
  response->output.assign(payload, output_size);
  response->error.assign(payload + output_size, error_size);
  return true;
}

bool
EncodeResponse(const Response& response, std::vector<unsigned char>* data, std::string* error)
{
  if (response.output.size() > std::numeric_limits<uint32_t>::max() ||
      response.error.size() > std::numeric_limits<uint32_t>::max() ||
      kResponseHeaderSize + response.output.size() + response.error.size() > kMaxResponseSize) {
    *error = "response is too large";
    return false;
  }
  data->assign(kResponseHeaderSize + response.output.size() + response.error.size(), 0);
  WriteU32(data, 0, kMagic);
  WriteU16(data, 4, kVersion);
  WriteU16(data, 6, kResponseHeaderSize);
  WriteU32(data, 8, static_cast<uint32_t>(response.cuda_status));
  WriteU32(data, 12, response.flags);
  WriteU32(data, 16, response.output.size());
  WriteU32(data, 20, response.error.size());
  std::memcpy(data->data() + kResponseHeaderSize, response.output.data(), response.output.size());
  std::memcpy(
      data->data() + kResponseHeaderSize + response.output.size(), response.error.data(), response.error.size());
  return true;
}

bool
ValidateProcessIdentity(const Request& request, const std::string& proc_root, std::string* error)
{
  const std::filesystem::path process_dir =
      std::filesystem::path(proc_root) / std::to_string(request.pid);
  std::ifstream stat(process_dir / "stat");
  std::string stat_line;
  if (!std::getline(stat, stat_line)) {
    *error = "failed to read current process stat";
    return false;
  }
  const size_t closing_paren = stat_line.rfind(')');
  if (closing_paren == std::string::npos || closing_paren + 2 >= stat_line.size()) {
    *error = "malformed current process stat";
    return false;
  }
  std::istringstream fields(stat_line.substr(closing_paren + 2));
  std::string field;
  uint64_t start_time_ticks = 0;
  for (size_t index = 0; index < 20; ++index) {
    if (!(fields >> field)) {
      *error = "malformed current process stat fields";
      return false;
    }
    if (index == 19) {
      char* end = nullptr;
      errno = 0;
      const unsigned long long parsed = std::strtoull(field.c_str(), &end, 10);
      if (errno != 0 || end == nullptr || *end != '\0') {
        *error = "invalid current process start time";
        return false;
      }
      start_time_ticks = parsed;
    }
  }
  std::ifstream cgroup_file(process_dir / "cgroup");
  std::ostringstream cgroup_contents;
  cgroup_contents << cgroup_file.rdbuf();
  const std::string cgroup = cgroup_contents.str();
  if ((!cgroup_file.good() && !cgroup_file.eof()) || cgroup.empty() || cgroup.size() > kMaxCgroupSize) {
    *error = "failed to read valid current process cgroup";
    return false;
  }
  if (start_time_ticks != request.expected_start_time_ticks || cgroup != request.expected_cgroup) {
    *error = "start time or cgroup mismatch";
    return false;
  }
  return true;
}

bool
ExecuteValidated(
    const Request& request, const std::string& proc_root, const OperationExecutor& executor,
    Response* response)
{
  if (request.action == Action::kHealth) {
    response->cuda_status = 1;
    response->error = "health requests must use the health socket";
    return true;
  }
  std::string identity_error;
  if (!ValidateProcessIdentity(request, proc_root, &identity_error)) {
    response->cuda_status = 1;
    response->error = "process identity changed before CUDA operation: " + identity_error;
    return true;
  }
  *response = executor(request);
  return ResponseAllowsServerContinue(*response);
}

bool
ResponseAllowsServerContinue(const Response& response)
{
  return (response.flags & kResponseFatal) == 0;
}

int32_t
FinishHandledOperation(
    bool post_handle_succeeded, int32_t failure_status,
    const CompletionExecutor& completion, OperationState* state)
{
  state->handle_returned = true;
  if (!post_handle_succeeded) {
    return failure_status;
  }
  const int32_t status = completion();
  state->completion_succeeded = status == 0;
  return status;
}

OperationHealth::OperationHealth(std::chrono::seconds max_operation_duration)
    : max_operation_duration_(max_operation_duration)
{
}

void
OperationHealth::MarkReady()
{
  std::lock_guard<std::mutex> lock(mutex_);
  ready_ = true;
}

void
OperationHealth::Begin(Action action, uint32_t pid)
{
  const auto now = Clock::now();
  std::lock_guard<std::mutex> lock(mutex_);
  busy_ = true;
  action_ = action;
  pid_ = pid;
  started_ = now;
  last_progress_ = now;
}

void
OperationHealth::Progress()
{
  std::lock_guard<std::mutex> lock(mutex_);
  if (busy_) {
    last_progress_ = Clock::now();
  }
}

void
OperationHealth::End()
{
  std::lock_guard<std::mutex> lock(mutex_);
  busy_ = false;
  action_ = Action::kHealth;
  pid_ = 0;
}

HealthSnapshot
OperationHealth::Snapshot() const
{
  const auto now = Clock::now();
  std::lock_guard<std::mutex> lock(mutex_);
  HealthSnapshot snapshot{
      .ready = ready_,
      .busy = busy_,
      .healthy = ready_,
      .action = action_,
      .pid = pid_,
      .deadline_seconds = static_cast<uint64_t>(max_operation_duration_.count()),
  };
  if (busy_) {
    snapshot.elapsed_seconds = DurationSeconds(started_, now);
    snapshot.seconds_since_progress = DurationSeconds(last_progress_, now);
    snapshot.healthy = ready_ && now - started_ <= max_operation_duration_;
  }
  return snapshot;
}

Response
HealthResponse(const OperationHealth& health)
{
  const HealthSnapshot snapshot = health.Snapshot();
  Response response;
  response.cuda_status = snapshot.healthy ? 0 : 1;
  if (snapshot.ready) {
    response.flags = kResponseCapabilityDeferredCUDA;
  }
  std::ostringstream output;
  output << "{\"ready\":" << (snapshot.ready ? "true" : "false")
         << ",\"busy\":" << (snapshot.busy ? "true" : "false")
         << ",\"healthy\":" << (snapshot.healthy ? "true" : "false")
         << ",\"action\":\"" << ActionName(snapshot.action) << "\""
         << ",\"pid\":" << snapshot.pid
         << ",\"elapsed_seconds\":" << snapshot.elapsed_seconds
         << ",\"seconds_since_progress\":" << snapshot.seconds_since_progress
         << ",\"deadline_seconds\":" << snapshot.deadline_seconds << "}\n";
  response.output = output.str();
  if (!snapshot.healthy) {
    response.error = snapshot.ready ? "operation exceeded watchdog deadline" : "daemon is not ready";
  }
  return response;
}

OwnedUnixSocket::~OwnedUnixSocket()
{
  Close();
}

bool
OwnedUnixSocket::Bind(const std::string& path, int backlog, std::string* error)
{
  sockaddr_un address{};
  if (fd_ >= 0 || !MakeSocketAddress(path, &address)) {
    *error = "invalid socket path";
    return false;
  }
  const std::string lock_path = path + ".lock";
  lock_fd_ = open(lock_path.c_str(), O_CREAT | O_CLOEXEC | O_RDWR, 0600);
  if (lock_fd_ < 0 || flock(lock_fd_, LOCK_EX | LOCK_NB) != 0) {
    *error = lock_fd_ < 0 ? std::strerror(errno) : "socket path is owned by another server";
    Close();
    return false;
  }
  fd_ = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  if (fd_ < 0) {
    *error = std::strerror(errno);
    return false;
  }
  struct stat existing {};
  if (lstat(path.c_str(), &existing) == 0) {
    if (!S_ISSOCK(existing.st_mode) || !SocketConfirmedStale(address)) {
      *error = "refusing to replace non-socket or live socket path";
      Close();
      return false;
    }
    if (unlink(path.c_str()) != 0) {
      *error = std::strerror(errno);
      Close();
      return false;
    }
  } else if (errno != ENOENT) {
    *error = std::strerror(errno);
    Close();
    return false;
  }
  if (bind(fd_, reinterpret_cast<sockaddr*>(&address), sizeof(address)) != 0) {
    *error = std::strerror(errno);
    Close();
    return false;
  }
  path_ = path;
  bound_ = true;
  struct stat bound_stat {};
  if (lstat(path.c_str(), &bound_stat) != 0 || !S_ISSOCK(bound_stat.st_mode)) {
    *error = "failed to inspect bound socket";
    Close();
    return false;
  }
  device_ = bound_stat.st_dev;
  inode_ = bound_stat.st_ino;
  if (backlog < 0 || chmod(path.c_str(), 0600) != 0 || listen(fd_, backlog) != 0) {
    *error = std::strerror(errno);
    Close();
    return false;
  }
  return true;
}

void
OwnedUnixSocket::UnlinkIfOwned()
{
  if (!bound_) {
    return;
  }
  struct stat current {};
  if (lstat(path_.c_str(), &current) == 0 &&
      static_cast<uint64_t>(current.st_dev) == device_ &&
      static_cast<uint64_t>(current.st_ino) == inode_) {
    (void)unlink(path_.c_str());
  }
  bound_ = false;
}

void
OwnedUnixSocket::Close()
{
  if (fd_ >= 0) {
    close(fd_);
    fd_ = -1;
  }
  UnlinkIfOwned();
  if (lock_fd_ >= 0) {
    close(lock_fd_);
    lock_fd_ = -1;
  }
}

int
PollForInputOrStop(int input_fd, int stop_fd)
{
  pollfd descriptors[2]{
      {.fd = input_fd, .events = POLLIN, .revents = 0},
      {.fd = stop_fd, .events = POLLIN, .revents = 0},
  };
  for (;;) {
    const int result = poll(descriptors, 2, -1);
    if (result < 0 && errno == EINTR) {
      continue;
    }
    if (result <= 0 || (descriptors[1].revents & POLLIN) != 0) {
      return 0;
    }
    if ((descriptors[0].revents & (POLLIN | POLLERR | POLLHUP | POLLNVAL)) != 0) {
      return 1;
    }
  }
}

const char*
ActionName(Action action)
{
  switch (action) {
    case Action::kHealth:
      return "none";
    case Action::kCheckpoint:
      return "checkpoint";
    case Action::kRestore:
      return "restore";
    case Action::kLock:
      return "lock";
    case Action::kUnlock:
      return "unlock";
  }
  return "unknown";
}

}  // namespace cuda_checkpoint_daemon
