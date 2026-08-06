// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "pagebroker.hpp"

#include <fcntl.h>
#include <linux/un.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace pagebroker {
namespace {
class FileDescriptor {
public:
  explicit FileDescriptor(int descriptor = -1) : descriptor_(descriptor) {}
  FileDescriptor(const FileDescriptor &) = delete;
  FileDescriptor &operator=(const FileDescriptor &) = delete;
  ~FileDescriptor() { reset(); }
  int get() const { return descriptor_; }
  explicit operator bool() const { return descriptor_ >= 0; }
  void reset(int descriptor = -1) {
    if (descriptor_ >= 0) close(descriptor_);
    descriptor_ = descriptor;
  }

private:
  int descriptor_;
};

class ProtoWriter {
public:
  void varint_field(int field, std::uint64_t value) {
    varint(static_cast<std::uint64_t>(field * 8));
    varint(value);
  }
  void string_field(int field, const std::string &value) {
    varint(static_cast<std::uint64_t>(field * 8 + 2));
    varint(value.size());
    data_ += value;
  }
  std::string take() { return std::move(data_); }

private:
  void varint(std::uint64_t value) {
    while (value >= 128) {
      data_.push_back(static_cast<char>((value & 127) | 128));
      value >>= 7;
    }
    data_.push_back(static_cast<char>(value));
  }
  std::string data_;
};

class ProtoReader {
public:
  ProtoReader(const void *data, std::size_t size)
      : cursor_(static_cast<const char *>(data)), end_(cursor_ + size) {}
  bool empty() const { return cursor_ == end_; }
  bool varint(std::uint64_t &value) {
    value = 0;
    for (int byte = 0; byte < 10 && cursor_ < end_; ++byte) {
      auto current = static_cast<unsigned char>(*cursor_++);
      if (byte == 9 && current > 1) return false;
      value |= static_cast<std::uint64_t>(current & 127) << (byte * 7);
      if (!(current & 128)) return true;
    }
    return false;
  }
  bool string(std::string &value) {
    std::uint64_t size;
    if (!varint(size) || size > static_cast<std::uint64_t>(end_ - cursor_))
      return false;
    value.assign(cursor_, cursor_ + size);
    cursor_ += size;
    return true;
  }
  bool skip(std::uint64_t tag) {
    switch (tag & 7) {
      case 0: {
        std::uint64_t value;
        return varint(value);
      }
      case 1:
        return advance(8);
      case 2: {
        std::uint64_t size;
        return varint(size) && advance(size);
      }
      case 3: {
        auto field = tag >> 3;
        while (!empty()) {
          std::uint64_t nested;
          if (!varint(nested) || (nested >> 3) == 0) return false;
          if ((nested & 7) == 4) return (nested >> 3) == field;
          if (!skip(nested)) return false;
        }
        return false;
      }
      case 5:
        return advance(4);
      default:
        return false;
    }
  }

private:
  bool advance(std::uint64_t size) {
    if (size > static_cast<std::uint64_t>(end_ - cursor_)) return false;
    cursor_ += size;
    return true;
  }
  const char *cursor_;
  const char *end_;
};

Response fail(const std::string &id, const std::string &message) {
  return {false, id, {}, {}, message};
}
fs::path tx_path(const fs::path &root, const std::string &id) {
  return root / "tx" / id;
}
bool safe_id(const std::string &id) {
  return !id.empty() && id.find_first_not_of(
                            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUV"
                            "WXYZ0123456789-_") == std::string::npos;
}
bool safe_relative(const fs::path &path) {
  return !path.empty() && !path.is_absolute() && *path.begin() != "..";
}

class CopyPlan {
public:
  explicit CopyPlan(const fs::path &source)
      : source_(open(source.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC |
                                         O_NOFOLLOW)) {
    if (!source_)
      throw std::runtime_error("cannot open checkpoint directory");
    for (const auto &entry : fs::recursive_directory_iterator(source)) {
      auto status = entry.symlink_status();
      if (fs::is_symlink(status))
        throw std::runtime_error("checkpoint contains a symlink");
      auto relative = entry.path().lexically_relative(source);
      if (!safe_relative(relative))
        throw std::runtime_error("checkpoint entry escapes its root");
      if (fs::is_directory(status)) {
        directories_.push_back(relative);
      } else if (fs::is_regular_file(status)) {
        auto bytes = entry.file_size();
        if (bytes > std::numeric_limits<std::uint64_t>::max() - bytes_)
          throw std::runtime_error("checkpoint size overflow");
        files_.push_back({relative, bytes});
        bytes_ += bytes;
      } else {
        throw std::runtime_error("checkpoint contains a special file");
      }
    }
  }
  std::uint64_t bytes() const { return bytes_; }
  bool copy_to(const fs::path &destination,
               const std::atomic_bool &cancelled) const {
    fs::create_directories(destination);
    for (const auto &directory : directories_) {
      if (cancelled) return false;
      fs::create_directories(destination / directory);
    }
    for (const auto &entry : files_) {
      if (cancelled) return false;
      auto target = destination / entry.relative;
      auto partial = target.string() + ".partial";
      fs::create_directories(target.parent_path());
      FileDescriptor input(open_source(entry.relative));
      struct stat status {};
      if (!input || fstat(input.get(), &status) != 0 ||
          !S_ISREG(status.st_mode) || status.st_size < 0 ||
          static_cast<std::uint64_t>(status.st_size) != entry.bytes)
        return false;
      FileDescriptor output(
          open(partial.c_str(),
               O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_NOFOLLOW, 0600));
      if (!output ||
          !copy_file(input.get(), output.get(), entry.bytes, cancelled) ||
          fstat(input.get(), &status) != 0 || status.st_size < 0 ||
          static_cast<std::uint64_t>(status.st_size) != entry.bytes)
        return false;
      output.reset();
      if (fs::file_size(partial) != entry.bytes) return false;
      fs::rename(partial, target);
    }
    return true;
  }

private:
  struct Entry {
    fs::path relative;
    std::uint64_t bytes;
  };
  int open_source(const fs::path &relative) const {
    FileDescriptor directory(fcntl(source_.get(), F_DUPFD_CLOEXEC, 0));
    if (!directory) return -1;
    auto component = relative.begin();
    while (component != relative.end()) {
      if (*component == "." || *component == "..") {
        errno = EINVAL;
        return -1;
      }
      auto next = component;
      ++next;
      if (next == relative.end())
        return openat(directory.get(), component->c_str(),
                      O_RDONLY | O_CLOEXEC | O_NOFOLLOW);
      directory.reset(openat(directory.get(), component->c_str(),
                             O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
      if (!directory) return -1;
      component = next;
    }
    errno = EINVAL;
    return -1;
  }
  static bool copy_file(int input, int output, std::uint64_t bytes,
                        const std::atomic_bool &cancelled) {
    std::array<char, 1 << 20> buffer;
    while (bytes > 0 && !cancelled) {
      auto wanted = std::min<std::uint64_t>(bytes, buffer.size());
      auto count = read(input, buffer.data(), wanted);
      if (count < 0 && errno == EINTR) continue;
      if (count <= 0) return false;
      std::size_t written = 0;
      while (written < static_cast<std::size_t>(count)) {
        auto result = write(output, buffer.data() + written,
                            static_cast<std::size_t>(count) - written);
        if (result < 0 && errno == EINTR) continue;
        if (result <= 0) return false;
        written += static_cast<std::size_t>(result);
      }
      bytes -= static_cast<std::uint64_t>(count);
    }
    return bytes == 0 && !cancelled;
  }
  std::vector<fs::path> directories_;
  std::vector<Entry> files_;
  std::uint64_t bytes_{};
  FileDescriptor source_;
};

} // namespace

void Server::serve_health() {
  FileDescriptor server(socket(AF_INET, SOCK_STREAM | SOCK_CLOEXEC, 0));
  if (!server) return;
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_addr.s_addr = htonl(INADDR_ANY);
  address.sin_port = htons(8080);
  int yes = 1;
  if (setsockopt(server.get(), SOL_SOCKET, SO_REUSEADDR, &yes, sizeof(yes)) <
          0 ||
      bind(server.get(), reinterpret_cast<sockaddr *>(&address),
           sizeof(address)) < 0 ||
      listen(server.get(), 8) < 0)
    return;
  for (;;) {
    FileDescriptor client(
        accept4(server.get(), nullptr, nullptr, SOCK_CLOEXEC));
    if (!client) continue;
    timeval timeout{1, 0};
    if (setsockopt(client.get(), SOL_SOCKET, SO_RCVTIMEO, &timeout,
                   sizeof(timeout)) < 0)
      continue;
    char request[256];
    auto size = read(client.get(), request, sizeof(request));
    std::string path =
        size > 0 ? std::string(request, request + size) : std::string();
    std::string body = "ok\n";
    std::string status = path.find("/healthz") != std::string::npos ||
                                 path.find("/readyz") != std::string::npos
                             ? "200 OK"
                             : "404 Not Found";
    std::string response = "HTTP/1.1 " + status + "\r\nContent-Length: " +
                           std::to_string(body.size()) +
                           "\r\nConnection: close\r\n\r\n" + body;
    send(client.get(), response.data(), response.size(), MSG_NOSIGNAL);
  }
}

bool decode_request(const void *data, std::size_t size, Request &request,
                    std::string &error) {
  request = {};
  error.clear();
  ProtoReader reader(data, size);
  bool operation = false;
  while (!reader.empty()) {
    std::uint64_t tag;
    if (!reader.varint(tag) || (tag >> 3) == 0) {
      error = "invalid protobuf tag";
      return false;
    }
    int field = tag >> 3;
    auto wire = tag & 7;
    if (field == 1 && wire == 0) {
      std::uint64_t value;
      if (!reader.varint(value)) return false;
      request.operation = static_cast<Request::Operation>(value);
      operation = true;
    } else if ((field == 2 || field == 3) && wire == 2) {
      std::string value;
      if (!reader.string(value)) return false;
      if (field == 2)
        request.transaction_id = std::move(value);
      else
        request.checkpoint_path = std::move(value);
    } else if (field == 4 && wire == 0) {
      std::uint64_t value;
      if (!reader.varint(value) || value > 1) return false;
      request.staging = value != 0;
    } else if (!reader.skip(tag)) {
      error = "invalid protobuf field";
      return false;
    }
  }
  if (!operation) {
    error = "operation is required";
    return false;
  }
  return true;
}

std::string encode_response(const Response &r) {
  ProtoWriter writer;
  writer.varint_field(1, r.ok);
  if (!r.transaction_id.empty()) writer.string_field(2, r.transaction_id);
  if (!r.staging_path.empty()) writer.string_field(3, r.staging_path);
  if (!r.scratch_path.empty()) writer.string_field(4, r.scratch_path);
  if (!r.error.empty()) writer.string_field(5, r.error);
  return writer.take();
}

std::uint64_t filesystem_budget(const fs::path &path) {
  struct statvfs stats {};
  if (statvfs(path.c_str(), &stats) != 0) return 0;
  if (stats.f_frsize != 0 &&
      stats.f_bavail >
          std::numeric_limits<std::uint64_t>::max() / stats.f_frsize)
    return 0;
  return static_cast<std::uint64_t>(stats.f_bavail) * stats.f_frsize;
}

TransactionManager::TransactionManager(fs::path staging, fs::path scratch,
                                       std::uint64_t budget)
    : staging_root_(std::move(staging)),
      scratch_root_(std::move(scratch)),
      budget_(budget) {
  cleanup();
}
void TransactionManager::cleanup() {
  std::lock_guard lock(mutex_);
  fs::remove_all(staging_root_ / "tx");
  fs::remove_all(scratch_root_);
  fs::create_directories(staging_root_ / "tx");
  fs::create_directories(scratch_root_);
  transactions_.clear();
  staged_bytes_ = 0;
}
void TransactionManager::release(
    const std::string &transaction_id,
    const std::shared_ptr<TransactionState> &state) {
  std::lock_guard lock(mutex_);
  auto transaction = transactions_.find(transaction_id);
  if (transaction == transactions_.end() || transaction->second != state)
    return;
  staged_bytes_ -= state->staged_bytes;
  transactions_.erase(transaction);
}
Response TransactionManager::submit(const Request &r) {
  if (!safe_id(r.transaction_id))
    return fail(r.transaction_id, "invalid transaction id");
  auto source_status = fs::symlink_status(r.checkpoint_path);
  if (fs::is_symlink(source_status) || !fs::is_directory(source_status))
    return fail(r.transaction_id, "checkpoint path is not a directory");
  auto path = tx_path(staging_root_, r.transaction_id);
  auto state = std::make_shared<TransactionState>();
  state->staging = r.staging;
  if (!r.staging) {
    std::lock_guard lock(mutex_);
    if (transactions_.contains(r.transaction_id))
      return fail(r.transaction_id, "transaction is already active");
    state->phase = TransactionState::Phase::Ready;
    transactions_.emplace(r.transaction_id, state);
    return {true, r.transaction_id, r.checkpoint_path,
            scratch_root_ / r.transaction_id, {}};
  }
  bool reserved = false;
  try {
    CopyPlan plan(r.checkpoint_path);
    state->staged_bytes = plan.bytes();
    {
      std::lock_guard lock(mutex_);
      if (transactions_.contains(r.transaction_id))
        return fail(r.transaction_id, "transaction is already active");
      if (staged_bytes_ > budget_ || plan.bytes() > budget_ - staged_bytes_)
        return fail(r.transaction_id, "staging budget exceeded");
      transactions_.emplace(r.transaction_id, state);
      staged_bytes_ += plan.bytes();
      reserved = true;
    }
    fs::remove_all(path);
    auto copied = plan.copy_to(path, state->cancelled);
    bool ready = false;
    {
      std::lock_guard lock(mutex_);
      auto transaction = transactions_.find(r.transaction_id);
      if (!copied || state->cancelled || transaction == transactions_.end() ||
          transaction->second != state) {
        if (transaction != transactions_.end() && transaction->second == state)
          state->phase = TransactionState::Phase::Cleaning;
      } else {
        state->phase = TransactionState::Phase::Ready;
        ready = true;
      }
    }
    if (!ready) {
      std::error_code cleanup_error;
      fs::remove_all(path, cleanup_error);
      release(r.transaction_id, state);
      return fail(r.transaction_id, state->cancelled
                                        ? "staging cancelled"
                                        : "checkpoint copy failed");
    }
    return {true, r.transaction_id, path, scratch_root_ / r.transaction_id, {}};
  } catch (const std::exception &e) {
    if (reserved) {
      {
        std::lock_guard lock(mutex_);
        auto transaction = transactions_.find(r.transaction_id);
        if (transaction != transactions_.end() && transaction->second == state)
          state->phase = TransactionState::Phase::Cleaning;
      }
      std::error_code cleanup_error;
      fs::remove_all(path, cleanup_error);
      release(r.transaction_id, state);
    }
    return fail(r.transaction_id, e.what());
  }
}
Response TransactionManager::commit(const Request &r) {
  std::shared_ptr<TransactionState> state;
  {
    std::lock_guard lock(mutex_);
    auto transaction = transactions_.find(r.transaction_id);
    if (transaction == transactions_.end())
      return fail(r.transaction_id, "transaction is not active");
    state = transaction->second;
    if (state->phase == TransactionState::Phase::Staging)
      return fail(r.transaction_id, "transaction is still staging");
    if (state->phase == TransactionState::Phase::Cleaning)
      return fail(r.transaction_id, "transaction is being cleaned up");
    state->phase = TransactionState::Phase::Cleaning;
  }
  try {
    if (state->staging) fs::remove_all(tx_path(staging_root_, r.transaction_id));
    fs::remove_all(scratch_root_ / r.transaction_id);
  } catch (const fs::filesystem_error &e) {
    std::lock_guard lock(mutex_);
    auto transaction = transactions_.find(r.transaction_id);
    if (transaction != transactions_.end() && transaction->second == state)
      state->phase = TransactionState::Phase::Ready;
    return fail(r.transaction_id, e.what());
  }
  release(r.transaction_id, state);
  return {true, r.transaction_id, {}, {}, {}};
}
Response TransactionManager::abort(const Request &r) {
  std::shared_ptr<TransactionState> state;
  {
    std::lock_guard lock(mutex_);
    auto transaction = transactions_.find(r.transaction_id);
    if (transaction == transactions_.end())
      return {true, r.transaction_id, {}, {}, {}};
    state = transaction->second;
    if (state->phase == TransactionState::Phase::Staging) {
      state->cancelled = true;
      return {true, r.transaction_id, {}, {}, {}};
    }
    if (state->phase == TransactionState::Phase::Cleaning)
      return fail(r.transaction_id, "transaction is being cleaned up");
    state->phase = TransactionState::Phase::Cleaning;
  }
  try {
    if (state->staging) fs::remove_all(tx_path(staging_root_, r.transaction_id));
    fs::remove_all(scratch_root_ / r.transaction_id);
  } catch (const fs::filesystem_error &e) {
    std::lock_guard lock(mutex_);
    auto transaction = transactions_.find(r.transaction_id);
    if (transaction != transactions_.end() && transaction->second == state)
      state->phase = TransactionState::Phase::Ready;
    return fail(r.transaction_id, e.what());
  }
  release(r.transaction_id, state);
  return {true, r.transaction_id, {}, {}, {}};
}

Server::Server(fs::path socket_path, fs::path staging, fs::path scratch,
               std::uint64_t budget)
    : socket_path_(std::move(socket_path)),
      transactions_(std::move(staging), std::move(scratch), budget) {}

void Server::handle_client(int descriptor) {
  FileDescriptor client(descriptor);
  timeval receive_timeout{30, 0};
  timeval send_timeout{5, 0};
  if (setsockopt(client.get(), SOL_SOCKET, SO_RCVTIMEO, &receive_timeout,
                 sizeof(receive_timeout)) < 0 ||
      setsockopt(client.get(), SOL_SOCKET, SO_SNDTIMEO, &send_timeout,
                 sizeof(send_timeout)) < 0)
    return;
  std::array<char, 65536> buffer;
  auto size = recv(client.get(), buffer.data(), buffer.size(), MSG_TRUNC);
  Request request;
  std::string error;
  Response response;
  if (size < 0) {
    response = fail({}, "read failed");
  } else if (static_cast<std::size_t>(size) > buffer.size()) {
    response = fail({}, "request is too large");
  } else if (!decode_request(buffer.data(), static_cast<std::size_t>(size),
                             request, error)) {
    response = fail({}, error.empty() ? "invalid request" : error);
  } else if (request.operation == Request::Operation::Submit) {
    response = transactions_.submit(request);
  } else if (request.operation == Request::Operation::Commit) {
    response = transactions_.commit(request);
  } else if (request.operation == Request::Operation::Abort) {
    response = transactions_.abort(request);
  } else {
    response = fail(request.transaction_id, "unknown operation");
  }
  auto encoded = encode_response(response);
  send(client.get(), encoded.data(), encoded.size(), MSG_NOSIGNAL);
}

int Server::run() {
  try {
    fs::create_directories(socket_path_.parent_path());
  } catch (const fs::filesystem_error &) {
    return 1;
  }
  auto socket_name = socket_path_.string();
  sockaddr_un address{};
  if (socket_name.size() >= sizeof(address.sun_path)) return 1;
  if (unlink(socket_name.c_str()) < 0 && errno != ENOENT) return 1;
  FileDescriptor server(socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0));
  if (!server) return 1;
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, socket_name.c_str(), socket_name.size() + 1);
  if (bind(server.get(), reinterpret_cast<sockaddr *>(&address),
           sizeof(address)) < 0 ||
      listen(server.get(), 8) < 0 || chmod(socket_name.c_str(), 0660) < 0)
    return 1;
  std::thread(serve_health).detach();
  for (;;) {
    int client = accept4(server.get(), nullptr, nullptr, SOCK_CLOEXEC);
    if (client < 0) continue;
    client_slots_.acquire();
    try {
      std::thread([this, client] {
        try {
          handle_client(client);
        } catch (...) {
        }
        client_slots_.release();
      }).detach();
    } catch (...) {
      close(client);
      client_slots_.release();
    }
  }
}

int serve(const fs::path &socket_path, const fs::path &staging,
          const fs::path &scratch, std::uint64_t budget) {
  return Server(socket_path, staging, scratch, budget).run();
}

} // namespace pagebroker

#ifndef PAGEBROKER_TEST
int main(int argc, char **argv) {
  if (argc != 2 || std::string(argv[1]) != "serve") return 2;
  auto budget = pagebroker::filesystem_budget("/staging");
  if (budget == 0) return 1;
  return pagebroker::serve("/run/pagebroker/pagebroker.sock", "/staging",
                           "/scratch", budget);
}
#endif
