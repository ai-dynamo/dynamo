// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "pagebroker.hpp"

#include <fcntl.h>
#include <linux/memfd.h>
#include <linux/un.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <sys/syscall.h>
#include <sys/time.h>
#include <unistd.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <syncstream>
#include <thread>
#include <utility>
#include <vector>

namespace fs = std::filesystem;
namespace pagebroker {

class CopyPool {
public:
  explicit CopyPool(unsigned workers) : worker_count_(workers) {
    workers_.reserve(workers);
    for (unsigned i = 0; i < workers; ++i) {
      workers_.emplace_back([this] {
        for (;;) {
          Task task;
          {
            std::unique_lock lock(mutex_);
            changed_.wait(lock, [this] { return stopping_ || !tasks_.empty(); });
            if (stopping_ && tasks_.empty())
              return;
            task = std::move(tasks_.front());
            tasks_.pop_front();
          }
          task.run();
        }
      });
    }
  }
  ~CopyPool() {
    {
      std::lock_guard lock(mutex_);
      stopping_ = true;
    }
    changed_.notify_all();
    for (auto &worker : workers_)
      worker.join();
  }
  void submit(std::string transaction_id, std::string relative,
              std::function<void()> task) {
    {
      std::lock_guard lock(mutex_);
      tasks_.push_back(
          {std::move(transaction_id), std::move(relative), std::move(task)});
    }
    changed_.notify_one();
  }
  bool prioritize(const std::string &transaction_id,
                  const std::string &relative) {
    std::lock_guard lock(mutex_);
    std::deque<Task> prioritized;
    for (auto task = tasks_.begin(); task != tasks_.end();) {
      if (task->transaction_id == transaction_id &&
          task->relative == relative) {
        prioritized.push_back(std::move(*task));
        task = tasks_.erase(task);
      } else {
        ++task;
      }
    }
    if (prioritized.empty())
      return false;
    while (!prioritized.empty()) {
      tasks_.push_front(std::move(prioritized.back()));
      prioritized.pop_back();
    }
    return true;
  }
  unsigned worker_count() const { return worker_count_; }

private:
  struct Task {
    std::string transaction_id;
    std::string relative;
    std::function<void()> run;
  };
  unsigned worker_count_;
  std::vector<std::thread> workers_;
  std::deque<Task> tasks_;
  std::mutex mutex_;
  std::condition_variable changed_;
  bool stopping_{};
};

#ifdef PAGEBROKER_TEST
bool test_copy_pool_priority() {
  CopyPool pool(1);
  std::mutex mutex;
  std::condition_variable changed;
  bool first_started = false;
  bool release_first = false;
  std::vector<std::string> order;
  pool.submit("tx", "first", [&] {
    std::unique_lock lock(mutex);
    first_started = true;
    changed.notify_one();
    changed.wait(lock, [&] { return release_first; });
    order.push_back("first");
  });
  {
    std::unique_lock lock(mutex);
    changed.wait(lock, [&] { return first_started; });
  }
  pool.submit("tx", "second", [&] {
    std::lock_guard lock(mutex);
    order.push_back("second");
  });
  pool.submit("tx", "requested", [&] {
    std::lock_guard lock(mutex);
    order.push_back("requested-1");
  });
  pool.submit("tx", "requested", [&] {
    std::lock_guard lock(mutex);
    order.push_back("requested-2");
  });
  assert(pool.prioritize("tx", "requested"));
  {
    std::lock_guard lock(mutex);
    release_first = true;
  }
  changed.notify_one();
  for (;;) {
    {
      std::lock_guard lock(mutex);
      if (order.size() == 4)
        return order == std::vector<std::string>{
                            "first", "requested-1", "requested-2", "second"};
    }
    std::this_thread::yield();
  }
}
#endif

namespace {
class FileDescriptor {
public:
  explicit FileDescriptor(int descriptor = -1) : descriptor_(descriptor) {}
  FileDescriptor(const FileDescriptor &) = delete;
  FileDescriptor &operator=(const FileDescriptor &) = delete;
  FileDescriptor(FileDescriptor &&other) noexcept
      : descriptor_(std::exchange(other.descriptor_, -1)) {}
  FileDescriptor &operator=(FileDescriptor &&other) noexcept {
    if (this != &other)
      reset(std::exchange(other.descriptor_, -1));
    return *this;
  }
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

class ProviderSession {
public:
  explicit ProviderSession(std::shared_ptr<StagingState> staging)
      : staging_(std::move(staging)) {
    if (!staging_) return;
    std::lock_guard lock(staging_->mutex);
    staging_->provider_running = true;
  }
  ~ProviderSession() {
    if (!staging_) return;
    {
      std::lock_guard lock(staging_->mutex);
      staging_->provider_running = false;
    }
    staging_->changed.notify_all();
  }

private:
  std::shared_ptr<StagingState> staging_;
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
bool path_within(const fs::path &root, const fs::path &path) {
  if (root.empty() || !root.is_absolute() || !path.is_absolute())
    return false;
  std::error_code error;
  auto canonical_root = fs::weakly_canonical(root, error);
  if (error)
    return false;
  auto canonical_path = fs::weakly_canonical(path, error);
  if (error)
    return false;
  auto relative = canonical_path.lexically_relative(canonical_root);
  return !relative.empty() && relative != "." && !relative.is_absolute() &&
         *relative.begin() != "..";
}
fs::path checkpoint_staging_path(const fs::path &destination,
                                 const std::string &transaction_id) {
  return destination.parent_path() /
         ("." + destination.filename().string() + ".pagebroker-" +
          transaction_id);
}
fs::path checkpoint_marker_path(const fs::path &staging_root,
                                const std::string &transaction_id) {
  return staging_root / "tx" / (".checkpoint-" + transaction_id);
}
bool valid_checkpoint_staging_path(const fs::path &checkpoint_root,
                                   const fs::path &path,
                                   const std::string &transaction_id) {
  auto filename = path.filename().string();
  auto suffix = ".pagebroker-" + transaction_id;
  return path_within(checkpoint_root, path) && filename.starts_with(".") &&
         filename.size() > suffix.size() && filename.ends_with(suffix);
}
struct CopyEntry {
  std::shared_ptr<FileDescriptor> source_root;
  fs::path destination;
  std::string relative;
  std::uint64_t bytes;
};
constexpr std::uint64_t copy_task_bytes = 16ULL << 20;
constexpr std::size_t copy_buffer_bytes = 1ULL << 20;
bool safe_relative(const fs::path &path) {
  return !path.empty() && !path.is_absolute() && *path.begin() != "..";
}
int open_beneath(int root, const fs::path &relative, int flags = O_RDONLY) {
  FileDescriptor directory(fcntl(root, F_DUPFD_CLOEXEC, 0));
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
                    flags | O_CLOEXEC | O_NOFOLLOW, 0600);
    directory.reset(openat(directory.get(), component->c_str(),
                           O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
    if (!directory) return -1;
    component = next;
  }
  errno = EINVAL;
  return -1;
}
class CopyPlan {
public:
  CopyPlan(const fs::path &source, const fs::path &destination) {
    auto source_root = std::make_shared<FileDescriptor>(
        open(source.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW));
    if (!*source_root)
      throw std::runtime_error("cannot open checkpoint directory");
    for (const auto &entry : fs::recursive_directory_iterator(source)) {
      auto status = entry.symlink_status();
      if (fs::is_symlink(status))
        throw std::runtime_error("checkpoint contains a symlink");
      auto relative = entry.path().lexically_relative(source);
      if (!safe_relative(relative))
        throw std::runtime_error("checkpoint entry escapes its root");
      auto target = destination / relative;
      if (fs::is_directory(status)) {
        directories_.push_back(target);
      } else if (fs::is_regular_file(status)) {
        auto bytes = entry.file_size();
        if (bytes > static_cast<std::uint64_t>(
                        std::numeric_limits<off_t>::max()))
          throw std::runtime_error("checkpoint file is too large");
        if (bytes > std::numeric_limits<std::uint64_t>::max() - bytes_)
          throw std::runtime_error("checkpoint size overflow");
        files_.push_back(
            {source_root, target, relative.generic_string(), bytes});
        bytes_ += bytes;
      } else {
        throw std::runtime_error("checkpoint contains a special file");
      }
    }
    std::sort(files_.begin(), files_.end(),
              [](const CopyEntry &left, const CopyEntry &right) {
                if (left.bytes != right.bytes)
                  return left.bytes < right.bytes;
                return left.relative < right.relative;
              });
  }

  std::uint64_t bytes() const { return bytes_; }
  std::size_t file_count() const { return files_.size(); }
  std::shared_ptr<StagingState> prepare(CopyPool &pool,
                                        const std::string &transaction_id) const;
  void schedule(CopyPool &pool, const std::string &transaction_id,
                const fs::path &staging,
                const std::shared_ptr<StagingState> &state,
                std::size_t &scheduled_tasks) const;

private:
  std::vector<fs::path> directories_;
  std::vector<CopyEntry> files_;
  std::uint64_t bytes_{};
};
unsigned copy_worker_count() {
  auto cpus = std::max(1u, std::thread::hardware_concurrency());
  return std::min(32u, std::max(8u, cpus * 2));
}
void set_staging_error(const std::shared_ptr<StagingState> &state, int error,
                       const std::string &message) {
  {
    std::lock_guard lock(state->mutex);
    if (state->error.empty()) {
      state->error_code = error ? error : EIO;
      state->error = message;
    }
  }
  state->changed.notify_all();
}
void finish_copy_task(const CopyEntry &entry,
                      const std::string &transaction_id,
                      const fs::path &staging,
                      const std::shared_ptr<StagingState> &state,
                      std::chrono::steady_clock::time_point started,
                      bool copied) {
  bool complete = false;
  bool file_ready = false;
  std::uint64_t copied_bytes = 0;
  std::string error;
  bool cancelled = false;
  {
    std::lock_guard lock(state->mutex);
    if (copied) {
      auto remaining = state->remaining_chunks.find(entry.relative);
      if (remaining != state->remaining_chunks.end() &&
          --remaining->second == 0) {
        auto partial = entry.destination.string() + ".partial";
        std::error_code rename_error;
        fs::rename(partial, entry.destination, rename_error);
        if (rename_error) {
          if (state->error.empty()) {
            state->error_code = rename_error.value();
            state->error = rename_error.message();
          }
        } else {
          state->copied_bytes += entry.bytes;
          state->ready_files.insert(entry.relative);
          file_ready = true;
        }
      }
    }
    if (state->remaining_tasks > 0)
      --state->remaining_tasks;
    if (state->remaining_tasks == 0) {
      state->complete = true;
      complete = true;
      copied_bytes = state->copied_bytes;
      error = state->error;
      cancelled = state->cancelled;
    }
  }
  state->changed.notify_all();
  if (file_ready)
    std::osyncstream(std::cerr)
        << "pagebroker stage file transaction="
        << std::quoted(transaction_id) << " path="
        << std::quoted(entry.relative) << " bytes=" << entry.bytes
        << std::endl;
  if (!complete)
    return;
  auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
      std::chrono::steady_clock::now() - started);
  if (error.empty() && !cancelled) {
    std::osyncstream(std::cerr)
        << "pagebroker stage complete transaction="
        << std::quoted(transaction_id) << " staging="
        << std::quoted(staging.string()) << " bytes=" << copied_bytes
        << " duration_ms=" << elapsed.count() << std::endl;
  } else if (!cancelled) {
    std::osyncstream(std::cerr)
        << "pagebroker stage failed transaction="
        << std::quoted(transaction_id) << " error="
        << std::quoted(error) << std::endl;
  }
}
void copy_chunk(CopyEntry entry, std::uint64_t offset, std::size_t bytes,
                const std::string &transaction_id, const fs::path &staging,
                const std::shared_ptr<StagingState> &state,
                std::chrono::steady_clock::time_point started) {
  bool skip_copy = false;
  {
    std::lock_guard lock(state->mutex);
    skip_copy = state->cancelled || !state->error.empty();
  }
  bool copied = false;
  if (!skip_copy) {
    auto partial = entry.destination.string() + ".partial";
    int source = open_beneath(entry.source_root->get(), entry.relative);
    struct stat source_stat {};
    int source_error = source < 0 ? errno : 0;
    if (!source_error &&
        (fstat(source, &source_stat) < 0 || !S_ISREG(source_stat.st_mode) ||
         static_cast<std::uint64_t>(source_stat.st_size) != entry.bytes))
      source_error = errno ? errno : ESTALE;
    int destination = source_error ? -1
                                   : open(partial.c_str(), O_WRONLY | O_CLOEXEC |
                                                               O_NOFOLLOW);
    int error = source_error ? source_error : (destination < 0 ? errno : 0);
    std::array<char, copy_buffer_bytes> buffer;
    std::size_t done = 0;
    while (!error && done < bytes) {
      auto wanted = std::min(buffer.size(), bytes - done);
      auto read_bytes = pread(source, buffer.data(), wanted,
                              static_cast<off_t>(offset + done));
      if (read_bytes <= 0) {
        error = read_bytes < 0 ? errno : EIO;
        break;
      }
      std::size_t written = 0;
      while (written < static_cast<std::size_t>(read_bytes)) {
        auto n = pwrite(destination, buffer.data() + written,
                        static_cast<std::size_t>(read_bytes) - written,
                        static_cast<off_t>(offset + done + written));
        if (n <= 0) {
          error = n < 0 ? errno : EIO;
          break;
        }
        written += static_cast<std::size_t>(n);
      }
      done += written;
    }
    if (source >= 0)
      close(source);
    if (destination >= 0)
      close(destination);
    if (error) {
      std::error_code ignored;
      fs::remove(partial, ignored);
      set_staging_error(state, error,
                        std::system_error(error, std::generic_category(),
                                          "copy checkpoint chunk")
                            .what());
    } else {
      copied = true;
    }
  }
  finish_copy_task(entry, transaction_id, staging, state, started, copied);
}

std::shared_ptr<StagingState>
CopyPlan::prepare(CopyPool &pool, const std::string &transaction_id) const {
  for (const auto &directory : directories_)
    fs::create_directories(directory);
  auto state = std::make_shared<StagingState>();
  state->prioritize_file = [&pool, transaction_id](const std::string &path) {
    return pool.prioritize(transaction_id, path);
  };
  for (const auto &entry : files_) {
    fs::create_directories(entry.destination.parent_path());
    state->planned_files.insert(entry.relative);
    auto chunks = static_cast<std::size_t>(std::max<std::uint64_t>(
        1, (entry.bytes + copy_task_bytes - 1) / copy_task_bytes));
    state->remaining_chunks.emplace(entry.relative, chunks);
    state->remaining_tasks += chunks;
    auto partial = entry.destination.string() + ".partial";
    FileDescriptor output(
        open(partial.c_str(),
             O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC | O_NOFOLLOW, 0600));
    if (!output || ftruncate(output.get(), entry.bytes) < 0)
      throw std::system_error(errno ? errno : EIO, std::generic_category(),
                              "create checkpoint partial");
  }
  return state;
}

void CopyPlan::schedule(CopyPool &pool, const std::string &transaction_id,
                        const fs::path &staging,
                        const std::shared_ptr<StagingState> &state,
                        std::size_t &scheduled_tasks) const {
  if (files_.empty()) {
    state->complete = true;
    state->changed.notify_all();
    return;
  }
  auto started = std::chrono::steady_clock::now();
  for (const auto &entry : files_) {
    auto chunks = state->remaining_chunks.at(entry.relative);
    for (std::size_t chunk = 0; chunk < chunks; ++chunk) {
      auto offset = static_cast<std::uint64_t>(chunk) * copy_task_bytes;
      auto bytes = static_cast<std::size_t>(std::min<std::uint64_t>(
          copy_task_bytes,
          entry.bytes > offset ? entry.bytes - offset : 0));
      pool.submit(transaction_id, entry.relative,
                  [entry, offset, bytes, transaction_id, staging, state,
                   started] {
                    copy_chunk(entry, offset, bytes, transaction_id, staging,
                               state, started);
                  });
      ++scheduled_tasks;
    }
  }
}

void response_status(std::string &out, std::int32_t status) {
  ProtoWriter writer;
  writer.varint_field(
      1, static_cast<std::uint64_t>(static_cast<std::int64_t>(status)));
  out = writer.take();
}
bool nested_string(const char *data, std::size_t size, int wanted,
                   std::string &value) {
  ProtoReader reader(data, size);
  while (!reader.empty()) {
    std::uint64_t tag;
    if (!reader.varint(tag) || (tag >> 3) == 0) return false;
    if ((tag >> 3) == static_cast<std::uint64_t>(wanted) && (tag & 7) == 2) {
      return reader.string(value);
    }
    if (!reader.skip(tag)) return false;
  }
  return false;
}
bool nested_varint(const char *data, std::size_t size, int wanted,
                   std::uint64_t &value) {
  ProtoReader reader(data, size);
  while (!reader.empty()) {
    std::uint64_t tag;
    if (!reader.varint(tag) || (tag >> 3) == 0) return false;
    if ((tag >> 3) == static_cast<std::uint64_t>(wanted) && (tag & 7) == 0)
      return reader.varint(value);
    if (!reader.skip(tag)) return false;
  }
  return false;
}
struct ProviderRequest {
  std::uint64_t operation{};
  std::string name;
  std::uint64_t flags{};
  std::uint64_t pid{};
  std::uint64_t vaddr{};
  std::uint64_t length{};
  std::uint64_t shared_id{};
};

bool decode_provider_request(const char *data, std::size_t size,
                             ProviderRequest &request) {
  request = {};
  ProtoReader reader(data, size);
  while (!reader.empty()) {
    std::uint64_t tag;
    if (!reader.varint(tag) || (tag >> 3) == 0) return false;
    if ((tag >> 3) == 1 && (tag & 7) == 0) {
      if (!reader.varint(request.operation)) return false;
    } else if ((tag >> 3) >= 2 && (tag >> 3) <= 4 && (tag & 7) == 2) {
      std::string nested;
      if (!reader.string(nested)) return false;
      if ((tag >> 3) == 2) {
        if (!nested_string(nested.data(), nested.size(), 1, request.name) ||
            !nested_varint(nested.data(), nested.size(), 2, request.flags))
          return false;
      } else if ((tag >> 3) == 3) {
        if (!nested_varint(nested.data(), nested.size(), 1, request.pid) ||
            !nested_varint(nested.data(), nested.size(), 2, request.vaddr) ||
            !nested_varint(nested.data(), nested.size(), 3, request.length))
          return false;
      } else {
        if (!nested_varint(nested.data(), nested.size(), 1,
                           request.shared_id) ||
            !nested_varint(nested.data(), nested.size(), 2, request.length))
          return false;
      }
    } else if (!reader.skip(tag)) {
      return false;
    }
  }
  return true;
}

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
                                       fs::path checkpoint,
                                       std::uint64_t budget)
    : staging_root_(std::move(staging)), scratch_root_(std::move(scratch)),
      checkpoint_root_(std::move(checkpoint)),
      budget_(budget), copy_pool_(std::make_unique<CopyPool>(copy_worker_count())) {
  cleanup();
}
TransactionManager::~TransactionManager() {
  for (auto &[_, state] : transactions_)
    stop_staging(state, true);
}
void TransactionManager::stop_staging(TransactionState &state, bool cancel) {
  if (state.staging && cancel) {
    {
      std::lock_guard lock(state.staging->mutex);
      state.staging->cancelled = true;
    }
    state.staging->changed.notify_all();
  }
  if (state.staging) {
    std::unique_lock lock(state.staging->mutex);
    state.staging->changed.wait(lock, [&] {
      return state.staging->complete && !state.staging->provider_running;
    });
  }
}
void TransactionManager::cleanup() {
  for (auto &[_, state] : transactions_)
    stop_staging(state, true);
  auto transaction_root = staging_root_ / "tx";
  if (fs::is_directory(transaction_root)) {
    for (const auto &entry : fs::directory_iterator(transaction_root)) {
      auto filename = entry.path().filename().string();
      constexpr std::string_view prefix = ".checkpoint-";
      if (!entry.is_regular_file() || !filename.starts_with(prefix))
        continue;
      auto transaction_id = filename.substr(prefix.size());
      std::ifstream marker(entry.path());
      std::string staging_path;
      std::getline(marker, staging_path);
      if (safe_id(transaction_id) &&
          valid_checkpoint_staging_path(checkpoint_root_, staging_path,
                                        transaction_id))
        fs::remove_all(staging_path);
      fs::remove(entry.path());
    }
  }
  fs::remove_all(staging_root_ / "tx");
  if (fs::is_directory(scratch_root_)) {
    for (const auto &entry : fs::directory_iterator(scratch_root_))
      fs::remove_all(entry.path());
  }
  fs::create_directories(staging_root_ / "tx");
  fs::create_directories(scratch_root_);
  transactions_.clear();
  staged_bytes_ = 0;
}
Response TransactionManager::submit(const Request &r) {
  if (!safe_id(r.transaction_id))
    return fail(r.transaction_id, "invalid transaction id");
  auto source_status = fs::symlink_status(r.checkpoint_path);
  if (fs::is_symlink(source_status) || !fs::is_directory(source_status))
    return fail(r.transaction_id, "checkpoint path is not a directory");
  auto path = tx_path(staging_root_, r.transaction_id);
  if (!r.staging) {
    std::lock_guard lock(mutex_);
    if (transactions_.contains(r.transaction_id))
      return fail(r.transaction_id, "transaction is already active");
    auto &state = transactions_[r.transaction_id];
    state.uses_staging = false;
    return {true, r.transaction_id, r.checkpoint_path,
            scratch_root_ / r.transaction_id, {}};
  }
  std::shared_ptr<StagingState> staging;
  std::size_t tasks = 0;
  std::size_t scheduled_tasks = 0;
  bool path_owned = false;
  bool transaction_inserted = false;
  try {
    auto submit_started = std::chrono::steady_clock::now();
    CopyPlan plan(r.checkpoint_path, path);
    std::lock_guard lock(mutex_);
    if (transactions_.contains(r.transaction_id))
      return fail(r.transaction_id, "transaction is already active");
    if (staged_bytes_ > budget_ || plan.bytes() > budget_ - staged_bytes_) {
      return fail(r.transaction_id, "staging budget exceeded");
    }
    path_owned = true;
    fs::remove_all(path);
    fs::create_directories(path);
    auto workers = copy_pool_->worker_count();
    auto files = plan.file_count();
    staging = plan.prepare(*copy_pool_, r.transaction_id);
    auto transaction = transactions_.emplace(
        std::piecewise_construct, std::forward_as_tuple(r.transaction_id),
        std::forward_as_tuple()).first;
    transaction_inserted = true;
    auto &state = transaction->second;
    state.uses_staging = true;
    state.staged_bytes = plan.bytes();
    state.staging = staging;
    tasks = staging->remaining_tasks;
    staged_bytes_ += plan.bytes();
    plan.schedule(*copy_pool_, r.transaction_id, path, staging,
                  scheduled_tasks);
    std::osyncstream(std::cerr)
        << "pagebroker stage scheduled transaction="
        << std::quoted(r.transaction_id) << " source="
        << std::quoted(r.checkpoint_path) << " staging="
        << std::quoted(path.string()) << " bytes=" << state.staged_bytes
        << " files=" << files << " tasks=" << tasks
        << " chunk_bytes=" << copy_task_bytes << " workers=" << workers
        << " submit_duration_ms="
        << std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::steady_clock::now() - submit_started)
               .count()
        << std::endl;
    return {true, r.transaction_id, path, scratch_root_ / r.transaction_id,
            {}};
  } catch (const std::exception &e) {
    std::cerr << "pagebroker stage failed transaction="
              << std::quoted(r.transaction_id) << " error="
              << std::quoted(e.what()) << std::endl;
    std::lock_guard lock(mutex_);
    auto transaction = transactions_.find(r.transaction_id);
    if (transaction_inserted && transaction != transactions_.end()) {
      {
        std::lock_guard staging_lock(staging->mutex);
        staging->cancelled = true;
        staging->remaining_tasks -= tasks - scheduled_tasks;
        if (staging->remaining_tasks == 0) staging->complete = true;
      }
      staging->changed.notify_all();
      stop_staging(transaction->second, false);
      std::error_code cleanup_error;
      fs::remove_all(path, cleanup_error);
      staged_bytes_ -= transaction->second.staged_bytes;
      transactions_.erase(transaction);
    } else if (path_owned && transaction == transactions_.end()) {
      std::error_code cleanup_error;
      fs::remove_all(path, cleanup_error);
    }
    return fail(r.transaction_id, e.what());
  }
}
Response TransactionManager::prepare_checkpoint(const Request &r) {
  std::lock_guard lock(mutex_);
  if (transactions_.contains(r.transaction_id))
    return fail(r.transaction_id, "transaction is already active");
  if (!safe_id(r.transaction_id))
    return fail(r.transaction_id, "invalid transaction id");
  fs::path destination = fs::path(r.checkpoint_path).lexically_normal();
  if (destination.empty() || !destination.is_absolute() ||
      destination.filename().empty() || destination.filename() == "." ||
      destination.filename() == "..")
    return fail(r.transaction_id,
                "checkpoint path must be an absolute directory path");
  try {
    fs::create_directories(destination.parent_path());
    if (!path_within(checkpoint_root_, destination))
      return fail(r.transaction_id, "checkpoint path is outside the configured root");
    auto staging = tx_path(staging_root_, r.transaction_id);
    auto durable_staging = checkpoint_staging_path(destination, r.transaction_id);
    if (fs::exists(staging) || fs::exists(durable_staging))
      return fail(r.transaction_id,
                  "checkpoint transaction path already exists");
    fs::create_directory(staging);
    std::ofstream marker(
        checkpoint_marker_path(staging_root_, r.transaction_id));
    marker << durable_staging.string() << '\n';
    if (!marker) {
      fs::remove_all(staging);
      return fail(r.transaction_id,
                  "failed to record checkpoint transaction");
    }
    auto transaction = transactions_.emplace(
        std::piecewise_construct, std::forward_as_tuple(r.transaction_id),
        std::forward_as_tuple()).first;
    auto &state = transaction->second;
    state.checkpoint = destination;
    state.promote = true;
    std::osyncstream(std::cerr)
        << "pagebroker checkpoint prepared transaction="
        << std::quoted(r.transaction_id) << " destination="
        << std::quoted(destination.string()) << " staging="
        << std::quoted(staging.string()) << std::endl;
    return {true, r.transaction_id, staging,
            scratch_root_ / r.transaction_id, {}};
  } catch (const fs::filesystem_error &e) {
    return fail(r.transaction_id, e.what());
  }
}
std::shared_ptr<StagingState>
TransactionManager::staging_state(const std::string &transaction_id) {
  std::lock_guard lock(mutex_);
  auto transaction = transactions_.find(transaction_id);
  return transaction == transactions_.end() ? nullptr
                                             : transaction->second.staging;
}
Response TransactionManager::wait_for_staging(const Request &request,
                                              Response response) {
  auto staging = staging_state(request.transaction_id);
  if (!staging) return fail(request.transaction_id, "transaction is not staging");
  std::unique_lock lock(staging->mutex);
  staging->changed.wait(lock, [&] {
    return staging->complete || staging->cancelled || !staging->error.empty();
  });
  if (!staging->error.empty()) return fail(request.transaction_id, staging->error);
  if (staging->cancelled) return fail(request.transaction_id, "staging cancelled");
  return response;
}
Response TransactionManager::commit(const Request &r) {
  auto started = std::chrono::steady_clock::now();
  std::lock_guard lock(mutex_);
  auto transaction = transactions_.find(r.transaction_id);
  if (transaction == transactions_.end())
    return fail(r.transaction_id, "transaction is not active");
  try {
    auto &state = transaction->second;
    if (!state.promote && state.uses_staging) {
      stop_staging(state, true);
      std::lock_guard staging_lock(state.staging->mutex);
      if (!state.staging->error.empty())
        return fail(r.transaction_id, state.staging->error);
    }
    if (state.promote) {
      auto staged = tx_path(staging_root_, r.transaction_id);
      auto durable_staged = checkpoint_staging_path(state.checkpoint,
                                                     r.transaction_id);
      fs::copy(staged, durable_staged, fs::copy_options::recursive);
      auto backup = state.checkpoint.parent_path() /
                    ("." + state.checkpoint.filename().string() +
                     ".pagebroker-old-" + r.transaction_id);
      if (fs::exists(backup))
        fs::remove_all(backup);
      if (fs::exists(state.checkpoint))
        fs::rename(state.checkpoint, backup);
      try {
        fs::rename(durable_staged, state.checkpoint);
      } catch (...) {
        if (fs::exists(backup) && !fs::exists(state.checkpoint))
          fs::rename(backup, state.checkpoint);
        throw;
      }
      std::error_code cleanup_error;
      fs::remove_all(backup, cleanup_error);
      cleanup_error.clear();
      fs::remove(checkpoint_marker_path(staging_root_, r.transaction_id),
                 cleanup_error);
      cleanup_error.clear();
      fs::remove_all(staged, cleanup_error);
      cleanup_error.clear();
      fs::remove_all(scratch_root_ / r.transaction_id, cleanup_error);
    } else {
      if (state.uses_staging)
        fs::remove_all(tx_path(staging_root_, r.transaction_id));
      fs::remove_all(scratch_root_ / r.transaction_id);
    }
  } catch (const fs::filesystem_error &e) {
    return fail(r.transaction_id, e.what());
  }
  auto promote = transaction->second.promote;
  auto checkpoint = transaction->second.checkpoint;
  staged_bytes_ -= transaction->second.staged_bytes;
  transactions_.erase(transaction);
  if (promote) {
    auto duration = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started);
    std::osyncstream(std::cerr)
        << "pagebroker checkpoint committed transaction="
        << std::quoted(r.transaction_id) << " destination="
        << std::quoted(checkpoint.string()) << " duration_s=" << std::fixed
        << std::setprecision(6) << duration.count() << std::endl;
  }
  return {true, r.transaction_id, {}, {}, {}};
}
Response TransactionManager::abort(const Request &r) {
  std::lock_guard lock(mutex_);
  auto transaction = transactions_.find(r.transaction_id);
  if (transaction != transactions_.end()) {
    try {
      stop_staging(transaction->second, true);
      if (transaction->second.promote)
        fs::remove_all(checkpoint_staging_path(transaction->second.checkpoint,
                                               r.transaction_id));
      else if (transaction->second.uses_staging)
        fs::remove_all(tx_path(staging_root_, r.transaction_id));
      fs::remove(checkpoint_marker_path(staging_root_, r.transaction_id));
      if (transaction->second.promote)
        fs::remove_all(tx_path(staging_root_, r.transaction_id));
      fs::remove_all(scratch_root_ / r.transaction_id);
    } catch (const fs::filesystem_error &e) {
      return fail(r.transaction_id, e.what());
    }
    staged_bytes_ -= transaction->second.staged_bytes;
    transactions_.erase(transaction);
  }
  return {true, r.transaction_id, {}, {}, {}};
}

Server::Server(fs::path socket_path, fs::path staging, fs::path scratch,
               fs::path checkpoint_root, std::uint64_t budget)
    : socket_path_(std::move(socket_path)),
      transactions_(std::move(staging), std::move(scratch),
                    std::move(checkpoint_root), budget) {}

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
  bool provider_session = false;
  fs::path provider_root;
  if (size < 0) {
    response = fail({}, "read failed");
  } else if (static_cast<std::size_t>(size) > buffer.size()) {
    response = fail({}, "request is too large");
  } else if (!decode_request(buffer.data(), static_cast<std::size_t>(size),
                             request, error)) {
    response = fail({}, error.empty() ? "invalid request" : error);
  } else if (request.operation == Request::Operation::Submit) {
    response = transactions_.submit(request);
    if (response.ok && request.staging)
      response = transactions_.wait_for_staging(request, std::move(response));
    if (response.ok) {
      provider_session = true;
      provider_root = response.staging_path;
    }
  } else if (request.operation == Request::Operation::PrepareCheckpoint) {
    response = transactions_.prepare_checkpoint(request);
  } else if (request.operation == Request::Operation::Commit) {
    response = transactions_.commit(request);
  } else if (request.operation == Request::Operation::Abort) {
    response = transactions_.abort(request);
  } else {
    response = fail(request.transaction_id, "unknown operation");
  }
  auto encoded = encode_response(response);
  auto sent = send(client.get(), encoded.data(), encoded.size(), MSG_NOSIGNAL);
  if (!provider_session || sent != static_cast<ssize_t>(encoded.size())) return;
  auto staging = transactions_.staging_state(request.transaction_id);
  ProviderSession session(staging);
  timeval no_timeout{};
  if (setsockopt(client.get(), SOL_SOCKET, SO_RCVTIMEO, &no_timeout,
                 sizeof(no_timeout)) < 0 ||
      setsockopt(client.get(), SOL_SOCKET, SO_SNDTIMEO, &no_timeout,
                 sizeof(no_timeout)) < 0)
    return;
  std::osyncstream(std::cerr)
      << "pagebroker provider session start transaction="
      << std::quoted(request.transaction_id) << " root="
      << std::quoted(provider_root.string()) << std::endl;
  auto status = serve_provider(provider_root, client.get(), -1, staging);
  std::osyncstream(std::cerr)
      << "pagebroker provider session stop transaction="
      << std::quoted(request.transaction_id) << " status=" << status
      << std::endl;
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
        } catch (const std::exception &error) {
          std::osyncstream(std::cerr)
              << "pagebroker client failed error=" << std::quoted(error.what())
              << std::endl;
        } catch (...) {
          std::osyncstream(std::cerr) << "pagebroker client failed" << std::endl;
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
          const fs::path &scratch, const fs::path &checkpoint_root,
          std::uint64_t budget) {
  return Server(socket_path, staging, scratch, checkpoint_root, budget).run();
}

namespace {
class Provider {
public:
  Provider(fs::path root, int socket_fd, int diagnostic_fd,
           std::shared_ptr<StagingState> staging)
      : root_(std::move(root)), socket_fd_(socket_fd),
        diagnostic_fd_(diagnostic_fd), staging_(std::move(staging)),
        root_fd_(open(root_.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC |
                                        O_NOFOLLOW)),
        root_error_(root_fd_ ? 0 : errno) {}

  int run() {
    struct stat socket_stat {};
    if (fstat(socket_fd_, &socket_stat) < 0)
      return failure("fstat socket", errno);
    if (!S_ISSOCK(socket_stat.st_mode))
      return failure("fstat socket", ENOTSOCK);
    if (!root_fd_)
      return failure("open root", root_error_);
    if (diagnostic_fd_ >= 0)
      dprintf(diagnostic_fd_, "ready socket_fd=%d fstat=ok\n", socket_fd_);

    std::array<char, 1 << 20> buffer;
    for (;;) {
      auto size = recv(socket_fd_, buffer.data(), buffer.size(), MSG_TRUNC);
      if (size == 0) {
        print_timings();
        return 0;
      }
      if (size < 0) {
        print_timings();
        return failure("recv", errno);
      }
      if (static_cast<std::size_t>(size) > buffer.size()) {
        print_timings();
        return failure("recv", EMSGSIZE);
      }

      auto started = std::chrono::steady_clock::now();
      ProviderRequest request;
      auto decode_started = std::chrono::steady_clock::now();
      auto decoded = decode_provider_request(buffer.data(), size, request);
      auto *timing = request.operation < timings_.size()
                         ? &timings_[request.operation]
                         : nullptr;
      if (timing) {
        ++timing->count;
        timing->decode_ns += elapsed_ns(decode_started);
        if (request.operation == 3 || request.operation == 4) {
          auto available = std::numeric_limits<std::uint64_t>::max() -
                           timing->bytes;
          timing->bytes += std::min(request.length, available);
        }
      }

      Reply reply;
      if (!decoded) {
        std::osyncstream(std::cerr)
            << "pagebroker provider request decode_failed root="
            << std::quoted(root_.string()) << " bytes=" << size << std::endl;
        reply.set_status(-EBADMSG);
      } else {
        handle(request, reply, timing);
      }

      auto send_started = std::chrono::steady_clock::now();
      auto send_error = send(reply);
      if (timing) {
        timing->send_ns += elapsed_ns(send_started);
        auto total_ns = elapsed_ns(started);
        timing->total_ns += total_ns;
        timing->max_ns = std::max(timing->max_ns, total_ns);
      }
      if (send_error) {
        print_timings();
        return failure("sendmsg", send_error);
      }
      if (request.operation == 5 || request.operation == 6)
        print_timings();
    }
  }

private:
  struct ProviderTiming {
    std::uint64_t count{};
    std::uint64_t total_ns{};
    std::uint64_t max_ns{};
    std::uint64_t bytes{};
    std::uint64_t decode_ns{};
    std::uint64_t log_ns{};
    std::uint64_t readiness_ns{};
    std::uint64_t open_ns{};
    std::uint64_t memfd_ns{};
    std::uint64_t truncate_ns{};
    std::uint64_t seal_ns{};
    std::uint64_t send_ns{};
  };

  struct Reply {
    void set_status(std::int32_t status) { response_status(data, status); }
    std::string data;
    FileDescriptor fd;
  };

  static std::uint64_t
  elapsed_ns(std::chrono::steady_clock::time_point started) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - started)
            .count());
  }

  static double seconds(std::uint64_t ns) {
    return static_cast<double>(ns) / 1'000'000'000.0;
  }

  static const char *operation_name(std::uint64_t operation) {
    switch (operation) {
    case 1: return "INIT";
    case 2: return "OPEN_IMAGE";
    case 3: return "GET_VMA";
    case 4: return "GET_SHARED";
    case 5: return "COMMIT";
    case 6: return "ABORT";
    default: return "UNKNOWN";
    }
  }

  int failure(const char *operation, int error) const {
    if (diagnostic_fd_ >= 0)
      dprintf(diagnostic_fd_, "failure operation=%s errno=%d (%s)\n",
              operation, error, std::strerror(error));
    return error ? error : 1;
  }

  void print_timings() {
    if (timings_printed_)
      return;
    timings_printed_ = true;
    for (std::size_t operation = 1; operation < timings_.size(); ++operation) {
      const auto &timing = timings_[operation];
      if (!timing.count)
        continue;
      std::osyncstream(std::cerr)
          << "pagebroker provider timing op=" << operation_name(operation)
          << " count=" << timing.count << std::fixed << std::setprecision(6)
          << " total_s=" << seconds(timing.total_ns)
          << " avg_s=" << seconds(timing.total_ns) / timing.count
          << " max_s=" << seconds(timing.max_ns)
          << " bytes=" << timing.bytes
          << " decode_s=" << seconds(timing.decode_ns)
          << " log_s=" << seconds(timing.log_ns)
          << " readiness_path_s=" << seconds(timing.readiness_ns)
          << " open_s=" << seconds(timing.open_ns)
          << " memfd_s=" << seconds(timing.memfd_ns)
          << " truncate_s=" << seconds(timing.truncate_ns)
          << " seal_s=" << seconds(timing.seal_ns)
          << " send_s=" << seconds(timing.send_ns) << std::endl;
    }
  }

  void log_request(const ProviderRequest &request, ProviderTiming &timing) {
    auto started = std::chrono::steady_clock::now();
    std::ostringstream log;
    log << "pagebroker provider request root=" << std::quoted(root_.string())
        << " op=" << operation_name(request.operation);
    switch (request.operation) {
    case 2:
      log << " path=" << std::quoted(request.name) << " flags=0x" << std::hex
          << request.flags << std::dec;
      break;
    case 3:
      log << " pid=" << request.pid << " vaddr=0x" << std::hex
          << request.vaddr << std::dec << " length=" << request.length;
      break;
    case 4:
      log << " shm_id=" << request.shared_id << " length=" << request.length;
      break;
    }
    std::osyncstream(std::cerr) << log.str() << std::endl;
    timing.log_ns += elapsed_ns(started);
  }

  void handle(const ProviderRequest &request, Reply &reply,
              ProviderTiming *timing) {
    if (request.operation >= timings_.size() || !timing) {
      reply.set_status(-ENOTSUP);
      return;
    }
    ++request_counts_[request.operation];
    log_request(request, *timing);
    switch (request.operation) {
    case 1:
    case 5:
    case 6:
      reply.set_status(0);
      break;
    case 2:
      open_image(request, reply, *timing);
      break;
    case 3:
      get_vma(request, reply, *timing);
      break;
    case 4:
      get_shared(request, reply, *timing);
      break;
    default:
      reply.set_status(-ENOTSUP);
      break;
    }
    if (request.operation == 5 || request.operation == 6) {
      std::cerr << "pagebroker provider requests init=" << request_counts_[1]
                << " image=" << request_counts_[2]
                << " vma=" << request_counts_[3]
                << " shared=" << request_counts_[4] << std::endl;
    }
  }

  FileDescriptor create_memfd(const char *name, std::uint64_t length,
                              ProviderTiming &timing, int &error) {
    if (length >
        static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
      error = EFBIG;
      return FileDescriptor();
    }
    auto started = std::chrono::steady_clock::now();
    FileDescriptor fd(syscall(SYS_memfd_create, name, MFD_ALLOW_SEALING));
    timing.memfd_ns += elapsed_ns(started);
    if (!fd) {
      error = errno;
      return FileDescriptor();
    }
    started = std::chrono::steady_clock::now();
    auto result = ftruncate(fd.get(), static_cast<off_t>(length));
    timing.truncate_ns += elapsed_ns(started);
    if (result < 0) {
      error = errno;
      return FileDescriptor();
    }
    error = 0;
    return fd;
  }

  void get_vma(const ProviderRequest &request, Reply &reply,
               ProviderTiming &timing) {
    int error;
    auto fd = create_memfd("pagebroker-extmem", request.length, timing, error);
    if (!fd) {
      reply.set_status(-error);
      return;
    }
    auto started = std::chrono::steady_clock::now();
    fcntl(fd.get(), F_ADD_SEALS, F_SEAL_GROW | F_SEAL_SHRINK);
    timing.seal_ns += elapsed_ns(started);
    reply.fd = std::move(fd);
    reply.set_status(0);
  }

  void get_shared(const ProviderRequest &request, Reply &reply,
                  ProviderTiming &timing) {
    if (request.length >
        static_cast<std::uint64_t>(std::numeric_limits<off_t>::max())) {
      reply.set_status(-EFBIG);
      return;
    }
    auto existing = shared_fds_.find(request.shared_id);
    if (existing == shared_fds_.end()) {
      int error;
      auto fd = create_memfd("pagebroker-extmem-shared", request.length,
                             timing, error);
      if (!fd) {
        reply.set_status(-error);
        return;
      }
      existing = shared_fds_.emplace(request.shared_id, std::move(fd)).first;
    } else {
      struct stat shared_stat {};
      if (fstat(existing->second.get(), &shared_stat) < 0) {
        reply.set_status(-errno);
        return;
      }
      if (shared_stat.st_size < static_cast<off_t>(request.length)) {
        auto started = std::chrono::steady_clock::now();
        auto result = ftruncate(existing->second.get(),
                                static_cast<off_t>(request.length));
        timing.truncate_ns += elapsed_ns(started);
        if (result < 0) {
          reply.set_status(-errno);
          return;
        }
      }
    }
    reply.fd.reset(fcntl(existing->second.get(), F_DUPFD_CLOEXEC, 0));
    reply.set_status(reply.fd ? 0 : -errno);
  }

  int wait_for_image(const fs::path &relative, ProviderTiming &timing) {
    if (!staging_)
      return 0;
    auto started = std::chrono::steady_clock::now();
    auto key = relative.generic_string();
    std::unique_lock lock(staging_->mutex);
    if (!staging_->planned_files.contains(key)) {
      timing.readiness_ns += elapsed_ns(started);
      return -ENOTSUP;
    }
    auto prioritize = staging_->prioritize_file;
    lock.unlock();
    auto prioritized = prioritize && prioritize(key);
    lock.lock();
    if (prioritized)
      std::osyncstream(std::cerr)
          << "pagebroker stage prioritized path=" << std::quoted(key)
          << std::endl;
    staging_->changed.wait(lock, [&] {
      return staging_->ready_files.contains(key) || staging_->complete ||
             staging_->cancelled || !staging_->error.empty();
    });

    int status = 0;
    if (!staging_->ready_files.contains(key)) {
      if (!staging_->error.empty())
        status = -(staging_->error_code ? staging_->error_code : EIO);
      else if (staging_->cancelled)
        status = -ECANCELED;
      else
        status = -ENOTSUP;
    }
    auto waited = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - started);
    timing.readiness_ns += elapsed_ns(started);
    if (waited.count() > 0)
      std::osyncstream(std::cerr)
          << "pagebroker provider file wait path=" << std::quoted(key)
          << " duration_ms=" << waited.count() << " status=" << status
          << std::endl;
    return status;
  }

  void open_image(const ProviderRequest &request, Reply &reply,
                  ProviderTiming &timing) {
    fs::path relative(request.name);
    if (!safe_relative(relative) ||
        std::find(relative.begin(), relative.end(), fs::path("..")) !=
            relative.end() ||
        request.flags > static_cast<std::uint64_t>(
                            std::numeric_limits<int>::max())) {
      reply.set_status(-EINVAL);
      return;
    }
    auto ready = wait_for_image(relative, timing);
    if (ready < 0) {
      reply.set_status(ready);
      return;
    }
    auto started = std::chrono::steady_clock::now();
    reply.fd.reset(open_beneath(root_fd_.get(), relative,
                                static_cast<int>(request.flags)));
    auto open_error = reply.fd ? 0 : errno;
    timing.open_ns += elapsed_ns(started);
    // Optional CRIU images must fall back to its normal local open path.
    reply.set_status(reply.fd ? 0
                              : (open_error == ENOENT ? -ENOTSUP
                                                      : -open_error));
  }

  int send(Reply &reply) const {
    struct iovec iov{reply.data.data(), reply.data.size()};
    std::array<char, CMSG_SPACE(sizeof(int))> control{};
    struct msghdr message {};
    message.msg_iov = &iov;
    message.msg_iovlen = 1;
    if (reply.fd) {
      message.msg_control = control.data();
      message.msg_controllen = control.size();
      auto *cmsg = CMSG_FIRSTHDR(&message);
      cmsg->cmsg_level = SOL_SOCKET;
      cmsg->cmsg_type = SCM_RIGHTS;
      cmsg->cmsg_len = CMSG_LEN(sizeof(int));
      auto fd = reply.fd.get();
      std::memcpy(CMSG_DATA(cmsg), &fd, sizeof(fd));
    }
    if (sendmsg(socket_fd_, &message, MSG_NOSIGNAL) < 0)
      return errno;
    return 0;
  }

  fs::path root_;
  int socket_fd_;
  int diagnostic_fd_;
  std::shared_ptr<StagingState> staging_;
  FileDescriptor root_fd_;
  int root_error_;
  std::map<std::uint64_t, FileDescriptor> shared_fds_;
  std::array<ProviderTiming, 7> timings_{};
  std::array<std::uint64_t, 7> request_counts_{};
  bool timings_printed_{};
};
} // namespace

int serve_provider(const fs::path &root, int socket_fd, int diagnostic_fd,
                   std::shared_ptr<StagingState> staging) {
  return Provider(root, socket_fd, diagnostic_fd, std::move(staging)).run();
}

} // namespace pagebroker

#ifndef PAGEBROKER_TEST
int main(int argc, char **argv) {
  if (argc == 5 && std::string(argv[1]) == "provider")
    return pagebroker::serve_provider(argv[2], std::stoi(argv[3]), std::stoi(argv[4]));
  if ((argc != 2 && argc != 3) || std::string(argv[1]) != "serve")
    return 2;
  auto budget = pagebroker::filesystem_budget("/staging");
  if (budget == 0) return 1;
  return pagebroker::serve("/run/pagebroker/pagebroker.sock", "/staging",
                           "/scratch", argc == 3 ? argv[2] : "", budget);
}
#endif
