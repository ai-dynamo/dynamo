// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>
#include <condition_variable>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <semaphore>
#include <set>
#include <string>

namespace pagebroker {

class CopyPool;

struct Request {
  enum class Operation : std::uint32_t {
    Submit = 1,
    PrepareCheckpoint = 2,
    Commit = 3,
    Abort = 4
  };
  Operation operation{};
  std::string transaction_id;
  std::string checkpoint_path;
};

struct Response {
  bool ok{};
  std::string transaction_id;
  std::string staging_path;
  std::string scratch_path;
  std::string error;
};

struct StagingState {
  std::mutex mutex;
  std::condition_variable changed;
  std::set<std::string> planned_files;
  std::set<std::string> ready_files;
  std::map<std::string, std::size_t> remaining_chunks;
  std::function<bool(const std::string &)> prioritize_file;
  std::uint64_t copied_bytes{};
  std::size_t remaining_tasks{};
  bool complete{};
  bool cancelled{};
  bool provider_running{};
  int error_code{};
  std::string error;
};

bool decode_request(const void *data, std::size_t size, Request &request,
                    std::string &error);
std::string encode_response(const Response &response);

class TransactionManager {
public:
  TransactionManager(std::filesystem::path staging_root,
                     std::filesystem::path scratch_root,
                     std::filesystem::path checkpoint_root,
                     std::uint64_t budget);
  ~TransactionManager();
  Response submit(const Request &request);
  Response prepare_checkpoint(const Request &request);
  Response commit(const Request &request);
  Response abort(const Request &request);
  std::shared_ptr<StagingState>
  staging_state(const std::string &transaction_id);
  void cleanup();

private:
  struct TransactionState {
    std::filesystem::path checkpoint;
    std::uint64_t staged_bytes{};
    bool promote{};
    std::shared_ptr<StagingState> staging;
  };
  static void stop_staging(TransactionState &state, bool cancel);
  std::filesystem::path staging_root_, scratch_root_, checkpoint_root_;
  std::uint64_t budget_;
  std::unique_ptr<CopyPool> copy_pool_;
  std::map<std::string, TransactionState> transactions_;
  std::uint64_t staged_bytes_{};
  std::mutex mutex_;
};

class Server {
public:
  Server(std::filesystem::path socket_path, std::filesystem::path staging_root,
         std::filesystem::path scratch_root,
         std::filesystem::path checkpoint_root, std::uint64_t budget);
  int run();

private:
  static void serve_health();
  void handle_client(int client);
  std::filesystem::path socket_path_;
  TransactionManager transactions_;
  std::counting_semaphore<64> client_slots_{64};
};

int serve(const std::filesystem::path &socket_path,
          const std::filesystem::path &staging_root,
          const std::filesystem::path &scratch_root,
          const std::filesystem::path &checkpoint_root,
          std::uint64_t budget);
std::uint64_t filesystem_budget(const std::filesystem::path &path);
int serve_provider(const std::filesystem::path &root, int socket_fd,
                   int diagnostic_fd,
                   std::shared_ptr<StagingState> staging = {});
#ifdef PAGEBROKER_TEST
bool test_copy_pool_priority();
#endif
} // namespace pagebroker
