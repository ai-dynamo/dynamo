// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <atomic>
#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <mutex>
#include <semaphore>
#include <string>

namespace pagebroker {

struct Request {
  enum class Operation : std::uint32_t { Submit = 1, Commit = 2, Abort = 3 };
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

bool decode_request(const void *data, std::size_t size, Request &request,
                    std::string &error);
std::string encode_response(const Response &response);

class TransactionManager {
public:
  TransactionManager(std::filesystem::path staging_root,
                     std::filesystem::path scratch_root, std::uint64_t budget);
  Response submit(const Request &request);
  Response commit(const Request &request);
  Response abort(const Request &request);

private:
  struct TransactionState {
    enum class Phase { Staging, Ready, Cleaning };
    std::uint64_t staged_bytes{};
    std::atomic_bool cancelled{};
    Phase phase{Phase::Staging};
  };
  void release(const std::string &transaction_id,
               const std::shared_ptr<TransactionState> &state);
  void cleanup();
  std::filesystem::path staging_root_, scratch_root_;
  std::uint64_t budget_;
  std::map<std::string, std::shared_ptr<TransactionState>> transactions_;
  std::uint64_t staged_bytes_{};
  std::mutex mutex_;
};

class Server {
public:
  Server(std::filesystem::path socket_path, std::filesystem::path staging_root,
         std::filesystem::path scratch_root, std::uint64_t budget);
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
          const std::filesystem::path &scratch_root, std::uint64_t budget);
std::uint64_t filesystem_budget(const std::filesystem::path &path);
} // namespace pagebroker
