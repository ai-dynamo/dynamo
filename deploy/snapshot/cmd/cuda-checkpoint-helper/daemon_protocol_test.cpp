/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "daemon_protocol.h"

#include <cassert>
#include <chrono>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <thread>
#include <unistd.h>

namespace {

using namespace cuda_checkpoint_daemon;

Request
TestRequest(Action action)
{
  return Request{
      .action = action,
      .pid = 123,
      .transfer_buffer_count =
          action == Action::kCheckpoint || action == Action::kRestore ? 2U : 0U,
      .transfer_chunk_bytes =
          action == Action::kCheckpoint || action == Action::kRestore ? 4096U : 0U,
      .expected_start_time_ticks = 987654,
      .device_map = action == Action::kRestore ? "GPU-a=GPU-b" : "",
      .storage_dir =
          action == Action::kCheckpoint || action == Action::kRestore
              ? "/checkpoints/cuda"
              : "",
      .expected_cgroup = "0::/kubepods/test\n",
  };
}

std::string
CreateProcRoot(const Request& request)
{
  char proc_template[] = "/tmp/cuda-daemon-proc-test-XXXXXX";
  const char* proc_root = mkdtemp(proc_template);
  assert(proc_root != nullptr);
  const std::filesystem::path process_dir =
      std::filesystem::path(proc_root) / std::to_string(request.pid);
  std::filesystem::create_directories(process_dir);
  {
    std::ofstream stat(process_dir / "stat");
    stat << "123 (worker with spaces) S 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 987654 20\n";
    std::ofstream cgroup(process_dir / "cgroup");
    cgroup << request.expected_cgroup;
  }
  return proc_root;
}

void
TestProtocol()
{
  Request request = TestRequest(Action::kRestore);
  std::vector<unsigned char> encoded;
  std::string error;
  assert(EncodeRequest(request, &encoded, &error));
  Request parsed;
  assert(ParseRequest(encoded.data(), encoded.size(), &parsed, &error));
  assert(
      parsed.pid == request.pid && parsed.storage_dir == request.storage_dir &&
      parsed.expected_start_time_ticks == request.expected_start_time_ticks &&
      parsed.expected_cgroup == request.expected_cgroup);

  encoded[4] = 99;
  assert(!ParseRequest(encoded.data(), encoded.size(), &parsed, &error));
  encoded.resize(kMaxRequestSize + 1);
  assert(!ParseRequest(encoded.data(), encoded.size(), &parsed, &error));

  Response response{.cuda_status = 17, .flags = kResponseFatal, .output = "stdout", .error = "stderr"};
  assert(EncodeResponse(response, &encoded, &error));
  Response parsed_response;
  assert(ParseResponse(encoded.data(), encoded.size(), &parsed_response, &error));
  assert(
      parsed_response.cuda_status == 17 && parsed_response.flags == kResponseFatal &&
      parsed_response.output == "stdout" && parsed_response.error == "stderr");
  encoded[16] = 200;
  assert(!ParseResponse(encoded.data(), encoded.size(), &parsed_response, &error));

  for (const Action action :
       {Action::kLock, Action::kCheckpoint, Action::kRestore, Action::kUnlock}) {
    request = TestRequest(action);
    assert(EncodeRequest(request, &encoded, &error));
    assert(ParseRequest(encoded.data(), encoded.size(), &parsed, &error));
    assert(parsed.action == action);
  }
}

void
TestExecutionIdentityAndFatalControlFlow()
{
  for (const Action action :
       {Action::kLock, Action::kCheckpoint, Action::kRestore, Action::kUnlock}) {
    Request request = TestRequest(action);
    const std::string proc_root = CreateProcRoot(request);
    int executions = 0;
    Response response;
    assert(ExecuteValidated(
        request, proc_root,
        [&executions](const Request&) {
          ++executions;
          return Response{};
        },
        &response));
    assert(executions == 1);
    request.expected_start_time_ticks++;
    assert(ExecuteValidated(
        request, proc_root,
        [&executions](const Request&) {
          ++executions;
          return Response{};
        },
        &response));
    assert(executions == 1);
    std::filesystem::remove_all(proc_root);
  }

  OperationState operation;
  int completions = 0;
  assert(FinishHandledOperation(
             false, 17, [&completions] {
               ++completions;
               return 0;
             },
             &operation) == 17);
  assert(completions == 0 && operation.fatal());
  operation = {};
  assert(FinishHandledOperation(
             true, 17, [&completions] {
               ++completions;
               return 0;
             },
             &operation) == 0);
  assert(completions == 1 && !operation.fatal());
  Response fatal{
      .cuda_status = 17,
      .flags = kResponseFatal,
      .output = "",
      .error = "",
  };
  assert(!ResponseAllowsServerContinue(fatal));
}

void
TestHealthStates()
{
  OperationHealth no_operation(std::chrono::seconds(1));
  HealthSnapshot snapshot = no_operation.Snapshot();
  assert(!snapshot.ready && !snapshot.busy && !snapshot.healthy);

  no_operation.MarkReady();
  snapshot = no_operation.Snapshot();
  assert(snapshot.ready && !snapshot.busy && snapshot.healthy);

  no_operation.Begin(Action::kCheckpoint, 123);
  snapshot = no_operation.Snapshot();
  assert(snapshot.ready && snapshot.busy && snapshot.healthy);
  assert(snapshot.action == Action::kCheckpoint && snapshot.pid == 123);
  std::this_thread::sleep_for(std::chrono::milliseconds(1100));
  snapshot = no_operation.Snapshot();
  assert(snapshot.busy && !snapshot.healthy);
  no_operation.End();
  snapshot = no_operation.Snapshot();
  assert(!snapshot.busy && snapshot.healthy);
}

void
TestSocketLifecycle()
{
  char socket_template[] = "/tmp/cuda-daemon-socket-test-XXXXXX";
  const char* directory = mkdtemp(socket_template);
  assert(directory != nullptr);
  const std::string path = std::string(directory) + "/helper.sock";
  std::string error;

  OwnedUnixSocket owner;
  assert(owner.Bind(path, 1, &error));
  struct stat owned {};
  assert(lstat(path.c_str(), &owned) == 0);
  OwnedUnixSocket contender;
  assert(!contender.Bind(path, 1, &error));
  struct stat after_contender {};
  assert(lstat(path.c_str(), &after_contender) == 0);
  assert(after_contender.st_ino == owned.st_ino && after_contender.st_dev == owned.st_dev);

  const std::string replacement = std::string(directory) + "/replacement";
  assert(unlink(path.c_str()) == 0);
  {
    std::ofstream file(replacement);
    file << "replacement";
  }
  assert(rename(replacement.c_str(), path.c_str()) == 0);
  owner.Close();
  struct stat replacement_stat {};
  assert(lstat(path.c_str(), &replacement_stat) == 0 && S_ISREG(replacement_stat.st_mode));
  assert(unlink(path.c_str()) == 0);

  const int stale_fd = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  assert(stale_fd >= 0);
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::memcpy(address.sun_path, path.c_str(), path.size() + 1);
  assert(bind(stale_fd, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == 0);
  close(stale_fd);
  OwnedUnixSocket stale_replacer;
  assert(stale_replacer.Bind(path, 1, &error));
  stale_replacer.Close();
  assert(lstat(path.c_str(), &replacement_stat) != 0 && errno == ENOENT);

  OwnedUnixSocket setup_failure;
  assert(!setup_failure.Bind(path, -1, &error));
  assert(lstat(path.c_str(), &replacement_stat) != 0 && errno == ENOENT);
  std::filesystem::remove_all(directory);
}

void
TestStopPipeWakesWaiters()
{
  int stop_pipe[2];
  assert(pipe(stop_pipe) == 0);
  const int listener = socket(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0);
  assert(listener >= 0);
  std::thread accept_waiter([&] { assert(PollForInputOrStop(listener, stop_pipe[0]) == 0); });
  const unsigned char wake = 1;
  assert(write(stop_pipe[1], &wake, sizeof(wake)) == sizeof(wake));
  accept_waiter.join();
  unsigned char drained = 0;
  assert(read(stop_pipe[0], &drained, sizeof(drained)) == sizeof(drained));

  int silent_client[2];
  assert(socketpair(AF_UNIX, SOCK_SEQPACKET | SOCK_CLOEXEC, 0, silent_client) == 0);
  std::thread client_waiter(
      [&] { assert(PollForInputOrStop(silent_client[0], stop_pipe[0]) == 0); });
  assert(write(stop_pipe[1], &wake, sizeof(wake)) == sizeof(wake));
  client_waiter.join();
  close(silent_client[0]);
  close(silent_client[1]);
  close(listener);
  close(stop_pipe[0]);
  close(stop_pipe[1]);
}

}  // namespace

int
main()
{
  TestProtocol();
  TestExecutionIdentityAndFatalControlFlow();
  TestHealthStates();
  TestSocketLifecycle();
  TestStopPipeWakesWaiters();
  return 0;
}
