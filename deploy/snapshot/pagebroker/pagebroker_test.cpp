// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "pagebroker.hpp"

#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cassert>
#include <cstdio>
#include <cstring>
#include <fstream>
int main() {
  pagebroker::Request decoded;
  std::string decode_error;
  const char request_with_unknown_fields[] = {
      '\x08', '\x01', '\x51', 0, 0, 0, 0, 0, 0, 0, 0, '\x5d', 0, 0, 0, 0};
  assert(pagebroker::decode_request(request_with_unknown_fields,
                                    sizeof(request_with_unknown_fields),
                                    decoded, decode_error));
  assert(decoded.operation == pagebroker::Request::Operation::Submit);
  const char request_with_unknown_group[] = {'\x08', '\x01', '\x63',
                                             '\x08', '\x01', '\x64'};
  assert(pagebroker::decode_request(request_with_unknown_group,
                                    sizeof(request_with_unknown_group),
                                    decoded, decode_error));
  decoded.transaction_id = "stale";
  const char request_without_id[] = {'\x08', '\x01'};
  assert(pagebroker::decode_request(
      request_without_id, sizeof(request_without_id), decoded, decode_error));
  assert(decoded.transaction_id.empty());
  const char invalid_field_zero[] = {'\x00', '\x00'};
  assert(!pagebroker::decode_request(
      invalid_field_zero, sizeof(invalid_field_zero), decoded, decode_error));
  const char overflowing_varint[] = {'\x08', '\x80', '\x80', '\x80',
                                     '\x80', '\x80', '\x80', '\x80',
                                     '\x80', '\x80', '\x02'};
  assert(!pagebroker::decode_request(
      overflowing_varint, sizeof(overflowing_varint), decoded, decode_error));
  auto root = std::filesystem::temp_directory_path() /
              ("pagebroker-test-" + std::to_string(getpid()));
  std::filesystem::remove_all(root);
  auto source = root / "source";
  std::filesystem::create_directories(source);
  std::ofstream(source / "image").write("checkpoint", 10);
  assert(pagebroker::filesystem_budget(root) > 0);

  auto outside = root / "outside";
  std::ofstream(outside) << "outside";
  std::filesystem::create_directory(source / "nested");
  std::filesystem::create_symlink(outside, source / "nested/link");
  auto unsafe = pagebroker::TransactionManager(root / "staging-unsafe",
                                               root / "scratch-unsafe", 100)
                    .submit({pagebroker::Request::Operation::Submit,
                             "tx-symlink", source});
  assert(!unsafe.ok);
  assert(!std::filesystem::exists(root / "staging-unsafe/tx/outside"));
  std::filesystem::remove(source / "nested/link");
  pagebroker::TransactionManager manager(root / "staging", root / "scratch",
                                         100);
  pagebroker::Request submit{pagebroker::Request::Operation::Submit, "tx-1",
                             source};
  auto ok = manager.submit(submit);
  assert(ok.ok);
  assert(std::filesystem::exists(root / "staging/tx/tx-1/image"));
  pagebroker::TransactionManager concurrent_manager(
      root / "staging-concurrent", root / "scratch-concurrent", 15);
  auto first = concurrent_manager.submit(submit);
  assert(first.ok);
  auto second = concurrent_manager.submit(pagebroker::Request{
      pagebroker::Request::Operation::Submit, "tx-2", source});
  assert(!second.ok);
  assert(concurrent_manager.abort(submit).ok);
  auto duplicate = manager.submit(submit);
  assert(!duplicate.ok);
  auto invalid_source = root / "invalid-source";
  std::filesystem::create_directories(invalid_source);
  std::filesystem::create_symlink(outside, invalid_source / "link");
  auto invalid_duplicate = manager.submit(
      {pagebroker::Request::Operation::Submit, "tx-1", invalid_source});
  assert(!invalid_duplicate.ok);
  assert(std::filesystem::exists(root / "staging/tx/tx-1/image"));
  auto committed = manager.commit(submit);
  assert(committed.ok);
  assert(!std::filesystem::exists(root / "staging/tx/tx-1"));
  auto too_big =
      pagebroker::TransactionManager(root / "staging2", root / "scratch2", 1)
          .submit(submit);
  assert(!too_big.ok);

  auto server_root = root / "server";
  auto server_socket = server_root / "pagebroker.sock";
  auto server_pid = fork();
  assert(server_pid >= 0);
  if (server_pid == 0)
    _exit(pagebroker::serve(server_socket, server_root / "staging",
                            server_root / "scratch", 1 << 20));
  for (int i = 0; i < 500 && !std::filesystem::exists(server_socket); ++i)
    usleep(10000);
  assert(std::filesystem::exists(server_socket));

  auto connect_control = [&] {
    int fd = socket(AF_UNIX, SOCK_SEQPACKET, 0);
    assert(fd >= 0);
    sockaddr_un address{};
    address.sun_family = AF_UNIX;
    std::snprintf(address.sun_path, sizeof(address.sun_path), "%s",
                  server_socket.c_str());
    assert(connect(fd, reinterpret_cast<sockaddr *>(&address),
                   sizeof(address)) == 0);
    return fd;
  };
  int idle_control = connect_control();
  int active_control = connect_control();
  assert(send(active_control, "\x08\x02", 2, 0) == 2);
  pollfd control_poll{active_control, POLLIN, 0};
  assert(poll(&control_poll, 1, 1000) == 1);
  char response[128];
  assert(recv(active_control, response, sizeof(response), 0) > 0);
  close(active_control);
  close(idle_control);
  int oversized_control = connect_control();
  std::string oversized_request(65537, '\0');
  oversized_request[0] = '\x08';
  oversized_request[1] = '\x02';
  assert(send(oversized_control, oversized_request.data(),
              oversized_request.size(),
              0) == static_cast<ssize_t>(oversized_request.size()));
  auto oversized_bytes = recv(oversized_control, response, sizeof(response), 0);
  assert(oversized_bytes > 0);
  assert(std::string(response, response + oversized_bytes)
             .find("request is too large") != std::string::npos);
  close(oversized_control);

  auto connect_health = [] {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    assert(fd >= 0);
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    address.sin_port = htons(8080);
    for (int i = 0; connect(fd, reinterpret_cast<sockaddr *>(&address),
                            sizeof(address)) != 0;
         ++i) {
      assert(i < 500);
      usleep(10000);
    }
    return fd;
  };
  int idle_health = connect_health();
  usleep(1200000);
  int health = connect_health();
  const char probe[] = "GET /healthz HTTP/1.1\r\n\r\n";
  assert(send(health, probe, sizeof(probe) - 1, 0) ==
         static_cast<ssize_t>(sizeof(probe) - 1));
  pollfd health_poll{health, POLLIN, 0};
  assert(poll(&health_poll, 1, 1000) == 1);
  auto health_bytes = recv(health, response, sizeof(response), 0);
  assert(health_bytes > 0);
  assert(std::string(response, response + health_bytes).find("200 OK") !=
         std::string::npos);
  close(health);
  int metrics = connect_health();
  const char metrics_probe[] = "GET /metrics HTTP/1.1\r\n\r\n";
  assert(send(metrics, metrics_probe, sizeof(metrics_probe) - 1, 0) ==
         static_cast<ssize_t>(sizeof(metrics_probe) - 1));
  auto metrics_bytes = recv(metrics, response, sizeof(response), 0);
  assert(metrics_bytes > 0);
  assert(
      std::string(response, response + metrics_bytes).find("404 Not Found") !=
      std::string::npos);
  close(metrics);
  close(idle_health);
  kill(server_pid, SIGTERM);
  assert(waitpid(server_pid, nullptr, 0) == server_pid);

  std::filesystem::remove_all(root);
}
