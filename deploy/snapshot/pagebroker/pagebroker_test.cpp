// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
#include "pagebroker.hpp"
#include <cassert>
#include <cstdio>
#include <fstream>
#include <netinet/in.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/wait.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <cstring>
#include <cerrno>
#include <unistd.h>
int main() {
  assert(pagebroker::test_copy_pool_priority());
  pagebroker::Request decoded;
  std::string decode_error;
  const char request_with_unknown_fields[] = {
      '\x08', '\x01', '\x51', 0, 0, 0, 0, 0, 0, 0, 0,
      '\x5d', 0,      0,      0, 0};
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

  auto provider_root = root / "provider";
  std::filesystem::create_directories(provider_root);
  std::ofstream(provider_root / "image.img") << "provider-image";
  int sockets[2];
  assert(socketpair(AF_UNIX, SOCK_SEQPACKET, 0, sockets) == 0);
  auto child = fork();
  assert(child >= 0);
  if (child == 0) {
    close(sockets[0]);
    _exit(pagebroker::serve_provider(provider_root, sockets[1], -1));
  }
  close(sockets[1]);
  auto receive_response = [](int connection, int *received_fd = nullptr) {
    char response[128] = {};
    char control[CMSG_SPACE(sizeof(int))] = {};
    iovec iov{response, sizeof(response)};
    msghdr msg{};
    msg.msg_iov = &iov;
    msg.msg_iovlen = 1;
    msg.msg_control = control;
    msg.msg_controllen = sizeof(control);
    auto n = recvmsg(connection, &msg, 0);
    assert(n > 0);
    if (received_fd) {
      auto *cmsg = CMSG_FIRSTHDR(&msg);
      assert(cmsg && cmsg->cmsg_type == SCM_RIGHTS);
      std::memcpy(received_fd, CMSG_DATA(cmsg), sizeof(int));
    }
    std::uint64_t value = 0;
    for (int i = 1, shift = 0; i < n && shift < 64; ++i, shift += 7) {
      auto byte = static_cast<unsigned char>(response[i]);
      value |= static_cast<std::uint64_t>(byte & 127) << shift;
      if (!(byte & 128))
        return static_cast<std::int32_t>(value);
    }
    std::abort();
  };
  auto roundtrip = [&](const std::string &request, int *received_fd = nullptr) {
    auto sent = send(sockets[0], request.data(), request.size(), 0);
    if (sent != static_cast<ssize_t>(request.size())) {
      std::fprintf(stderr, "provider send failed: %s\n", std::strerror(errno));
      std::abort();
    }
    return receive_response(sockets[0], received_fd);
  };
  assert(roundtrip("\x08\x01\x63\x08\x01\x64") == 0);
  std::string open = "\x08\x02\x12\x0d\x0a\x09";
  open += "image.img";
  open.push_back('\x10');
  open.push_back('\0');
  int image_fd = -1;
  assert(roundtrip(open, &image_fd) == 0);
  char contents[32] = {};
  assert(read(image_fd, contents, sizeof(contents)) == 14);
  assert(std::string(contents, 14) == "provider-image");
  close(image_fd);
  std::string missing = "\x08\x02\x12\x13\x0a\x0f";
  missing += "not-present.img";
  missing.push_back('\x10');
  missing.push_back('\0');
  assert(roundtrip(missing) == -ENOTSUP);
  std::filesystem::create_directories(root / "outside-provider");
  std::ofstream(root / "outside-provider/secret") << "secret";
  std::filesystem::create_directory_symlink(root / "outside-provider",
                                            provider_root / "escape");
  std::string escaping = "\x08\x02\x12\x11\x0a\x0d";
  escaping += "escape/secret";
  escaping.push_back('\x10');
  escaping.push_back('\0');
  auto escaping_status = roundtrip(escaping);
  assert(escaping_status == -ELOOP || escaping_status == -ENOTDIR);
  int vma_fd = -1;
  assert(roundtrip("\x08\x03\x1a\x07\x08\x01\x10\x80\x20\x18\x10",
                   &vma_fd) == 0);
  struct stat vma_stat = {};
  assert(fstat(vma_fd, &vma_stat) == 0 && vma_stat.st_size == 16);
  assert(vma_stat.st_blocks == 0);
  assert((fcntl(vma_fd, F_GET_SEALS) & (F_SEAL_GROW | F_SEAL_SHRINK)) ==
         (F_SEAL_GROW | F_SEAL_SHRINK));
  close(vma_fd);
  const std::string oversized_vma =
      "\x08\x03\x1a\x10\x08\x01\x10\x80\x20\x18\xff\xff\xff\xff\xff"
      "\xff\xff\xff\xff\x01";
  assert(roundtrip(oversized_vma) == -EFBIG);
  int shared_fd = -1;
  assert(roundtrip("\x08\x04\x22\x04\x08\x07\x10\x20", &shared_fd) == 0);
  struct stat shared_stat = {};
  assert(fstat(shared_fd, &shared_stat) == 0 && shared_stat.st_size == 32);
  assert(shared_stat.st_blocks == 0);
  assert(fcntl(shared_fd, F_GET_SEALS) >= 0);
  assert(write(shared_fd, "shared", 6) == 6);
  int same_shared_fd = -1;
  assert(roundtrip("\x08\x04\x22\x04\x08\x07\x10\x40", &same_shared_fd) == 0);
  assert(fstat(same_shared_fd, &shared_stat) == 0 && shared_stat.st_size == 64);
  char shared_contents[7] = {};
  assert(pread(same_shared_fd, shared_contents, 6, 0) == 6);
  assert(std::string(shared_contents, 6) == "shared");
  close(same_shared_fd);
  int other_shared_fd = -1;
  assert(roundtrip("\x08\x04\x22\x04\x08\x08\x10\x20", &other_shared_fd) == 0);
  std::memset(shared_contents, 0xff, sizeof(shared_contents));
  assert(pread(other_shared_fd, shared_contents, 6, 0) == 6);
  assert(std::string(shared_contents, 6) == std::string(6, '\0'));
  close(other_shared_fd);
  close(shared_fd);
  assert(roundtrip("\x08\x05") == 0);
  close(sockets[0]);
  int status = 0;
  assert(waitpid(child, &status, 0) == child && WIFEXITED(status));

  auto delayed_root = root / "delayed-provider";
  std::filesystem::create_directories(delayed_root);
  int delayed_sockets[2];
  assert(socketpair(AF_UNIX, SOCK_SEQPACKET, 0, delayed_sockets) == 0);
  auto delayed_state = std::make_shared<pagebroker::StagingState>();
  delayed_state->planned_files.insert("image.img");
  std::thread delayed_provider([&] {
    assert(pagebroker::serve_provider(delayed_root, delayed_sockets[1], -1,
                                      delayed_state) == 0);
    close(delayed_sockets[1]);
  });
  assert(send(delayed_sockets[0], open.data(), open.size(), 0) ==
         static_cast<ssize_t>(open.size()));
  pollfd delayed_poll{delayed_sockets[0], POLLIN, 0};
  assert(poll(&delayed_poll, 1, 25) == 0);
  std::ofstream(delayed_root / "image.img.partial") << "delayed-image";
  std::filesystem::rename(delayed_root / "image.img.partial",
                          delayed_root / "image.img");
  {
    std::lock_guard lock(delayed_state->mutex);
    delayed_state->ready_files.insert("image.img");
  }
  delayed_state->changed.notify_all();
  int delayed_fd = -1;
  assert(receive_response(delayed_sockets[0], &delayed_fd) == 0);
  std::memset(contents, 0, sizeof(contents));
  assert(read(delayed_fd, contents, sizeof(contents)) == 13);
  assert(std::string(contents, 13) == "delayed-image");
  close(delayed_fd);

  assert(send(delayed_sockets[0], missing.data(), missing.size(), 0) ==
         static_cast<ssize_t>(missing.size()));
  assert(receive_response(delayed_sockets[0]) == -ENOTSUP);
  close(delayed_sockets[0]);
  delayed_provider.join();

  int cancelled_sockets[2];
  assert(socketpair(AF_UNIX, SOCK_SEQPACKET, 0, cancelled_sockets) == 0);
  auto cancelled_state = std::make_shared<pagebroker::StagingState>();
  cancelled_state->planned_files.insert("not-present.img");
  std::thread cancelled_provider([&] {
    assert(pagebroker::serve_provider(delayed_root, cancelled_sockets[1], -1,
                                      cancelled_state) == 0);
    close(cancelled_sockets[1]);
  });
  assert(send(cancelled_sockets[0], missing.data(), missing.size(), 0) ==
         static_cast<ssize_t>(missing.size()));
  pollfd cancelled_poll{cancelled_sockets[0], POLLIN, 0};
  assert(poll(&cancelled_poll, 1, 25) == 0);
  {
    std::lock_guard lock(cancelled_state->mutex);
    cancelled_state->cancelled = true;
  }
  cancelled_state->changed.notify_all();
  assert(receive_response(cancelled_sockets[0]) == -ECANCELED);
  close(cancelled_sockets[0]);
  cancelled_provider.join();

  auto source = root / "source";
  std::filesystem::create_directories(source);
  std::ofstream(source / "image").write("checkpoint", 10);
  assert(pagebroker::filesystem_budget(root) > 0);

  auto outside_file = root / "symlink-target";
  std::ofstream(outside_file) << "outside";
  std::filesystem::create_directory(source / "nested");
  std::filesystem::create_symlink(outside_file, source / "nested/link");
  auto unsafe = pagebroker::TransactionManager(
                    root / "staging-unsafe", root / "scratch-unsafe",
                    root / "checkpoints", 100)
                    .submit({pagebroker::Request::Operation::Submit,
                             "tx-symlink", source, true});
  assert(!unsafe.ok);
  assert(!std::filesystem::exists(root / "staging-unsafe/tx/outside"));
  std::filesystem::remove(source / "nested/link");
  pagebroker::TransactionManager manager(root / "staging", root / "scratch",
                                         root / "checkpoints", 10);
  pagebroker::Request submit{pagebroker::Request::Operation::Submit, "tx-1",
                             source, true};
  auto ok = manager.submit(submit);
  assert(ok.ok);
  pagebroker::TransactionManager concurrent_manager(root / "staging-concurrent",
                                                    root / "scratch-concurrent",
                                                    root / "checkpoints", 15);
  auto first = concurrent_manager.submit(submit);
  assert(first.ok);
  auto second = concurrent_manager.submit(
      pagebroker::Request{pagebroker::Request::Operation::Submit, "tx-2",
                          source, true});
  assert(!second.ok);
  assert(concurrent_manager.abort(submit).ok);
  auto duplicate = manager.submit(submit);
  assert(!duplicate.ok);
  auto invalid_source = root / "invalid-source";
  std::filesystem::create_directories(invalid_source);
  std::filesystem::create_symlink(outside_file, invalid_source / "link");
  auto invalid_duplicate = manager.submit(
      {pagebroker::Request::Operation::Submit, "tx-1", invalid_source,
       true});
  assert(!invalid_duplicate.ok);
  assert(std::filesystem::exists(root / "staging/tx/tx-1/image"));
  auto second_rejected = manager.submit(
      pagebroker::Request{pagebroker::Request::Operation::Submit, "tx-2",
                          source, true});
  assert(!second_rejected.ok);
  auto committed = manager.commit(submit);
  assert(committed.ok);
  assert(!std::filesystem::exists(root / "staging/tx/tx-1"));
  auto too_big =
      pagebroker::TransactionManager(root / "staging2", root / "scratch2",
                                     root / "checkpoints", 1)
          .submit(submit);
  assert(!too_big.ok);

  pagebroker::TransactionManager direct_manager(
      root / "staging-direct", root / "scratch-direct", root / "checkpoints", 0);
  pagebroker::Request direct{pagebroker::Request::Operation::Submit,
                             "tx-direct", source};
  auto direct_result = direct_manager.submit(direct);
  assert(direct_result.ok);
  assert(direct_result.staging_path == source);
  assert(!std::filesystem::exists(root / "staging-direct/tx/tx-direct"));
  assert(direct_manager.commit(direct).ok);
  assert(std::filesystem::exists(source / "image"));

  auto checkpoint_root = root / "checkpoints";
  auto destination = checkpoint_root / "nested" / "final";
  pagebroker::TransactionManager checkpoint_manager(
      root / "staging3", root / "scratch3", checkpoint_root, 100);
  std::filesystem::create_directories(destination);
  std::ofstream(destination / "old-manifest").write("old", 3);
  pagebroker::Request prepare{
      pagebroker::Request::Operation::PrepareCheckpoint, "tx-checkpoint",
      destination};
  auto prepared = checkpoint_manager.prepare_checkpoint(prepare);
  assert(prepared.ok);
  std::ofstream(std::filesystem::path(prepared.staging_path) / "manifest")
      .write("ok", 2);
  assert(checkpoint_manager.commit(prepare).ok);
  assert(std::filesystem::exists(destination / "manifest"));
  assert(!std::filesystem::exists(destination / "old-manifest"));
  assert(!std::filesystem::exists(
      destination.parent_path() / ".final.pagebroker-old-tx-checkpoint"));

  auto outside = checkpoint_manager.prepare_checkpoint(pagebroker::Request{
      pagebroker::Request::Operation::PrepareCheckpoint, "tx-outside",
      root / "outside"});
  assert(!outside.ok);

  auto leaked = checkpoint_manager.prepare_checkpoint(pagebroker::Request{
      pagebroker::Request::Operation::PrepareCheckpoint, "tx-leaked",
      checkpoint_root / "leaked"});
  assert(leaked.ok);
  pagebroker::TransactionManager restarted(
      root / "staging3", root / "scratch3", checkpoint_root, 100);
  assert(!std::filesystem::exists(leaked.staging_path));

  auto outside_staging = root / "outside" / ".keep.pagebroker-tx-unsafe";
  std::filesystem::create_directories(outside_staging);
  std::filesystem::create_directories(root / "staging-unsafe" / "tx");
  std::ofstream(root / "staging-unsafe" / "tx" /
                ".checkpoint-tx-unsafe")
      << outside_staging.string() << '\n';
  pagebroker::TransactionManager safe_cleanup(
      root / "staging-unsafe", root / "scratch-unsafe", checkpoint_root, 100);
  assert(std::filesystem::exists(outside_staging));

  auto server_root = root / "server";
  auto server_source = server_root / "source";
  auto server_socket = server_root / "pagebroker.sock";
  std::filesystem::create_directories(server_source);
  std::ofstream(server_source / "image.img") << "sidecar-provider";
  auto server_pid = fork();
  assert(server_pid >= 0);
  if (server_pid == 0)
    _exit(pagebroker::serve(server_socket, server_root / "staging",
                            server_root / "scratch",
                            server_root / "checkpoints", 1 << 20));
  for (int i = 0; i < 500 && !std::filesystem::exists(server_socket); ++i)
    usleep(10000);
  assert(std::filesystem::exists(server_socket));

  auto connect_server = [&] {
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
  int idle_control = connect_server();
  int active_control = connect_server();
  assert(send(active_control, "\x08\x02", 2, 0) == 2);
  pollfd control_poll{active_control, POLLIN, 0};
  assert(poll(&control_poll, 1, 1000) == 1);
  char concurrent_response[128];
  assert(recv(active_control, concurrent_response,
              sizeof(concurrent_response), 0) > 0);
  close(active_control);
  close(idle_control);
  int oversized_control = connect_server();
  std::string oversized_request(65537, '\0');
  oversized_request[0] = '\x08';
  oversized_request[1] = '\x02';
  assert(send(oversized_control, oversized_request.data(),
              oversized_request.size(),
              0) == static_cast<ssize_t>(oversized_request.size()));
  auto oversized_bytes = recv(oversized_control, concurrent_response,
                              sizeof(concurrent_response), 0);
  assert(oversized_bytes > 0);
  assert(std::string(concurrent_response,
                     concurrent_response + oversized_bytes)
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
                           sizeof(address)) != 0; ++i) {
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
  auto health_bytes = recv(health, concurrent_response,
                           sizeof(concurrent_response), 0);
  assert(health_bytes > 0);
  assert(std::string(concurrent_response,
                     concurrent_response + health_bytes)
             .find("200 OK") != std::string::npos);
  close(health);
  int metrics = connect_health();
  const char metrics_probe[] = "GET /metrics HTTP/1.1\r\n\r\n";
  assert(send(metrics, metrics_probe, sizeof(metrics_probe) - 1, 0) ==
         static_cast<ssize_t>(sizeof(metrics_probe) - 1));
  auto metrics_bytes = recv(metrics, concurrent_response,
                            sizeof(concurrent_response), 0);
  assert(metrics_bytes > 0);
  assert(std::string(concurrent_response,
                     concurrent_response + metrics_bytes)
             .find("404 Not Found") != std::string::npos);
  close(metrics);
  close(idle_health);
  auto append_varint = [](std::string &message, std::uint64_t value) {
    while (value >= 128) {
      message.push_back(static_cast<char>((value & 127) | 128));
      value >>= 7;
    }
    message.push_back(static_cast<char>(value));
  };
  auto append_string = [&](std::string &message, int field,
                           const std::string &value) {
    append_varint(message, field * 8 + 2);
    append_varint(message, value.size());
    message += value;
  };

  int provider = connect_server();
  std::string submit_message = "\x08\x01";
  append_string(submit_message, 2, "tx-sidecar");
  append_string(submit_message, 3, server_source.string());
  assert(send(provider, submit_message.data(), submit_message.size(), 0) ==
         static_cast<ssize_t>(submit_message.size()));
  char control_response[1024] = {};
  assert(recv(provider, control_response, sizeof(control_response), 0) > 0);
  assert(control_response[0] == 0x08 && control_response[1] == 0x01);

  assert(send(provider, "\x08\x01", 2, 0) == 2);
  char provider_response[128] = {};
  assert(recv(provider, provider_response, sizeof(provider_response), 0) == 2);
  assert(provider_response[0] == 0x08 && provider_response[1] == 0x00);

  std::string open_fields;
  append_string(open_fields, 1, "image.img");
  append_varint(open_fields, 2 * 8);
  append_varint(open_fields, 0);
  std::string open_request = "\x08\x02";
  append_string(open_request, 2, open_fields);
  assert(send(provider, open_request.data(), open_request.size(), 0) ==
         static_cast<ssize_t>(open_request.size()));
  char open_response[128] = {};
  char open_control[CMSG_SPACE(sizeof(int))] = {};
  iovec open_iov{open_response, sizeof(open_response)};
  msghdr open_message{};
  open_message.msg_iov = &open_iov;
  open_message.msg_iovlen = 1;
  open_message.msg_control = open_control;
  open_message.msg_controllen = sizeof(open_control);
  assert(recvmsg(provider, &open_message, 0) == 2);
  assert(open_response[0] == 0x08 && open_response[1] == 0x00);
  auto *open_cmsg = CMSG_FIRSTHDR(&open_message);
  assert(open_cmsg && open_cmsg->cmsg_type == SCM_RIGHTS);
  int staged_image = -1;
  std::memcpy(&staged_image, CMSG_DATA(open_cmsg), sizeof(staged_image));
  char staged_contents[32] = {};
  assert(read(staged_image, staged_contents, sizeof(staged_contents)) == 16);
  assert(std::string(staged_contents, 16) == "sidecar-provider");
  close(staged_image);

  assert(send(provider, "\x08\x05", 2, 0) == 2);
  assert(recv(provider, provider_response, sizeof(provider_response), 0) == 2);
  close(provider);

  int control = connect_server();
  std::string commit = "\x08\x04";
  append_string(commit, 2, "tx-sidecar");
  assert(send(control, commit.data(), commit.size(), 0) ==
         static_cast<ssize_t>(commit.size()));
  assert(recv(control, control_response, sizeof(control_response), 0) > 0);
  assert(control_response[0] == 0x08 && control_response[1] == 0x01);
  close(control);
  kill(server_pid, SIGTERM);
  assert(waitpid(server_pid, nullptr, 0) == server_pid);

  std::filesystem::remove_all(root);
}
