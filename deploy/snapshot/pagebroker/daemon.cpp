#include "daemon.hpp"

#include <arpa/inet.h>
#include <errno.h>
#include <fcntl.h>
#include <poll.h>
#include <signal.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <chrono>
#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <string>
#include <string_view>
#include <syncstream>
#include <system_error>
#include <utility>
#include <vector>

#include "broker.hpp"
#include "file_descriptor.hpp"

namespace fs = std::filesystem;
using snapshot::pagebroker::Broker;
using snapshot::pagebroker::Failure;
using snapshot::pagebroker::Request;
using snapshot::pagebroker::Response;

namespace {
constexpr uint32_t kMaxMessageSize = 64 << 10;  // 64 KB
constexpr timeval kConnectionTimeout{30, 0};
constexpr int kShutdownPollTimeoutMs = 1000;
constexpr auto kTransactionReapInterval = std::chrono::minutes(2);
volatile sig_atomic_t shutting_down;

void
Stop(int)
{
  shutting_down = 1;
}

void
LogError(std::string_view operation, const std::error_code& error)
{
  std::cerr << operation << ": " << error.message() << '\n';
}

ExitCode
Fail(std::string_view operation, const std::error_code& error)
{
  LogError(operation, error);
  return ExitCode::FAILURE;
}

std::error_code
InstallSignalHandler(int signal)
{
  struct sigaction action {};
  action.sa_handler = Stop;
  sigemptyset(&action.sa_mask);
  if (sigaction(signal, &action, nullptr) < 0)
    return {errno, std::generic_category()};
  return {};
}

std::error_code
InstallSignalHandlers()
{
  if (const auto error = InstallSignalHandler(SIGINT); error)
    return error;
  return InstallSignalHandler(SIGTERM);
}

std::error_code
PrepareDirectories(const fs::path& socket_path, const fs::path& staging_directory)
{
  std::error_code error;
  fs::create_directories(socket_path.parent_path(), error);
  if (error)
    return error;
  fs::create_directories(staging_directory, error);
  return error;
}

std::error_code
ConfigureConnection(int connection)
{
  if (setsockopt(connection, SOL_SOCKET, SO_RCVTIMEO, &kConnectionTimeout, sizeof(kConnectionTimeout)) < 0 ||
      setsockopt(connection, SOL_SOCKET, SO_SNDTIMEO, &kConnectionTimeout, sizeof(kConnectionTimeout)) < 0)
    return {errno, std::generic_category()};
  return {};
}

std::error_code
ConfigureListener(int listener)
{
  const int flags = fcntl(listener, F_GETFL);
  if (flags < 0 || fcntl(listener, F_SETFL, flags | O_NONBLOCK) < 0)
    return {errno, std::generic_category()};
  return {};
}

std::pair<FileDescriptor, std::error_code>
CreateListener(const fs::path& socket_path)
{
  std::error_code error;
  fs::remove(socket_path, error);
  if (error)
    return std::make_pair(FileDescriptor(-1), error);

  FileDescriptor listener(socket(AF_UNIX, SOCK_STREAM, 0));
  if (listener.get() < 0)
    return std::make_pair(std::move(listener), std::error_code(errno, std::generic_category()));
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::strcpy(address.sun_path, socket_path.c_str());
  if (bind(listener.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) < 0 ||
      listen(listener.get(), SOMAXCONN) < 0)
    return std::make_pair(std::move(listener), std::error_code(errno, std::generic_category()));
  if (error = ConfigureListener(listener.get()); error)
    return std::make_pair(std::move(listener), error);
  return std::make_pair(std::move(listener), std::error_code{});
}

bool
ReadAll(int fd, void* buffer, size_t size)
{
  auto* bytes = static_cast<char*>(buffer);
  while (size > 0) {
    ssize_t read;
    do {
      read = recv(fd, bytes, size, 0);
    } while (read < 0 && errno == EINTR);
    if (read <= 0)
      return false;
    bytes += read;
    size -= read;
  }
  return true;
}

bool
WriteAll(int fd, const void* buffer, size_t size)
{
  const auto* bytes = static_cast<const char*>(buffer);
  while (size > 0) {
    ssize_t written;
    do {
      written = send(fd, bytes, size, MSG_NOSIGNAL);
    } while (written < 0 && errno == EINTR);
    if (written <= 0)
      return false;
    bytes += written;
    size -= written;
  }
  return true;
}

Response
InvalidRequest()
{
  Response response;
  response.set_request_id("");
  response.set_transaction_id("");
  response.mutable_failure()->set_code(Failure::INVALID_REQUEST);
  response.mutable_failure()->set_message("invalid request");
  return response;
}

const char*
CommandName(Request::CommandCase command)
{
  switch (command) {
    case Request::kStagedRestore:
      return "staged_restore";
    case Request::kPrepareStagedCheckpoint:
      return "prepare_staged_checkpoint";
    case Request::kCommit:
      return "commit";
    case Request::kAbort:
      return "abort";
    default:
      return "invalid";
  }
}

const char*
ResultName(const Response& response)
{
  switch (response.result_case()) {
    case Response::kStagedRestoreDirectory:
      return "staged_restore";
    case Response::kStagedCheckpointDirectory:
      return "staged_checkpoint";
    case Response::kCommitComplete:
      return "committed";
    case Response::kAbortComplete:
      return "aborted";
    case Response::kFailure:
      return "failed";
    default:
      return "invalid";
  }
}

void
HandleConnection(int connection, Broker& broker)
{
  uint32_t size = 0;
  if (!ReadAll(connection, &size, sizeof(size)))
    return;
  size = ntohl(size);

  Response response;
  if (size > kMaxMessageSize) {
    response = InvalidRequest();
  } else {
    std::string message(size, '\0');
    Request request;
    if (!ReadAll(connection, message.data(), size) || !request.ParseFromString(message) || !request.IsInitialized()) {
      response = InvalidRequest();
    } else {
      const auto request_start = std::chrono::steady_clock::now();
      response = broker.HandleRequest(request);
      const auto duration =
          std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now() - request_start);
      std::osyncstream(std::cerr) << "transaction=" << request.transaction_id()
                                  << " command=" << CommandName(request.command_case())
                                  << " result=" << ResultName(response) << " duration_ms=" << duration.count()
                                  << (response.has_failure() ? " error=" + response.failure().message() : "") << '\n';
    }
  }

  std::string message = response.SerializeAsString();
  size = htonl(message.size());
  WriteAll(connection, &size, sizeof(size));
  WriteAll(connection, message.data(), message.size());
}

void
ServeConnection(int connection, Broker& broker)
{
  FileDescriptor descriptor(connection);
  if (const auto error = ConfigureConnection(descriptor.get()); error) {
    LogError("set connection timeout", error);
    return;
  }
  HandleConnection(descriptor.get(), broker);
}

void
WaitForHandler(std::future<void>& handler)
{
  try {
    handler.get();
  }
  catch (const std::exception& error) {
    std::cerr << "connection handler: " << error.what() << '\n';
  }
  catch (...) {
    std::cerr << "connection handler: unknown exception\n";
  }
}

void
ReapHandlers(std::vector<std::future<void>>& handlers)
{
  for (auto handler = handlers.begin(); handler != handlers.end();) {
    if (handler->wait_for(std::chrono::seconds(0)) != std::future_status::ready) {
      ++handler;
      continue;
    }
    WaitForHandler(*handler);
    handler = handlers.erase(handler);
  }
}

void
WaitForHandlers(std::vector<std::future<void>>& handlers)
{
  for (auto& handler : handlers) WaitForHandler(handler);
}

void
Serve(FileDescriptor& listener, Broker& broker, size_t max_concurrency)
{
  std::vector<std::future<void>> handlers;
  auto next_transaction_reap = std::chrono::steady_clock::now();
  while (!shutting_down) {
    ReapHandlers(handlers);
    const auto now = std::chrono::steady_clock::now();
    if (now >= next_transaction_reap) {
      broker.ReapExpiredTransactions(now);
      next_transaction_reap = now + kTransactionReapInterval;
    }
    pollfd poll_descriptor{listener.get(), POLLIN, 0};
    const int ready = poll(&poll_descriptor, 1, kShutdownPollTimeoutMs);
    if (ready == 0)
      continue;
    if (ready < 0) {
      if (errno != EINTR)
        LogError("poll", {errno, std::generic_category()});
      continue;
    }
    const int connection = accept(listener.get(), nullptr, nullptr);
    if (connection < 0) {
      if (errno != EINTR && errno != EAGAIN && errno != EWOULDBLOCK)
        LogError("accept", {errno, std::generic_category()});
      continue;
    }
    if (handlers.size() == max_concurrency) {
      FileDescriptor descriptor(connection);
      std::cerr << "connection limit reached\n";
      continue;
    }
    try {
      handlers.emplace_back(
          std::async(std::launch::async, [connection, &broker] { ServeConnection(connection, broker); }));
    }
    catch (const std::exception& error) {
      FileDescriptor descriptor(connection);
      std::cerr << "start connection: " << error.what() << '\n';
    }
  }
  WaitForHandlers(handlers);
}
}  // namespace

ExitCode
RunDaemon(const fs::path& socket_path, const fs::path& staging_directory, size_t max_concurrency)
{
  shutting_down = 0;
  if (const auto error = InstallSignalHandlers(); error)
    return Fail("install signal handlers", error);
  if (const auto error = PrepareDirectories(socket_path, staging_directory); error)
    return Fail("create daemon directories", error);
  if (socket_path.string().size() >= sizeof(sockaddr_un::sun_path)) {
    std::cerr << "socket path is too long\n";
    return ExitCode::INVALID_ARGUMENTS;
  }
  auto [listener, error] = CreateListener(socket_path);
  if (error)
    return Fail("create listener", error);

  Broker broker(staging_directory);
  Serve(listener, broker, max_concurrency);
  return ExitCode::SUCCESS;
}
