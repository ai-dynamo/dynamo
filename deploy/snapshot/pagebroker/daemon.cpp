#include <arpa/inet.h>
#include <errno.h>
#include <sys/socket.h>
#include <sys/un.h>

#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>

#include "broker.hpp"
#include "file_descriptor.hpp"

namespace fs = std::filesystem;
using snapshot::pagebroker::Broker;
using snapshot::pagebroker::Failure;
using snapshot::pagebroker::Request;
using snapshot::pagebroker::Response;

namespace {
constexpr uint32_t kMaxMessageSize = 64 << 10;  // 64 KB
enum ArgumentIndex { kSocketPath = 1, kStagingDirectory, kArgumentCount };

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
      response = broker.HandleRequest(request);
    }
  }

  std::string message = response.SerializeAsString();
  size = htonl(message.size());
  WriteAll(connection, &size, sizeof(size));
  WriteAll(connection, message.data(), message.size());
}
}  // namespace

int
main(int argc, char** argv)
{
  if (argc != kArgumentCount) {
    std::cerr << "usage: pagebroker-daemon SOCKET STAGING_DIRECTORY\n";
    return 2;
  }

  const fs::path socket_path(argv[kSocketPath]);
  std::error_code error;
  fs::create_directories(socket_path.parent_path(), error);
  if (error) {
    std::cerr << "create socket directory: " << error.message() << '\n';
    return 1;
  }
  fs::create_directories(argv[kStagingDirectory], error);
  if (error) {
    std::cerr << "create staging directory: " << error.message() << '\n';
    return 1;
  }
  if (socket_path.string().size() >= sizeof(sockaddr_un::sun_path)) {
    std::cerr << "socket path is too long\n";
    return 2;
  }
  unlink(socket_path.c_str());

  FileDescriptor listener(socket(AF_UNIX, SOCK_STREAM, 0));
  if (listener.get() < 0) {
    std::cerr << "create listener: " << std::strerror(errno) << '\n';
    return 1;
  }
  sockaddr_un address{};
  address.sun_family = AF_UNIX;
  std::strcpy(address.sun_path, socket_path.c_str());
  if (bind(listener.get(), reinterpret_cast<const sockaddr*>(&address), sizeof(address)) < 0 ||
      listen(listener.get(), 16) < 0) {
    std::cerr << "listen: " << std::strerror(errno) << '\n';
    return 1;
  }

  Broker broker(argv[kStagingDirectory]);
  for (;;) {
    FileDescriptor connection(accept(listener.get(), nullptr, nullptr));
    if (connection.get() < 0) {
      if (errno != EINTR)
        std::cerr << "accept: " << std::strerror(errno) << '\n';
      continue;
    }
    HandleConnection(connection.get(), broker);
  }
}
