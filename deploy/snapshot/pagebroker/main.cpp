#include <charconv>
#include <iostream>
#include <string_view>

#include "daemon.hpp"

namespace {
bool
ParseMaxConcurrency(std::string_view value, size_t& max_concurrency)
{
  const auto [end, error] = std::from_chars(value.data(), value.data() + value.size(), max_concurrency);
  return error == std::errc{} && end == value.data() + value.size() && max_concurrency > 0;
}
}  // namespace

int
main(int argc, char** argv)
{
  size_t max_concurrency;
  if (argc != 5 || std::string_view(argv[3]) != "--max-concurrency" ||
      !ParseMaxConcurrency(argv[4], max_concurrency)) {
    std::cerr << "usage: pagebroker socket_path staging_directory --max-concurrency max_concurrency\n";
    return static_cast<int>(ExitCode::INVALID_ARGUMENTS);
  }
  return static_cast<int>(RunDaemon(argv[1], argv[2], max_concurrency));
}
