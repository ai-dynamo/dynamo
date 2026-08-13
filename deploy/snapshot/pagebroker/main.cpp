#include <iostream>

#include "daemon.hpp"

int
main(int argc, char** argv)
{
  if (argc != 3) {
    std::cerr << "usage: pagebroker-daemon SOCKET STAGING_DIRECTORY\n";
    return static_cast<int>(ExitCode::INVALID_ARGUMENTS);
  }
  return static_cast<int>(RunDaemon(argv[1], argv[2]));
}
