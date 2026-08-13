#pragma once

#include <filesystem>

enum class ExitCode { SUCCESS = 0, FAILURE = 1, INVALID_ARGUMENTS = 2 };

ExitCode RunDaemon(const std::filesystem::path& socket_path, const std::filesystem::path& staging_directory);
