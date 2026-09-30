/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

#include "roundtrip_common.hpp"
#include "storage.hpp"
#include "workload.hpp"

namespace cuda_custom_storage_benchmark { namespace {

bool
ParseUnsigned(std::string_view value, size_t* result)
{
  if (value.empty()) {
    return false;
  }
  for (const char character : value) {
    if (character < '0' || character > '9') {
      return false;
    }
  }
  const std::string input(value);
  char* end = nullptr;
  errno = 0;
  const unsigned long long parsed = std::strtoull(input.c_str(), &end, 10);
  if (errno != 0 || end == nullptr || *end != '\0' || parsed > std::numeric_limits<size_t>::max()) {
    return false;
  }
  *result = static_cast<size_t>(parsed);
  return true;
}

[[noreturn]] void
ThrowUsage()
{
  throw std::runtime_error(
      "usage: cuda-custom-storage-roundtrip --artifact-dir <absolute-empty-directory> "
      "[--bytes N] [--device N] [--truncate-before-restore | --corrupt-before-restore]");
}

Options
ParseOptions(int argc, char** argv)
{
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    if (argument == "--artifact-dir" && ++index < argc) {
      options.artifact_dir = argv[index];
    } else if (argument == "--bytes" && ++index < argc) {
      size_t bytes = 0;
      if (!ParseUnsigned(argv[index], &bytes) || bytes == 0) {
        ThrowUsage();
      }
      options.bytes = bytes;
    } else if (argument == "--device" && ++index < argc) {
      size_t device = 0;
      if (!ParseUnsigned(argv[index], &device) || device > static_cast<size_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("--device must be a non-negative CUDA device ordinal");
      }
      options.device = static_cast<int>(device);
    } else if (argument == "--truncate-before-restore" && options.corruption == CorruptionMode::kNone) {
      options.corruption = CorruptionMode::kTruncate;
    } else if (argument == "--corrupt-before-restore" && options.corruption == CorruptionMode::kNone) {
      options.corruption = CorruptionMode::kSameSize;
    } else {
      ThrowUsage();
    }
  }
  if (options.artifact_dir.empty() || !options.artifact_dir.is_absolute()) {
    throw std::runtime_error("--artifact-dir must name an absolute directory");
  }

  const mode_t previous_umask = umask(0077);
  std::error_code error;
  const bool is_created = fs::create_directory(options.artifact_dir, error);
  (void)umask(previous_umask);
  if (!is_created || error) {
    throw std::runtime_error("artifact directory must not already exist and its parent must be writable");
  }
  return options;
}

int
Run(const Options& options)
{
  int child_to_parent[2] = {-1, -1};
  int parent_to_child[2] = {-1, -1};
  CreatePipe(child_to_parent);
  try {
    CreatePipe(parent_to_child);
  }
  catch (...) {
    (void)close(child_to_parent[0]);
    (void)close(child_to_parent[1]);
    throw;
  }

  Watchdog watchdog;
  const pid_t child_pid = fork();
  if (child_pid < 0) {
    const int saved_errno = errno;
    (void)close(child_to_parent[0]);
    (void)close(child_to_parent[1]);
    (void)close(parent_to_child[0]);
    (void)close(parent_to_child[1]);
    errno = saved_errno;
    ThrowErrno("fork");
  }
  if (child_pid == 0) {
    (void)close(child_to_parent[0]);
    (void)close(parent_to_child[1]);
    const int status = RunWorkload(child_to_parent[1], parent_to_child[0], options);
    _exit(status);
  }
  ChildProcess child(child_pid);
  watchdog.Arm(child.pid());
  const Clock::time_point deadline = Clock::now() + std::chrono::seconds(kWatchdogSeconds);

  ScopedFd ready_read(child_to_parent[0]);
  ScopedFd command_write(parent_to_child[1]);
  (void)close(child_to_parent[1]);
  (void)close(parent_to_child[0]);

  ReadyMessage ready{};
  ReadAll(ready_read.get(), &ready, sizeof(ready), deadline);
  CheckCUDA(cuInit(0), "controller cuInit");
  RetainedPrimaryContext retained_context(options.device);
  const std::string device_uuid = DeviceUUID(retained_context.device());
  ExpectProcessState(child.pid(), CU_PROCESS_STATE_RUNNING, "before_lock");

  const Artifact artifact = CheckpointProcess(
      child.pid(), options.artifact_dir, ready.device_ptr, ready.bytes, options.device, device_uuid,
      retained_context.context());
  if (artifact.application_ptr != ready.device_ptr || artifact.application_bytes != ready.bytes ||
      artifact.extent.device != options.device || artifact.extent.device_uuid != device_uuid) {
    throw std::runtime_error("checkpoint application or device identity changed");
  }

  std::cout << "extent device=" << artifact.extent.device << " device_uuid=" << artifact.extent.device_uuid
            << " checkpoint_ptr=0x" << std::hex << artifact.extent.checkpoint_ptr << " stream=0x"
            << artifact.extent.stream << std::dec << " bytes=" << artifact.extent.size << " file=" << kExtentFilename
            << '\n';

  const fs::path checkpoint_file = options.artifact_dir / kExtentFilename;
  ValidateExtentFile(checkpoint_file, artifact.extent.size);
  if (options.corruption == CorruptionMode::kTruncate) {
    const size_t truncated_size = artifact.extent.size - 1;
    fs::resize_file(checkpoint_file, truncated_size);
    if (ExtentFileSize(checkpoint_file) != truncated_size) {
      throw std::runtime_error("truncation did not produce the expected artifact size");
    }
    bool is_rejected = false;
    try {
      ValidateExtentFile(checkpoint_file, artifact.extent.size);
    }
    catch (const ExtentSizeMismatch&) {
      is_rejected = true;
    }
    if (!is_rejected) {
      throw std::runtime_error("truncated artifact was incorrectly accepted");
    }
    child.KillAndWait();
    std::cout << "corruption_check=passed mode=truncated detection=file_size_validation\n";
    return 0;
  }
  if (options.corruption == CorruptionMode::kSameSize) {
    CorruptExtentSameSize(checkpoint_file, artifact.extent.size);
    ValidateExtentFile(checkpoint_file, artifact.extent.size);
  }

  try {
    RestoreProcess(child.pid(), options.artifact_dir, artifact, retained_context.context());
  }
  catch (const RestoreRejected& error) {
    if (options.corruption == CorruptionMode::kSameSize) {
      child.KillAndWait();
      std::cout << "corruption_check=passed mode=same_size detection=cuda_restore_error error=" << error.what() << '\n';
      return 0;
    }
    throw;
  }

  const WorkloadCommand command = WorkloadCommand::kVerify;
  WriteAll(command_write.get(), &command, sizeof(command));
  WorkloadResult result = WorkloadResult::kUnexpectedFailure;
  ReadAll(ready_read.get(), &result, sizeof(result), deadline);
  const int child_status = child.Wait(deadline);
  const bool has_child_succeeded = WIFEXITED(child_status) && WEXITSTATUS(child_status) == 0;
  if (options.corruption == CorruptionMode::kSameSize) {
    if (result != WorkloadResult::kRestoredBytesMismatch || has_child_succeeded) {
      throw std::runtime_error("corrupted restore did not produce an explicit restored-byte mismatch");
    }
    std::cout << "corruption_check=passed mode=same_size detection=workload_verification\n";
    return 0;
  }
  if (result != WorkloadResult::kPassed || !has_child_succeeded) {
    throw std::runtime_error("workload did not verify restored bytes and post-restore CUDA execution");
  }

  std::cout << "roundtrip=passed application_ptr=0x" << std::hex << ready.device_ptr << std::dec
            << " application_bytes=" << ready.bytes << " storage_bytes=" << artifact.extent.size << '\n';
  return 0;
}

}}  // namespace cuda_custom_storage_benchmark

int
main(int argc, char** argv)
{
  try {
    cuda_custom_storage_benchmark::IgnoreBrokenPipeSignal();
    const cuda_custom_storage_benchmark::Options options = cuda_custom_storage_benchmark::ParseOptions(argc, argv);
    return cuda_custom_storage_benchmark::Run(options);
  }
  catch (const std::exception& exception) {
    std::fprintf(stderr, "cuda-custom-storage-roundtrip failed: %s\n", exception.what());
    return 1;
  }
}
