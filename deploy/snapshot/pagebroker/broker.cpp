#include "broker.hpp"

#include <sys/statvfs.h>

#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>

#include "posix_copy_engine.hpp"

namespace snapshot::pagebroker {
namespace fs = std::filesystem;
namespace {

Response
Reply(const Request& request)
{
  Response response;
  response.set_request_id(request.request_id());
  response.set_transaction_id(request.transaction_id());
  return response;
}

Response
Fail(const Request& request, Failure::Code code, const std::string& message)
{
  auto response = Reply(request);
  response.mutable_failure()->set_code(code);
  response.mutable_failure()->set_message(message);
  return response;
}

Response
CommitSucceeded(const Request& request)
{
  auto response = Reply(request);
  response.mutable_commit_complete();
  return response;
}

Response
AbortSucceeded(const Request& request)
{
  auto response = Reply(request);
  response.mutable_abort_complete();
  return response;
}

bool
IsSafePathComponent(const std::string& value)
{
  return !value.empty() && value != "." && value != ".." && value.find('/') == std::string::npos &&
         value.find('\\') == std::string::npos && value.find('\0') == std::string::npos;
}

bool
IsFilesystemPosix(const StorageBackend& storage, const IOEngine& engine)
{
  return storage.has_filesystem() && !storage.filesystem().directory().empty() && engine.has_posix_copy();
}

uintmax_t
TreeSize(const Path& path)
{
  uintmax_t bytes = 0;
  for (const auto& entry : fs::recursive_directory_iterator(path)) {
    if (entry.is_symlink())
      throw std::runtime_error("checkpoint contains symlink");
    if (entry.is_regular_file())
      bytes += entry.file_size();
  }
  return bytes;
}

bool
HasAvailableSpace(const Path& filesystem, uintmax_t required_bytes)
{
  struct statvfs stat {};
  return statvfs(filesystem.c_str(), &stat) == 0 && uintmax_t(stat.f_bavail) * stat.f_frsize >= required_bytes;
}

Path
TransactionDirectory(const Path& transaction_root, const std::string& transaction_id)
{
  if (!IsSafePathComponent(transaction_id))
    throw std::runtime_error("invalid transaction path component");
  return transaction_root / transaction_id;
}

}  // namespace

Broker::Broker(Path staging_root) : staging_root_(fs::weakly_canonical(std::move(staging_root)))
{
  io_engines_.push_back(std::make_unique<PosixCopyEngine>());
  fs::create_directories(staging_root_ / "restore");
  fs::create_directories(staging_root_ / "checkpoint");
}

const TransferEngine&
Broker::Engine(TransferEngineType engine_type) const
{
  for (const auto& candidate : io_engines_) {
    if (candidate->type() == engine_type)
      return *candidate;
  }
  throw std::runtime_error("configured I/O engine not found");
}

Response
Broker::HandleRequest(const Request& request)
{
  if (!request.has_request_id() || request.request_id().empty() || !request.has_transaction_id() ||
      !IsSafePathComponent(request.transaction_id()))
    return Fail(request, Failure::INVALID_REQUEST, "request and transaction IDs are required");

  try {
    switch (request.command_case()) {
      case Request::kStagedRestore:
        return Restore(request);
      case Request::kPrepareStagedCheckpoint:
        return PrepareCheckpoint(request);
      case Request::kCommit:
        return Commit(request);
      case Request::kAbort:
        return Abort(request);
      default:
        return Fail(request, Failure::INVALID_REQUEST, "unsupported operation");
    }
  }
  catch (const std::exception& error) {
    return Fail(request, Failure::STORAGE_ERROR, error.what());
  }
}

Response
Broker::Restore(const Request& request)
{
  const auto& operation = request.staged_restore();
  if (!IsFilesystemPosix(operation.source(), operation.io_engine()))
    return Fail(request, Failure::INVALID_REQUEST, "filesystem storage and POSIX copy are required");
  const auto& engine = Engine(TransferEngineType::POSIX_COPY);

  const Path source(operation.source().filesystem().directory());
  const Path restore_root = staging_root_ / "restore";
  const Path staging_directory = TransactionDirectory(restore_root, request.transaction_id());
  if (!source.is_absolute() || fs::is_symlink(source) || !fs::is_directory(source))
    return Fail(request, Failure::INVALID_REQUEST, "source must be an absolute storage directory");
  if (fs::exists(staging_directory) || transaction_states_.contains(request.transaction_id()))
    return Fail(request, Failure::TRANSACTION_CONFLICT, "restore transaction conflicts");
  if (!HasAvailableSpace(staging_root_, TreeSize(source)))
    return Fail(request, Failure::INSUFFICIENT_STORAGE, "insufficient tmpfs capacity");

  try {
    transaction_states_.emplace(request.transaction_id(), TransactionState::LIVE);
    engine.CopyDirectory(source, staging_directory);
    restore_transactions_.emplace(request.transaction_id(), RestoreTransactionDescriptor(staging_directory));
  }
  catch (...) {
    fs::remove_all(staging_directory);
    transaction_states_.erase(request.transaction_id());
    restore_transactions_.erase(request.transaction_id());
    throw;
  }
  auto response = Reply(request);
  response.mutable_staged_restore_directory()->set_image_directory(staging_directory.string());
  return response;
}

Response
Broker::PrepareCheckpoint(const Request& request)
{
  const auto& operation = request.prepare_staged_checkpoint();
  if (!IsFilesystemPosix(operation.destination(), operation.io_engine()))
    return Fail(request, Failure::INVALID_REQUEST, "filesystem storage and POSIX copy are required");
  const auto& engine = Engine(TransferEngineType::POSIX_COPY);

  const Path destination(operation.destination().filesystem().directory());
  const Path checkpoint_root = staging_root_ / "checkpoint";
  const Path staging_directory = TransactionDirectory(checkpoint_root, request.transaction_id());
  if (!destination.is_absolute())
    return Fail(request, Failure::INVALID_REQUEST, "destination must be an absolute storage directory");
  if (fs::exists(staging_directory) || transaction_states_.contains(request.transaction_id()))
    return Fail(request, Failure::TRANSACTION_CONFLICT, "checkpoint transaction conflicts");
  try {
    transaction_states_.emplace(request.transaction_id(), TransactionState::LIVE);
    fs::create_directory(staging_directory);
    checkpoint_transactions_.emplace(
        request.transaction_id(),
        CheckpointTransactionDescriptor(staging_directory, operation.destination(), engine.type()));
  }
  catch (...) {
    fs::remove_all(staging_directory);
    transaction_states_.erase(request.transaction_id());
    checkpoint_transactions_.erase(request.transaction_id());
    throw;
  }
  auto response = Reply(request);
  response.mutable_staged_checkpoint_directory()->set_image_directory(staging_directory.string());
  return response;
}

Response
Broker::Commit(const Request& request)
{
  const auto state = transaction_states_.find(request.transaction_id());
  if (state == transaction_states_.end() || state->second == TransactionState::ABORTED)
    return Fail(request, Failure::TRANSACTION_NOT_FOUND, "transaction not found");
  if (state->second == TransactionState::COMMITTED)
    return CommitSucceeded(request);

  const auto restore = restore_transactions_.find(request.transaction_id());
  if (restore != restore_transactions_.end())
    return CleanupRestore(request, restore->second);

  const auto checkpoint = checkpoint_transactions_.find(request.transaction_id());
  if (checkpoint == checkpoint_transactions_.end())
    return Fail(request, Failure::INTERNAL_ERROR, "live transaction has no descriptor");

  return PublishCheckpoint(request, checkpoint->second);
}

Response
Broker::CleanupRestore(const Request& request, const RestoreTransactionDescriptor& transaction)
{
  fs::remove_all(transaction.staging_directory());
  restore_transactions_.erase(request.transaction_id());
  transaction_states_.at(request.transaction_id()) = TransactionState::COMMITTED;
  return CommitSucceeded(request);
}

Response
Broker::PublishCheckpoint(const Request& request, const CheckpointTransactionDescriptor& transaction)
{
  const Path staging_directory = transaction.staging_directory();
  if (!fs::is_directory(staging_directory))
    return Fail(request, Failure::TRANSACTION_NOT_FOUND, "checkpoint staging directory not found");
  const Path published_directory(transaction.destination_storage().filesystem().directory());
  Path partial = published_directory;
  partial += ".pagebroker-partial";
  if (fs::exists(partial))
    return Fail(request, Failure::TRANSACTION_CONFLICT, "checkpoint destination conflicts");
  TreeSize(staging_directory);

  try {
    fs::create_directories(published_directory.parent_path());
    Engine(transaction.engine_type()).CopyDirectory(staging_directory, partial);
    fs::remove_all(published_directory);
    fs::rename(partial, published_directory);
    checkpoint_transactions_.erase(request.transaction_id());
    transaction_states_.at(request.transaction_id()) = TransactionState::COMMITTED;
    std::error_code cleanup_error;
    fs::remove_all(staging_directory, cleanup_error);
  }
  catch (...) {
    fs::remove_all(partial);
    throw;
  }
  return CommitSucceeded(request);
}

Response
Broker::Abort(const Request& request)
{
  const auto state = transaction_states_.find(request.transaction_id());
  if (state == transaction_states_.end() || state->second == TransactionState::COMMITTED)
    return Fail(request, Failure::TRANSACTION_NOT_FOUND, "transaction not found");
  if (state->second == TransactionState::ABORTED)
    return AbortSucceeded(request);

  const Path restore_root = staging_root_ / "restore";
  const Path checkpoint_root = staging_root_ / "checkpoint";
  fs::remove_all(TransactionDirectory(restore_root, request.transaction_id()));
  fs::remove_all(TransactionDirectory(checkpoint_root, request.transaction_id()));
  checkpoint_transactions_.erase(request.transaction_id());
  restore_transactions_.erase(request.transaction_id());
  state->second = TransactionState::ABORTED;
  return AbortSucceeded(request);
}

}  // namespace snapshot::pagebroker
