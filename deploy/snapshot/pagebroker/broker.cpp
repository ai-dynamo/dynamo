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

const StorageBackend&
ValidateStagedRestore(const StagedRestoreRequest& request)
{
  if (!request.has_source() || request.source().kind_case() == StorageBackend::KIND_NOT_SET)
    throw std::invalid_argument("restore source is required");
  return request.source();
}

const StorageBackend&
ValidateStagedCheckpoint(const PrepareStagedCheckpointRequest& request)
{
  if (!request.has_destination() || request.destination().kind_case() == StorageBackend::KIND_NOT_SET)
    throw std::invalid_argument("checkpoint destination is required");
  return request.destination();
}

void
RejectSymlinks(const Path& directory)
{
  for (const auto& entry : fs::recursive_directory_iterator(directory)) {
    if (entry.is_symlink())
      throw std::runtime_error("checkpoint contains symlink");
  }
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

Transaction&
Broker::GetTransaction(const std::string& transaction_id)
{
  std::lock_guard lock(transactions_mutex_);
  return transactions_.try_emplace(transaction_id).first->second;
}

bool
Broker::ReserveStaging(uintmax_t bytes)
{
  std::lock_guard lock(transactions_mutex_);
  if (!HasAvailableSpace(staging_root_, bytes + reserved_staging_bytes_))
    return false;
  reserved_staging_bytes_ += bytes;
  return true;
}

void
Broker::ReleaseStaging(uintmax_t bytes)
{
  std::lock_guard lock(transactions_mutex_);
  reserved_staging_bytes_ -= bytes;
}

Response
Broker::AbortStaging(
    const Request& request, Transaction& transaction, const Path& staging_directory, const std::exception& error)
{
  std::error_code cleanup_error;
  fs::remove_all(staging_directory, cleanup_error);
  if (cleanup_error)
    return Fail(request, Failure::STORAGE_ERROR, std::string(error.what()) + "; cleanup: " + cleanup_error.message());
  transaction.clear_descriptor();
  transaction.set_state(Transaction::State::ABORTED);
  return Fail(request, Failure::STORAGE_ERROR, error.what());
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

const TransferEngine&
Broker::Engine(const IOEngine& engine) const
{
  if (engine.has_posix_copy())
    return Engine(TransferEngineType::POSIX_COPY);
  throw std::invalid_argument("unsupported I/O engine");
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
  catch (const std::invalid_argument& error) {
    return Fail(request, Failure::INVALID_REQUEST, error.what());
  }
  catch (const std::exception& error) {
    return Fail(request, Failure::STORAGE_ERROR, error.what());
  }
}

Response
Broker::Restore(const Request& request)
{
  const auto& operation = request.staged_restore();
  const auto& source = ValidateStagedRestore(operation);
  const auto& engine = Engine(operation.io_engine());
  return StageRestore(request, source, engine);
}

Response
Broker::StageRestore(const Request& request, const StorageBackend& source, const TransferEngine& engine)
{
  const Path restore_root = staging_root_ / "restore";
  const Path staging_directory = TransactionDirectory(restore_root, request.transaction_id());
  Transaction& transaction = GetTransaction(request.transaction_id());
  std::lock_guard lock(transaction.mutex());
  if (transaction.state() != Transaction::State::NEW || fs::exists(staging_directory))
    return Fail(request, Failure::TRANSACTION_CONFLICT, "restore transaction conflicts");
  const uintmax_t bytes = engine.RestoreSize(source);
  if (!ReserveStaging(bytes))
    return Fail(request, Failure::INSUFFICIENT_STORAGE, "insufficient tmpfs capacity");
  bool staging_reserved = true;
  try {
    transaction.set_state(Transaction::State::PREPARING);
    engine.StageRestore(source, staging_directory);
    ReleaseStaging(bytes);
    staging_reserved = false;
    transaction.set_descriptor(RestoreTransactionDescriptor(staging_directory));
    transaction.set_state(Transaction::State::STAGED);
  }
  catch (const std::exception& error) {
    if (staging_reserved)
      ReleaseStaging(bytes);
    return AbortStaging(request, transaction, staging_directory, error);
  }
  auto response = Reply(request);
  response.mutable_staged_restore_directory()->set_image_directory(staging_directory.string());
  return response;
}

Response
Broker::PrepareCheckpoint(const Request& request)
{
  const auto& operation = request.prepare_staged_checkpoint();
  const auto& destination = ValidateStagedCheckpoint(operation);
  const auto& engine = Engine(operation.io_engine());
  return StageCheckpoint(request, destination, engine);
}

Response
Broker::StageCheckpoint(const Request& request, const StorageBackend& destination, const TransferEngine& engine)
{
  const Path checkpoint_root = staging_root_ / "checkpoint";
  const Path staging_directory = TransactionDirectory(checkpoint_root, request.transaction_id());
  engine.ValidateCheckpointDestination(destination);
  Transaction& transaction = GetTransaction(request.transaction_id());
  std::lock_guard lock(transaction.mutex());
  if (transaction.state() != Transaction::State::NEW || fs::exists(staging_directory))
    return Fail(request, Failure::TRANSACTION_CONFLICT, "checkpoint transaction conflicts");
  try {
    transaction.set_state(Transaction::State::PREPARING);
    fs::create_directory(staging_directory);
    transaction.set_descriptor(CheckpointTransactionDescriptor(staging_directory, destination, engine.type()));
    transaction.set_state(Transaction::State::STAGED);
  }
  catch (const std::exception& error) {
    return AbortStaging(request, transaction, staging_directory, error);
  }
  auto response = Reply(request);
  response.mutable_staged_checkpoint_directory()->set_image_directory(staging_directory.string());
  return response;
}

Response
Broker::Commit(const Request& request)
{
  Transaction& transaction = GetTransaction(request.transaction_id());
  std::lock_guard lock(transaction.mutex());
  if (transaction.state() == Transaction::State::NEW || transaction.state() == Transaction::State::ABORTED)
    return Fail(request, Failure::TRANSACTION_NOT_FOUND, "transaction not found");
  if (transaction.state() == Transaction::State::PREPARING)
    return Fail(request, Failure::TRANSACTION_CONFLICT, "transaction is preparing");
  if (transaction.state() == Transaction::State::COMMITTED)
    return CommitSucceeded(request);

  if (const auto* restore = std::get_if<RestoreTransactionDescriptor>(&transaction.descriptor()))
    return CleanupRestore(request, *restore);

  const auto* checkpoint = std::get_if<CheckpointTransactionDescriptor>(&transaction.descriptor());
  if (checkpoint == nullptr)
    return Fail(request, Failure::INTERNAL_ERROR, "live transaction has no descriptor");

  return PublishCheckpoint(request, *checkpoint);
}

Response
Broker::CleanupRestore(const Request& request, const RestoreTransactionDescriptor& transaction)
{
  fs::remove_all(transaction.staging_directory());
  Transaction& current = GetTransaction(request.transaction_id());
  current.clear_descriptor();
  current.set_state(Transaction::State::COMMITTED);
  return CommitSucceeded(request);
}

Response
Broker::PublishCheckpoint(const Request& request, const CheckpointTransactionDescriptor& transaction)
{
  const Path staging_directory = transaction.staging_directory();
  if (!fs::is_directory(staging_directory))
    return Fail(request, Failure::TRANSACTION_NOT_FOUND, "checkpoint staging directory not found");
  const auto& engine = Engine(transaction.engine_type());
  if (engine.CheckpointDestinationConflicts(transaction.destination_storage()))
    return Fail(request, Failure::TRANSACTION_CONFLICT, "checkpoint destination conflicts");
  RejectSymlinks(staging_directory);
  try {
    engine.PublishCheckpoint(staging_directory, transaction.destination_storage());
    Transaction& current = GetTransaction(request.transaction_id());
    current.clear_descriptor();
    current.set_state(Transaction::State::COMMITTED);
    std::error_code cleanup_error;
    fs::remove_all(staging_directory, cleanup_error);
  }
  catch (const std::exception& error) {
    return Fail(request, Failure::STORAGE_ERROR, error.what());
  }
  return CommitSucceeded(request);
}

Response
Broker::Abort(const Request& request)
{
  Transaction& transaction = GetTransaction(request.transaction_id());
  std::lock_guard lock(transaction.mutex());
  if (transaction.state() == Transaction::State::NEW || transaction.state() == Transaction::State::COMMITTED)
    return Fail(request, Failure::TRANSACTION_NOT_FOUND, "transaction not found");
  if (transaction.state() == Transaction::State::ABORTED)
    return AbortSucceeded(request);

  const Path restore_root = staging_root_ / "restore";
  const Path checkpoint_root = staging_root_ / "checkpoint";
  fs::remove_all(TransactionDirectory(restore_root, request.transaction_id()));
  fs::remove_all(TransactionDirectory(checkpoint_root, request.transaction_id()));
  transaction.clear_descriptor();
  transaction.set_state(Transaction::State::ABORTED);
  return AbortSucceeded(request);
}

}  // namespace snapshot::pagebroker
