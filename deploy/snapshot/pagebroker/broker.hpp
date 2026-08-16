#pragma once

#include <chrono>
#include <deque>
#include <exception>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "checkpoint_transaction_descriptor.hpp"
#include "pagebroker_types.hpp"
#include "restore_transaction_descriptor.hpp"
#include "transaction.hpp"
#include "transfer_engine.hpp"

namespace snapshot::pagebroker {
class Broker {
 public:
  explicit Broker(Path staging_root);
  Response HandleRequest(const Request& request);
  void ReapExpiredTransactions(std::chrono::steady_clock::time_point now);

 private:
  using Engines = std::vector<std::unique_ptr<TransferEngine>>;
  using TransactionHandle = std::shared_ptr<Transaction>;
  using Transactions = std::unordered_map<std::string, TransactionHandle>;
  struct TerminalTransaction {
    std::string id;
    TransactionHandle transaction;
    std::chrono::steady_clock::time_point completed;
  };

  const TransferEngine& Engine(TransferEngineType engine_type) const;
  const TransferEngine& Engine(const IOEngine& engine) const;
  TransactionHandle CreateOrGetTransaction(const std::string& transaction_id);
  TransactionHandle FindTransaction(const std::string& transaction_id);
  void RetainTerminalTransaction(const std::string& transaction_id);
  void ReapTerminalTransactions();
  bool ReserveStaging(uintmax_t bytes);
  void ReleaseStaging(uintmax_t bytes);
  Response AbortStaging(
      const Request& request, Transaction& transaction, const Path& staging_directory, const std::exception& error);
  Response Restore(const Request& request);
  Response StageRestore(const Request& request, const StorageBackend& source, const TransferEngine& engine);
  Response PrepareCheckpoint(const Request& request);
  Response StageCheckpoint(const Request& request, const StorageBackend& destination, const TransferEngine& engine);
  // The Snapshot Agent sends COMMIT after CRIU returns; the provider will send it directly later.
  Response Commit(const Request& request);
  Response CleanupRestore(
      const Request& request, Transaction& transaction, const RestoreTransactionDescriptor& descriptor);
  Response PublishCheckpoint(
      const Request& request, Transaction& transaction, const CheckpointTransactionDescriptor& descriptor);
  Response Abort(const Request& request);
  Path staging_root_;
  Engines io_engines_;
  std::mutex transactions_mutex_;
  Transactions transactions_;
  std::mutex terminal_transactions_mutex_;
  std::deque<TerminalTransaction> terminal_transactions_;
  uintmax_t reserved_staging_bytes_ = 0;
};
}  // namespace snapshot::pagebroker
