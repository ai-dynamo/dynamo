#pragma once

#include <memory>
#include <unordered_map>
#include <vector>

#include "checkpoint_transaction_descriptor.hpp"
#include "pagebroker_types.hpp"
#include "restore_transaction_descriptor.hpp"
#include "transfer_engine.hpp"

namespace snapshot::pagebroker {
class Broker {
 public:
  explicit Broker(Path staging_root);
  Response HandleRequest(const Request& request);

 private:
  using Engines = std::vector<std::unique_ptr<TransferEngine>>;
  using CheckpointTransactions = std::unordered_map<std::string, CheckpointTransactionDescriptor>;
  using RestoreTransactions = std::unordered_map<std::string, RestoreTransactionDescriptor>;
  enum class TransactionState { LIVE, COMMITTED, ABORTED };
  using TransactionStates = std::unordered_map<std::string, TransactionState>;

  const TransferEngine& Engine(TransferEngineType engine_type) const;
  Response Restore(const Request& request);
  Response PrepareCheckpoint(const Request& request);
  // The Snapshot Agent sends COMMIT after CRIU returns; the provider will send it directly later.
  Response Commit(const Request& request);
  Response CleanupRestore(const Request& request, const RestoreTransactionDescriptor& transaction);
  Response PublishCheckpoint(const Request& request, const CheckpointTransactionDescriptor& transaction);
  Response Abort(const Request& request);
  Path staging_root_;
  Engines io_engines_;
  CheckpointTransactions checkpoint_transactions_;
  RestoreTransactions restore_transactions_;
  TransactionStates transaction_states_;
};
}  // namespace snapshot::pagebroker
