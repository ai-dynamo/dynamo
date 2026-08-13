#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "checkpoint_transaction_descriptor.hpp"
#include "pagebroker_types.hpp"
#include "restore_transaction_descriptor.hpp"
#include "transfer_engine.hpp"
#include "transaction.hpp"

namespace snapshot::pagebroker {
class Broker {
 public:
  explicit Broker(Path staging_root);
  Response HandleRequest(const Request& request);

 private:
  using Engines = std::vector<std::unique_ptr<TransferEngine>>;
  using Transactions = std::unordered_map<std::string, Transaction>;

  const TransferEngine& Engine(TransferEngineType engine_type) const;
  Transaction& GetTransaction(const std::string& transaction_id);
  bool ReserveStaging(uintmax_t bytes);
  void ReleaseStaging(uintmax_t bytes);
  Response Restore(const Request& request);
  Response PrepareCheckpoint(const Request& request);
  // The Snapshot Agent sends COMMIT after CRIU returns; the provider will send it directly later.
  Response Commit(const Request& request);
  Response CleanupRestore(const Request& request, const RestoreTransactionDescriptor& transaction);
  Response PublishCheckpoint(const Request& request, const CheckpointTransactionDescriptor& transaction);
  Response Abort(const Request& request);
  Path staging_root_;
  Engines io_engines_;
  std::mutex transactions_mutex_;
  Transactions transactions_;
  uintmax_t reserved_staging_bytes_ = 0;
};
}  // namespace snapshot::pagebroker
