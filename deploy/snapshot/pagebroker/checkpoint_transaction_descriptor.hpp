#pragma once

#include "pagebroker_types.hpp"
#include "transfer_engine.hpp"

namespace snapshot::pagebroker {
class CheckpointTransactionDescriptor {
 public:
  CheckpointTransactionDescriptor(
      Path staging_directory, StorageBackend destination_storage, TransferEngineType engine_type);

  const Path& staging_directory() const;
  const StorageBackend& destination_storage() const;
  TransferEngineType engine_type() const;

 private:
  Path staging_directory_;
  StorageBackend destination_storage_;
  TransferEngineType engine_type_;
};
}  // namespace snapshot::pagebroker
