#include "checkpoint_transaction_descriptor.hpp"

#include <utility>

namespace snapshot::pagebroker {
CheckpointTransactionDescriptor::CheckpointTransactionDescriptor(
    Path staging_directory, StorageBackend destination_storage, TransferEngineType engine_type)
    : staging_directory_(std::move(staging_directory)), destination_storage_(std::move(destination_storage)),
      engine_type_(engine_type)
{
}

const Path&
CheckpointTransactionDescriptor::staging_directory() const
{
  return staging_directory_;
}

const StorageBackend&
CheckpointTransactionDescriptor::destination_storage() const
{
  return destination_storage_;
}

TransferEngineType
CheckpointTransactionDescriptor::engine_type() const
{
  return engine_type_;
}
}  // namespace snapshot::pagebroker
