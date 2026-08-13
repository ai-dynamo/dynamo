#pragma once

#include "transfer_engine.hpp"

namespace snapshot::pagebroker {
class PosixCopyEngine final : public TransferEngine {
 public:
  TransferEngineType type() const override;
  uintmax_t RestoreSize(const StorageBackend& source) const override;
  void StageRestore(const StorageBackend& source, const Path& destination) const override;
  bool CheckpointDestinationConflicts(const StorageBackend& destination) const override;
  void PublishCheckpoint(const Path& source, const StorageBackend& destination) const override;
  void CopyDirectory(const Path& source, const Path& destination) const override;
};
}  // namespace snapshot::pagebroker
