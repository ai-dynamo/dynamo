#pragma once

#include <cstdint>
#include <filesystem>

#include "pagebroker_types.hpp"

namespace snapshot::pagebroker {
using Path = std::filesystem::path;

enum class TransferEngineType { POSIX_COPY };

class TransferEngine {
 public:
  virtual ~TransferEngine();
  virtual TransferEngineType type() const = 0;
  virtual uintmax_t RestoreSize(const StorageBackend& source) const = 0;
  virtual void StageRestore(const StorageBackend& source, const Path& destination) const = 0;
  virtual void ValidateCheckpointDestination(const StorageBackend& destination) const = 0;
  virtual bool CheckpointDestinationConflicts(const StorageBackend& destination) const = 0;
  virtual void PublishCheckpoint(const Path& source, const StorageBackend& destination) const = 0;
  virtual void CopyDirectory(const Path& source, const Path& destination) const = 0;
};
}  // namespace snapshot::pagebroker
