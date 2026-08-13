#pragma once

#include "transfer_engine.hpp"

namespace snapshot::pagebroker {
class PosixCopyEngine final : public TransferEngine {
 public:
  TransferEngineType type() const override;
  void CopyDirectory(const Path& source, const Path& destination) const override;
};
}  // namespace snapshot::pagebroker
