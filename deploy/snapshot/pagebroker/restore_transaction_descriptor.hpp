#pragma once

#include "transfer_engine.hpp"

namespace snapshot::pagebroker {
class RestoreTransactionDescriptor {
 public:
  explicit RestoreTransactionDescriptor(Path staging_directory);

  const Path& staging_directory() const;

 private:
  Path staging_directory_;
};
}  // namespace snapshot::pagebroker
