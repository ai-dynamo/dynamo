#pragma once

#include <mutex>
#include <cstdint>
#include <variant>

#include "checkpoint_transaction_descriptor.hpp"
#include "restore_transaction_descriptor.hpp"

namespace snapshot::pagebroker {
class Transaction {
 public:
  enum class State { NEW, PREPARING, STAGED, COMMITTED, ABORTED };
  using Descriptor = std::variant<std::monostate, RestoreTransactionDescriptor, CheckpointTransactionDescriptor>;

  std::mutex& mutex();
  State state() const;
  void set_state(State state);
  const Descriptor& descriptor() const;
  void set_descriptor(Descriptor descriptor);
  void clear_descriptor();
  uintmax_t reserved_bytes() const;
  void set_reserved_bytes(uintmax_t bytes);

 private:
  std::mutex mutex_;
  State state_ = State::NEW;
  Descriptor descriptor_;
  uintmax_t reserved_bytes_ = 0;
};
}  // namespace snapshot::pagebroker
