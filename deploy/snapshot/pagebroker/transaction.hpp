#pragma once

#include <mutex>
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
  bool retain_terminal();

 private:
  std::mutex mutex_;
  State state_ = State::NEW;
  Descriptor descriptor_;
  bool terminal_retained_ = false;
};
}  // namespace snapshot::pagebroker
