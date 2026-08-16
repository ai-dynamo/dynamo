#pragma once

#include <chrono>
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
  bool expired(std::chrono::steady_clock::time_point now, std::chrono::steady_clock::duration lifetime) const;

 private:
  std::mutex mutex_;
  State state_ = State::NEW;
  Descriptor descriptor_;
  std::chrono::steady_clock::time_point staging_started_at_;
  bool terminal_retained_ = false;
};
}  // namespace snapshot::pagebroker
