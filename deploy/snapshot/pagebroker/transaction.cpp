#include "transaction.hpp"

#include <utility>

namespace snapshot::pagebroker {
std::mutex&
Transaction::mutex()
{
  return mutex_;
}

Transaction::State
Transaction::state() const
{
  return state_;
}

void
Transaction::set_state(State state)
{
  state_ = state;
  if (state == State::PREPARING)
    staging_started_at_ = std::chrono::steady_clock::now();
}

const Transaction::Descriptor&
Transaction::descriptor() const
{
  return descriptor_;
}

void
Transaction::set_descriptor(Descriptor descriptor)
{
  descriptor_ = std::move(descriptor);
}

void
Transaction::clear_descriptor()
{
  descriptor_ = std::monostate();
}

bool
Transaction::retain_terminal()
{
  if (terminal_retained_ || (state_ != State::COMMITTED && state_ != State::ABORTED))
    return false;
  terminal_retained_ = true;
  return true;
}

bool
Transaction::expired(std::chrono::steady_clock::time_point now, std::chrono::steady_clock::duration lifetime) const
{
  return state_ == State::STAGED && now - staging_started_at_ >= lifetime;
}

}  // namespace snapshot::pagebroker
