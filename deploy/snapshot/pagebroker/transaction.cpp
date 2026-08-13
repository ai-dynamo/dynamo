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

}  // namespace snapshot::pagebroker
