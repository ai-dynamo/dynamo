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

uintmax_t
Transaction::reserved_bytes() const
{
  return reserved_bytes_;
}

void
Transaction::set_reserved_bytes(uintmax_t bytes)
{
  reserved_bytes_ = bytes;
}
}  // namespace snapshot::pagebroker
