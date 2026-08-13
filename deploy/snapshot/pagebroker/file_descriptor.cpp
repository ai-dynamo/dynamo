#include "file_descriptor.hpp"

#include <unistd.h>

FileDescriptor::FileDescriptor(int value) : value_(value) {}

FileDescriptor::~FileDescriptor() noexcept
{
  if (value_ >= 0)
    close(value_);
}

int
FileDescriptor::get() const
{
  return value_;
}
