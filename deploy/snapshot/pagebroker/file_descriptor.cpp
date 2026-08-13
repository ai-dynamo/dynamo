#include "file_descriptor.hpp"

#include <unistd.h>

#include <utility>

FileDescriptor::FileDescriptor(int value) : value_(value) {}

FileDescriptor::~FileDescriptor() noexcept
{
  if (value_ >= 0)
    close(value_);
}

FileDescriptor::FileDescriptor(FileDescriptor&& other) noexcept : value_(std::exchange(other.value_, -1)) {}

FileDescriptor&
FileDescriptor::operator=(FileDescriptor&& other) noexcept
{
  if (this != &other) {
    if (value_ >= 0)
      close(value_);
    value_ = std::exchange(other.value_, -1);
  }
  return *this;
}

int
FileDescriptor::get() const
{
  return value_;
}
