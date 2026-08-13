#pragma once

class FileDescriptor {
 public:
  explicit FileDescriptor(int value);
  ~FileDescriptor() noexcept;

  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;

  int get() const;

 private:
  int value_;
};
