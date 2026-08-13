#pragma once

class FileDescriptor {
 public:
  explicit FileDescriptor(int value);
  ~FileDescriptor() noexcept;

  FileDescriptor(const FileDescriptor&) = delete;
  FileDescriptor& operator=(const FileDescriptor&) = delete;
  FileDescriptor(FileDescriptor&& other) noexcept;
  FileDescriptor& operator=(FileDescriptor&& other) noexcept;

  int get() const;

 private:
  int value_;
};
