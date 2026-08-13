#pragma once

#include <condition_variable>
#include <mutex>

class ConnectionTracker {
 public:
  void Start();
  void Finish();
  void Wait();

 private:
  std::mutex mutex_;
  std::condition_variable finished_;
  unsigned int active_ = 0;
};
