#include "connection_tracker.hpp"

void
ConnectionTracker::Start()
{
  std::lock_guard lock(mutex_);
  ++active_;
}

void
ConnectionTracker::Finish()
{
  std::lock_guard lock(mutex_);
  if (--active_ == 0)
    finished_.notify_all();
}

void
ConnectionTracker::Wait()
{
  std::unique_lock lock(mutex_);
  finished_.wait(lock, [this] { return active_ == 0; });
}
