#include "posix_copy_engine.hpp"

#include <filesystem>

namespace snapshot::pagebroker {
TransferEngineType
PosixCopyEngine::type() const
{
  return TransferEngineType::POSIX_COPY;
}

void
PosixCopyEngine::CopyDirectory(const Path& source, const Path& destination) const
{
  std::filesystem::copy(source, destination, std::filesystem::copy_options::recursive);
}
}  // namespace snapshot::pagebroker
