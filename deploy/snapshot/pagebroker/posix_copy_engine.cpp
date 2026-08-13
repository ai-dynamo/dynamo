#include "posix_copy_engine.hpp"

#include <filesystem>
#include <stdexcept>

namespace snapshot::pagebroker {
namespace {
Path
SourcePath(const StorageBackend& source)
{
  if (!source.has_filesystem() || source.filesystem().directory().empty())
    throw std::invalid_argument("filesystem source is required");
  const Path path(source.filesystem().directory());
  if (!path.is_absolute() || std::filesystem::is_symlink(path) || !std::filesystem::is_directory(path))
    throw std::invalid_argument("source must be an absolute storage directory");
  return path;
}

Path
DestinationPath(const StorageBackend& destination)
{
  if (!destination.has_filesystem() || destination.filesystem().directory().empty())
    throw std::invalid_argument("filesystem destination is required");
  const Path path(destination.filesystem().directory());
  if (!path.is_absolute())
    throw std::invalid_argument("destination must be an absolute storage directory");
  return path;
}

Path
PartialPath(const Path& destination)
{
  Path partial = destination;
  partial += ".pagebroker-partial";
  return partial;
}

uintmax_t
DirectorySize(const Path& path)
{
  uintmax_t bytes = 0;
  for (const auto& entry : std::filesystem::recursive_directory_iterator(path)) {
    if (entry.is_symlink())
      throw std::runtime_error("checkpoint contains symlink");
    if (entry.is_regular_file())
      bytes += entry.file_size();
  }
  return bytes;
}
}  // namespace

TransferEngineType
PosixCopyEngine::type() const
{
  return TransferEngineType::POSIX_COPY;
}

uintmax_t
PosixCopyEngine::RestoreSize(const StorageBackend& source) const
{
  return DirectorySize(SourcePath(source));
}

void
PosixCopyEngine::StageRestore(const StorageBackend& source, const Path& destination) const
{
  CopyDirectory(SourcePath(source), destination);
}

bool
PosixCopyEngine::CheckpointDestinationConflicts(const StorageBackend& destination) const
{
  return std::filesystem::exists(PartialPath(DestinationPath(destination)));
}

void
PosixCopyEngine::PublishCheckpoint(const Path& source, const StorageBackend& destination) const
{
  const Path published = DestinationPath(destination);
  const Path partial = PartialPath(published);
  try {
    std::filesystem::create_directories(published.parent_path());
    CopyDirectory(source, partial);
    std::filesystem::remove_all(published);
    std::filesystem::rename(partial, published);
  }
  catch (...) {
    std::error_code cleanup_error;
    std::filesystem::remove_all(partial, cleanup_error);
    throw;
  }
}

void
PosixCopyEngine::CopyDirectory(const Path& source, const Path& destination) const
{
  std::filesystem::copy(source, destination, std::filesystem::copy_options::recursive);
}
}  // namespace snapshot::pagebroker
