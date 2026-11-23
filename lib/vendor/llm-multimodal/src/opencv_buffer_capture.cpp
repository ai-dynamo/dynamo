#include <opencv2/core/version.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/videoio/registry.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <limits>
#include <vector>

namespace {

void set_error(char* output, size_t capacity, const char* message) {
  if (output == nullptr || capacity == 0) {
    return;
  }
  std::snprintf(output, capacity, "%s", message);
}

}  // namespace

#if CV_VERSION_MAJOR > 4 || \
    (CV_VERSION_MAJOR == 4 && CV_VERSION_MINOR >= 11)

namespace {

class MemoryStreamReader final : public cv::IStreamReader {
 public:
  MemoryStreamReader(const uint8_t* data, size_t size)
      : data_(data), size_(size) {}

  long long read(char* buffer, long long size) override {
    if (size <= 0 || position_ >= size_) {
      return 0;
    }
    const size_t count =
        std::min(static_cast<size_t>(size), size_ - position_);
    std::memcpy(buffer, data_ + position_, count);
    position_ += count;
    return static_cast<long long>(count);
  }

  long long seek(long long offset, int origin) override {
    if (size_ > static_cast<size_t>(std::numeric_limits<long long>::max())) {
      return -1;
    }

    const long long end = static_cast<long long>(size_);
    long long base = 0;
    if (origin == SEEK_CUR) {
      base = static_cast<long long>(position_);
    } else if (origin == SEEK_END) {
      base = end;
    } else if (origin != SEEK_SET) {
      return -1;
    }

    if (offset > end - base || offset < -base) {
      return -1;
    }
    const long long next = base + offset;
    position_ = static_cast<size_t>(next);
    return next;
  }

 private:
  const uint8_t* data_;
  size_t size_;
  size_t position_ = 0;
};

}  // namespace

extern "C" void* smg_opencv_capture_from_buffer(const uint8_t* data,
                                                 size_t size,
                                                 int decoder_threads,
                                                 char* error,
                                                 size_t error_capacity) {
  try {
    if (data == nullptr || size == 0) {
      set_error(error, error_capacity, "video buffer is empty");
      return nullptr;
    }

    for (const auto backend :
         cv::videoio_registry::getStreamBufferedBackends()) {
      if (!cv::videoio_registry::hasBackend(backend)) {
        continue;
      }
      cv::Ptr<cv::IStreamReader> reader =
          cv::makePtr<MemoryStreamReader>(data, size);
      auto* capture = new cv::VideoCapture(
          reader, static_cast<int>(backend),
          std::vector<int>{cv::CAP_PROP_N_THREADS, decoder_threads});
      if (capture->isOpened()) {
        return capture;
      }
      delete capture;
    }
    set_error(error, error_capacity,
              "OpenCV has no usable buffered video backend");
  } catch (const std::exception& exception) {
    set_error(error, error_capacity, exception.what());
  } catch (...) {
    set_error(error, error_capacity, "unknown OpenCV buffered capture error");
  }
  return nullptr;
}

#else

extern "C" void* smg_opencv_capture_from_buffer(const uint8_t*, size_t, int,
                                                 char* error,
                                                 size_t error_capacity) {
  set_error(error, error_capacity,
            "buffered video capture requires OpenCV 4.11 or newer");
  return nullptr;
}

#endif
