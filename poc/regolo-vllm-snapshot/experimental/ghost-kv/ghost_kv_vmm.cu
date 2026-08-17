#include <cuda.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>

static void check(CUresult result, const char* action) {
  if (result == CUDA_SUCCESS) return;
  const char* name = nullptr;
  const char* text = nullptr;
  cuGetErrorName(result, &name);
  cuGetErrorString(result, &text);
  std::fprintf(stderr, "%s: %s (%s)\n", action, name ? name : "unknown",
               text ? text : "unknown");
  std::exit(1);
}

int main() {
  check(cuInit(0), "cuInit");
  CUdevice device;
  check(cuDeviceGet(&device, 0), "cuDeviceGet");
  CUcontext context = nullptr;
  check(cuCtxCreate(&context, 0, device), "cuCtxCreate");

  CUmemAllocationProp prop{};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;

  size_t granularity = 0;
  check(cuMemGetAllocationGranularity(
            &granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM),
        "cuMemGetAllocationGranularity");
  size_t wanted = 64ULL * 1024 * 1024;
  if (const char* raw = std::getenv("GHOST_KV_BYTES")) {
    char* end = nullptr;
    const unsigned long long parsed = std::strtoull(raw, &end, 10);
    if (!end || *end != '\0' || parsed == 0) {
      std::fprintf(stderr, "invalid GHOST_KV_BYTES\n");
      return 2;
    }
    wanted = static_cast<size_t>(parsed);
  }
  const size_t size = ((wanted + granularity - 1) / granularity) * granularity;

  size_t free_before = 0, total = 0;
  check(cuMemGetInfo(&free_before, &total), "cuMemGetInfo(before)");

  CUdeviceptr address = 0;
  check(cuMemAddressReserve(&address, size, 0, 0, 0), "cuMemAddressReserve");

  CUmemAccessDesc access{};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = device;
  access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  CUmemGenericAllocationHandle first = 0;
  check(cuMemCreate(&first, size, &prop, 0), "cuMemCreate(first)");
  check(cuMemMap(address, size, 0, first, 0), "cuMemMap(first)");
  check(cuMemSetAccess(address, size, &access, 1), "cuMemSetAccess(first)");
  check(cuMemsetD8(address, 0xa5, size), "cuMemsetD8(secret)");
  unsigned char observed = 0;
  check(cuMemcpyDtoH(&observed, address, 1), "cuMemcpyDtoH(secret)");
  if (observed != 0xa5) {
    std::fprintf(stderr, "initial write mismatch: %u\n", observed);
    return 1;
  }
  size_t free_mapped = 0;
  check(cuMemGetInfo(&free_mapped, &total), "cuMemGetInfo(mapped)");

  check(cuMemUnmap(address, size), "cuMemUnmap");
  check(cuMemRelease(first), "cuMemRelease(first)");
  size_t free_after_unmap = 0;
  check(cuMemGetInfo(&free_after_unmap, &total), "cuMemGetInfo(after_unmap)");

  CUmemGenericAllocationHandle second = 0;
  check(cuMemCreate(&second, size, &prop, 0), "cuMemCreate(second)");
  check(cuMemMap(address, size, 0, second, 0), "cuMemMap(second)");
  check(cuMemSetAccess(address, size, &access, 1), "cuMemSetAccess(second)");
  check(cuMemsetD8(address, 0, size), "cuMemsetD8(zero)");
  check(cuMemcpyDtoH(&observed, address, 1), "cuMemcpyDtoH(zero)");
  if (observed != 0) {
    std::fprintf(stderr, "zeroization mismatch: %u\n", observed);
    return 1;
  }

  check(cuMemUnmap(address, size), "cuMemUnmap(final)");
  check(cuMemRelease(second), "cuMemRelease(second)");
  check(cuMemAddressFree(address, size), "cuMemAddressFree");
  size_t free_final = 0;
  check(cuMemGetInfo(&free_final, &total), "cuMemGetInfo(final)");
  std::printf("{\"va_stable\":true,\"zeroed\":true,\"bytes\":%zu,\"granularity\":%zu,\"free_before\":%zu,\"free_mapped\":%zu,\"free_after_unmap\":%zu,\"free_final\":%zu}\n",
              size, granularity, free_before, free_mapped, free_after_unmap,
              free_final);
  check(cuCtxDestroy(context), "cuCtxDestroy");
  return 0;
}
