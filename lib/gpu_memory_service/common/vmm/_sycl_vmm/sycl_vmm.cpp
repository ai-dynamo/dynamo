// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

// _sycl_vmm — pybind11 module exposing SYCL experimental VMM, IPC,
// host-register, and queue/memcpy APIs to Python.
//
// Compiled with:  icpx -fsycl -shared -fPIC $(python3 -m pybind11 --includes) \
//                 sycl_vmm.cpp -o _sycl_vmm$(python3-config --extension-suffix)
//
// All SYCL VMM/IPC/host-register APIs live in sycl::ext::oneapi::experimental.
// Standard queue, device, context, and memcpy are core SYCL.

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <atomic>

// torch C++ API for tensor_from_device_ptr (at::from_blob with XPU device)
#include <torch/csrc/autograd/python_variable.h>
#include <torch/torch.h>

#include <sycl/sycl.hpp>

// --- VMM / physical_mem headers ---
// Older oneAPI:    sycl/ext/oneapi/virtual_mem/
// oneAPI nightly:   sycl/ext/oneapi/experimental/virtual_mem/
// The namespace is sycl::ext::oneapi::experimental in BOTH versions.
#if __has_include(<sycl/ext/oneapi/experimental/virtual_mem/virtual_mem.hpp>)
#include <sycl/ext/oneapi/experimental/virtual_mem/physical_mem.hpp>
#include <sycl/ext/oneapi/experimental/virtual_mem/virtual_mem.hpp>
#elif __has_include(<sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>)
#include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#else
#error \
    "SYCL VMM headers not found. Requires a oneAPI DPC++ compiler with sycl::ext::oneapi::experimental virtual_mem support."
#endif

// --- IPC: L0 interop path ---
// Uses Level-Zero APIs directly for IPC export/import of physical memory.
// zePhysicalMemGetProperties (export) and zePhysicalMemCreate (import)
// with external memory FD extension.
#include <level_zero/ze_api.h>

#include <sycl/ext/oneapi/backend/level_zero.hpp>
#define SYCL_HAS_IPC 1


// --- Host register (compile-time guarded) ---
#if __has_include(<sycl/ext/oneapi/experimental/register_host_memory.hpp>)
#include <sycl/ext/oneapi/experimental/register_host_memory.hpp>
#define SYCL_HAS_HOST_REGISTER 1
#else
#define SYCL_HAS_HOST_REGISTER 0
#endif

#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

// POSIX headers for memfd-based IPC handle transport
#include <sys/mman.h>
#include <unistd.h>

namespace py = pybind11;
namespace syclex = sycl::ext::oneapi::experimental;

// ============================================================================
// Internal state
// ============================================================================

// Per-device cached SYCL objects.
struct DeviceState {
  sycl::device device;
  sycl::context context;
  // Cached aspect flags — checked once at init, immutable after.
  bool has_vmm;
  bool has_ipc;
  bool has_host_register;
};

static std::mutex g_mutex;
static bool g_initialized = false;
static std::vector<DeviceState> g_devices;
static int g_active_device = 0;

// Handle tables — map integer IDs to SYCL objects so Python holds plain ints.
static std::mutex g_handle_mutex;
static int64_t g_next_phys_id = 1;
static std::unordered_map<int64_t, syclex::physical_mem> g_phys_handles;

static int64_t g_next_stream_id = 1;
static std::unordered_map<int64_t, sycl::queue> g_stream_handles;

// Workaround: Intel NEO driver does not reflect zePhysicalMemCreate allocations
// in any free-memory query (zesMemoryGetState, sycl::ext::intel::free_memory,
// torch.xpu.mem_get_info). Track committed bytes internally until the driver
// accounting is fixed.
static std::atomic<size_t> g_total_allocated_bytes{0};

// L0 native physical_mem handles — for IPC-capable allocations created via L0
// directly (because SYCL physical_mem constructor cannot pass export flags).
struct L0PhysMem {
  ze_context_handle_t context;
  ze_device_handle_t device;
  ze_physical_mem_handle_t phys;
  size_t size;
  bool is_imported;           // true = imported via IPC; skip zePhysicalMemDestroy on release
  int cached_export_fd = -1;  // NEO driver returns stale fd on repeat export; cache+dup
};
static std::unordered_map<int64_t, L0PhysMem> g_l0_phys_handles;

// ============================================================================
// Helpers
// ============================================================================

static DeviceState&
dev_state(int dev_idx)
{
  if (!g_initialized)
    throw std::runtime_error("_sycl_vmm: not initialized — call ensure_initialized() first");
  if (dev_idx < 0 || dev_idx >= static_cast<int>(g_devices.size()))
    throw std::out_of_range("_sycl_vmm: device index out of range");
  return g_devices[static_cast<size_t>(dev_idx)];
}

static DeviceState&
active_dev()
{
  return dev_state(g_active_device);
}

// ============================================================================
// Module functions
// ============================================================================

static void
finalize()
{
  // Destroy all SYCL objects (queues, contexts, physical_mem) before process
  // exit.  Python atexit runs before C++ global destructors, so calling this
  // from atexit prevents the SYCL runtime from hitting already-finalized L0
  // handles during its own static destruction (UR_RESULT_ERROR_UNINITIALIZED).
  std::lock_guard<std::mutex> lock(g_mutex);
  if (!g_initialized)
    return;
  {
    std::lock_guard<std::mutex> hlock(g_handle_mutex);
    g_stream_handles.clear();
    g_phys_handles.clear();
    g_l0_phys_handles.clear();
  }
  g_devices.clear();
  g_initialized = false;
  g_total_allocated_bytes.store(0);
}

static void
ensure_initialized()
{
  std::lock_guard<std::mutex> lock(g_mutex);
  if (g_initialized)
    return;

  auto all_gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  if (all_gpu_devices.empty())
    throw std::runtime_error("_sycl_vmm: no GPU devices found");

  // Filter to VMM-capable devices only
  // Only the Level-Zero device supports VMM (virtual memory management).
  g_devices.clear();
  g_devices.reserve(all_gpu_devices.size());
  for (auto& dev : all_gpu_devices) {
    if (!dev.has(sycl::aspect::ext_oneapi_virtual_mem))
      continue;
    auto ctx = dev.get_platform().khr_get_default_context();
    bool vmm = dev.has(sycl::aspect::ext_oneapi_virtual_mem);
    bool ipc = false;
    bool hreg = false;
    ipc = true;  // L0 interop IPC always available
#if SYCL_HAS_HOST_REGISTER
    hreg = dev.has(sycl::aspect::ext_oneapi_register_host_memory);
#endif
    g_devices.push_back(DeviceState{dev, ctx, vmm, ipc, hreg});
  }
  if (g_devices.empty())
    throw std::runtime_error(
        "_sycl_vmm: no VMM-capable GPU devices found. "
        "VMM requires the Level-Zero backend for XPU.");
  g_initialized = true;
}

static int
device_count()
{
  if (!g_initialized)
    ensure_initialized();
  return static_cast<int>(g_devices.size());
}

static std::string
device_uuid(int dev_idx)
{
  auto& ds = dev_state(dev_idx);
  if (!ds.device.has(sycl::aspect::ext_intel_device_info_uuid))
    throw std::runtime_error(
        "_sycl_vmm: device does not support UUID query "
        "(aspect::ext_intel_device_info_uuid)");
  auto raw = ds.device.get_info<sycl::ext::intel::info::device::uuid>();
  // Format as "GPU-xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx" matching pynvml style.
  char buf[48];
  std::snprintf(
      buf, sizeof(buf), "GPU-%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x", raw[0], raw[1],
      raw[2], raw[3], raw[4], raw[5], raw[6], raw[7], raw[8], raw[9], raw[10], raw[11], raw[12], raw[13], raw[14],
      raw[15]);
  return std::string(buf);
}

static void
set_device(int dev_idx)
{
  dev_state(dev_idx);  // validate index
  g_active_device = dev_idx;
}

static int
get_mem_granularity(int dev_idx)
{
  auto& ds = dev_state(dev_idx);
  // Level-Zero physical_mem requires 2MB page alignment for allocations > ~1MB.
  // SYCL get_mem_granularity returns 64KB (VA granularity) which is insufficient.
  // Return the effective physical page size that physical_mem_create uses.
  constexpr size_t PHYS_PAGE_SIZE = 2 * 1024 * 1024;  // 2MB
  size_t sycl_gran = syclex::get_mem_granularity(ds.device, ds.context, syclex::granularity_mode::recommended);
  size_t gran = std::max(PHYS_PAGE_SIZE, static_cast<size_t>(sycl_gran));
  return static_cast<int>(gran);
}

// --- VA reservation --------------------------------------------------------

static uintptr_t
reserve_virtual_mem(size_t size)
{
  auto& ds = active_dev();
  uintptr_t ptr = syclex::reserve_virtual_mem(/*Start=*/0, size, ds.context);
  return ptr;
}

static void
free_virtual_mem(uintptr_t ptr, size_t size)
{
  auto& ds = active_dev();
  syclex::free_virtual_mem(ptr, size, ds.context);
}

static void
set_access_mode(uintptr_t ptr, size_t size, int dev_idx, int mode)
{
  auto& ds = dev_state(dev_idx);
  syclex::address_access_mode am;
  switch (mode) {
    case 0:
      am = syclex::address_access_mode::none;
      break;
    case 1:
      am = syclex::address_access_mode::read;
      break;
    case 2:
      am = syclex::address_access_mode::read_write;
      break;
    default:
      throw std::invalid_argument("_sycl_vmm: invalid access mode");
  }
  syclex::set_access_mode(reinterpret_cast<void*>(ptr), size, am, ds.context);
}

static void
unmap(uintptr_t ptr, size_t size)
{
  auto& ds = active_dev();
  syclex::unmap(reinterpret_cast<void*>(ptr), size, ds.context);
}


// --- L0 interop helpers (forward declarations for physical_mem_create) --------
static ze_context_handle_t
_get_l0_context(const sycl::context& ctx)
{
  return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(ctx);
}

static ze_device_handle_t
_get_l0_device(const sycl::device& dev)
{
  return sycl::get_native<sycl::backend::ext_oneapi_level_zero>(dev);
}

// --- Physical memory -------------------------------------------------------

static py::tuple
physical_mem_create(int dev_idx, size_t size, bool want_ipc)
{
  auto& ds = dev_state(dev_idx);

  // Round up size to physical memory page alignment.
  // Level-Zero uses 2MB pages for physical_mem allocations above ~1MB.
  // The SYCL get_mem_granularity API returns 64KB (the VA granularity) for both
  // minimum and recommended modes — it does NOT expose the physical page size.
  // zeVirtualMemQueryPageSize(ctx, dev, size) returns 2MB for larger allocations,
  // but SYCL has no equivalent. Hard-code 2MB as the safe alignment.
  constexpr size_t PHYS_PAGE_SIZE = 2 * 1024 * 1024;  // 2MB
  size_t gran = std::max(
      PHYS_PAGE_SIZE,
      static_cast<size_t>(syclex::get_mem_granularity(ds.device, ds.context, syclex::granularity_mode::recommended)));
  size = ((size + gran - 1) / gran) * gran;

  try {
    if (want_ipc) {
      // L0 interop: create physical_mem via L0 directly with export flag.
      ze_context_handle_t l0_ctx = _get_l0_context(ds.context);
      ze_device_handle_t l0_dev = _get_l0_device(ds.device);

      ze_external_memory_export_desc_t export_desc = {};
      export_desc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_DESC;
      export_desc.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD;

      ze_physical_mem_desc_t phys_desc = {};
      phys_desc.stype = ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC;
      phys_desc.pNext = &export_desc;
      phys_desc.flags = 0;
      phys_desc.size = size;

      ze_physical_mem_handle_t l0_phys = nullptr;
      ze_result_t rc = zePhysicalMemCreate(l0_ctx, l0_dev, &phys_desc, &l0_phys);
      if (rc != ZE_RESULT_SUCCESS) {
        if (rc == ZE_RESULT_ERROR_OUT_OF_DEVICE_MEMORY || rc == ZE_RESULT_ERROR_OUT_OF_HOST_MEMORY)
          return py::make_tuple(false, static_cast<int64_t>(0));
        throw std::runtime_error(
            "_sycl_vmm: zePhysicalMemCreate failed: 0x" + std::to_string(static_cast<unsigned>(rc)));
      }

      std::lock_guard<std::mutex> lock(g_handle_mutex);
      int64_t id = g_next_phys_id++;
      g_l0_phys_handles[id] = {l0_ctx, l0_dev, l0_phys, size, /*is_imported=*/false};
      g_total_allocated_bytes.fetch_add(size, std::memory_order_relaxed);
      return py::make_tuple(true, id);
    }
    // Non-IPC allocation: use SYCL physical_mem
    {
      syclex::physical_mem pmem(ds.device, ds.context, size);
      std::lock_guard<std::mutex> lock(g_handle_mutex);
      int64_t id = g_next_phys_id++;
      g_phys_handles.emplace(id, std::move(pmem));
      g_total_allocated_bytes.fetch_add(size, std::memory_order_relaxed);
      return py::make_tuple(true, id);
    }
  }
  catch (const sycl::exception& e) {
    // OOM: return (false, 0) — let the Python layer handle it.
    if (e.code() == sycl::errc::memory_allocation)
      return py::make_tuple(false, static_cast<int64_t>(0));
    throw;  // re-raise non-OOM errors
  }
}

static void
physical_mem_release(int64_t handle_id)
{
  std::lock_guard<std::mutex> lock(g_handle_mutex);
  auto l0_it = g_l0_phys_handles.find(handle_id);
  if (l0_it != g_l0_phys_handles.end()) {
    if (!l0_it->second.is_imported) {
      g_total_allocated_bytes.fetch_sub(l0_it->second.size, std::memory_order_relaxed);
      if (l0_it->second.cached_export_fd >= 0)
        ::close(l0_it->second.cached_export_fd);
      zePhysicalMemDestroy(l0_it->second.context, l0_it->second.phys);
    }
    // Imported handles: do NOT call zePhysicalMemDestroy (GPU runtime driver DMA-BUF
    // refcount sighting — destroying an imported copy corrupts the shared backing).
    g_l0_phys_handles.erase(l0_it);
    return;
  }
  auto it = g_phys_handles.find(handle_id);
  if (it == g_phys_handles.end())
    throw std::invalid_argument("_sycl_vmm: unknown physical_mem handle");
  g_total_allocated_bytes.fetch_sub(it->second.size(), std::memory_order_relaxed);
  g_phys_handles.erase(it);
}

static void
physical_mem_map(int64_t handle_id, uintptr_t ptr, size_t size, int mode)
{
  syclex::address_access_mode am;
  switch (mode) {
    case 1:
      am = syclex::address_access_mode::read;
      break;
    case 2:
      am = syclex::address_access_mode::read_write;
      break;
    default:
      am = syclex::address_access_mode::read_write;
      break;
  }

  std::lock_guard<std::mutex> lock(g_handle_mutex);
  auto l0_it = g_l0_phys_handles.find(handle_id);
  if (l0_it != g_l0_phys_handles.end()) {
    ze_memory_access_attribute_t l0_access = ZE_MEMORY_ACCESS_ATTRIBUTE_READWRITE;
    if (am == syclex::address_access_mode::read)
      l0_access = ZE_MEMORY_ACCESS_ATTRIBUTE_READONLY;
    else if (am == syclex::address_access_mode::none)
      l0_access = ZE_MEMORY_ACCESS_ATTRIBUTE_NONE;
    ze_result_t rc = zeVirtualMemMap(
        l0_it->second.context, reinterpret_cast<const void*>(ptr), size, l0_it->second.phys, 0, l0_access);
    if (rc != ZE_RESULT_SUCCESS)
      throw std::runtime_error("_sycl_vmm: zeVirtualMemMap failed: 0x" + std::to_string(static_cast<unsigned>(rc)));
    return;
  }
  auto it = g_phys_handles.find(handle_id);
  if (it == g_phys_handles.end())
    throw std::invalid_argument("_sycl_vmm: unknown physical_mem handle");
  it->second.map(ptr, size, am);
}

static size_t
physical_mem_size(int64_t handle_id)
{
  std::lock_guard<std::mutex> lock(g_handle_mutex);
  auto l0_it = g_l0_phys_handles.find(handle_id);
  if (l0_it != g_l0_phys_handles.end())
    return l0_it->second.size;
  auto it = g_phys_handles.find(handle_id);
  if (it == g_phys_handles.end())
    throw std::invalid_argument("_sycl_vmm: unknown physical_mem handle");
  return it->second.size();
}

// --- IPC -----------------------------------------------------------------

// ======== L0 interop path (Level-Zero APIs for IPC export/import) ========
// Uses zePhysicalMemGetProperties (export) and zePhysicalMemCreate (import)
// with external memory FD extension. Validated by ipc_spike_vmm.

// Export: extract POSIX FD from a L0 physical_mem via zePhysicalMemGetProperties.
// The allocation must have been created with want_ipc=true (which uses L0 with export flag).
static int
ipc_export_fd(int64_t handle_id)
{
  std::lock_guard<std::mutex> lock(g_handle_mutex);
  auto l0_it = g_l0_phys_handles.find(handle_id);
  if (l0_it == g_l0_phys_handles.end())
    throw std::invalid_argument("_sycl_vmm: ipc_export_fd requires a handle created with want_ipc=true");

  auto& lm = l0_it->second;

  // GPU runtime sighting: zePhysicalMemGetProperties caches the export fd internally
  // and returns the SAME (stale) number on repeat calls even after the fd was
  // closed. Workaround: call GetProperties only once, cache the fd, and return
  // dup() copies to callers.
  if (lm.cached_export_fd < 0) {
    ze_external_memory_export_fd_t export_fd_prop = {};
    export_fd_prop.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_EXPORT_FD;
    export_fd_prop.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD;
    export_fd_prop.fd = -1;

    ze_physical_mem_properties_t phys_props = {};
    phys_props.stype = ZE_STRUCTURE_TYPE_PHYSICAL_MEM_PROPERTIES;
    phys_props.pNext = &export_fd_prop;

    ze_result_t rc = zePhysicalMemGetProperties(lm.context, lm.phys, &phys_props);
    if (rc != ZE_RESULT_SUCCESS || export_fd_prop.fd < 0)
      throw std::runtime_error(
          "_sycl_vmm: zePhysicalMemGetProperties(export_fd) failed: rc=0x" + std::to_string(static_cast<unsigned>(rc)) +
          ", fd=" + std::to_string(export_fd_prop.fd));

    lm.cached_export_fd = export_fd_prop.fd;
  }

  // Return a dup so the caller (and L0 import) can consume/close it without
  // invalidating our cached copy.
  int dup_fd = ::dup(lm.cached_export_fd);
  if (dup_fd < 0)
    throw std::runtime_error("_sycl_vmm: dup(cached_export_fd) failed");
  return dup_fd;
}

// Import: create a L0 physical_mem from an imported FD.
// Returns a positive handle_id (stored in g_l0_phys_handles).
// Caller must use physical_mem_map() to bind it to a VA, then unmap/release.
static int64_t
ipc_import_fd(int fd, int dev_idx, size_t import_size = 0)
{
  auto& ds = dev_state(dev_idx);
  ze_context_handle_t l0_ctx = _get_l0_context(ds.context);
  ze_device_handle_t l0_dev = _get_l0_device(ds.device);

  // The size must be provided by the caller (GMS metadata has it).
  size_t size = import_size;
  if (size == 0)
    throw std::runtime_error("_sycl_vmm: ipc_import_fd: import_size is required for L0 interop path.");

  // Create L0 physical_mem from imported FD.
  ze_external_memory_import_fd_t import_fd_desc = {};
  import_fd_desc.stype = ZE_STRUCTURE_TYPE_EXTERNAL_MEMORY_IMPORT_FD;
  import_fd_desc.flags = ZE_EXTERNAL_MEMORY_TYPE_FLAG_OPAQUE_FD;
  import_fd_desc.fd = fd;

  ze_physical_mem_desc_t phys_desc = {};
  phys_desc.stype = ZE_STRUCTURE_TYPE_PHYSICAL_MEM_DESC;
  phys_desc.pNext = &import_fd_desc;
  phys_desc.flags = 0;
  phys_desc.size = size;

  ze_physical_mem_handle_t imported_phys = nullptr;
  ze_result_t rc = zePhysicalMemCreate(l0_ctx, l0_dev, &phys_desc, &imported_phys);
  if (rc != ZE_RESULT_SUCCESS)
    throw std::runtime_error(
        "_sycl_vmm: zePhysicalMemCreate(import_fd) failed: 0x" + std::to_string(static_cast<unsigned>(rc)));

  // Store in g_l0_phys_handles (same table as server-side L0 allocations).
  // physical_mem_map/unmap/release work on these handles directly.
  std::lock_guard<std::mutex> lock(g_handle_mutex);
  int64_t id = g_next_phys_id++;
  g_l0_phys_handles[id] = {l0_ctx, l0_dev, imported_phys, size, /*is_imported=*/true};
  return id;
}


// --- Host register ---------------------------------------------------------

#if SYCL_HAS_HOST_REGISTER
static void
host_register(uintptr_t ptr, size_t size)
{
  auto& ds = active_dev();
  if (!ds.has_host_register)
    throw std::runtime_error(
        "_sycl_vmm: device does not support host memory registration "
        "(aspect::ext_oneapi_register_host_memory not advertised by this device).");
  syclex::register_host_memory(reinterpret_cast<void*>(ptr), size, ds.context);
}
static void
host_unregister(uintptr_t ptr)
{
  auto& ds = active_dev();
  syclex::unregister_host_memory(reinterpret_cast<void*>(ptr), ds.context);
}
#else
static void
host_register(uintptr_t, size_t)
{
  throw std::runtime_error("_sycl_vmm: register_host_memory not available (oneAPI too old)");
}
static void
host_unregister(uintptr_t)
{
  throw std::runtime_error("_sycl_vmm: unregister_host_memory not available (oneAPI too old)");
}
#endif

// --- Queues / streams ------------------------------------------------------

static int64_t
stream_create(int dev_idx)
{
  auto& ds = dev_state(dev_idx);
  sycl::queue q{ds.context, ds.device, sycl::property::queue::in_order{}};

  std::lock_guard<std::mutex> lock(g_handle_mutex);
  int64_t id = g_next_stream_id++;
  g_stream_handles.emplace(id, std::move(q));
  return id;
}

static void
stream_destroy(int64_t stream_id)
{
  std::lock_guard<std::mutex> lock(g_handle_mutex);
  auto it = g_stream_handles.find(stream_id);
  if (it == g_stream_handles.end())
    throw std::invalid_argument("_sycl_vmm: unknown stream handle");
  g_stream_handles.erase(it);
}

static void
stream_synchronize(int64_t stream_id)
{
  sycl::queue* q = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    auto it = g_stream_handles.find(stream_id);
    if (it == g_stream_handles.end())
      throw std::invalid_argument("_sycl_vmm: unknown stream handle");
    q = &it->second;
  }
  q->wait_and_throw();
}

// --- Memcpy ----------------------------------------------------------------

static void
memcpy_async(uintptr_t dst, uintptr_t src, size_t size, int64_t stream_id)
{
  sycl::queue* q = nullptr;
  {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    auto it = g_stream_handles.find(stream_id);
    if (it == g_stream_handles.end())
      throw std::invalid_argument("_sycl_vmm: unknown stream handle");
    q = &it->second;
  }
  q->memcpy(reinterpret_cast<void*>(dst), reinterpret_cast<const void*>(src), size);
}

// --- Pointer validation ----------------------------------------------------

static int
get_pointer_type(uintptr_t ptr, int dev_idx)
{
  auto& ds = dev_state(dev_idx);

  // First try SYCL USM pointer type query.
  sycl::usm::alloc kind = sycl::get_pointer_type(reinterpret_cast<const void*>(ptr), ds.context);

  if (kind != sycl::usm::alloc::unknown) {
    return static_cast<int>(kind);
  }

  // For VMM-mapped pointers, USM returns 'unknown'. Fall back to
  // get_access_mode which IS VMM-aware.
  try {
    auto mode = syclex::get_access_mode(reinterpret_cast<const void*>(ptr), 1, ds.context);
    // If we get here without exception, the pointer is in a mapped VMM range.
    // Return a synthetic value (4) to distinguish from USM types (0-3).
    return 4;  // VMM-mapped pointer
  }
  catch (...) {
    // Not a VMM pointer either — genuinely unknown.
    return static_cast<int>(sycl::usm::alloc::unknown);
  }
}


// ============================================================================
// pybind11 module definition
// ============================================================================

// --- Allocation tracking -----------------------------------------------------

// Create a torch.Tensor aliasing existing device memory (no copy, no ownership).
// dtype_code is the integer value of c10::ScalarType (e.g. float16=5).
static py::object
tensor_from_device_ptr(
    uintptr_t data_ptr, std::vector<int64_t> shape, std::vector<int64_t> stride, int64_t dtype_code, int device_index)
{
  auto options =
      at::TensorOptions().device(c10::Device(c10::kXPU, device_index)).dtype(static_cast<c10::ScalarType>(dtype_code));

  at::Tensor tensor = at::from_blob(
      reinterpret_cast<void*>(data_ptr), shape, stride,
      /*deleter=*/[](void*) {},  // no-op: GMS owns the memory
      options);

  return py::reinterpret_steal<py::object>(THPVariable_Wrap(std::move(tensor)));
}

static size_t
total_allocated_bytes()
{
  return g_total_allocated_bytes.load(std::memory_order_relaxed);
}

PYBIND11_MODULE(_sycl_vmm, m)
{
  m.doc() = "SYCL VMM/IPC/host-register native module for GMS XPU backend";

  // --- runtime / discovery ---
  m.def("ensure_initialized", &ensure_initialized, "Initialize SYCL runtime and cache GPU devices/contexts");
  m.def("device_count", &device_count, "Return number of GPU devices");
  m.def("device_uuid", &device_uuid, "Return GPU UUID string for a device (GPU-xxxx-... format)", py::arg("dev_idx"));
  m.def("set_device", &set_device, "Set the active device for subsequent operations", py::arg("dev_idx"));

  // --- VMM core ---
  m.def(
      "get_mem_granularity", &get_mem_granularity, "Return minimum allocation granularity in bytes",
      py::arg("dev_idx"));
  m.def(
      "reserve_virtual_mem", &reserve_virtual_mem, "Reserve a contiguous VA range; returns base address",
      py::arg("size"));
  m.def("free_virtual_mem", &free_virtual_mem, "Release a VA reservation", py::arg("ptr"), py::arg("size"));
  m.def(
      "set_access_mode", &set_access_mode, "Set device-side access mode for a mapped VA range", py::arg("ptr"),
      py::arg("size"), py::arg("dev_idx"), py::arg("mode"));
  m.def("unmap", &unmap, "Unbind a VA range from its physical memory", py::arg("ptr"), py::arg("size"));

  // --- physical_mem ---
  m.def(
      "physical_mem_create", &physical_mem_create, "Create physical memory; returns (success, handle_id)",
      py::arg("dev_idx"), py::arg("size"), py::arg("enable_ipc") = true);
  m.def("physical_mem_release", &physical_mem_release, "Release a physical memory handle", py::arg("handle_id"));
  m.def(
      "total_allocated_bytes", &total_allocated_bytes,
      "Return total bytes committed via physical_mem_create (internal tracking; "
      "workaround for GPU runtime not reflecting VMM in free-memory queries)");
  m.def(
      "physical_mem_map", &physical_mem_map, "Map physical memory to a VA range", py::arg("handle_id"), py::arg("ptr"),
      py::arg("size"), py::arg("mode") = 2);
  m.def(
      "physical_mem_size", &physical_mem_size, "Return the size of a physical memory allocation", py::arg("handle_id"));

  // --- IPC ---
  m.def(
      "ipc_export_fd", &ipc_export_fd, "Export a physical_mem handle as a memfd FD for cross-process sharing",
      py::arg("handle_id"));
  m.def(
      "ipc_import_fd", &ipc_import_fd, "Import a physical_mem handle from a memfd FD; closes the FD", py::arg("fd"),
      py::arg("dev_idx"), py::arg("import_size") = 0);

  // --- host register ---
  m.def("host_register", &host_register, "Pin host memory for DMA access", py::arg("ptr"), py::arg("size"));
  m.def("host_unregister", &host_unregister, "Unpin previously registered host memory", py::arg("ptr"));

  // --- queues / streams ---
  m.def("stream_create", &stream_create, "Create a non-blocking in-order queue; returns stream_id", py::arg("dev_idx"));
  m.def("stream_destroy", &stream_destroy, "Destroy a stream", py::arg("stream_id"));
  m.def("stream_synchronize", &stream_synchronize, "Wait for all work on a stream to complete", py::arg("stream_id"));

  // --- memcpy ---
  m.def(
      "memcpy_async", &memcpy_async, "Async memcpy on a stream (works for H2D, D2H, D2D)", py::arg("dst"),
      py::arg("src"), py::arg("size"), py::arg("stream_id"));

  // --- pointer validation ---
  m.def(
      "get_pointer_type", &get_pointer_type, "Query pointer type: 0=host, 1=device, 2=shared, 3=unknown, 4=vmm_mapped",
      py::arg("ptr"), py::arg("dev_idx"));

  // --- runtime aspect queries ---
  m.def(
      "has_ipc_support", [](int dev_idx) -> bool { return dev_state(dev_idx).has_ipc; },
      "Check if device supports IPC physical memory (cached at init)", py::arg("dev_idx"));

  m.def(
      "has_host_register_support", [](int dev_idx) -> bool { return dev_state(dev_idx).has_host_register; },
      "Check if device supports host memory registration (cached at init)", py::arg("dev_idx"));

  m.def(
      "has_vmm_support", [](int dev_idx) -> bool { return dev_state(dev_idx).has_vmm; },
      "Check if device supports virtual memory management (cached at init)", py::arg("dev_idx"));

  // --- compile-time capability flags ---
  // --- tensor creation from raw pointer ---
  m.def(
      "tensor_from_device_ptr", &tensor_from_device_ptr,
      "Create a torch.Tensor aliasing device memory (no copy). "
      "dtype_code is int(torch_dtype) e.g. torch.float16 -> 5",
      py::arg("data_ptr"), py::arg("shape"), py::arg("stride"), py::arg("dtype_code"), py::arg("device_index"));

  m.def(
      "finalize", &finalize,
      "Destroy all SYCL/L0 resources.  Called via atexit to prevent "
      "crash during C++ static destruction at process exit.");

  m.attr("HAS_SYCL_IPC") = py::bool_(SYCL_HAS_IPC != 0);
  m.attr("HAS_SYCL_HOST_REGISTER") = py::bool_(SYCL_HAS_HOST_REGISTER != 0);
#ifdef __INTEL_LLVM_COMPILER
  m.attr("ONEAPI_VERSION") = py::int_(__INTEL_LLVM_COMPILER);
#else
  m.attr("ONEAPI_VERSION") = py::int_(0);
#endif
}
