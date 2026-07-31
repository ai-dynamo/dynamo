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

#include <sycl/sycl.hpp>

// --- VMM / physical_mem headers ---
// oneAPI 2025.3.x: sycl/ext/oneapi/virtual_mem/
// oneAPI nightly:   sycl/ext/oneapi/experimental/virtual_mem/
// The namespace is sycl::ext::oneapi::experimental in BOTH versions.
#if __has_include(<sycl/ext/oneapi/experimental/virtual_mem/virtual_mem.hpp>)
#  include <sycl/ext/oneapi/experimental/virtual_mem/virtual_mem.hpp>
#  include <sycl/ext/oneapi/experimental/virtual_mem/physical_mem.hpp>
#elif __has_include(<sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>)
#  include <sycl/ext/oneapi/virtual_mem/virtual_mem.hpp>
#  include <sycl/ext/oneapi/virtual_mem/physical_mem.hpp>
#else
#  error "SYCL VMM headers not found. Requires oneAPI 2025.3+ or DPC++ nightly."
#endif

// --- IPC physical memory (compile-time guarded) ---
#if __has_include(<sycl/ext/oneapi/experimental/ipc_physical_memory.hpp>)
#  include <sycl/ext/oneapi/experimental/ipc_physical_memory.hpp>
#  define SYCL_HAS_IPC_PHYSICAL_MEM 1
#else
#  define SYCL_HAS_IPC_PHYSICAL_MEM 0
#endif

// --- Host register (compile-time guarded) ---
#if __has_include(<sycl/ext/oneapi/experimental/register_host_memory.hpp>)
#  include <sycl/ext/oneapi/experimental/register_host_memory.hpp>
#  define SYCL_HAS_HOST_REGISTER 1
#else
#  define SYCL_HAS_HOST_REGISTER 0
#endif

// --- Intel free_memory extension (compile-time guarded) ---
#if __has_include(<sycl/ext/intel/info/device.hpp>)
#  include <sycl/ext/intel/info/device.hpp>
#  define SYCL_HAS_FREE_MEMORY 1
#else
#  define SYCL_HAS_FREE_MEMORY 0
#endif

#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
namespace syclex = sycl::ext::oneapi::experimental;

// ============================================================================
// Internal state
// ============================================================================

// Per-device cached SYCL objects.
struct DeviceState {
    sycl::device   device;
    sycl::context  context;
    // Default in-order queue for synchronize() and fallback memcpy.
    sycl::queue    default_queue;
    // Cached aspect flags — checked once at init, immutable after.
    bool           has_vmm;
    bool           has_ipc;
    bool           has_host_register;
};

static std::mutex                          g_mutex;
static bool                                g_initialized = false;
static std::vector<DeviceState>            g_devices;
static int                                 g_active_device = 0;

// Handle tables — map integer IDs to SYCL objects so Python holds plain ints.
static std::mutex                          g_handle_mutex;
static int64_t                             g_next_phys_id = 1;
static std::unordered_map<int64_t, syclex::physical_mem> g_phys_handles;

static int64_t                             g_next_stream_id = 1;
static std::unordered_map<int64_t, sycl::queue>          g_stream_handles;

// ============================================================================
// Helpers
// ============================================================================

static DeviceState& dev_state(int dev_idx) {
    if (!g_initialized)
        throw std::runtime_error("_sycl_vmm: not initialized — call ensure_initialized() first");
    if (dev_idx < 0 || dev_idx >= static_cast<int>(g_devices.size()))
        throw std::out_of_range("_sycl_vmm: device index out of range");
    return g_devices[static_cast<size_t>(dev_idx)];
}

static DeviceState& active_dev() {
    return dev_state(g_active_device);
}

// ============================================================================
// Module functions
// ============================================================================

static void ensure_initialized() {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (g_initialized) return;

    auto all_gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
    if (all_gpu_devices.empty())
        throw std::runtime_error("_sycl_vmm: no GPU devices found");

    // Filter to VMM-capable devices only. On systems with both Level-Zero and
    // OpenCL backends, the same GPU appears twice — only the Level-Zero device
    // supports VMM (virtual memory management).
    g_devices.clear();
    g_devices.reserve(all_gpu_devices.size());
    for (auto& dev : all_gpu_devices) {
        if (!dev.has(sycl::aspect::ext_oneapi_virtual_mem))
            continue;  // skip OpenCL / non-VMM devices
        auto ctx = dev.get_platform().khr_get_default_context();
        sycl::queue q{ctx, dev, sycl::property::queue::in_order{}};
        bool vmm  = dev.has(sycl::aspect::ext_oneapi_virtual_mem);
        bool ipc  = false;
        bool hreg = false;
#if SYCL_HAS_IPC_PHYSICAL_MEM
        ipc  = dev.has(sycl::aspect::ext_oneapi_ipc_physical_memory);
#endif
#if SYCL_HAS_HOST_REGISTER
        hreg = dev.has(sycl::aspect::ext_oneapi_register_host_memory);
#endif
        g_devices.push_back(DeviceState{dev, ctx, std::move(q), vmm, ipc, hreg});
    }
    if (g_devices.empty())
        throw std::runtime_error(
            "_sycl_vmm: no VMM-capable GPU devices found. "
            "VMM requires the Level-Zero backend (not OpenCL).");
    g_initialized = true;
}

static int device_count() {
    if (!g_initialized) ensure_initialized();
    return static_cast<int>(g_devices.size());
}

static py::tuple device_memory_info(int dev_idx) {
    auto& ds = dev_state(dev_idx);
    uint64_t total = ds.device.get_info<sycl::info::device::global_mem_size>();
    uint64_t free_bytes = total;  // default: report total as free
#if SYCL_HAS_FREE_MEMORY
    try {
        free_bytes = ds.device.get_info<sycl::ext::intel::info::device::free_memory>();
    } catch (...) {
        // Fallback to total if the extension query fails on this device/driver.
    }
#endif
    return py::make_tuple(free_bytes, total);
}

static void set_device(int dev_idx) {
    dev_state(dev_idx);  // validate index
    g_active_device = dev_idx;
}

static int get_mem_granularity(int dev_idx) {
    auto& ds = dev_state(dev_idx);
    size_t gran = syclex::get_mem_granularity(ds.device, ds.context,
                                              syclex::granularity_mode::minimum);
    return static_cast<int>(gran);
}

// --- VA reservation --------------------------------------------------------

static uintptr_t reserve_virtual_mem(size_t size) {
    auto& ds = active_dev();
    uintptr_t ptr = syclex::reserve_virtual_mem(/*Start=*/0, size, ds.context);
    return ptr;
}

static void free_virtual_mem(uintptr_t ptr, size_t size) {
    auto& ds = active_dev();
    syclex::free_virtual_mem(ptr, size, ds.context);
}

static void set_access_mode(uintptr_t ptr, size_t size, int dev_idx, int mode) {
    auto& ds = dev_state(dev_idx);
    syclex::address_access_mode am;
    switch (mode) {
        case 0: am = syclex::address_access_mode::none;       break;
        case 1: am = syclex::address_access_mode::read;       break;
        case 2: am = syclex::address_access_mode::read_write; break;
        default:
            throw std::invalid_argument("_sycl_vmm: invalid access mode");
    }
    syclex::set_access_mode(reinterpret_cast<void*>(ptr), size, am, ds.context);
}

static void unmap(uintptr_t ptr, size_t size) {
    auto& ds = active_dev();
    syclex::unmap(reinterpret_cast<void*>(ptr), size, ds.context);
}

// --- Physical memory -------------------------------------------------------

static py::tuple physical_mem_create(int dev_idx, size_t size, bool want_ipc) {
    auto& ds = dev_state(dev_idx);
    try {
        // oneAPI 2025.3: physical_mem(device, context, size) — no properties param.
        // oneAPI nightly: physical_mem(device, context, size, properties{enable_ipc}).
        // The enable_ipc property only exists when IPC headers are present.
        syclex::physical_mem pmem = [&]() {
#if SYCL_HAS_IPC_PHYSICAL_MEM
            if (want_ipc) {
                return syclex::physical_mem(
                    ds.device, ds.context, size,
                    syclex::properties{syclex::enable_ipc});
            }
#else
            (void)want_ipc;  // IPC not available on this oneAPI version
#endif
            return syclex::physical_mem(ds.device, ds.context, size);
        }();

        std::lock_guard<std::mutex> lock(g_handle_mutex);
        int64_t id = g_next_phys_id++;
        g_phys_handles.emplace(id, std::move(pmem));
        return py::make_tuple(true, id);
    } catch (const sycl::exception& e) {
        // OOM: return (false, 0) — let the Python layer handle it.
        if (e.code() == sycl::errc::memory_allocation)
            return py::make_tuple(false, static_cast<int64_t>(0));
        throw;  // re-raise non-OOM errors
    }
}

static void physical_mem_release(int64_t handle_id) {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    auto it = g_phys_handles.find(handle_id);
    if (it == g_phys_handles.end())
        throw std::invalid_argument("_sycl_vmm: unknown physical_mem handle");
    g_phys_handles.erase(it);  // destructor releases the physical memory
}

static void physical_mem_map(int64_t handle_id, uintptr_t ptr, size_t size, int mode) {
    syclex::address_access_mode am;
    switch (mode) {
        case 1: am = syclex::address_access_mode::read;       break;
        case 2: am = syclex::address_access_mode::read_write; break;
        default: am = syclex::address_access_mode::read_write; break;
    }

    std::lock_guard<std::mutex> lock(g_handle_mutex);
    auto it = g_phys_handles.find(handle_id);
    if (it == g_phys_handles.end())
        throw std::invalid_argument("_sycl_vmm: unknown physical_mem handle");
    it->second.map(ptr, size, am);
}

static size_t physical_mem_size(int64_t handle_id) {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    auto it = g_phys_handles.find(handle_id);
    if (it == g_phys_handles.end())
        throw std::invalid_argument("_sycl_vmm: unknown physical_mem handle");
    return it->second.size();
}

// --- IPC -------------------------------------------------------------------

#if SYCL_HAS_IPC_PHYSICAL_MEM
namespace sycl_ipc = sycl::ext::oneapi::experimental::ipc;

static void _check_ipc_aspect(int dev_idx) {
    auto& ds = dev_state(dev_idx);
    if (!ds.has_ipc)
        throw std::runtime_error(
            "_sycl_vmm: device does not support IPC physical memory "
            "(aspect::ext_oneapi_ipc_physical_memory). "
            "Ensure oneAPI 2026.1+ and Level-Zero v2 adapter are in use.");
}

static int ipc_export_fd(int64_t handle_id) {
    _check_ipc_aspect(g_active_device);

    syclex::physical_mem* pmem = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_handle_mutex);
        auto it = g_phys_handles.find(handle_id);
        if (it == g_phys_handles.end())
            throw std::invalid_argument("_sycl_vmm: unknown physical_mem handle");
        pmem = &it->second;
    }

    auto ipc_handle = sycl_ipc::physical_memory::get(*pmem);
    auto handle_data = ipc_handle.data();  // std::vector<std::byte>

    // Transport the handle data via memfd so Python receives a plain FD.
    int fd = memfd_create("sycl_ipc_phys", MFD_CLOEXEC);
    if (fd < 0)
        throw std::runtime_error("_sycl_vmm: memfd_create failed");

    ssize_t written = write(fd, handle_data.data(), handle_data.size());
    if (written < 0 || static_cast<size_t>(written) != handle_data.size()) {
        close(fd);
        throw std::runtime_error("_sycl_vmm: failed to write IPC handle to memfd");
    }
    lseek(fd, 0, SEEK_SET);
    return fd;
}

static int64_t ipc_import_fd(int fd, int dev_idx) {
    _check_ipc_aspect(dev_idx);
    auto& ds = dev_state(dev_idx);

    // Read handle data from the memfd.
    off_t sz = lseek(fd, 0, SEEK_END);
    lseek(fd, 0, SEEK_SET);
    std::vector<std::byte> handle_data(static_cast<size_t>(sz));
    ssize_t rd = read(fd, handle_data.data(), handle_data.size());
    close(fd);  // Always close — matches the CUDA contract.

    if (rd < 0 || static_cast<size_t>(rd) != handle_data.size())
        throw std::runtime_error("_sycl_vmm: failed to read IPC handle from fd");

    syclex::physical_mem imported =
        sycl_ipc::physical_memory::open(handle_data, ds.context, ds.device);

    std::lock_guard<std::mutex> lock(g_handle_mutex);
    int64_t id = g_next_phys_id++;
    g_phys_handles.emplace(id, std::move(imported));
    return id;
}

static void ipc_put_handle(py::bytes handle_bytes_py) {
    std::string raw = handle_bytes_py;
    // Construct a handle from the raw bytes and release it.
    std::vector<std::byte> hdata(raw.size());
    std::memcpy(hdata.data(), raw.data(), raw.size());

    sycl_ipc::handle h(std::move(hdata));
    auto& ds = active_dev();
    sycl_ipc::physical_memory::put(h, ds.context);
}
#else
// Stubs when IPC headers are not available.
static int ipc_export_fd(int64_t) {
    throw std::runtime_error("_sycl_vmm: IPC physical memory not available (oneAPI too old)");
}
static int64_t ipc_import_fd(int, int) {
    throw std::runtime_error("_sycl_vmm: IPC physical memory not available (oneAPI too old)");
}
static void ipc_put_handle(py::bytes) {
    throw std::runtime_error("_sycl_vmm: IPC physical memory not available (oneAPI too old)");
}
#endif

// --- Host register ---------------------------------------------------------

#if SYCL_HAS_HOST_REGISTER
static void host_register(uintptr_t ptr, size_t size) {
    auto& ds = active_dev();
    if (!ds.has_host_register)
        throw std::runtime_error(
            "_sycl_vmm: device does not support host memory registration "
            "(aspect::ext_oneapi_register_host_memory). "
            "Ensure oneAPI 2026.1+ is in use.");
    syclex::register_host_memory(reinterpret_cast<void*>(ptr), size, ds.context);
}
static void host_unregister(uintptr_t ptr) {
    auto& ds = active_dev();
    syclex::unregister_host_memory(reinterpret_cast<void*>(ptr), ds.context);
}
#else
static void host_register(uintptr_t, size_t) {
    throw std::runtime_error("_sycl_vmm: register_host_memory not available (oneAPI too old)");
}
static void host_unregister(uintptr_t) {
    throw std::runtime_error("_sycl_vmm: unregister_host_memory not available (oneAPI too old)");
}
#endif

// --- Queues / streams ------------------------------------------------------

static int64_t stream_create(int dev_idx) {
    auto& ds = dev_state(dev_idx);
    sycl::queue q{ds.context, ds.device, sycl::property::queue::in_order{}};

    std::lock_guard<std::mutex> lock(g_handle_mutex);
    int64_t id = g_next_stream_id++;
    g_stream_handles.emplace(id, std::move(q));
    return id;
}

static void stream_destroy(int64_t stream_id) {
    std::lock_guard<std::mutex> lock(g_handle_mutex);
    auto it = g_stream_handles.find(stream_id);
    if (it == g_stream_handles.end())
        throw std::invalid_argument("_sycl_vmm: unknown stream handle");
    g_stream_handles.erase(it);
}

static void stream_synchronize(int64_t stream_id) {
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

static void memcpy_async(uintptr_t dst, uintptr_t src, size_t size, int64_t stream_id) {
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

static int get_pointer_type(uintptr_t ptr, int dev_idx) {
    auto& ds = dev_state(dev_idx);

    // First try SYCL USM pointer type query.
    sycl::usm::alloc kind = sycl::get_pointer_type(
        reinterpret_cast<const void*>(ptr), ds.context);

    if (kind != sycl::usm::alloc::unknown) {
        return static_cast<int>(kind);
    }

    // For VMM-mapped pointers, USM returns 'unknown'. Fall back to
    // get_access_mode which IS VMM-aware.
    try {
        auto mode = syclex::get_access_mode(
            reinterpret_cast<const void*>(ptr), 1, ds.context);
        // If we get here without exception, the pointer is in a mapped VMM range.
        // Return a synthetic value (4) to distinguish from USM types (0-3).
        return 4;  // VMM-mapped pointer
    } catch (...) {
        // Not a VMM pointer either — genuinely unknown.
        return static_cast<int>(sycl::usm::alloc::unknown);
    }
}

// --- Global synchronize ---------------------------------------------------

static void synchronize() {
    active_dev().default_queue.wait_and_throw();
}

// ============================================================================
// pybind11 module definition
// ============================================================================

PYBIND11_MODULE(_sycl_vmm, m) {
    m.doc() = "SYCL VMM/IPC/host-register native module for GMS XPU backend";

    // --- runtime / discovery ---
    m.def("ensure_initialized", &ensure_initialized,
          "Initialize SYCL runtime and cache GPU devices/contexts");
    m.def("device_count", &device_count,
          "Return number of GPU devices");
    m.def("device_memory_info", &device_memory_info,
          "Return (free_bytes, total_bytes) for a device",
          py::arg("dev_idx"));
    m.def("set_device", &set_device,
          "Set the active device for subsequent operations",
          py::arg("dev_idx"));
    m.def("synchronize", &synchronize,
          "Wait for all work on the active device's default queue");

    // --- VMM core ---
    m.def("get_mem_granularity", &get_mem_granularity,
          "Return minimum allocation granularity in bytes",
          py::arg("dev_idx"));
    m.def("reserve_virtual_mem", &reserve_virtual_mem,
          "Reserve a contiguous VA range; returns base address",
          py::arg("size"));
    m.def("free_virtual_mem", &free_virtual_mem,
          "Release a VA reservation",
          py::arg("ptr"), py::arg("size"));
    m.def("set_access_mode", &set_access_mode,
          "Set device-side access mode for a mapped VA range",
          py::arg("ptr"), py::arg("size"), py::arg("dev_idx"), py::arg("mode"));
    m.def("unmap", &unmap,
          "Unbind a VA range from its physical memory",
          py::arg("ptr"), py::arg("size"));

    // --- physical_mem ---
    m.def("physical_mem_create", &physical_mem_create,
          "Create physical memory; returns (success, handle_id)",
          py::arg("dev_idx"), py::arg("size"), py::arg("enable_ipc") = true);
    m.def("physical_mem_release", &physical_mem_release,
          "Release a physical memory handle",
          py::arg("handle_id"));
    m.def("physical_mem_map", &physical_mem_map,
          "Map physical memory to a VA range",
          py::arg("handle_id"), py::arg("ptr"), py::arg("size"), py::arg("mode") = 2);
    m.def("physical_mem_size", &physical_mem_size,
          "Return the size of a physical memory allocation",
          py::arg("handle_id"));

    // --- IPC ---
    m.def("ipc_export_fd", &ipc_export_fd,
          "Export a physical_mem handle as a memfd FD for cross-process sharing",
          py::arg("handle_id"));
    m.def("ipc_import_fd", &ipc_import_fd,
          "Import a physical_mem handle from a memfd FD; closes the FD",
          py::arg("fd"), py::arg("dev_idx"));
    m.def("ipc_put_handle", &ipc_put_handle,
          "Release an IPC handle's resources",
          py::arg("handle_bytes"));

    // --- host register ---
    m.def("host_register", &host_register,
          "Pin host memory for DMA access",
          py::arg("ptr"), py::arg("size"));
    m.def("host_unregister", &host_unregister,
          "Unpin previously registered host memory",
          py::arg("ptr"));

    // --- queues / streams ---
    m.def("stream_create", &stream_create,
          "Create a non-blocking in-order queue; returns stream_id",
          py::arg("dev_idx"));
    m.def("stream_destroy", &stream_destroy,
          "Destroy a stream",
          py::arg("stream_id"));
    m.def("stream_synchronize", &stream_synchronize,
          "Wait for all work on a stream to complete",
          py::arg("stream_id"));

    // --- memcpy ---
    m.def("memcpy_async", &memcpy_async,
          "Async memcpy on a stream (works for H2D, D2H, D2D)",
          py::arg("dst"), py::arg("src"), py::arg("size"), py::arg("stream_id"));

    // --- pointer validation ---
    m.def("get_pointer_type", &get_pointer_type,
          "Query pointer type: 0=host, 1=device, 2=shared, 3=unknown, 4=vmm_mapped",
          py::arg("ptr"), py::arg("dev_idx"));

    // --- runtime aspect queries ---
    m.def("has_ipc_support", [](int dev_idx) -> bool {
        return dev_state(dev_idx).has_ipc;
    }, "Check if device supports IPC physical memory (cached at init)",
       py::arg("dev_idx"));

    m.def("has_host_register_support", [](int dev_idx) -> bool {
        return dev_state(dev_idx).has_host_register;
    }, "Check if device supports host memory registration (cached at init)",
       py::arg("dev_idx"));

    m.def("has_vmm_support", [](int dev_idx) -> bool {
        return dev_state(dev_idx).has_vmm;
    }, "Check if device supports virtual memory management (cached at init)",
       py::arg("dev_idx"));

    // --- compile-time capability flags ---
    m.attr("HAS_SYCL_IPC") = py::bool_(SYCL_HAS_IPC_PHYSICAL_MEM != 0);
    m.attr("HAS_SYCL_HOST_REGISTER") = py::bool_(SYCL_HAS_HOST_REGISTER != 0);
    m.attr("HAS_SYCL_FREE_MEMORY") = py::bool_(SYCL_HAS_FREE_MEMORY != 0);
#ifdef __INTEL_LLVM_COMPILER
    m.attr("ONEAPI_VERSION") = py::int_(__INTEL_LLVM_COMPILER);
#else
    m.attr("ONEAPI_VERSION") = py::int_(0);
#endif
}
