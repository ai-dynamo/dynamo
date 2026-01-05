// CUDAPluggableAllocator for the GPU Memory Service.
//
// This extension provides two key pieces:
// 1) A CUDAPluggableAllocator backend (my_malloc/my_free) that reserves VA and
//    relies on a Python callback to RPC-allocate physical memory elsewhere and
//    map it into the reserved VA before my_malloc returns.
// 2) Python-callable helpers for mapping/unmapping imported allocations and
//    flipping VA permissions (RO/RW) to enforce publish semantics.
//
// Intended usage (writer):
// - install malloc/free callbacks that RPC allocate + map RW (import_and_map(..., read_only=false))
// - load weights
// - cudaDeviceSynchronize(); cumem.set_access_all(true)  // flip to RO
// - Commit() to the Allocation Server
//
// Readers import allocations and map RO (import_and_map(..., read_only=true)) and
// should keep the RO connection open during inference.

#include <cuda.h>

#include <iostream>
#include <unordered_map>

#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Error handling
static char error_msg[10240];
static CUresult error_code = CUDA_SUCCESS;

#define CUDA_CHECK(condition)                                                                                         \
  do {                                                                                                                \
    CUresult error = condition;                                                                                       \
    if (error != CUDA_SUCCESS) {                                                                                      \
      error_code = error;                                                                                             \
      const char* error_string;                                                                                       \
      cuGetErrorString(error, &error_string);                                                                         \
      snprintf(                                                                                                       \
          error_msg, sizeof(error_msg), "CUDA Error: %s at %s:%d", error_string ? error_string : "unknown", __FILE__, \
          __LINE__);                                                                                                  \
      std::cerr << error_msg << std::endl;                                                                            \
    }                                                                                                                 \
  } while (0)

// Allocation tracking
struct AllocationInfo {
  CUdeviceptr va;
  size_t size;
  size_t aligned_size;
  int device;
  CUmemGenericAllocationHandle handle;  // 0 if not mapped yet
  bool is_imported;                     // true if memory was imported from external FD
};

static std::unordered_map<CUdeviceptr, AllocationInfo> g_allocations;

// Python callbacks used for remote allocation tracking
static PyObject* g_python_malloc_callback = nullptr;
static PyObject* g_python_free_callback = nullptr;

// Note: this extension never creates physical allocations in-process; physical memory is
// always provided via RPC + FD import.

// Helper to ensure CUDA context
static void
ensure_context(int device)
{
  CUcontext ctx;
  CUDA_CHECK(cuCtxGetCurrent(&ctx));
  if (!ctx) {
    CUDA_CHECK(cuDevicePrimaryCtxRetain(&ctx, device));
    CUDA_CHECK(cuCtxSetCurrent(ctx));
  }
}

// Get allocation granularity
static size_t
get_granularity(int device)
{
  CUmemAllocationProp prop = {};
  prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
  prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  prop.location.id = device;
  prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

  size_t granularity;
  CUDA_CHECK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));
  return granularity;
}

// Align size to granularity
static size_t
align_size(size_t size, size_t granularity)
{
  return ((size + granularity - 1) / granularity) * granularity;
}

extern "C" {

// ---------------------------------------------------------------------------
// Pluggable allocator functions

void*
my_malloc(ssize_t size, int device, CUstream stream)
{
  error_code = CUDA_SUCCESS;
  ensure_context(device);

  size_t granularity = get_granularity(device);
  size_t aligned_size = align_size(size, granularity);

  // Reserve virtual address
  CUdeviceptr va;
  CUDA_CHECK(cuMemAddressReserve(&va, aligned_size, granularity, 0, 0));
  if (error_code != CUDA_SUCCESS) {
    return nullptr;
  }

  AllocationInfo info = {};
  info.va = va;
  info.size = size;
  info.aligned_size = aligned_size;
  info.device = device;
  info.handle = 0;
  info.is_imported = false;

  g_allocations[va] = info;

  // Require Python callback to allocate + map remote memory for this VA.
  if (!g_python_malloc_callback) {
    std::cerr << "allocator: init_module(malloc_cb, free_cb) must be called before using my_malloc" << std::endl;
    cuMemAddressFree(va, aligned_size);
    g_allocations.erase(va);
    return nullptr;
  }

  // Call Python callback to perform RPC allocate + map.
  PyGILState_STATE gstate = PyGILState_Ensure();

  PyObject* args = Py_BuildValue(
      "(KKKiK)", (unsigned long long)va, (unsigned long long)size, (unsigned long long)aligned_size, device,
      (unsigned long long)(uintptr_t)stream);

  PyObject* result = PyObject_CallObject(g_python_malloc_callback, args);
  Py_DECREF(args);
  Py_XDECREF(result);

  if (PyErr_Occurred()) {
    PyErr_Print();
    // Treat callback failure as allocation failure.
    // Clean up the VA reservation (and any mapping/handle if present).
    auto it2 = g_allocations.find(va);
    if (it2 != g_allocations.end()) {
      AllocationInfo& info2 = it2->second;
      if (info2.handle != 0) {
        cuMemUnmap(info2.va, info2.aligned_size);
        cuMemRelease(info2.handle);
      }
      cuMemAddressFree(info2.va, info2.aligned_size);
      g_allocations.erase(it2);
    }
    PyGILState_Release(gstate);
    return nullptr;
  }

  PyGILState_Release(gstate);

  return (void*)va;
}

void
my_free(void* ptr, ssize_t size, int device, CUstream stream)
{
  CUdeviceptr va = (CUdeviceptr)ptr;

  auto it = g_allocations.find(va);
  if (it == g_allocations.end()) {
    std::cerr << "my_free: unknown pointer " << ptr << std::endl;
    return;
  }

  AllocationInfo& info = it->second;
  ensure_context(info.device);

  // Call Python callback if set
  if (g_python_free_callback) {
    PyGILState_STATE gstate = PyGILState_Ensure();

    PyObject* args = Py_BuildValue("(K)", (unsigned long long)va);
    PyObject* result = PyObject_CallObject(g_python_free_callback, args);
    Py_DECREF(args);
    Py_XDECREF(result);

    if (PyErr_Occurred()) {
      PyErr_Print();
    }

    PyGILState_Release(gstate);
  }

  // Unmap if mapped
  if (info.handle != 0) {
    CUDA_CHECK(cuMemUnmap(va, info.aligned_size));

    // Always release the handle reference (imported or locally created).
    CUDA_CHECK(cuMemRelease(info.handle));
  }

  // Free VA reservation
  CUDA_CHECK(cuMemAddressFree(va, info.aligned_size));

  g_allocations.erase(it);
}

// ---------------------------------------------------------------------------
// Python-exposed functions

// init_module(malloc_callback, free_callback)
static PyObject*
py_init_module(PyObject* self, PyObject* args)
{
  PyObject* malloc_cb = nullptr;
  PyObject* free_cb = nullptr;

  if (!PyArg_ParseTuple(args, "OO", &malloc_cb, &free_cb)) {
    return nullptr;
  }

  if (!PyCallable_Check(malloc_cb) || !PyCallable_Check(free_cb)) {
    PyErr_SetString(PyExc_TypeError, "Both arguments must be callables");
    return nullptr;
  }

  // Hold strong references so callbacks remain valid after the caller scope exits.
  Py_XINCREF(malloc_cb);
  Py_XINCREF(free_cb);
  Py_XDECREF(g_python_malloc_callback);
  Py_XDECREF(g_python_free_callback);

  g_python_malloc_callback = malloc_cb;
  g_python_free_callback = free_cb;

  Py_RETURN_NONE;
}

// import_and_map(va, fd, size, device, read_only) -> handle
// Import external FD and map to reserved VA
static PyObject*
py_import_and_map(PyObject* self, PyObject* args)
{
  unsigned long long va;
  int fd;
  unsigned long long size;
  int device;
  int read_only;

  // read_only controls access permissions: true -> RO, false -> RW.
  if (!PyArg_ParseTuple(args, "KiKip", &va, &fd, &size, &device, &read_only)) {
    return nullptr;
  }

  auto it = g_allocations.find((CUdeviceptr)va);
  if (it == g_allocations.end()) {
    PyErr_SetString(PyExc_ValueError, "VA not found in allocations");
    return nullptr;
  }

  AllocationInfo& info = it->second;

  if (info.handle != 0) {
    PyErr_SetString(PyExc_RuntimeError, "VA already mapped");
    return nullptr;
  }

  ensure_context(device);
  error_code = CUDA_SUCCESS;

  // Import the shareable handle
  CUmemGenericAllocationHandle handle;
  CUDA_CHECK(cuMemImportFromShareableHandle(&handle, (void*)(intptr_t)fd, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));
  if (error_code != CUDA_SUCCESS) {
    PyErr_SetString(PyExc_RuntimeError, error_msg);
    return nullptr;
  }

  // Map to reserved VA
  size_t granularity = get_granularity(device);
  size_t aligned_size = align_size(size, granularity);

  CUDA_CHECK(cuMemMap((CUdeviceptr)va, aligned_size, 0, handle, 0));
  if (error_code != CUDA_SUCCESS) {
    cuMemRelease(handle);
    PyErr_SetString(PyExc_RuntimeError, error_msg);
    return nullptr;
  }

  // Set access permissions
  CUmemAccessDesc access = {};
  access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = device;
  access.flags = read_only ? CU_MEM_ACCESS_FLAGS_PROT_READ : CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

  CUDA_CHECK(cuMemSetAccess((CUdeviceptr)va, aligned_size, &access, 1));
  if (error_code != CUDA_SUCCESS) {
    cuMemUnmap((CUdeviceptr)va, aligned_size);
    cuMemRelease(handle);
    PyErr_SetString(PyExc_RuntimeError, error_msg);
    return nullptr;
  }

  info.handle = handle;
  info.is_imported = true;
  info.device = device;
  // Update sizes to reflect what was actually mapped
  info.size = size;
  info.aligned_size = aligned_size;

  return PyLong_FromUnsignedLongLong((unsigned long long)handle);
}

// get_all_allocations() -> list of (va, size, aligned_size, handle, is_imported, is_mapped, device)
// Get info about all tracked allocations
static PyObject*
py_get_all_allocations(PyObject* self, PyObject* args)
{
  PyObject* result = PyList_New(0);
  if (!result)
    return nullptr;

  for (const auto& pair : g_allocations) {
    const AllocationInfo& info = pair.second;
    bool is_mapped = (info.handle != 0);

    PyObject* item = Py_BuildValue(
        "(KKKKiii)", (unsigned long long)info.va, (unsigned long long)info.size, (unsigned long long)info.aligned_size,
        (unsigned long long)info.handle, (int)info.is_imported, (int)is_mapped, info.device);

    if (!item) {
      Py_DECREF(result);
      return nullptr;
    }

    if (PyList_Append(result, item) < 0) {
      Py_DECREF(item);
      Py_DECREF(result);
      return nullptr;
    }
    Py_DECREF(item);
  }

  return result;
}

// set_access_all(read_only) -> int
// Flip access for all mapped allocations. Returns number updated.
static PyObject*
py_set_access_all(PyObject* self, PyObject* args)
{
  int read_only;
  if (!PyArg_ParseTuple(args, "p", &read_only)) {
    return nullptr;
  }

  int updated = 0;
  for (auto& pair : g_allocations) {
    AllocationInfo& info = pair.second;
    if (info.handle == 0) {
      continue;
    }

    ensure_context(info.device);
    error_code = CUDA_SUCCESS;

    CUmemAccessDesc access = {};
    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access.location.id = info.device;
    access.flags = read_only ? CU_MEM_ACCESS_FLAGS_PROT_READ : CU_MEM_ACCESS_FLAGS_PROT_READWRITE;

    CUDA_CHECK(cuMemSetAccess(info.va, info.aligned_size, &access, 1));
    if (error_code == CUDA_SUCCESS) {
      updated += 1;
    } else {
      // Best-effort: keep going.
      error_code = CUDA_SUCCESS;
    }
  }

  return PyLong_FromLong(updated);
}

// ---------------------------------------------------------------------------
// Module definition

static PyMethodDef module_methods[] = {
    {"init_module", py_init_module, METH_VARARGS, "Initialize module with malloc/free callbacks"},
    {"import_and_map", py_import_and_map, METH_VARARGS, "Import external FD and map to reserved VA"},
    {"set_access_all", py_set_access_all, METH_VARARGS, "Flip access permissions for all mapped allocations"},
    {"get_all_allocations", py_get_all_allocations, METH_NOARGS, "Get info about all tracked allocations"},
    {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef allocator_module = {
    PyModuleDef_HEAD_INIT, "_allocator_ext", "CUDA memory allocator using VMM for GPU Memory Service integrations", -1,
    module_methods};

PyMODINIT_FUNC
PyInit__allocator_ext(void)
{
  return PyModule_Create(&allocator_module);
}

}  // extern "C"
