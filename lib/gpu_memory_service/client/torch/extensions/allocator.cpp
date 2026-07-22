// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Minimal CUDAPluggableAllocator shim for GPU Memory Service.
//
// This extension provides the my_malloc/my_free function pointers required by
// PyTorch's CUDAPluggableAllocator. All actual CUDA VMM operations are delegated
// to Python callbacks which use cuda.bindings.
//
// Note: The stream parameter is unused because CUDA VMM operations (cuMemMap,
// cuMemUnmap) are synchronous and globally visible - they don't have per-stream
// semantics like cudaMallocAsync. We keep the parameter to match PyTorch's
// CUDAPluggableAllocator interface signature.
//
// PEP 703 (free-threaded CPython) note: the callback pair is held in a magic-
// statics singleton, so reads from my_malloc/my_free are data-race-free without
// explicit synchronization. The C++ contract is "first call to callbacks() wins";
// in practice that is init_module, because the Python wrapper
// _ensure_callbacks_initialized invokes init_module synchronously before any
// allocation path can reach my_malloc / my_free. Subsequent calls to callbacks()
// with new arguments are silent no-ops on the stored pointers.

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>

namespace {

struct Callbacks {
  PyObject* malloc_cb;
  PyObject* free_cb;
};

// Magic-statics singleton: thread-safe lazy construction is guaranteed by
// C++11 [stmt.dcl]/4, and the resulting happens-before makes all subsequent
// reads data-race-free. New refs to the captured callables are taken inside
// the lambda so the singleton owns them for the program lifetime.
const Callbacks&
callbacks(PyObject* m = nullptr, PyObject* f = nullptr)
{
  static const Callbacks instance = [&]() {
    Py_XINCREF(m);
    Py_XINCREF(f);
    return Callbacks{m, f};
  }();
  return instance;
}

}  // namespace

extern "C" {

void*
my_malloc(ssize_t size, int device, void* stream)
{
  const Callbacks& cb = callbacks();
  if (!cb.malloc_cb) {
    return nullptr;
  }

  PyGILState_STATE gstate = PyGILState_Ensure();

  PyObject* args = Py_BuildValue("(niK)", size, device, (unsigned long long)stream);
  PyObject* result = PyObject_CallObject(cb.malloc_cb, args);
  Py_DECREF(args);

  void* ptr = nullptr;
  if (result && PyLong_Check(result)) {
    ptr = (void*)PyLong_AsUnsignedLongLong(result);
  }
  Py_XDECREF(result);

  if (PyErr_Occurred()) {
    PyErr_Print();
  }

  PyGILState_Release(gstate);
  return ptr;
}

void
my_free(void* ptr, ssize_t size, int device, void* stream)
{
  const Callbacks& cb = callbacks();
  if (!cb.free_cb) {
    return;
  }

  PyGILState_STATE gstate = PyGILState_Ensure();

  PyObject* args = Py_BuildValue("(KniK)", (unsigned long long)ptr, size, device, (unsigned long long)stream);
  PyObject* result = PyObject_CallObject(cb.free_cb, args);
  Py_DECREF(args);
  Py_XDECREF(result);

  if (PyErr_Occurred()) {
    PyErr_Print();
  }

  PyGILState_Release(gstate);
}

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

  // First call to callbacks() wins; subsequent calls do not rebind the stored
  // pointers. In practice this is invoked exactly once by the Python wrapper
  // _ensure_callbacks_initialized.
  callbacks(malloc_cb, free_cb);

  Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {"init_module", py_init_module, METH_VARARGS, "Set malloc/free callbacks"}, {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef allocator_module = {
    PyModuleDef_HEAD_INIT, "_allocator_ext", "CUDAPluggableAllocator shim for GPU Memory Service", -1, module_methods};

PyMODINIT_FUNC
PyInit__allocator_ext(void)
{
  PyObject* m = PyModule_Create(&allocator_module);
  if (m == nullptr) {
    return nullptr;
  }

#ifdef Py_GIL_DISABLED
  PyUnstable_Module_SetGIL(m, Py_MOD_GIL_NOT_USED);
#endif

  return m;
}

}  // extern "C"
