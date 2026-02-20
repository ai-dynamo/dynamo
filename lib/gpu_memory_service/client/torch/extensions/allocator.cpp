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

#define PY_SSIZE_T_CLEAN
#include <Python.h>

#include <cstdint>

static PyObject* g_malloc_callback = nullptr;
static PyObject* g_free_callback = nullptr;

extern "C" {

void*
my_malloc(ssize_t size, int device, void* stream)
{
  if (!g_malloc_callback) {
    return nullptr;
  }

  PyGILState_STATE gstate = PyGILState_Ensure();

  PyObject* args = Py_BuildValue("(niK)", size, device, (unsigned long long)stream);
  PyObject* result = PyObject_CallObject(g_malloc_callback, args);
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
  if (!g_free_callback) {
    return;
  }

  PyGILState_STATE gstate = PyGILState_Ensure();

  PyObject* args = Py_BuildValue("(KniK)", (unsigned long long)ptr, size, device, (unsigned long long)stream);
  PyObject* result = PyObject_CallObject(g_free_callback, args);
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

  Py_XINCREF(malloc_cb);
  Py_XINCREF(free_cb);
  Py_XDECREF(g_malloc_callback);
  Py_XDECREF(g_free_callback);

  g_malloc_callback = malloc_cb;
  g_free_callback = free_cb;

  Py_RETURN_NONE;
}

static PyMethodDef module_methods[] = {
    {"init_module", py_init_module, METH_VARARGS, "Set malloc/free callbacks"}, {nullptr, nullptr, 0, nullptr}};

static struct PyModuleDef allocator_module = {
    PyModuleDef_HEAD_INIT, "_allocator_ext", "CUDAPluggableAllocator shim for GPU Memory Service", -1, module_methods};

PyMODINIT_FUNC
PyInit__allocator_ext(void)
{
  return PyModule_Create(&allocator_module);
}

}  // extern "C"
