// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct reads of plain Python data into Rust values, for the request fields that carry
//! prompt-sized lists of ints. pythonize converts each int through pyo3's abi3 128-bit
//! extraction and each list item through the generic sequence protocol; these read exact
//! builtin types straight through the C API instead.
//!
//! Each function returns `None` when the input holds anything outside the exact builtin
//! types it reads. The caller then converts with pythonize as before, so results and
//! errors for those inputs are unchanged. For every input accepted, the result equals
//! pythonize's.

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyFloat, PyList, PyString, PyTuple};
use serde_json::{Map, Number, Value};

/// The map key serde_json's `raw_value` feature reads as embedded raw JSON
/// (`serde_json::raw::TOKEN`, which is private).
const SERDE_JSON_RAW_VALUE_TOKEN: &str = "$serde_json::private::RawValue";

/// Nesting depth past which [`exact_json_value`] declines rather than recursing further.
const MAX_DEPTH: usize = 128;

/// An exact `list` of exact `int`s as `Vec<u32>`; `None` for anything else, including a
/// negative id or one past `u32::MAX`.
pub(crate) fn read_exact_u32_list(obj: &Bound<'_, PyAny>) -> Option<Vec<u32>> {
    let list = obj.downcast_exact::<PyList>().ok()?;
    let mut out = Vec::with_capacity(list.len());
    // SAFETY: the GIL is held and the loop runs no Python code (only exact `int`s are
    // converted, and `PyLong_AsUnsignedLong` does not call `__index__`), so the exact
    // list cannot change, every index is in range, and the borrowed items stay valid.
    unsafe {
        for index in 0..list.len() {
            let item = ffi::PyList_GetItem(list.as_ptr(), index as ffi::Py_ssize_t);
            if ffi::PyLong_CheckExact(item) == 0 {
                return None;
            }
            // A negative int sets OverflowError and returns `c_ulong::MAX`; both it and
            // values past `u32::MAX` fail `try_from`.
            let Ok(id) = u32::try_from(ffi::PyLong_AsUnsignedLong(item)) else {
                ffi::PyErr_Clear();
                return None;
            };
            out.push(id);
        }
    }
    Some(out)
}

/// Plain JSON-shaped Python data (exact `dict` with `str` keys, `list`, `tuple`, `str`,
/// `int`, `float`, `bool`, `None`) as a `serde_json::Value`; `None` for anything else.
pub(crate) fn exact_json_value(obj: &Bound<'_, PyAny>) -> Option<Value> {
    json_value(obj, 0)
}

fn json_value(obj: &Bound<'_, PyAny>, depth: usize) -> Option<Value> {
    if depth > MAX_DEPTH {
        return None;
    }
    let ptr = obj.as_ptr();
    // SAFETY: `ptr` is a live object; these are type checks and reads only.
    unsafe {
        if ffi::PyLong_CheckExact(ptr) != 0 {
            return exact_int(ptr);
        }
        if ffi::PyList_CheckExact(ptr) != 0 {
            return sequence(obj, ptr, depth, ffi::PyList_GetItem);
        }
    }
    if obj.is_none() {
        return Some(Value::Null);
    }
    // `bool` cannot be subclassed, so this is exact; pythonize reads it before `int`.
    if let Ok(flag) = obj.downcast_exact::<pyo3::types::PyBool>() {
        return Some(Value::Bool(flag.is_true()));
    }
    if let Ok(text) = obj.downcast_exact::<PyString>() {
        return text.to_str().ok().map(|s| Value::String(s.to_owned()));
    }
    if let Ok(number) = obj.downcast_exact::<PyFloat>() {
        // NaN and the infinities become `Null`, as serde_json's own `Value` visitor does.
        return Some(Number::from_f64(number.value()).map_or(Value::Null, Value::Number));
    }
    if obj.downcast_exact::<PyTuple>().is_ok() {
        // SAFETY: as above.
        return unsafe { sequence(obj, obj.as_ptr(), depth, ffi::PyTuple_GetItem) };
    }
    if let Ok(dict) = obj.downcast_exact::<PyDict>() {
        let mut map = Map::with_capacity(dict.len());
        for (key, value) in dict.iter() {
            let key = key.downcast_exact::<PyString>().ok()?.to_str().ok()?;
            if key == SERDE_JSON_RAW_VALUE_TOKEN {
                return None;
            }
            map.insert(key.to_owned(), json_value(&value, depth + 1)?);
        }
        return Some(Value::Object(map));
    }
    None
}

/// An exact `int` in `i64` or `u64` range, as pythonize gives it to serde_json: the same
/// `Number`, whichever of the two ranges it falls in. Larger ints decline.
///
/// # Safety
/// `ptr` must be a live exact `int`, with the GIL held.
unsafe fn exact_int(ptr: *mut ffi::PyObject) -> Option<Value> {
    unsafe {
        let signed = ffi::PyLong_AsLongLong(ptr);
        if signed != -1 || ffi::PyErr_Occurred().is_null() {
            return Some(Value::from(signed));
        }
        ffi::PyErr_Clear();
        let unsigned = ffi::PyLong_AsUnsignedLongLong(ptr);
        if unsigned == u64::MAX && !ffi::PyErr_Occurred().is_null() {
            ffi::PyErr_Clear();
            return None;
        }
        Some(Value::from(unsigned))
    }
}

/// An exact `list` or `tuple` as an array, reading items by index.
///
/// # Safety
/// `ptr` must be `obj`'s pointer and `get_item` the matching borrowed-item getter, with the
/// GIL held.
unsafe fn sequence(
    obj: &Bound<'_, PyAny>,
    ptr: *mut ffi::PyObject,
    depth: usize,
    get_item: unsafe extern "C" fn(*mut ffi::PyObject, ffi::Py_ssize_t) -> *mut ffi::PyObject,
) -> Option<Value> {
    let len = obj.len().ok()?;
    let mut items = Vec::with_capacity(len);
    // SAFETY: the conversion runs no Python code, so the exact list or tuple cannot change,
    // every index is in range, and the borrowed items stay valid while they are read.
    unsafe {
        for index in 0..len {
            let item = get_item(ptr, index as ffi::Py_ssize_t);
            // Ints, by far the commonest item, skip building a `Bound`.
            let value = if ffi::PyLong_CheckExact(item) != 0 {
                exact_int(item)?
            } else {
                json_value(&Bound::from_borrowed_ptr(obj.py(), item), depth + 1)?
            };
            items.push(value);
        }
    }
    Some(Value::Array(items))
}
