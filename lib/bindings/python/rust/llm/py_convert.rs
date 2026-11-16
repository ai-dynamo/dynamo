// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Direct reads of exact builtin Python types, for request fields that carry prompt-sized
//! lists of ints, where pythonize pays tens of ns per int. Accepted inputs convert exactly
//! as pythonize would; anything else returns `None`, and the caller falls back to pythonize.

use pyo3::ffi;
use pyo3::prelude::*;
use pyo3::types::{PyBool, PyDict, PyFloat, PyInt, PyList, PyString, PyTuple};
use serde_json::{Map, Number, Value};

/// The map key serde_json's `raw_value` feature reads as embedded raw JSON
/// (`serde_json::raw::TOKEN`, which is private).
const SERDE_JSON_RAW_VALUE_TOKEN: &str = "$serde_json::private::RawValue";

/// An exact `list` of exact `int`s as `Vec<u32>`; `None` for anything else, including a
/// negative id or one past `u32::MAX`.
pub(crate) fn read_exact_u32_list(obj: &Bound<'_, PyAny>) -> Option<Vec<u32>> {
    let list = obj.downcast_exact::<PyList>().ok()?;
    // SAFETY: `PyLong_AsUnsignedLong` on an exact `int` calls no `__index__`, so the read
    // runs no Python code.
    unsafe {
        read_list(list, |item| {
            let int = item.downcast_exact::<PyInt>().ok()?;
            // A negative int sets OverflowError and returns `c_ulong::MAX`; both it and
            // values past `u32::MAX` fail `try_from`.
            let id = u32::try_from(ffi::PyLong_AsUnsignedLong(int.as_ptr()));
            if id.is_err() {
                ffi::PyErr_Clear();
            }
            id.ok()
        })
    }
}

/// Exact `dict` (with `str` keys), `list`, `tuple`, `str`, `int` (in `i64` or `u64` range),
/// `float`, `bool` or `None` as a `serde_json::Value`; `None` for anything else.
pub(crate) fn exact_json_value(obj: &Bound<'_, PyAny>) -> Option<Value> {
    if let Ok(int) = obj.downcast_exact::<PyInt>() {
        return exact_int(int);
    }
    if let Ok(list) = obj.downcast_exact::<PyList>() {
        // SAFETY: this function reads only exact builtin types, which runs no Python code.
        let items = unsafe {
            read_list(list, |item| match item.downcast_exact::<PyInt>() {
                // Ints, by far the commonest item, skip the recursive call.
                Ok(int) => exact_int(int),
                Err(_) => exact_json_value(item),
            })
        };
        return items.map(Value::Array);
    }
    if obj.is_none() {
        return Some(Value::Null);
    }
    if let Ok(flag) = obj.downcast_exact::<PyBool>() {
        return Some(Value::Bool(flag.is_true()));
    }
    if let Ok(text) = obj.downcast_exact::<PyString>() {
        return text.to_str().ok().map(|s| Value::String(s.to_owned()));
    }
    if let Ok(number) = obj.downcast_exact::<PyFloat>() {
        // NaN and the infinities become `Null`, as serde_json's own `Value` visitor does.
        return Some(Number::from_f64(number.value()).map_or(Value::Null, Value::Number));
    }
    if let Ok(tuple) = obj.downcast_exact::<PyTuple>() {
        return tuple
            .iter_borrowed()
            .map(|item| exact_json_value(&item))
            .collect::<Option<_>>()
            .map(Value::Array);
    }
    let dict = obj.downcast_exact::<PyDict>().ok()?;
    let mut map = Map::with_capacity(dict.len());
    for (key, value) in dict.iter() {
        let key = key.downcast_exact::<PyString>().ok()?.to_str().ok()?;
        if key == SERDE_JSON_RAW_VALUE_TOKEN {
            return None;
        }
        map.insert(key.to_owned(), exact_json_value(&value)?);
    }
    Some(Value::Object(map))
}

/// An `int` in `i64` or `u64` range as the same `Number` pythonize gives; larger ints decline.
fn exact_int(int: &Bound<'_, PyInt>) -> Option<Value> {
    let ptr = int.as_ptr();
    // SAFETY: `ptr` is a live `int` and the GIL is held; these calls run no Python code.
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

/// Each item of `list` through `read`, borrowed without touching its refcount; `None` if
/// `read` declines any item.
///
/// # Safety
/// `read` must run no Python code, so that nothing can change the list while it is read.
unsafe fn read_list<T>(
    list: &Bound<'_, PyList>,
    read: impl Fn(&Bound<'_, PyAny>) -> Option<T>,
) -> Option<Vec<T>> {
    let mut out = Vec::with_capacity(list.len());
    for index in 0..list.len() {
        // SAFETY: the GIL is held and, per the contract, the list cannot change, so every
        // index is in range and the borrowed item stays valid while `read` uses it.
        let item = unsafe {
            Borrowed::from_ptr(
                list.py(),
                ffi::PyList_GetItem(list.as_ptr(), index as ffi::Py_ssize_t),
            )
        };
        out.push(read(&item)?);
    }
    Some(out)
}
