// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use bytes::Bytes;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};
use serde::{Deserialize, Serialize};

/// A type that can hold either JSON data or raw bytes.
#[derive(Clone, Debug, PartialEq)]
pub enum Payload {
    Json(serde_json::Value),
    Bytes(Bytes),
}

impl Payload {
    pub fn from_json(value: serde_json::Value) -> Self {
        Self::Json(value)
    }

    pub fn from_bytes(bytes: impl Into<Bytes>) -> Self {
        Self::Bytes(bytes.into())
    }

    pub fn is_json(&self) -> bool {
        matches!(self, Self::Json(_))
    }

    pub fn is_bytes(&self) -> bool {
        matches!(self, Self::Bytes(_))
    }

    pub fn as_json(&self) -> Option<&serde_json::Value> {
        match self {
            Self::Json(v) => Some(v),
            Self::Bytes(_) => None,
        }
    }

    pub fn as_bytes(&self) -> Option<&Bytes> {
        match self {
            Self::Json(_) => None,
            Self::Bytes(b) => Some(b),
        }
    }

    pub fn into_json(self) -> serde_json::Result<serde_json::Value> {
        match self {
            Self::Json(v) => Ok(v),
            Self::Bytes(b) => serde_json::from_slice(&b),
        }
    }
}

impl From<serde_json::Value> for Payload {
    fn from(value: serde_json::Value) -> Self {
        Self::Json(value)
    }
}

impl Serialize for Payload {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Json(v) => v.serialize(serializer),
            Self::Bytes(b) => b.serialize(serializer),
        }
    }
}

impl<'de> Deserialize<'de> for Payload {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = serde_json::Value::deserialize(deserializer)?;
        Ok(Self::Json(value))
    }
}

impl<'py> FromPyObject<'py> for Payload {
    fn extract_bound(ob: &Bound<'py, PyAny>) -> PyResult<Self> {
        if let Ok(bytes) = ob.downcast::<PyBytes>() {
            return Ok(Self::Bytes(Bytes::from(bytes.as_bytes().to_vec())));
        }
        match pythonize::depythonize(ob) {
            Ok(value) => Ok(Self::Json(value)),
            Err(e) => Err(PyErr::new::<pyo3::exceptions::PyTypeError, _>(format!(
                "Failed to convert Python object to Payload: {}",
                e
            ))),
        }
    }
}

impl IntoPy<PyObject> for Payload {
    fn into_py(self, py: Python<'_>) -> PyObject {
        match self {
            Self::Json(v) => match pythonize::pythonize(py, &v) {
                Ok(obj) => obj.into(),
                Err(e) => PyString::new(py, &format!("Error: {}", e)).into(),
            },
            Self::Bytes(b) => PyBytes::new(py, &b).into(),
        }
    }
}

impl Default for Payload {
    fn default() -> Self {
        Self::Json(serde_json::Value::Null)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_is_json() {
        let json = Payload::from_json(json!("test"));
        assert!(json.is_json());
        assert!(!json.is_bytes());
    }

    #[test]
    fn test_is_bytes() {
        let bytes = Payload::from_bytes(vec![1, 2, 3]);
        assert!(bytes.is_bytes());
        assert!(!bytes.is_json());
    }

    #[test]
    fn test_serialize_json() {
        let json = Payload::from_json(json!({"key": "value"}));
        let serialized = serde_json::to_string(&json).unwrap();
        assert_eq!(serialized, r#"{"key":"value"}"#);
    }

    #[test]
    fn test_default() {
        let default = Payload::default();
        assert!(default.is_json());
        assert_eq!(default.as_json().unwrap(), &serde_json::Value::Null);
    }
}
