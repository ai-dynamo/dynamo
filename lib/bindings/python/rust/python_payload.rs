// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use bytes::Bytes;
use dynamo_runtime::pipeline::PipelineError;
use dynamo_runtime::pipeline::network::{
    EncodedResponseFrame, IngressRequestDecoder, IngressResponseEncoder, NetworkStreamWrapper,
    RequestPlanePayloadCodec,
};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::protocols::maybe_error::MaybeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use pythonize::{Depythonizer, Pythonizer, depythonize};
use serde::de::Error as _;
use serde::ser::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::engine::map_python_exception;

/// Python-owned request value used only by the network ingress fast path.
/// Serde events are transcoded directly to or from Python objects without an
/// intermediate Rust value tree.
#[derive(Clone)]
pub(crate) struct PythonPayload(Py<PyAny>);

impl PythonPayload {
    pub(crate) fn into_inner(self) -> Py<PyAny> {
        self.0
    }
}

impl std::fmt::Debug for PythonPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("PythonPayload(<PyAny>)")
    }
}

impl<'de> Deserialize<'de> for PythonPayload {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        Python::with_gil(|py| {
            serde_transcode::transcode(deserializer, Pythonizer::new(py))
                .map(|value| Self(value.unbind()))
                .map_err(D::Error::custom)
        })
    }
}

impl Serialize for PythonPayload {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Python::with_gil(|py| {
            let mut depythonizer = Depythonizer::from_object(self.0.bind(py));
            serde_transcode::transcode(&mut depythonizer, serializer).map_err(S::Error::custom)
        })
    }
}

/// One raw item yielded by a Python async generator.
pub(crate) struct PythonResponseItem(PyResult<Py<PyAny>>);

impl PythonResponseItem {
    pub(crate) fn new(item: PyResult<Py<PyAny>>) -> Self {
        Self(item)
    }

    fn into_result(self) -> PyResult<Py<PyAny>> {
        self.0
    }
}

impl std::fmt::Debug for PythonResponseItem {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            Ok(_) => f.write_str("PythonResponseItem::Data(<PyAny>)"),
            Err(_) => f.write_str("PythonResponseItem::Error(<PyErr>)"),
        }
    }
}

#[derive(Debug, Default)]
pub(crate) struct PythonIngressPayloadAdapter;

impl IngressRequestDecoder<PythonPayload> for PythonIngressPayloadAdapter {
    async fn decode_request(
        &self,
        payload_codec: RequestPlanePayloadCodec,
        bytes: Bytes,
    ) -> Result<PythonPayload, PipelineError> {
        tokio::task::spawn_blocking(move || payload_codec.decode::<PythonPayload>(&bytes))
            .await
            .map_err(|error| {
                PipelineError::DeserializationError(format!(
                    "failed to offload {} Python request decode: {error}",
                    payload_codec.name()
                ))
            })?
            .map_err(|error| {
                PipelineError::DeserializationError(format!(
                    "Failed deserializing {} Python request payload: {error}",
                    payload_codec.name()
                ))
            })
    }
}

impl IngressResponseEncoder<PythonResponseItem> for PythonIngressPayloadAdapter {
    async fn encode_response(
        &self,
        payload_codec: RequestPlanePayloadCodec,
        response: Option<PythonResponseItem>,
        complete_final: bool,
    ) -> Result<EncodedResponseFrame, PipelineError> {
        if complete_final {
            let wrapper = NetworkStreamWrapper::<Annotated<()>> {
                data: None,
                complete_final: true,
            };
            let bytes = payload_codec.encode(&wrapper).map_err(|error| {
                PipelineError::SerializationError(format!(
                    "Failed serializing {} request-plane final response: {error}",
                    payload_codec.name()
                ))
            })?;
            return Ok(EncodedResponseFrame {
                bytes: bytes.into(),
                is_error: false,
                stop_stream: false,
            });
        }

        let response = response.ok_or_else(|| {
            PipelineError::SerializationError(
                "request-plane response item missing before final frame".to_string(),
            )
        })?;
        tokio::task::spawn_blocking(move || encode_python_response(payload_codec, response))
            .await
            .map_err(|error| {
                PipelineError::SerializationError(format!(
                    "failed to offload {} Python response encode: {error}",
                    payload_codec.name()
                ))
            })?
    }
}

fn encode_python_response(
    payload_codec: RequestPlanePayloadCodec,
    response: PythonResponseItem,
) -> Result<EncodedResponseFrame, PipelineError> {
    let (annotated, stop_stream) = match response.into_result() {
        Ok(item) => match Python::with_gil(|py| parse_python_response(item, py)) {
            Ok(annotated) => (annotated, false),
            Err(error) => (
                Annotated::from_error(format!(
                    "critical error: invalid response object from Python async generator; \
                     application-logic-mismatch: {error}"
                )),
                true,
            ),
        },
        Err(error) => (Annotated::from_err(map_python_exception(error)), true),
    };
    let is_error = annotated.is_error();
    let wrapper = NetworkStreamWrapper {
        data: Some(annotated),
        complete_final: false,
    };

    match payload_codec.encode(&wrapper) {
        Ok(bytes) => Ok(EncodedResponseFrame {
            bytes: bytes.into(),
            is_error,
            stop_stream,
        }),
        Err(error) => {
            let fallback = NetworkStreamWrapper {
                data: Some(Annotated::<()>::from_error(format!(
                    "critical error: failed serializing Python response as {}: {error}",
                    payload_codec.name()
                ))),
                complete_final: false,
            };
            let bytes = payload_codec.encode(&fallback).map_err(|fallback_error| {
                PipelineError::SerializationError(format!(
                    "failed to serialize Python response and fallback error as {}: {fallback_error}",
                    payload_codec.name()
                ))
            })?;
            Ok(EncodedResponseFrame {
                bytes: bytes.into(),
                is_error: true,
                stop_stream: true,
            })
        }
    }
}

fn parse_python_response(
    item: Py<PyAny>,
    py: Python<'_>,
) -> Result<Annotated<PythonPayload>, String> {
    let bound = item.bind(py);
    let Some(dict) = bound.downcast::<PyDict>().ok() else {
        return Ok(Annotated::from_data(PythonPayload(item)));
    };
    let is_envelope = dict
        .get_item(pyo3::intern!(py, "_dynamo_annotated"))
        .map_err(|error| error.to_string())?
        .and_then(|value| value.is_truthy().ok())
        .unwrap_or(false);
    if !is_envelope {
        return Ok(Annotated::from_data(PythonPayload(item)));
    }

    // Keep the payload itself as the original Python object. Fully
    // depythonizing `Annotated<PythonPayload>` would rebuild the nested data
    // subtree and defeat the direct request-plane path's ownership reuse.
    // Intern the fixed envelope keys: converting an `&str` for every lookup
    // otherwise creates and hashes a temporary Python string for every frame.
    let data =
        optional_item(dict, pyo3::intern!(py, "data"))?.map(|value| PythonPayload(value.unbind()));
    let id = extract_optional(dict, pyo3::intern!(py, "id"))?;
    let event = extract_optional(dict, pyo3::intern!(py, "event"))?;
    let comment = extract_optional(dict, pyo3::intern!(py, "comment"))?;
    let error = optional_item(dict, pyo3::intern!(py, "error"))?
        .map(|value| depythonize(&value).map_err(|error| error.to_string()))
        .transpose()?;

    Ok(Annotated {
        data,
        id,
        event,
        comment,
        error,
    })
}

fn optional_item<'py>(
    dict: &Bound<'py, PyDict>,
    name: &Bound<'py, PyString>,
) -> Result<Option<Bound<'py, PyAny>>, String> {
    dict.get_item(name)
        .map_err(|error| error.to_string())
        .map(|value| value.filter(|value| !value.is_none()))
}

fn extract_optional<'py, T>(
    dict: &Bound<'py, PyDict>,
    name: &Bound<'py, PyString>,
) -> Result<Option<T>, String>
where
    T: FromPyObject<'py>,
{
    optional_item(dict, name)?
        .map(|value| value.extract().map_err(|error| error.to_string()))
        .transpose()
}

#[cfg(test)]
mod tests {
    // Keep Rust unit tests here free of Python C API calls. This crate uses
    // PyO3's `extension-module` feature, so standalone `cargo test` binaries
    // intentionally do not link libpython. Python behavior is covered by
    // tests/test_request_plane_python_payload.py against the built extension.
    #[test]
    fn network_ingress_types_do_not_contain_serde_json_value() {
        let unary = std::any::type_name::<crate::PythonServerStreamingIngress>();
        let bidirectional = std::any::type_name::<crate::PythonBidirectionalIngress>();
        assert!(!unary.contains("serde_json::value::Value"), "{unary}");
        assert!(
            !bidirectional.contains("serde_json::value::Value"),
            "{bidirectional}"
        );
    }
}
