// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use bytes::Bytes;
use dynamo_runtime::error::DynamoError;
use dynamo_runtime::pipeline::PipelineError;
use dynamo_runtime::pipeline::network::{
    EncodedResponseFrame, IngressRequestDecoder, IngressResponseEncoder, NetworkStreamWrapper,
    RequestPlanePayloadCodec,
};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::protocols::maybe_error::MaybeError;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::PyDict;
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

// These compatibility implementations retain the bounds used to distinguish
// unary from bidirectional ingress blanket implementations. The network path
// always serializes this type through `PythonIngressPayloadAdapter`.
impl Serialize for PythonResponseItem {
    fn serialize<S>(&self, _serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Err(S::Error::custom(
            "PythonResponseItem must be serialized by PythonIngressPayloadAdapter",
        ))
    }
}

impl MaybeError for PythonResponseItem {
    fn from_err(error: impl std::error::Error + 'static) -> Self {
        Self(Err(PyRuntimeError::new_err(error.to_string())))
    }

    fn err(&self) -> Option<DynamoError> {
        // The direct adapter must see Python exceptions so it can classify and
        // serialize them in the same blocking operation as the response frame.
        // Returning an error here would make the generic pipeline terminate the
        // stream before `encode_response` receives the item.
        None
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
        tokio::task::spawn_blocking(move || {
            encode_python_response(payload_codec, response, complete_final)
        })
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
        .get_item("_dynamo_annotated")
        .map_err(|error| error.to_string())?
        .and_then(|value| value.is_truthy().ok())
        .unwrap_or(false);
    if !is_envelope {
        return Ok(Annotated::from_data(PythonPayload(item)));
    }

    let data = dict
        .get_item("data")
        .map_err(|error| error.to_string())?
        .map(|value| PythonPayload(value.unbind()));
    let id = extract_optional(dict, "id")?;
    let event = extract_optional(dict, "event")?;
    let comment = extract_optional(dict, "comment")?;
    let error = dict
        .get_item("error")
        .map_err(|error| error.to_string())?
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

fn extract_optional<'py, T>(dict: &Bound<'py, PyDict>, name: &str) -> Result<Option<T>, String>
where
    T: FromPyObject<'py>,
{
    dict.get_item(name)
        .map_err(|error| error.to_string())?
        .map(|value| value.extract().map_err(|error| error.to_string()))
        .transpose()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pyo3::exceptions::PyValueError;
    use serde_json::json;

    fn nested_value() -> serde_json::Value {
        json!({
            "request": {
                "prompt": "hello \u{4e2d}",
                "tokens": [1, 2, 65535],
                "options": {"stream": true, "temperature": 0.25}
            },
            "nullable": null
        })
    }

    async fn decode_request(
        codec: RequestPlanePayloadCodec,
        value: &serde_json::Value,
    ) -> PythonPayload {
        let bytes = codec.encode(value).expect("request should encode");
        PythonIngressPayloadAdapter
            .decode_request(codec, bytes.into())
            .await
            .expect("request should decode directly to Python")
    }

    fn assert_python_value(payload: &PythonPayload, expected: &serde_json::Value) {
        Python::with_gil(|py| {
            let actual: serde_json::Value =
                depythonize(payload.0.bind(py)).expect("Python payload should be JSON-compatible");
            assert_eq!(&actual, expected);
        });
    }

    fn decode_response(
        codec: RequestPlanePayloadCodec,
        bytes: &[u8],
    ) -> NetworkStreamWrapper<Annotated<serde_json::Value>> {
        codec
            .decode(bytes)
            .expect("encoded Python response should decode")
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn direct_request_and_plain_response_round_trip_both_codecs() {
        for codec in [
            RequestPlanePayloadCodec::Json,
            RequestPlanePayloadCodec::Msgpack,
        ] {
            let expected = nested_value();
            let payload = decode_request(codec, &expected).await;
            assert_python_value(&payload, &expected);

            let encoded = PythonIngressPayloadAdapter
                .encode_response(
                    codec,
                    Some(PythonResponseItem::new(Ok(payload.into_inner()))),
                    false,
                )
                .await
                .expect("plain Python response should encode");
            assert!(!encoded.is_error);
            assert!(!encoded.stop_stream);
            let wrapper = decode_response(codec, &encoded.bytes);
            assert!(!wrapper.complete_final);
            assert_eq!(wrapper.data.and_then(|data| data.data), Some(expected));
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn annotated_response_preserves_metadata_both_codecs() {
        for codec in [
            RequestPlanePayloadCodec::Json,
            RequestPlanePayloadCodec::Msgpack,
        ] {
            let item = Python::with_gil(|py| {
                let envelope = PyDict::new(py);
                envelope.set_item("_dynamo_annotated", true).unwrap();
                let data = pythonize::pythonize(py, &nested_value()).unwrap();
                envelope.set_item("data", data).unwrap();
                envelope.set_item("id", "chunk-7").unwrap();
                envelope.set_item("event", "delta").unwrap();
                envelope
                    .set_item("comment", vec!["first", "second"])
                    .unwrap();
                envelope.into_any().unbind()
            });
            let encoded = PythonIngressPayloadAdapter
                .encode_response(codec, Some(PythonResponseItem::new(Ok(item))), false)
                .await
                .expect("annotated Python response should encode");
            let annotated = decode_response(codec, &encoded.bytes)
                .data
                .expect("response data should be present");
            assert_eq!(annotated.data, Some(nested_value()));
            assert_eq!(annotated.id.as_deref(), Some("chunk-7"));
            assert_eq!(annotated.event.as_deref(), Some("delta"));
            assert_eq!(
                annotated.comment,
                Some(vec!["first".into(), "second".into()])
            );
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn python_exception_and_malformed_output_become_terminal_errors() {
        for codec in [
            RequestPlanePayloadCodec::Json,
            RequestPlanePayloadCodec::Msgpack,
        ] {
            let exception = PythonResponseItem::new(Err(PyValueError::new_err("bad request")));
            let encoded = PythonIngressPayloadAdapter
                .encode_response(codec, Some(exception), false)
                .await
                .expect("Python exception should encode as an error frame");
            assert!(encoded.is_error);
            assert!(encoded.stop_stream);
            let annotated = decode_response(codec, &encoded.bytes).data.unwrap();
            assert_eq!(annotated.event.as_deref(), Some("error"));
            assert!(
                annotated
                    .error
                    .expect("typed error should be present")
                    .to_string()
                    .contains("bad request")
            );

            let malformed = Python::with_gil(|py| {
                py.import("builtins")
                    .unwrap()
                    .getattr("object")
                    .unwrap()
                    .call0()
                    .unwrap()
                    .unbind()
            });
            let encoded = PythonIngressPayloadAdapter
                .encode_response(codec, Some(PythonResponseItem::new(Ok(malformed))), false)
                .await
                .expect("malformed output should encode as a fallback error");
            assert!(encoded.is_error);
            assert!(encoded.stop_stream);
            let annotated = decode_response(codec, &encoded.bytes).data.unwrap();
            assert_eq!(annotated.event.as_deref(), Some("error"));
        }
    }

    #[tokio::test(flavor = "multi_thread")]
    async fn terminal_frame_round_trips_both_codecs() {
        for codec in [
            RequestPlanePayloadCodec::Json,
            RequestPlanePayloadCodec::Msgpack,
        ] {
            let encoded = PythonIngressPayloadAdapter
                .encode_response(codec, None, true)
                .await
                .expect("terminal frame should encode");
            let wrapper = decode_response(codec, &encoded.bytes);
            assert!(wrapper.complete_final);
            assert!(wrapper.data.is_none());
            assert!(!encoded.is_error);
            assert!(!encoded.stop_stream);
        }
    }

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
