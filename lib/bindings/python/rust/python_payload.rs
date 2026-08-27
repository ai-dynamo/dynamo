// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use bytes::Bytes;
use dynamo_runtime::pipeline::PipelineError;
use dynamo_runtime::pipeline::network::{
    EncodedResponseFrame, IngressRequestDecoder, IngressResponseEncoder, NetworkStreamWrapper,
    RequestPlanePayloadCodec, ResponseFrameKind,
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
                kind: ResponseFrameKind::Data,
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

/// Response encoder for the push egress path (`push_egress.rs`).
///
/// The push path encodes each response all the way to request-plane bytes on
/// the Python thread that produced it, under the GIL that thread already holds
/// (`push_egress::PushFrame`). By the time a frame reaches here there is
/// nothing left to do but forward it: no GIL, no second serde pass, and —
/// unlike the [`PythonResponseItem`] encoder above — no `spawn_blocking` hop.
///
/// The only work left is the terminal complete-final frame, which carries no
/// Python data at all, and the rare codec re-encode described on
/// [`push_egress::PushFrame`].
impl IngressResponseEncoder<crate::push_egress::PushFrame> for PythonIngressPayloadAdapter {
    async fn encode_response(
        &self,
        payload_codec: RequestPlanePayloadCodec,
        response: Option<crate::push_egress::PushFrame>,
        complete_final: bool,
    ) -> Result<EncodedResponseFrame, PipelineError> {
        if complete_final {
            let wrapper = NetworkStreamWrapper::<Annotated<()>> {
                data: None,
                complete_final: true,
            };
            let bytes = payload_codec.encode(&wrapper).map_err(|error| {
                PipelineError::SerializationError(format!(
                    "Failed serializing {} push-egress final response: {error}",
                    payload_codec.name()
                ))
            })?;
            return Ok(EncodedResponseFrame {
                bytes: bytes.into(),
                kind: ResponseFrameKind::Data,
                stop_stream: false,
            });
        }

        let frame = response.ok_or_else(|| {
            PipelineError::SerializationError(
                "push-egress response item missing before final frame".to_string(),
            )
        })?;
        frame.into_encoded(payload_codec)
    }
}

/// Convert an `Annotated` value into request-plane bytes and its frame kind,
/// using the canonical non-terminal wrapper shape (`complete_final: false`).
///
/// Both the pull path ([`encode_python_response`]) and the push path
/// ([`crate::push_egress::PushFrame::encode`]) call this, so neither can silently
/// change the wrapper shape, classification, or codec invocation without also
/// breaking this function — and the tests below that exercise it directly with
/// concrete non-Python types.
///
/// An error envelope whose `error` field is empty carries no type to classify
/// on, so it counts as an engine error rather than being silently excused.
pub(crate) fn encode_annotated_response<T: Serialize>(
    codec: RequestPlanePayloadCodec,
    annotated: Annotated<T>,
) -> Result<(Vec<u8>, ResponseFrameKind), anyhow::Error> {
    let kind = match (annotated.is_error(), annotated.error.as_ref()) {
        (false, _) => ResponseFrameKind::Data,
        (true, Some(err)) => ResponseFrameKind::classify_error(err),
        (true, None) => ResponseFrameKind::EngineError,
    };
    let wrapper = NetworkStreamWrapper {
        data: Some(annotated),
        complete_final: false,
    };
    let bytes = codec.encode(&wrapper)?;
    Ok((bytes, kind))
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

    match encode_annotated_response(payload_codec, annotated) {
        Ok((bytes, kind)) => Ok(EncodedResponseFrame {
            bytes: bytes.into(),
            kind,
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
                kind: ResponseFrameKind::EngineError,
                stop_stream: true,
            })
        }
    }
}

/// Interpret one Python response object as a wire `Annotated<PythonPayload>`.
///
/// Shared by both egress paths — pull via [`encode_python_response`], push via
/// `push_egress::PushFrame::encode` — so the two cannot drift and start
/// disagreeing about what a given Python object means on the wire. The payload
/// stays a [`PythonPayload`], which transcodes straight into the request-plane
/// codec and therefore preserves everything that codec can represent (binary
/// values, non-finite floats, non-string mapping keys). The GIL is already held
/// on both paths.
pub(crate) fn parse_python_response(
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

    use super::{
        Annotated, NetworkStreamWrapper, RequestPlanePayloadCodec, ResponseFrameKind,
        encode_annotated_response,
    };
    use dynamo_runtime::error::{BackendError, DynamoError, ErrorType};
    use dynamo_runtime::protocols::maybe_error::MaybeError;

    // ── encode_annotated_response contract ───────────────────────────────────
    //
    // Both egress paths (pull via encode_python_response, push via
    // PushFrame::encode) call encode_annotated_response. These tests pin every
    // field of the output so that a change to the wrapper shape, frame
    // classification, or complete_final flag in either path would be caught here.
    //
    // serde_json::Value is used as the concrete payload type because it is
    // Serialize without touching the Python C API.

    /// The frame kind must reflect `annotated.is_error()` — an error envelope
    /// carries `event: "error"`, a data envelope does not. A swap of the two
    /// would let error frames be forwarded as healthy responses and vice versa.
    #[test]
    fn encode_annotated_response_is_error_true_for_error_annotated() {
        let (_, kind) = encode_annotated_response(
            RequestPlanePayloadCodec::Json,
            Annotated::<serde_json::Value>::from_error("oops"),
        )
        .unwrap();
        assert!(kind.is_error());
    }

    #[test]
    fn encode_annotated_response_is_error_false_for_data_annotated() {
        let (_, kind) = encode_annotated_response(
            RequestPlanePayloadCodec::Json,
            Annotated::from_data(serde_json::json!({"ok": true})),
        )
        .unwrap();
        assert!(!kind.is_error());
    }

    /// An untyped error frame is an engine failure: nothing about it says the
    /// request was torn down.
    #[test]
    fn encode_annotated_response_classifies_untyped_error_as_engine_error() {
        let (_, kind) = encode_annotated_response(
            RequestPlanePayloadCodec::Json,
            Annotated::<serde_json::Value>::from_error("upstream returned 503"),
        )
        .unwrap();
        assert_eq!(kind, ResponseFrameKind::EngineError);
    }

    /// `map_python_exception` turns `PyGeneratorExit` into this error, which is
    /// how a draining Python worker ends its open streams. It must stay
    /// distinguishable from an engine fault all the way to the wire, or a
    /// rolling restart lights up the mid-stream failure counter.
    #[test]
    fn encode_annotated_response_classifies_engine_shutdown_as_cancellation() {
        let shutdown = DynamoError::builder()
            .error_type(ErrorType::Backend(BackendError::EngineShutdown))
            .message("engine shutting down")
            .build();
        let (_, kind) = encode_annotated_response(
            RequestPlanePayloadCodec::Json,
            Annotated::<serde_json::Value>::from_err(shutdown),
        )
        .unwrap();
        assert_eq!(kind, ResponseFrameKind::Cancellation);
        assert!(
            kind.is_error(),
            "a cancellation frame is still an error frame on the wire"
        );
    }

    /// Cancellation reaches the wire as an ordinary typed error frame too.
    #[test]
    fn encode_annotated_response_classifies_cancelled_as_cancellation() {
        let cancelled = DynamoError::builder()
            .error_type(ErrorType::Cancelled)
            .message("client went away")
            .build();
        let (_, kind) = encode_annotated_response(
            RequestPlanePayloadCodec::Json,
            Annotated::<serde_json::Value>::from_err(cancelled),
        )
        .unwrap();
        assert_eq!(kind, ResponseFrameKind::Cancellation);
    }

    /// Non-terminal frames must have `complete_final: false` on the wire.
    /// A stray `true` would tell the caller the stream has ended even when
    /// the Python generator is still running.
    #[test]
    fn encode_annotated_response_complete_final_is_always_false() {
        for codec in [
            RequestPlanePayloadCodec::Json,
            RequestPlanePayloadCodec::Msgpack,
        ] {
            let (bytes, _) =
                encode_annotated_response(codec, Annotated::from_data(serde_json::json!(null)))
                    .unwrap();
            let wrapper: NetworkStreamWrapper<Annotated<serde_json::Value>> =
                codec.decode(&bytes).unwrap();
            assert!(!wrapper.complete_final, "codec={}", codec.name());
        }
    }

    /// The payload must survive the encode → decode round-trip intact.
    /// A `data: None` in the wrapper or a wrong serde path would drop it.
    #[test]
    fn encode_annotated_response_data_survives_roundtrip() {
        let payload = serde_json::json!({"text": "hello", "n": 42});
        let (bytes, kind) = encode_annotated_response(
            RequestPlanePayloadCodec::Json,
            Annotated::from_data(payload.clone()),
        )
        .unwrap();
        assert!(!kind.is_error());
        let wrapper: NetworkStreamWrapper<Annotated<serde_json::Value>> =
            RequestPlanePayloadCodec::Json.decode(&bytes).unwrap();
        let data = wrapper.data.unwrap().data.unwrap();
        assert_eq!(data, payload);
    }

    /// Encoding is deterministic: the same `Annotated` value with the same
    /// codec must produce byte-identical frames from both egress paths.
    /// Non-determinism would mean the pull and push paths could silently
    /// diverge on map ordering or float representation.

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

    /// The Python `Client` router path carries requests/responses through the
    /// request-plane codec as a dynamic value. It uses `rmpv::Value` (not
    /// `serde_json::Value`) so that a `bytes` field survives the (default)
    /// msgpack codec as a `Binary` marker instead of erroring — which is what
    /// lets workers emit raw bytes instead of base64. This pins that property
    /// at the wire level; the full Python `Client` round-trip is covered by
    /// `tests/test_request_plane_python_payload.py` against the built extension.
    #[test]
    fn rmpv_value_carries_bytes_through_msgpack_request_plane() {
        use super::{Annotated, NetworkStreamWrapper, RequestPlanePayloadCodec};
        let payload = rmpv::Value::Map(vec![
            (
                rmpv::Value::String("img".into()),
                rmpv::Value::Binary(vec![0xDE, 0xAD, 0xBE, 0xEF]),
            ),
            (rmpv::Value::String("n".into()), rmpv::Value::from(7i64)),
        ]);
        let wrapper = NetworkStreamWrapper {
            data: Some(Annotated::from_data(payload)),
            complete_final: false,
        };
        let wire = RequestPlanePayloadCodec::Msgpack
            .encode(&wrapper)
            .expect("encode");
        let back: NetworkStreamWrapper<Annotated<rmpv::Value>> = RequestPlanePayloadCodec::Msgpack
            .decode(&wire)
            .expect("bytes field must not error on decode");
        let data = back.data.expect("data").data.expect("annotated data");
        let img = data
            .as_map()
            .expect("map")
            .iter()
            .find(|(k, _)| k.as_str() == Some("img"))
            .map(|(_, v)| v)
            .expect("img field");
        assert!(
            matches!(img, rmpv::Value::Binary(b) if b == &[0xDE, 0xAD, 0xBE, 0xEF]),
            "msgpack must preserve bytes as Binary, got {img:?}"
        );
    }
}
