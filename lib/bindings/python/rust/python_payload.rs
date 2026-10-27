// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::cell::Cell;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};

use bytes::Bytes;
use dynamo_llm::admission_gate::{AdmissionRequest, AdmissionResponse, ResponseOutput};
use dynamo_runtime::pipeline::PipelineError;
use dynamo_runtime::pipeline::network::{
    EncodedResponseFrame, IngressRequestDecoder, IngressResponseEncoder, NetworkStreamWrapper,
    RESPONSE_ENCODE_CAPACITY_HINT, RequestPlanePayloadCodec,
};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::protocols::maybe_error::MaybeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyString};
use pythonize::{Depythonizer, Pythonizer, depythonize};
use serde::de::{DeserializeOwned, Error as _};
use serde::ser::Error as _;
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::engine::map_python_exception;

mod dp_rank;

use dp_rank::DpRankProbe;

/// Python-owned value used only by the network ingress fast path: a unary
/// request, a bidirectional request frame, or a response's data. Serde events
/// are transcoded directly to or from the Python object without an
/// intermediate Rust value tree, and serializing emits that object alone.
///
/// Decoding also records the request-plane codec the value arrived in, which a
/// unary request's responses are encoded with whatever rank it names, and,
/// for a unary request alone, the DP rank its body names in `routing.dp_rank`,
/// read during that same transcoding, so backend admission reads it without
/// touching Python. The handler receives the object unchanged either way. Only
/// a unary request's metadata is ever read: a bidirectional frame is never
/// admitted, so it is decoded without the rank probe and names no rank, and a
/// value built from a handler's response keeps the defaults.
#[derive(Clone)]
pub(crate) struct PythonPayload {
    object: Py<PyAny>,
    dp_rank: Option<u32>,
    payload_codec: RequestPlanePayloadCodec,
}

impl PythonPayload {
    /// A handler's Python object, with default metadata.
    fn new(object: Py<PyAny>) -> Self {
        Self {
            object,
            dp_rank: None,
            payload_codec: RequestPlanePayloadCodec::default(),
        }
    }

    pub(crate) fn into_inner(self) -> Py<PyAny> {
        self.object
    }

    /// The codec this request arrived in, which the ingress encodes its
    /// responses with.
    pub(crate) fn payload_codec(&self) -> RequestPlanePayloadCodec {
        self.payload_codec
    }
}

impl std::fmt::Debug for PythonPayload {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PythonPayload")
            .field("object", &format_args!("<PyAny>"))
            .field("dp_rank", &self.dp_rank)
            .field("payload_codec", &self.payload_codec.name())
            .finish()
    }
}

impl<'de> Deserialize<'de> for PythonPayload {
    /// A unary request body: transcode straight into the Python object, through
    /// a probe that records the body's rank as it passes, in the same pass and
    /// under the same GIL acquisition.
    ///
    /// A deserializer does not name its format, so this assumes the default
    /// codec; the request-plane decoder replaces it with the codec the request
    /// actually arrived in.
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let dp_rank = Cell::new(None);
        let object = Python::with_gil(|py| {
            serde_transcode::transcode(
                DpRankProbe::body(deserializer, &dp_rank),
                Pythonizer::new(py),
            )
            .map(Bound::unbind)
            .map_err(D::Error::custom)
        })?;
        Ok(Self {
            dp_rank: dp_rank.get(),
            ..Self::new(object)
        })
    }
}

/// A bidirectional request frame, decoded into a [`PythonPayload`] by the same
/// transcoding as a unary request but without the rank probe: no bidirectional
/// frame is ever admitted, so nothing would read its rank.
struct BidirectionalFrame(PythonPayload);

impl<'de> Deserialize<'de> for BidirectionalFrame {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let object = Python::with_gil(|py| {
            serde_transcode::transcode(deserializer, Pythonizer::new(py))
                .map(Bound::unbind)
                .map_err(D::Error::custom)
        })?;
        Ok(Self(PythonPayload::new(object)))
    }
}

impl Serialize for PythonPayload {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        Python::with_gil(|py| {
            let mut depythonizer = Depythonizer::from_object(self.object.bind(py));
            serde_transcode::transcode(&mut depythonizer, serializer).map_err(S::Error::custom)
        })
    }
}

impl AdmissionRequest for PythonPayload {
    fn dp_rank(&self) -> Option<u32> {
        self.dp_rank
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

/// Payload adapter for the unary ingresses, pull and push: its requests may be
/// admitted, so their rank is read as they are decoded.
#[derive(Debug, Default)]
pub(crate) struct PythonIngressPayloadAdapter;

/// Payload adapter for the bidirectional ingress, which backend admission never
/// sees: its request frames are decoded without the rank probe, and its
/// responses encoded without classifying them.
#[derive(Debug, Default)]
pub(crate) struct PythonBidirectionalPayloadAdapter;

/// Decode one request body, or bidirectional frame, as `T` off the async
/// runtime, since building its Python object takes the GIL.
async fn decode_python_request<T>(
    payload_codec: RequestPlanePayloadCodec,
    bytes: Bytes,
) -> Result<T, PipelineError>
where
    T: DeserializeOwned + Send + 'static,
{
    tokio::task::spawn_blocking(move || payload_codec.decode::<T>(&bytes))
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

/// Unary requests, decoded into a Python object that records the rank its body
/// names and the codec it arrived in.
impl IngressRequestDecoder<PythonPayload> for PythonIngressPayloadAdapter {
    async fn decode_request(
        &self,
        payload_codec: RequestPlanePayloadCodec,
        bytes: Bytes,
    ) -> Result<PythonPayload, PipelineError> {
        let mut payload = decode_python_request::<PythonPayload>(payload_codec, bytes).await?;
        // Unconditionally: a request's responses go back in the codec it arrived
        // in, whether its body names a usable rank or none.
        payload.payload_codec = payload_codec;
        Ok(payload)
    }
}

/// Bidirectional request frames, decoded into the same Python object without
/// the rank probe; each records the codec it arrived in.
impl IngressRequestDecoder<PythonPayload> for PythonBidirectionalPayloadAdapter {
    async fn decode_request(
        &self,
        payload_codec: RequestPlanePayloadCodec,
        bytes: Bytes,
    ) -> Result<PythonPayload, PipelineError> {
        let BidirectionalFrame(mut payload) =
            decode_python_request::<BidirectionalFrame>(payload_codec, bytes).await?;
        payload.payload_codec = payload_codec;
        Ok(payload)
    }
}

/// Response encoder for the bidirectional path, whose stream yields the raw
/// Python items. They are never admitted, so never classified either.
impl IngressResponseEncoder<PythonResponseItem> for PythonBidirectionalPayloadAdapter {
    async fn encode_response(
        &self,
        payload_codec: RequestPlanePayloadCodec,
        response: Option<PythonResponseItem>,
        complete_final: bool,
    ) -> Result<EncodedResponseFrame, PipelineError> {
        if complete_final {
            return Ok(EncodedResponseFrame {
                bytes: terminal_frame_bytes(payload_codec)?,
                is_error: false,
                stop_stream: false,
            });
        }

        let response = response.ok_or_else(|| {
            PipelineError::SerializationError(
                "request-plane response item missing before final frame".to_string(),
            )
        })?;
        encode_python_item(payload_codec, response, false)
            .await
            .frame
    }
}

/// One unary Python response, encoded for the request plane by the response
/// stream that produced it, with what backend admission reads from it.
///
/// The unary pull path encodes each item before it leaves the engine's stream,
/// so the classification exists before anything downstream — the backend
/// admission gate first — sees the item, with no second GIL acquisition: it is
/// read during the same parse that the encoding needs anyway, and only while
/// the gate still reads it (see [`WaitClassification`]). The frame is exactly
/// what the ingress encoder used to produce for the raw item, error and stop
/// flags included, under the codec the request arrived in.
pub(crate) struct EncodedPythonResponse {
    frame: Result<EncodedResponseFrame, PipelineError>,
    output: ResponseOutput,
}

impl std::fmt::Debug for EncodedPythonResponse {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut debug = f.debug_struct("EncodedPythonResponse");
        match &self.frame {
            Ok(frame) => debug
                .field("len", &frame.bytes.len())
                .field("is_error", &frame.is_error)
                .field("stop_stream", &frame.stop_stream),
            Err(error) => debug.field("error", error),
        };
        debug.field("output", &self.output).finish()
    }
}

impl AdmissionResponse for EncodedPythonResponse {
    fn admission_output(&self) -> ResponseOutput {
        self.output
    }
}

/// Encode one raw Python item off the async runtime, since parsing and
/// serializing it take the GIL, classifying it for backend admission only when
/// `classify` says the gate still reads it.
pub(crate) async fn encode_python_item(
    payload_codec: RequestPlanePayloadCodec,
    item: PythonResponseItem,
    classify: bool,
) -> EncodedPythonResponse {
    tokio::task::spawn_blocking(move || {
        encode_python_response(payload_codec, item.into_result(), classify)
    })
    .await
    .unwrap_or_else(|error| EncodedPythonResponse {
        frame: Err(PipelineError::SerializationError(format!(
            "failed to offload {} Python response encode: {error}",
            payload_codec.name()
        ))),
        output: ResponseOutput::INTERIM,
    })
}

/// Which of one request's responses backend admission still reads the
/// classification of: each, in stream order, up to and including the first
/// that ends the request's wait by the gate's own rule,
/// [`ResponseOutput::ends_wait`]. Once that one is in the stream, the gate
/// reads no later classification, so the producer stops classifying; a request
/// that does not use the gate — on an endpoint that is no model backend — has
/// none of its responses classified. Only the gate releases capacity, as it
/// observes the decisive response: noting one here releases nothing.
///
/// The decisive response carries its own classification to the gate, however
/// far ahead of the gate the producer runs, and only a response the stream
/// accepted counts: one that failed to encode or to send never reaches the
/// gate, so classification goes on for the next. Its state is a flag rather
/// than a lock, as the push path consults it on the handler's thread, where
/// encoding can run Python that re-enters `send`.
pub(crate) struct WaitClassification {
    /// Whether the request uses the gate.
    uses_admission: bool,
    /// Set once the stream has accepted a response that ends the wait.
    ended: AtomicBool,
}

impl WaitClassification {
    pub(crate) fn new(uses_admission: bool) -> Self {
        Self {
            uses_admission,
            ended: AtomicBool::new(false),
        }
    }

    /// Whether the next response must be classified. Decided before it is
    /// encoded: a response goes unclassified only behind, in stream order, one
    /// the stream has already accepted that ended the wait.
    pub(crate) fn classifies(&self) -> bool {
        self.uses_admission && !self.ended.load(Ordering::Acquire)
    }

    /// Note that the stream accepted, in order, a response classified as
    /// `output`.
    pub(crate) fn accepted(&self, output: ResponseOutput) {
        if output.ends_wait() {
            self.ended.store(true, Ordering::Release);
        }
    }
}

/// The unary pull path's items are already encoded: forward them.
impl IngressResponseEncoder<EncodedPythonResponse> for PythonIngressPayloadAdapter {
    async fn encode_response(
        &self,
        payload_codec: RequestPlanePayloadCodec,
        response: Option<EncodedPythonResponse>,
        complete_final: bool,
    ) -> Result<EncodedResponseFrame, PipelineError> {
        if complete_final {
            return Ok(EncodedResponseFrame {
                bytes: terminal_frame_bytes(payload_codec)?,
                is_error: false,
                stop_stream: false,
            });
        }

        response
            .ok_or_else(|| {
                PipelineError::SerializationError(
                    "request-plane response item missing before final frame".to_string(),
                )
            })?
            .frame
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
            return Ok(EncodedResponseFrame {
                bytes: terminal_frame_bytes(payload_codec)?,
                is_error: false,
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

/// Convert an `Annotated` value into request-plane bytes and the `is_error` flag,
/// using the canonical non-terminal wrapper shape (`complete_final: false`).
///
/// Both the pull path ([`encode_python_response`]) and the push path
/// ([`crate::push_egress::PushFrame::encode`]) call this, so neither can silently
/// change the wrapper shape, `is_error` logic, or codec invocation without also
/// breaking this function — and the tests below that exercise it directly with
/// concrete non-Python types.
pub(crate) fn encode_annotated_response<T: Serialize>(
    codec: RequestPlanePayloadCodec,
    annotated: Annotated<T>,
) -> Result<(Vec<u8>, bool), anyhow::Error> {
    // `with_capacity`, not `new`: starting from zero would regress JSON
    // responses to pay the reallocations this change exists to remove.
    let mut bytes = Vec::with_capacity(RESPONSE_ENCODE_CAPACITY_HINT);
    let is_error = write_annotated_response(codec, annotated, &mut bytes)?;
    Ok((bytes, is_error))
}

/// Encode the canonical non-terminal wrapper into a caller-owned writer, so the
/// push path can reuse one allocation across a request's frames.
///
/// The wrapper shape is defined here and nowhere else; `encode_annotated_response`
/// delegates to it, so the two cannot disagree.
pub(crate) fn write_annotated_response<T: Serialize, W: std::io::Write>(
    codec: RequestPlanePayloadCodec,
    annotated: Annotated<T>,
    writer: &mut W,
) -> Result<bool, anyhow::Error> {
    let is_error = annotated.is_error();
    let wrapper = NetworkStreamWrapper {
        data: Some(annotated),
        complete_final: false,
    };
    codec.encode_into(&wrapper, writer)?;
    Ok(is_error)
}

/// Memoized separately per codec, since encoding is codec-specific.
fn terminal_frame_bytes(codec: RequestPlanePayloadCodec) -> Result<Bytes, PipelineError> {
    static JSON: OnceLock<Bytes> = OnceLock::new();
    static MSGPACK: OnceLock<Bytes> = OnceLock::new();

    let cell = match codec {
        RequestPlanePayloadCodec::Json => &JSON,
        RequestPlanePayloadCodec::Msgpack => &MSGPACK,
    };
    if let Some(bytes) = cell.get() {
        return Ok(bytes.clone());
    }

    let wrapper = NetworkStreamWrapper::<Annotated<()>> {
        data: None,
        complete_final: true,
    };
    let bytes: Bytes = codec
        .encode(&wrapper)
        .map_err(|error| {
            PipelineError::SerializationError(format!(
                "Failed serializing {} request-plane final response: {error}",
                codec.name()
            ))
        })?
        .into();
    Ok(cell.get_or_init(|| bytes).clone())
}

fn encode_python_response(
    payload_codec: RequestPlanePayloadCodec,
    item: PyResult<Py<PyAny>>,
    classify: bool,
) -> EncodedPythonResponse {
    let (annotated, stop_stream, output) = match item {
        Ok(item) => match Python::with_gil(|py| {
            parse_python_response(item, py).map(|annotated| {
                let output = if classify {
                    response_output(&annotated, py)
                } else {
                    ResponseOutput::INTERIM
                };
                (annotated, output)
            })
        }) {
            Ok((annotated, output)) => (annotated, false, output),
            Err(error) => (
                Annotated::from_error(format!(
                    "critical error: invalid response object from Python async generator; \
                     application-logic-mismatch: {error}"
                )),
                true,
                ResponseOutput::INTERIM,
            ),
        },
        Err(error) => (
            Annotated::from_err(map_python_exception(error)),
            true,
            ResponseOutput::INTERIM,
        ),
    };

    match encode_annotated_response(payload_codec, annotated) {
        Ok((bytes, is_error)) => EncodedPythonResponse {
            frame: Ok(EncodedResponseFrame {
                bytes: bytes.into(),
                is_error,
                stop_stream,
            }),
            output,
        },
        Err(error) => {
            // What reaches the caller is an error, whatever the object held.
            let fallback = NetworkStreamWrapper {
                data: Some(Annotated::<()>::from_error(format!(
                    "critical error: failed serializing Python response as {}: {error}",
                    payload_codec.name()
                ))),
                complete_final: false,
            };
            let frame = payload_codec
                .encode(&fallback)
                .map(|bytes| EncodedResponseFrame {
                    bytes: bytes.into(),
                    is_error: true,
                    stop_stream: true,
                })
                .map_err(|fallback_error| {
                    PipelineError::SerializationError(format!(
                        "failed to serialize Python response and fallback error as {}: \
                         {fallback_error}",
                        payload_codec.name()
                    ))
                });
            EncodedPythonResponse {
                frame,
                output: ResponseOutput::INTERIM,
            }
        }
    }
}

/// What one parsed Python response carries for backend admission, read under
/// the GIL its caller already holds for the parse: the backend contract's own
/// fields, never a second pass over the payload. Called only while the gate
/// still reads the classification (see [`WaitClassification`]).
///
/// A generated token is a non-empty `token_ids`, or a text-mode chunk's
/// generated text (see [`choices_output`]); an empty one, or none, is not. The
/// completion response is the one carrying a `finish_reason`, at its top level
/// or in a text-mode chunk's choice, or — a vLLM prefill's handoff result,
/// which reports no finish reason — the prefill handoff carrying
/// `disaggregated_params` that reports its `completion_usage`, which vLLM
/// builds only once the prefill has run. Usage alone completes nothing. An
/// error, an annotation without data, a payload that is not a mapping, and
/// every other response is interim: a disaggregation bootstrap announcement,
/// which SGLang's prefill sends before its prefill runs, carries neither.
pub(crate) fn response_output(
    annotated: &Annotated<PythonPayload>,
    py: Python<'_>,
) -> ResponseOutput {
    let data = match &annotated.data {
        Some(data) if !annotated.is_error() => data,
        _ => return ResponseOutput::INTERIM,
    };
    let Ok(response) = data.object.bind(py).downcast::<PyDict>() else {
        return ResponseOutput::INTERIM;
    };
    let generated_token = non_null_item(response, pyo3::intern!(py, "token_ids"))
        .and_then(|token_ids| token_ids.len().ok())
        .is_some_and(|count| count > 0);
    let prefill_handoff_usage = non_null_item(response, pyo3::intern!(py, "completion_usage"))
        .is_some()
        && response
            .contains(pyo3::intern!(py, "disaggregated_params"))
            .unwrap_or(false);
    let text = choices_output(response, py);
    ResponseOutput {
        generated_token: generated_token || text.generated_token,
        completion: non_null_item(response, pyo3::intern!(py, "finish_reason")).is_some()
            || prefill_handoff_usage
            || text.completion,
    }
}

/// What a text-mode chunk's `choices` carry, read as an OpenAI chat completion
/// chunk's: generated text in any choice's non-empty string `delta.content`,
/// and the completion in any choice's non-null `finish_reason`. A role-only or
/// empty delta carries no generated text, and a part not of that shape carries
/// nothing.
fn choices_output(response: &Bound<'_, PyDict>, py: Python<'_>) -> ResponseOutput {
    let mut output = ResponseOutput::INTERIM;
    let Some(choices) = non_null_item(response, pyo3::intern!(py, "choices")) else {
        return output;
    };
    let Ok(choices) = choices.downcast::<PyList>() else {
        return output;
    };
    for choice in choices.iter() {
        let Ok(choice) = choice.downcast::<PyDict>() else {
            continue;
        };
        output.completion |= non_null_item(choice, pyo3::intern!(py, "finish_reason")).is_some();
        output.generated_token |= non_null_item(choice, pyo3::intern!(py, "delta"))
            .and_then(|delta| {
                let delta = delta.downcast::<PyDict>().ok()?;
                non_null_item(delta, pyo3::intern!(py, "content"))
            })
            .is_some_and(|content| {
                content.downcast::<PyString>().is_ok() && content.len().is_ok_and(|len| len > 0)
            });
    }
    output
}

/// The value at `key`, when present and not `None`.
fn non_null_item<'py>(
    dict: &Bound<'py, PyDict>,
    key: &Bound<'py, PyString>,
) -> Option<Bound<'py, PyAny>> {
    dict.get_item(key)
        .ok()
        .flatten()
        .filter(|value| !value.is_none())
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
        return Ok(Annotated::from_data(PythonPayload::new(item)));
    };
    let is_envelope = dict
        .get_item(pyo3::intern!(py, "_dynamo_annotated"))
        .map_err(|error| error.to_string())?
        .and_then(|value| value.is_truthy().ok())
        .unwrap_or(false);
    if !is_envelope {
        return Ok(Annotated::from_data(PythonPayload::new(item)));
    }

    // Keep the payload itself as the original Python object. Fully
    // depythonizing `Annotated<PythonPayload>` would rebuild the nested data
    // subtree and defeat the direct request-plane path's ownership reuse.
    // Intern the fixed envelope keys: converting an `&str` for every lookup
    // otherwise creates and hashes a temporary Python string for every frame.
    let data = optional_item(dict, pyo3::intern!(py, "data"))?
        .map(|value| PythonPayload::new(value.unbind()));
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
        Annotated, NetworkStreamWrapper, RequestPlanePayloadCodec, encode_annotated_response,
        terminal_frame_bytes,
    };

    /// Each codec's terminal frame must decode back to `data: None,
    /// complete_final: true` — the contract both egress paths rely on to
    /// signal end-of-stream.
    #[test]
    fn terminal_frame_bytes_decodes_to_complete_final() {
        for codec in [
            RequestPlanePayloadCodec::Json,
            RequestPlanePayloadCodec::Msgpack,
        ] {
            let bytes = terminal_frame_bytes(codec).unwrap();
            let wrapper: NetworkStreamWrapper<Annotated<serde_json::Value>> =
                codec.decode(&bytes).unwrap();
            assert!(wrapper.complete_final, "codec={}", codec.name());
            assert!(wrapper.data.is_none(), "codec={}", codec.name());
        }
    }

    /// Two calls for the same codec must return the same cached `Bytes`
    /// storage, not just equal contents. `assert_eq!` alone would still pass
    /// if a bug re-encoded a byte-identical buffer on every call; comparing
    /// `as_ptr()` is what actually pins the memoization.
    #[test]
    fn terminal_frame_bytes_is_memoized_per_codec() {
        for codec in [
            RequestPlanePayloadCodec::Json,
            RequestPlanePayloadCodec::Msgpack,
        ] {
            let first = terminal_frame_bytes(codec).unwrap();
            let second = terminal_frame_bytes(codec).unwrap();
            assert!(!first.is_empty(), "codec={}", codec.name());
            assert_eq!(first, second, "codec={}", codec.name());
            assert_eq!(
                first.as_ptr(),
                second.as_ptr(),
                "codec={} must reuse the cached Bytes storage",
                codec.name()
            );
        }
    }

    // ── encode_annotated_response contract ───────────────────────────────────
    //
    // Both egress paths (pull via encode_python_response, push via
    // PushFrame::encode) call encode_annotated_response. These tests pin every
    // field of the output so that a change to the wrapper shape, is_error
    // logic, or complete_final flag in either path would be caught here.
    //
    // serde_json::Value is used as the concrete payload type because it is
    // Serialize without touching the Python C API.

    /// `is_error` must reflect `annotated.is_error()` — true when the envelope
    /// carries `event: "error"`, false otherwise. A swap of the two would let
    /// error frames be forwarded as healthy responses and vice versa.
    #[test]
    fn encode_annotated_response_is_error_true_for_error_annotated() {
        let (_, is_error) = encode_annotated_response(
            RequestPlanePayloadCodec::Json,
            Annotated::<serde_json::Value>::from_error("oops"),
        )
        .unwrap();
        assert!(is_error);
    }

    #[test]
    fn encode_annotated_response_is_error_false_for_data_annotated() {
        let (_, is_error) = encode_annotated_response(
            RequestPlanePayloadCodec::Json,
            Annotated::from_data(serde_json::json!({"ok": true})),
        )
        .unwrap();
        assert!(!is_error);
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
        let (bytes, is_error) = encode_annotated_response(
            RequestPlanePayloadCodec::Json,
            Annotated::from_data(payload.clone()),
        )
        .unwrap();
        assert!(!is_error);
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

    /// Bidirectional frames are never admitted, so the bidirectional ingress
    /// has its own payload adapter, which decodes without the rank probe and
    /// encodes without classifying; both unary ingresses keep the adapter that
    /// reads a request's rank.
    #[test]
    fn only_the_unary_ingresses_use_the_admission_adapter() {
        let bidirectional = std::any::type_name::<crate::PythonBidirectionalIngress>();
        assert!(
            bidirectional.ends_with("PythonBidirectionalPayloadAdapter>"),
            "{bidirectional}"
        );
        for unary in [
            std::any::type_name::<crate::PythonServerStreamingIngress>(),
            std::any::type_name::<crate::PythonPushEgressIngress>(),
        ] {
            assert!(unary.ends_with("PythonIngressPayloadAdapter>"), "{unary}");
        }
    }

    // ── WaitClassification ───────────────────────────────────────────────────
    //
    // Which responses the pull stream and the push sink classify. Driven here
    // the way both drive it: `classifies` before a response is encoded,
    // `accepted` once the stream has it.

    use super::WaitClassification;
    use dynamo_llm::admission_gate::ResponseOutput;

    const INTERIM: ResponseOutput = ResponseOutput::INTERIM;
    const TOKEN: ResponseOutput = ResponseOutput {
        generated_token: true,
        completion: false,
    };
    const COMPLETION: ResponseOutput = ResponseOutput {
        generated_token: false,
        completion: true,
    };
    const LAST_TOKEN: ResponseOutput = ResponseOutput {
        generated_token: true,
        completion: true,
    };

    /// Whether each response of `outputs` is classified, accepting each in
    /// turn as the stream would.
    fn classified(classification: &WaitClassification, outputs: &[ResponseOutput]) -> Vec<bool> {
        outputs
            .iter()
            .map(|&output| {
                let classifies = classification.classifies();
                classification.accepted(if classifies { output } else { INTERIM });
                classifies
            })
            .collect()
    }

    /// A request that does not use the gate — on an endpoint that is no model
    /// backend, or a bidirectional one — has none of its responses classified,
    /// whatever they carry.
    #[test]
    fn a_request_not_using_the_gate_classifies_nothing() {
        let classification = WaitClassification::new(false);
        assert_eq!(
            classified(
                &classification,
                &[INTERIM, TOKEN, COMPLETION, LAST_TOKEN, TOKEN]
            ),
            [false; 5]
        );
    }

    /// A request that uses the gate classifies every response up to and
    /// including the first that carries a token or is the completion, and none
    /// after it.
    #[test]
    fn a_request_using_the_gate_classifies_through_its_first_token_or_completion() {
        for (outputs, expected) in [
            // A token first, as a decode's, or a prefill's before its completion.
            (
                vec![INTERIM, INTERIM, TOKEN, COMPLETION, TOKEN],
                vec![true, true, true, false, false],
            ),
            // A completion without a token first.
            (
                vec![INTERIM, COMPLETION, TOKEN, LAST_TOKEN],
                vec![true, true, false, false],
            ),
            // A completion carrying the last token.
            (
                vec![LAST_TOKEN, TOKEN, COMPLETION],
                vec![true, false, false],
            ),
        ] {
            let classification = WaitClassification::new(true);
            assert_eq!(
                classified(&classification, &outputs),
                expected,
                "{outputs:?}"
            );
        }
    }

    /// A response that failed to encode or to send never reaches the gate, so
    /// it is never accepted, and classification goes on: the valid token sent
    /// after it still ends the wait.
    #[test]
    fn a_response_never_accepted_leaves_classification_on() {
        let classification = WaitClassification::new(true);
        assert!(
            classification.classifies(),
            "the failing token is classified"
        );
        // ...as a token, then fails: `accepted` is never called for it.
        assert!(classification.classifies(), "so the next one is classified");
        classification.accepted(TOKEN);
        assert!(!classification.classifies());
    }

    /// A push handler's encode can run Python that re-enters `send`. Each
    /// send decides before its encode, so the outer frame is classified even
    /// when the inner one, accepted first, ends the wait; frames after both are
    /// behind the decisive one in stream order and go unclassified.
    #[test]
    fn a_reentrant_send_accepted_first_keeps_stream_order() {
        let classification = WaitClassification::new(true);
        let outer = classification.classifies();
        let inner = classification.classifies();
        assert!(outer && inner);
        classification.accepted(TOKEN);
        classification.accepted(TOKEN);
        assert!(!classification.classifies());
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
