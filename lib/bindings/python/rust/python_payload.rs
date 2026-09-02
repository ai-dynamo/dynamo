// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use bytes::{BufMut, Bytes, BytesMut};
use dynamo_runtime::pipeline::PipelineError;
use dynamo_runtime::pipeline::network::{
    EncodedResponseFrame, IngressRequestDecoder, IngressResponseEncoder, NetworkStreamWrapper,
    RequestPlanePayloadCodec,
};
use dynamo_runtime::protocols::annotated::Annotated;
use dynamo_runtime::protocols::maybe_error::MaybeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyInt, PyList, PyString};
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
    // `with_capacity`, not `new`: this is still the pull path's encoder, and
    // `serde_json::to_vec` — which it used before — starts at 128 bytes. Growing
    // from zero here would regress JSON responses to pay the reallocations this
    // change exists to remove.
    let mut bytes = Vec::with_capacity(128);
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

// ── Steady-state token-frame fast path ───────────────────────────────────────
//
// Writes standard TRT-LLM token frames directly to bypass costly generic
// serialization.
//
// Streaming decode predominantly produces a two-key dict (`token_ids`, `index`).
// Using the generic `pythonize` encoder creates massive overhead due to
// sequential type-checking and integer downcasting for every single token ID.
//
// This fast path skips that ladder. Byte-equivalence with the generic encoder is
// verified by the `token_frame_matches_generic_encoding` test.

/// Justifies the hardcoded map lengths below based on envelope serialization:
///
/// * **`Annotated` (1 key):** Serializes only the `data` key (the token payload).
///   Other fields (`id`, `event`, etc.) are `None` on token frames and skipped by `serde`.
/// * **`NetworkStreamWrapper` (2 keys):** Serializes the `data` key (holding the
///   `Annotated` object) plus the always-written `complete_final` boolean.
const KEY_DATA: &str = "data";
const KEY_COMPLETE_FINAL: &str = "complete_final";
const KEY_TOKEN_IDS: &str = "token_ids";
const KEY_INDEX: &str = "index";

/// Try to encode a token frame straight to msgpack, skipping serde entirely.
///
/// Returns `true` if `dict` matched and the frame was written. Returns `false`,
/// having written nothing, if it did not — the caller must then use the generic
/// path. To match, `dict` must be exactly `token_ids` then `index`, holding a
/// list of non-negative ints and a non-negative int. Msgpack only; JSON always
/// takes the generic path.
pub(crate) fn try_write_token_frame(dict: &Bound<'_, PyDict>, out: &mut BytesMut) -> bool {
    // We can only tell a frame is ineligible partway through writing it — a bad
    // token id in the middle of the list, say — so remember where the caller
    // left off and rewind on failure. Half a frame followed by a whole one is
    // undecodable garbage on the wire.
    let rewind = out.len();
    if parse_and_write_token_frame(dict, out).is_none() {
        out.truncate(rewind);
        return false;
    }
    true
}

fn parse_and_write_token_frame(dict: &Bound<'_, PyDict>, out: &mut BytesMut) -> Option<()> {
    // Match on iteration ORDER, not by key lookup.
    //
    // A msgpack map stores its entries in the order the serializer visits them,
    // which for a Python dict is insertion order. So a dict holding these same
    // two keys the other way round has to encode the other way round too — if
    // we matched it here, the fast and generic paths would disagree on bytes.
    //
    // Matching on order also makes this a *complete* check of the shape: a third
    // key fails the `next()` below, and a `_dynamo_annotated` envelope can never
    // match. That is what makes it safe to skip `parse_python_response`.
    let mut entries = dict.iter();
    let (token_ids_key, token_ids) = entries.next()?;
    let (index_key, index) = entries.next()?;
    if entries.next().is_some()
        || !key_is(&token_ids_key, KEY_TOKEN_IDS)
        || !key_is(&index_key, KEY_INDEX)
    {
        return None;
    }

    let token_ids = token_ids.downcast::<PyList>().ok()?;
    let index = exact_uint(&index)?;

    write_token_frame_bytes(
        token_ids.iter().map(|token_id| exact_uint(&token_id)),
        u32::try_from(token_ids.len()).ok()?,
        index,
        &mut (&mut *out).writer(),
    )
}

/// Write the msgpack bytes for one token frame.
///
/// Kept separate from the Python-facing code above so the wire layout — the part
/// that must match the generic encoder exactly — can be unit-tested. This crate
/// builds as a PyO3 extension module and so does not link libpython, meaning
/// `cargo test` cannot touch Python objects. See the module tests below.
///
/// `token_count` must equal the number of items `token_ids` yields. msgpack
/// writes an array's length *before* its elements, so a mismatch produces a
/// frame that cannot be decoded. A `None` item aborts the frame mid-write, which
/// is why the caller has to be able to rewind.
fn write_token_frame_bytes<W, I>(
    token_ids: I,
    token_count: u32,
    index: u64,
    writer: &mut W,
) -> Option<()>
where
    W: std::io::Write,
    I: Iterator<Item = Option<u64>>,
{
    // Three nested maps, matching what serde would emit:
    //   NetworkStreamWrapper { data, complete_final }
    //     -> Annotated { data }   (id/event/comment/error are None, so skipped)
    //          -> { token_ids, index }
    rmp::encode::write_map_len(writer, 2).ok()?;
    rmp::encode::write_str(writer, KEY_DATA).ok()?;
    rmp::encode::write_map_len(writer, 1).ok()?;
    rmp::encode::write_str(writer, KEY_DATA).ok()?;
    rmp::encode::write_map_len(writer, 2).ok()?;
    rmp::encode::write_str(writer, KEY_TOKEN_IDS).ok()?;
    rmp::encode::write_array_len(writer, token_count).ok()?;
    let mut written = 0u32;
    for token_id in token_ids {
        rmp::encode::write_uint(writer, token_id?).ok()?;
        written += 1;
    }
    if written != token_count {
        return None;
    }
    rmp::encode::write_str(writer, KEY_INDEX).ok()?;
    rmp::encode::write_uint(writer, index).ok()?;
    rmp::encode::write_str(writer, KEY_COMPLETE_FINAL).ok()?;
    rmp::encode::write_bool(writer, false).ok()?;
    Some(())
}

/// Whether `key` is exactly the `str` `name`.
fn key_is(key: &Bound<'_, PyAny>, name: &str) -> bool {
    key.downcast_exact::<PyString>()
        .ok()
        .and_then(|key| key.to_str().ok())
        .is_some_and(|key| key == name)
}

/// `value` as a `u64` — but only if the generic path would also encode it as a
/// plain unsigned integer.
///
/// `downcast_exact` is load-bearing. In Python `bool` is a subclass of `int`,
/// and pythonize checks `PyBool` *before* `PyInt`, so generically a `True` in a
/// token list becomes a msgpack bool. It must not silently become the integer 1
/// here. Requiring the exact type keeps bools (and any other `int` subclass) on
/// the generic path.
///
/// `extract` then rejects negatives and anything larger than `u64`, which the
/// generic path also encodes differently — as a signed int, and as msgpack
/// *bytes* respectively.
fn exact_uint(value: &Bound<'_, PyAny>) -> Option<u64> {
    value.downcast_exact::<PyInt>().ok()?.extract().ok()
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
        Ok((bytes, is_error)) => Ok(EncodedResponseFrame {
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
        Annotated, NetworkStreamWrapper, RequestPlanePayloadCodec, encode_annotated_response,
        write_token_frame_bytes,
    };
    use serde::Serialize;

    // ── token-frame fast path: wire equivalence ──────────────────────────────
    //
    // The fast path skips serde. That is only safe if it produces the SAME bytes
    // the generic path would, so these tests encode one logical response both
    // ways and compare.
    //
    // Byte equality, not just "both decode": the two paths are picked per frame
    // *within a single response stream*, so any difference in map ordering or
    // integer width would be a wire mismatch between consecutive frames of one
    // response.
    //
    // The Python half — the shape and type checks in `parse_and_write_token_frame`
    // — cannot be tested here, since these tests do not link libpython. It is
    // covered by pytest against the built extension instead.

    /// Stands in for the dict `handler_base.py` builds per token, with the fields
    /// in the same order Python inserts them. msgpack writes a Rust struct as a
    /// map, so this serializes exactly as that Python dict does.
    ///
    /// `u64`, not `u32`, to match what the fast path actually accepts: `exact_uint`
    /// takes any value up to `u64::MAX`, so the comparison has to reach that far
    /// too or the widest markers would go unchecked.
    #[derive(Serialize)]
    struct TokenFramePayload {
        token_ids: Vec<u64>,
        index: u64,
    }

    /// What the generic path puts on the wire for this frame.
    fn generic_encoding(token_ids: &[u64], index: u64) -> Vec<u8> {
        let (bytes, is_error) = encode_annotated_response(
            RequestPlanePayloadCodec::Msgpack,
            Annotated::from_data(TokenFramePayload {
                token_ids: token_ids.to_vec(),
                index,
            }),
        )
        .expect("generic encode must succeed");
        assert!(!is_error, "a token frame is never an error frame");
        bytes
    }

    /// What the fast path puts on the wire for this frame.
    fn fast_encoding(token_ids: &[u64], index: u64) -> Vec<u8> {
        let mut bytes = Vec::new();
        write_token_frame_bytes(
            token_ids.iter().copied().map(Some),
            u32::try_from(token_ids.len()).expect("length fits u32"),
            index,
            &mut bytes,
        )
        .expect("fast encode must succeed");
        bytes
    }

    /// Values on both sides of every msgpack uint marker boundary. msgpack picks
    /// a different marker byte per magnitude, so these are where a hand-written
    /// encoder would drift from `write_uint`. Runs to `u64::MAX` because that is
    /// the largest value the fast path will accept.
    const MAGNITUDES: [u64; 12] = [
        0,
        1,
        127,           // largest positive fixint
        128,           // first needing uint8
        255,           // largest uint8
        256,           // first needing uint16
        65_535,        // largest uint16
        65_536,        // first needing uint32
        4_294_967_295, // largest uint32
        4_294_967_296, // first needing uint64
        u64::MAX - 1,
        u64::MAX,
    ];

    fn assert_encodings_agree(token_ids: &[u64], index: u64) {
        assert_eq!(
            fast_encoding(token_ids, index),
            generic_encoding(token_ids, index),
            "fast and generic encodings diverged for token_ids={token_ids:?} index={index}"
        );
    }

    /// Covers token id widths, plus list lengths on both sides of the fixarray
    /// boundary where `write_array_len` switches marker.
    #[test]
    fn token_frame_matches_generic_encoding() {
        let mut cases: Vec<Vec<u64>> = vec![
            vec![],              // engine returned no new tokens
            vec![5],             // the steady-state case
            MAGNITUDES.to_vec(), // mixed widths in one list
            (0..15).collect(),   // fixarray, at the limit
            (0..16).collect(),   // first length needing array16
            (0..40).collect(),   // comfortably past it
        ];
        cases.extend(MAGNITUDES.map(|magnitude| vec![magnitude]));

        for token_ids in &cases {
            assert_encodings_agree(token_ids, 0);
        }
    }

    /// `index` is encoded independently of the token list, so it only needs its
    /// own uint boundaries — no cross product with the cases above.
    #[test]
    fn token_frame_index_matches_generic_encoding() {
        for index in MAGNITUDES {
            assert_encodings_agree(&[5], index);
        }
    }

    /// The frame must decode to the envelope the rest of the request plane
    /// expects: `data` present, no error, not the end-of-stream marker.
    ///
    /// The byte-equality tests above compare the two paths to each other, so they
    /// would pass if both were wrong in the same way. This one checks the result
    /// independently.
    #[test]
    fn token_frame_decodes_as_a_non_terminal_data_frame() {
        let bytes = fast_encoding(&[7, 8], 3);
        let wrapper: NetworkStreamWrapper<Annotated<rmpv::Value>> =
            RequestPlanePayloadCodec::Msgpack
                .decode(&bytes)
                .expect("fast-path frame must decode");

        assert!(!wrapper.complete_final, "not the end-of-stream marker");
        let annotated = wrapper.data.expect("frame carries data");
        assert!(annotated.error.is_none(), "a token frame carries no error");
        assert!(annotated.id.is_none() && annotated.event.is_none());

        let data = annotated.data.expect("annotated data");
        let fields = data.as_map().expect("payload is a map");
        let field = |name: &str| {
            fields
                .iter()
                .find(|(key, _)| key.as_str() == Some(name))
                .map(|(_, value)| value)
                .unwrap_or_else(|| panic!("payload must carry {name}"))
        };
        assert_eq!(
            field("token_ids")
                .as_array()
                .expect("token_ids is an array")
                .iter()
                .map(|id| id.as_u64().expect("token id is a uint"))
                .collect::<Vec<_>>(),
            vec![7, 8]
        );
        assert_eq!(field("index").as_u64(), Some(3));
    }

    /// A token the Python side could not convert must abort the whole frame,
    /// never report a partial one as written. The caller rewinds on `None`.
    #[test]
    fn token_frame_aborts_on_an_unconvertible_token() {
        let mut bytes = Vec::new();
        assert!(
            write_token_frame_bytes([Some(1), None, Some(3)].into_iter(), 3, 0, &mut bytes)
                .is_none(),
            "an unconvertible token must abort the frame"
        );
    }

    /// If the declared count and the tokens actually yielded disagree, the array
    /// header would lie about its length and the frame would not decode.
    #[test]
    fn token_frame_rejects_a_token_count_mismatch() {
        let mut bytes = Vec::new();
        assert!(
            write_token_frame_bytes([Some(1), Some(2)].into_iter(), 3, 0, &mut bytes).is_none(),
            "too few tokens for the declared count must abort the frame"
        );
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
