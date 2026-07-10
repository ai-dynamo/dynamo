// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Validation and phase merging for vLLM's routed-expert wire payload.
//!
//! The worker serializes an ndarray with `numpy.save` and base64 encodes the
//! resulting `.npy` bytes. This module is deliberately protocol-owned so both
//! prefill admission and the HTTP response assembler enforce exactly the same
//! process-boundary limits.

use std::mem;

use base64::Engine as _;
use ndarray::{ArrayViewD, Axis};
use ndarray_npy::{ViewElement, ViewNpyExt, WritableElement, WriteNpyExt};

use super::generate::GenerateProtocolError;

pub(crate) const MAX_NPY_BYTES: usize = 32 * 1024 * 1024;
const MAX_ENCODED_NPY_BYTES: usize = MAX_NPY_BYTES.div_ceil(3) * 4;
pub(crate) const MAX_CUMULATIVE_ROUTED_EXPERT_ENCODED_BYTES: usize = 64 * 1024 * 1024;
pub(crate) const MAX_CUMULATIVE_ROUTED_EXPERT_DECODED_BYTES: usize = 64 * 1024 * 1024;
const MAX_NPY_ELEMENTS: usize = 8 * 1024 * 1024;
const MAX_NPY_RANK: usize = 8;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct RoutedExpertPayloadStats {
    pub(crate) encoded_bytes: usize,
    pub(crate) decoded_bytes: usize,
}

#[derive(Debug)]
pub(crate) struct MergedRoutedExpertPayload {
    pub(crate) payload: String,
    pub(crate) stats: RoutedExpertPayloadStats,
}

#[derive(Debug, Default)]
pub(crate) struct RoutedExpertResponseBudget {
    encoded_bytes: usize,
    decoded_bytes: usize,
}

impl RoutedExpertResponseBudget {
    pub(crate) fn record(
        &mut self,
        stats: RoutedExpertPayloadStats,
    ) -> Result<(), GenerateProtocolError> {
        self.encoded_bytes = self
            .encoded_bytes
            .checked_add(stats.encoded_bytes)
            .ok_or_else(|| invalid_response("routed-expert encoded byte count overflowed"))?;
        if self.encoded_bytes > MAX_CUMULATIVE_ROUTED_EXPERT_ENCODED_BYTES {
            return Err(invalid_response(format!(
                "routed-expert metadata exceeded the {MAX_CUMULATIVE_ROUTED_EXPERT_ENCODED_BYTES}-byte cumulative encoded limit"
            )));
        }
        self.decoded_bytes = self
            .decoded_bytes
            .checked_add(stats.decoded_bytes)
            .ok_or_else(|| invalid_response("routed-expert decoded byte count overflowed"))?;
        if self.decoded_bytes > MAX_CUMULATIVE_ROUTED_EXPERT_DECODED_BYTES {
            return Err(invalid_response(format!(
                "routed-expert metadata exceeded the {MAX_CUMULATIVE_ROUTED_EXPERT_DECODED_BYTES}-byte cumulative decoded limit"
            )));
        }
        Ok(())
    }
}

fn invalid_response(message: impl Into<String>) -> GenerateProtocolError {
    GenerateProtocolError::InvalidResponse(message.into())
}

pub(crate) fn validate_routed_expert_payload(
    label: &str,
    payload: &str,
) -> Result<RoutedExpertPayloadStats, GenerateProtocolError> {
    let decoded = decode_payload(label, payload)?;
    macro_rules! valid_dtype {
        ($dtype:ty) => {
            if valid_npy::<$dtype>(&decoded) {
                return Ok(RoutedExpertPayloadStats {
                    encoded_bytes: payload.len(),
                    decoded_bytes: decoded.len(),
                });
            }
        };
    }
    valid_dtype!(i8);
    valid_dtype!(u8);
    valid_dtype!(i16);
    valid_dtype!(u16);
    valid_dtype!(i32);
    valid_dtype!(u32);
    valid_dtype!(i64);
    valid_dtype!(u64);
    valid_dtype!(f32);
    valid_dtype!(f64);
    valid_dtype!(bool);
    Err(invalid_response(format!(
        "invalid {label} routed-expert NumPy payload"
    )))
}

#[cfg(test)]
pub(crate) fn merge_routed_expert_payloads(
    prefill: Option<String>,
    decode: Option<String>,
) -> Result<Option<String>, GenerateProtocolError> {
    Ok(merge_routed_expert_payloads_with_stats(prefill, decode)?.map(|merged| merged.payload))
}

pub(crate) fn merge_routed_expert_payloads_with_stats(
    prefill: Option<String>,
    decode: Option<String>,
) -> Result<Option<MergedRoutedExpertPayload>, GenerateProtocolError> {
    let (prefill, decode) = match (prefill, decode) {
        (Some(prefill), Some(decode)) => (prefill, decode),
        (Some(payload), None) => {
            let stats = validate_routed_expert_payload("prefill", &payload)?;
            return Ok(Some(MergedRoutedExpertPayload { payload, stats }));
        }
        (None, Some(payload)) => {
            let stats = validate_routed_expert_payload("decode", &payload)?;
            return Ok(Some(MergedRoutedExpertPayload { payload, stats }));
        }
        (None, None) => return Ok(None),
    };
    let prefill = decode_payload("prefill", &prefill)?;
    let decode = decode_payload("decode", &decode)?;

    macro_rules! try_dtype {
        ($dtype:ty) => {
            if let Some(bytes) = concat_npy_rows::<$dtype>(&prefill, &decode) {
                let decoded_bytes = bytes.len();
                let payload = base64::engine::general_purpose::STANDARD.encode(bytes);
                return Ok(Some(MergedRoutedExpertPayload {
                    stats: RoutedExpertPayloadStats {
                        encoded_bytes: payload.len(),
                        decoded_bytes,
                    },
                    payload,
                }));
            }
        };
    }
    try_dtype!(i8);
    try_dtype!(u8);
    try_dtype!(i16);
    try_dtype!(u16);
    try_dtype!(i32);
    try_dtype!(u32);
    try_dtype!(i64);
    try_dtype!(u64);
    try_dtype!(f32);
    try_dtype!(f64);
    try_dtype!(bool);
    Err(invalid_response(
        "prefill/decode routed-expert NumPy payloads have incompatible dtype or shape",
    ))
}

fn decode_payload(label: &str, payload: &str) -> Result<Vec<u8>, GenerateProtocolError> {
    validate_payload_size(label, payload.len(), MAX_ENCODED_NPY_BYTES)?;
    let decoded = base64::engine::general_purpose::STANDARD
        .decode(payload)
        .map_err(|error| invalid_response(format!("invalid {label} routed experts: {error}")))?;
    validate_payload_size(label, decoded.len(), MAX_NPY_BYTES)?;
    Ok(decoded)
}

fn valid_npy<T>(bytes: &[u8]) -> bool
where
    T: ViewElement,
{
    let Some(storage) = NpyInput::for_element::<T>(bytes) else {
        return false;
    };
    let Ok(array) = ArrayViewD::<T>::view_npy(storage.as_slice()) else {
        return false;
    };
    valid_shape(array.shape(), array.len())
}

fn validate_payload_size(
    label: &str,
    actual: usize,
    maximum: usize,
) -> Result<(), GenerateProtocolError> {
    if actual > maximum {
        return Err(invalid_response(format!(
            "{label} routed-expert payload exceeds the {maximum}-byte limit"
        )));
    }
    Ok(())
}

fn concat_npy_rows<T>(prefill: &[u8], decode: &[u8]) -> Option<Vec<u8>>
where
    T: ViewElement + WritableElement + Clone,
{
    let prefill_storage = NpyInput::for_element::<T>(prefill)?;
    let decode_storage = NpyInput::for_element::<T>(decode)?;
    let prefill = ArrayViewD::<T>::view_npy(prefill_storage.as_slice()).ok()?;
    let decode = ArrayViewD::<T>::view_npy(decode_storage.as_slice()).ok()?;
    if !valid_shape(prefill.shape(), prefill.len())
        || !valid_shape(decode.shape(), decode.len())
        || prefill.shape()[1..] != decode.shape()[1..]
        || prefill.len().checked_add(decode.len())? > 2 * MAX_NPY_ELEMENTS
    {
        return None;
    }

    let merged = ndarray::concatenate(Axis(0), &[prefill, decode]).ok()?;
    let mut output = Vec::new();
    merged.write_npy(&mut output).ok()?;
    Some(output)
}

fn valid_shape(shape: &[usize], elements: usize) -> bool {
    !shape.is_empty() && shape.len() <= MAX_NPY_RANK && elements <= MAX_NPY_ELEMENTS
}

enum NpyInput<'a> {
    Borrowed(&'a [u8]),
    Aligned {
        storage: Vec<u8>,
        offset: usize,
        len: usize,
    },
}

impl<'a> NpyInput<'a> {
    fn for_element<T>(bytes: &'a [u8]) -> Option<Self> {
        let alignment = mem::align_of::<T>();
        if bytes.as_ptr().align_offset(alignment) == 0 {
            return Some(Self::Borrowed(bytes));
        }

        let mut storage = vec![0; bytes.len().checked_add(alignment - 1)?];
        let offset = storage.as_ptr().align_offset(alignment);
        if offset == usize::MAX || offset.checked_add(bytes.len())? > storage.len() {
            return None;
        }
        storage[offset..offset + bytes.len()].copy_from_slice(bytes);
        Some(Self::Aligned {
            storage,
            offset,
            len: bytes.len(),
        })
    }

    fn as_slice(&self) -> &[u8] {
        match self {
            Self::Borrowed(bytes) => bytes,
            Self::Aligned {
                storage,
                offset,
                len,
            } => &storage[*offset..*offset + *len],
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{Array, IxDyn};

    fn payload<T>(values: Vec<T>, shape: &[usize]) -> String
    where
        T: WritableElement,
    {
        let array = Array::from_shape_vec(IxDyn(shape), values).unwrap();
        let mut bytes = Vec::new();
        array.write_npy(&mut bytes).unwrap();
        base64::engine::general_purpose::STANDARD.encode(bytes)
    }

    #[test]
    fn validates_supported_payload_and_reports_wire_size() {
        let payload = payload(vec![1i16, 2], &[2, 1]);
        let stats = validate_routed_expert_payload("prefill", &payload).unwrap();
        assert_eq!(stats.encoded_bytes, payload.len());
        assert!(stats.decoded_bytes < stats.encoded_bytes);
    }

    #[test]
    fn rejects_invalid_base64_and_npy() {
        assert!(validate_routed_expert_payload("prefill", "not-base64").is_err());
        let malformed = base64::engine::general_purpose::STANDARD.encode(b"not-npy");
        assert!(validate_routed_expert_payload("prefill", &malformed).is_err());
    }

    #[test]
    fn validates_but_preserves_single_phase_wire_bytes() {
        let original = payload(vec![1i16, 2], &[2, 1]);
        let merged = merge_routed_expert_payloads(Some(original.clone()), None)
            .unwrap()
            .unwrap();
        assert_eq!(merged, original);
    }

    #[test]
    fn rejects_mismatched_dtype_or_shape() {
        assert!(
            merge_routed_expert_payloads(
                Some(payload(vec![1i16], &[1, 1])),
                Some(payload(vec![2i32], &[1, 1])),
            )
            .is_err()
        );
        assert!(
            merge_routed_expert_payloads(
                Some(payload(vec![1i16, 2], &[1, 2])),
                Some(payload(vec![3i16, 4, 5], &[1, 3])),
            )
            .is_err()
        );
    }

    #[test]
    fn rejects_scalar_and_excessive_shape_metadata() {
        assert!(!valid_shape(&[], 1));
        assert!(!valid_shape(&[1; MAX_NPY_RANK + 1], 1));
        assert!(!valid_shape(&[1], MAX_NPY_ELEMENTS + 1));
    }

    #[test]
    fn rejects_encoded_or_decoded_payload_over_bound() {
        assert!(
            validate_payload_size("prefill", MAX_ENCODED_NPY_BYTES + 1, MAX_ENCODED_NPY_BYTES,)
                .is_err()
        );
        assert!(validate_payload_size("decode", MAX_NPY_BYTES + 1, MAX_NPY_BYTES).is_err());
    }
}
