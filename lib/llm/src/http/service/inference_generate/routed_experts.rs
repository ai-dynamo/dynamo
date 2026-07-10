// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Merge the vLLM routed-expert wire format across prefill and decode.
//!
//! vLLM serializes each expert trace with `numpy.save` and then base64-encodes
//! the resulting `.npy` bytes. P/D generation produces one such array per
//! phase, so the HTTP boundary concatenates axis 0 while preserving the
//! remaining dimensions and dtype.

use std::mem;

use base64::Engine as _;
use ndarray::{ArrayViewD, Axis};
use ndarray_npy::{ViewElement, ViewNpyExt, WritableElement, WriteNpyExt};

use super::invalid_response;
use crate::protocols::inference::generate::GenerateProtocolError;

// A trace is backend-generated rather than user-supplied, but it crosses a
// process boundary. Bound both parsing and the merged allocation so a corrupt
// NPY header cannot turn one response into an unbounded allocation.
const MAX_NPY_BYTES: usize = 32 * 1024 * 1024;
const MAX_ENCODED_NPY_BYTES: usize = MAX_NPY_BYTES.div_ceil(3) * 4;
const MAX_NPY_ELEMENTS: usize = 8 * 1024 * 1024;
const MAX_NPY_RANK: usize = 8;

pub(super) fn merge_routed_expert_payloads(
    prefill: Option<String>,
    decode: Option<String>,
) -> Result<Option<String>, GenerateProtocolError> {
    let (prefill, decode) = match (prefill, decode) {
        (Some(prefill), Some(decode)) => (prefill, decode),
        (Some(payload), None) => {
            validate_single_payload("prefill", &payload)?;
            return Ok(Some(payload));
        }
        (None, Some(payload)) => {
            validate_single_payload("decode", &payload)?;
            return Ok(Some(payload));
        }
        (None, None) => return Ok(None),
    };
    let prefill = decode_payload("prefill", &prefill)?;
    let decode = decode_payload("decode", &decode)?;

    macro_rules! try_dtype {
        ($dtype:ty) => {
            if let Some(bytes) = concat_npy_rows::<$dtype>(&prefill, &decode) {
                return Ok(Some(
                    base64::engine::general_purpose::STANDARD.encode(bytes),
                ));
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

fn validate_single_payload(label: &str, payload: &str) -> Result<(), GenerateProtocolError> {
    let decoded = decode_payload(label, payload)?;
    macro_rules! valid_dtype {
        ($dtype:ty) => {
            if valid_npy::<$dtype>(&decoded) {
                return Ok(());
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
    use ndarray_npy::WriteNpyExt;

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
    fn rejects_invalid_base64() {
        let error = merge_routed_expert_payloads(
            Some("not-base64".to_string()),
            Some(payload(vec![1i16], &[1, 1])),
        )
        .unwrap_err();
        assert!(error.to_string().contains("invalid prefill routed experts"));
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
    fn rejects_malformed_single_phase_payload() {
        let malformed = base64::engine::general_purpose::STANDARD.encode(b"not-npy");
        let error = merge_routed_expert_payloads(None, Some(malformed)).unwrap_err();
        assert!(
            error
                .to_string()
                .contains("invalid decode routed-expert NumPy payload")
        );
    }

    #[test]
    fn rejects_malformed_npy() {
        let malformed = base64::engine::general_purpose::STANDARD.encode(b"not-npy");
        let error =
            merge_routed_expert_payloads(Some(malformed.clone()), Some(malformed)).unwrap_err();
        assert!(error.to_string().contains("incompatible dtype or shape"));
    }

    #[test]
    fn rejects_mismatched_dtype() {
        let error = merge_routed_expert_payloads(
            Some(payload(vec![1i16], &[1, 1])),
            Some(payload(vec![2i32], &[1, 1])),
        )
        .unwrap_err();
        assert!(error.to_string().contains("incompatible dtype or shape"));
    }

    #[test]
    fn rejects_mismatched_non_row_shape() {
        let error = merge_routed_expert_payloads(
            Some(payload(vec![1i16, 2], &[1, 2])),
            Some(payload(vec![3i16, 4, 5], &[1, 3])),
        )
        .unwrap_err();
        assert!(error.to_string().contains("incompatible dtype or shape"));
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
            validate_payload_size("prefill", MAX_ENCODED_NPY_BYTES + 1, MAX_ENCODED_NPY_BYTES)
                .is_err()
        );
        assert!(validate_payload_size("decode", MAX_NPY_BYTES + 1, MAX_NPY_BYTES).is_err());
    }
}
