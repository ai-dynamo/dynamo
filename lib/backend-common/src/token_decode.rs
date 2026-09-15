// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0

//! Decode large MessagePack token arrays without per-element Serde dispatch.
//! The normal request deserializer still validates all other fields.

use std::io::Cursor;

use dynamo_runtime::pipeline::{
    PipelineError,
    network::{
        EncodedResponseFrame, IngressRequestDecoder, IngressResponseEncoder,
        RequestPlanePayloadCodec, SerdeIngressPayloadAdapter,
    },
};
use dynamo_runtime::protocols::annotated::Annotated;
use serde::Deserialize;

use crate::{LLMEngineOutput, PreprocessedRequest};

fn take<const N: usize>(input: &mut &[u8]) -> Option<[u8; N]> {
    let part = input.get(..N)?.try_into().ok()?;
    *input = &input[N..];
    Some(part)
}

fn array(input: &mut &[u8]) -> Option<Vec<u32>> {
    let marker = take::<1>(input)?[0];
    let count = match marker {
        0x90..=0x9f => (marker & 15) as usize,
        0xdc => u16::from_be_bytes(take(input)?) as usize,
        0xdd => u32::from_be_bytes(take(input)?) as usize,
        _ => return None,
    };
    if count > input.len() {
        return None;
    }
    // Match Serde's cautious allocation limit of one MiB.
    let mut out = Vec::with_capacity(count.min(262_144));
    for _ in 0..count {
        let number: i128 = match take::<1>(input)?[0] {
            n @ 0x00..=0x7f => n.into(),
            0xcc => take::<1>(input)?[0].into(),
            0xcd => u16::from_be_bytes(take(input)?).into(),
            0xce => u32::from_be_bytes(take(input)?).into(),
            0xcf => u64::from_be_bytes(take(input)?).into(),
            0xd0 => i8::from_be_bytes(take(input)?).into(),
            0xd1 => i16::from_be_bytes(take(input)?).into(),
            0xd2 => i32::from_be_bytes(take(input)?).into(),
            0xd3 => i64::from_be_bytes(take(input)?).into(),
            _ => return None,
        };
        out.push(u32::try_from(number).ok()?);
    }
    Some(out)
}

fn try_request(bytes: &[u8]) -> Option<PreprocessedRequest> {
    // Copying metadata costs more than this fast path saves on small requests.
    if bytes.len() < 4096 {
        return None;
    }
    let mut remaining = bytes;
    let count = match take::<1>(&mut remaining)?[0] {
        n @ 0x80..=0x8f => (n & 15) as u32,
        0xde => u16::from_be_bytes(take(&mut remaining)?) as u32,
        0xdf => u32::from_be_bytes(take(&mut remaining)?),
        _ => return None,
    };
    let mut cursor = Cursor::new(bytes);
    cursor.set_position((bytes.len() - remaining.len()) as u64);
    let mut decoder = rmp_serde::Deserializer::new(cursor);
    for _ in 0..count {
        let key = String::deserialize(&mut decoder).ok()?;
        if key == "token_ids" {
            let start = decoder.position() as usize;
            let mut tail = &bytes[start..];
            let ids = array(&mut tail)?;
            let mut metadata = Vec::with_capacity(start + 1 + tail.len());
            metadata.extend_from_slice(&bytes[..start]);
            // Replace only the array with an empty one. Serde still checks
            // required fields, duplicate fields, and all remaining metadata.
            metadata.push(0x90);
            metadata.extend_from_slice(tail);
            let mut request: PreprocessedRequest = rmp_serde::from_slice(&metadata).ok()?;
            request.token_ids = ids;
            return Some(request);
        }
        serde::de::IgnoredAny::deserialize(&mut decoder).ok()?;
    }
    None
}

fn decode(bytes: &[u8]) -> Result<PreprocessedRequest, rmp_serde::decode::Error> {
    // This also retains the stock error and visitor behavior on unsupported or
    // malformed inputs, duplicate fields, and positional struct encodings.
    match try_request(bytes) {
        Some(request) => Ok(request),
        None => rmp_serde::from_slice(bytes),
    }
}

pub(crate) struct TokenPayloadAdapter;

impl IngressRequestDecoder<PreprocessedRequest> for TokenPayloadAdapter {
    fn decode_request(
        &self,
        codec: RequestPlanePayloadCodec,
        bytes: bytes::Bytes,
    ) -> impl std::future::Future<Output = Result<PreprocessedRequest, PipelineError>> + Send {
        let decoded = match codec {
            RequestPlanePayloadCodec::Msgpack => decode(&bytes).map_err(anyhow::Error::from),
            _ => codec.decode(&bytes),
        };
        std::future::ready(decoded.map_err(|err| {
            PipelineError::DeserializationError(format!(
                "Failed deserializing {} request payload: {}",
                codec.name(),
                err
            ))
        }))
    }
}

impl IngressResponseEncoder<Annotated<LLMEngineOutput>> for TokenPayloadAdapter {
    fn encode_response(
        &self,
        codec: RequestPlanePayloadCodec,
        response: Option<Annotated<LLMEngineOutput>>,
        complete_final: bool,
    ) -> impl std::future::Future<Output = Result<EncodedResponseFrame, PipelineError>> + Send {
        SerdeIngressPayloadAdapter.encode_response(codec, response, complete_final)
    }
}

#[cfg(test)]
mod tests;
