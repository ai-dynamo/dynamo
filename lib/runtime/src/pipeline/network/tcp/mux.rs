// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Multiplexed TCP response-stream protocol.
//!
//! [`MuxCodec`] carries both the connection handshake and logical response
//! streams. Its compact fixed-width header lets connection writers batch frames
//! without copying their payloads.

use std::{io, sync::OnceLock, time::Duration};

use bytes::{BufMut, Bytes, BytesMut};
use uuid::Uuid;

use tokio_util::codec::{Decoder, Encoder};

pub mod client;

pub const RESPONSE_MUX_VERSION: u8 = 1;
pub const RESPONSE_MUX_POOL_SIZE: usize = 4;
pub const RESPONSE_MUX_WRITER_QUEUE: usize = 4096;
pub const RESPONSE_MUX_STREAM_WRITER_QUEUE: usize = 8;
pub const RESPONSE_MUX_CONNECTION_QUEUE_BYTES: usize = 262_144;
pub const RESPONSE_MUX_CONNECT_TIMEOUT_SECS: u64 = 5;

pub const RESPONSE_MUX_DEFAULT_BATCH_INTERVAL_MS: u64 = 1;
pub const RESPONSE_MUX_MAX_BATCH_INTERVAL_MS: u64 = 100;
pub const RESPONSE_MUX_DEFAULT_BATCH_MAX_BYTES: usize = 65_536;
pub const RESPONSE_MUX_DEFAULT_BATCH_MAX_FRAMES: usize = 64;
pub const RESPONSE_MUX_DEFAULT_STREAM_WINDOW_BYTES: usize = 262_144;
pub const RESPONSE_MUX_CREDIT_UPDATE_BYTES: usize = 65_536;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResponseMuxConfig {
    pub packet_metrics: bool,
    pub batch_interval: Duration,
    pub batch_max_bytes: usize,
    pub batch_max_frames: usize,
    pub stream_window_bytes: usize,
}

impl ResponseMuxConfig {
    fn parse_with(mut read: impl FnMut(&str) -> Option<String>) -> anyhow::Result<Self> {
        use crate::config::environment_names::tcp_response_stream as env;

        fn parse<T>(
            read: &mut impl FnMut(&str) -> Option<String>,
            name: &str,
            default: T,
        ) -> anyhow::Result<T>
        where
            T: std::str::FromStr,
            T::Err: std::fmt::Display,
        {
            match read(name) {
                None => Ok(default),
                Some(value) => value
                    .parse::<T>()
                    .map_err(|err| anyhow::anyhow!("invalid {name}={value:?}: {err}")),
            }
        }

        let parse_bool = |name, value: Option<String>| match value.as_deref() {
            None | Some("") | Some("0") | Some("false") => Ok(false),
            Some("1") | Some("true") => Ok(true),
            Some(value) => anyhow::bail!("invalid {name}={value:?}; expected 0, 1, false, or true"),
        };
        let packet_metrics = parse_bool(
            env::DYN_TCP_RESPONSE_PACKET_METRICS,
            read(env::DYN_TCP_RESPONSE_PACKET_METRICS),
        )?;
        let interval_ms = parse(
            &mut read,
            env::DYN_TCP_RESPONSE_BATCH_INTERVAL_MS,
            RESPONSE_MUX_DEFAULT_BATCH_INTERVAL_MS,
        )?;
        if interval_ms > RESPONSE_MUX_MAX_BATCH_INTERVAL_MS {
            anyhow::bail!(
                "{} must be at most {} ms; got {}",
                env::DYN_TCP_RESPONSE_BATCH_INTERVAL_MS,
                RESPONSE_MUX_MAX_BATCH_INTERVAL_MS,
                interval_ms
            );
        }
        let batch_max_bytes = parse(
            &mut read,
            env::DYN_TCP_RESPONSE_BATCH_MAX_BYTES,
            RESPONSE_MUX_DEFAULT_BATCH_MAX_BYTES,
        )?;
        let batch_max_frames = parse(
            &mut read,
            env::DYN_TCP_RESPONSE_BATCH_MAX_FRAMES,
            RESPONSE_MUX_DEFAULT_BATCH_MAX_FRAMES,
        )?;
        let stream_window_bytes = parse(
            &mut read,
            env::DYN_TCP_RESPONSE_STREAM_WINDOW_BYTES,
            RESPONSE_MUX_DEFAULT_STREAM_WINDOW_BYTES,
        )?;
        for (name, value) in [
            (env::DYN_TCP_RESPONSE_BATCH_MAX_BYTES, batch_max_bytes),
            (env::DYN_TCP_RESPONSE_BATCH_MAX_FRAMES, batch_max_frames),
            (
                env::DYN_TCP_RESPONSE_STREAM_WINDOW_BYTES,
                stream_window_bytes,
            ),
        ] {
            if value == 0 {
                anyhow::bail!("{name} must be greater than zero");
            }
        }
        for (name, value) in [(
            env::DYN_TCP_RESPONSE_STREAM_WINDOW_BYTES,
            stream_window_bytes,
        )] {
            if value > u32::MAX as usize {
                anyhow::bail!("{name} must fit in an unsigned 32-bit credit update");
            }
        }
        Ok(Self {
            packet_metrics,
            batch_interval: Duration::from_millis(interval_ms),
            batch_max_bytes,
            batch_max_frames,
            stream_window_bytes,
        })
    }

    pub fn from_env() -> anyhow::Result<Self> {
        Self::parse_with(|name| std::env::var(name).ok())
    }
}

static RESPONSE_MUX_CONFIG: OnceLock<ResponseMuxConfig> = OnceLock::new();

pub fn initialize_response_mux_config() -> anyhow::Result<ResponseMuxConfig> {
    if let Some(config) = RESPONSE_MUX_CONFIG.get() {
        return Ok(*config);
    }
    let config = ResponseMuxConfig::from_env()?;
    let _ = RESPONSE_MUX_CONFIG.set(config);
    Ok(*RESPONSE_MUX_CONFIG.get().expect("response mux config set"))
}

pub const MUX_HEADER_LEN: usize = 21;
const CONNECTION_HELLO_LEN: usize = 17;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum MuxFrameKind {
    Prologue = 1,
    Data = 2,
    End = 3,
    Stop = 4,
    Kill = 5,
    WindowUpdate = 6,
    Reset = 7,
    ConnectionHello = 8,
    ConnectionReady = 9,
}

impl TryFrom<u8> for MuxFrameKind {
    type Error = io::Error;

    fn try_from(value: u8) -> Result<Self, Self::Error> {
        match value {
            1 => Ok(Self::Prologue),
            2 => Ok(Self::Data),
            3 => Ok(Self::End),
            4 => Ok(Self::Stop),
            5 => Ok(Self::Kill),
            6 => Ok(Self::WindowUpdate),
            7 => Ok(Self::Reset),
            8 => Ok(Self::ConnectionHello),
            9 => Ok(Self::ConnectionReady),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unknown response mux frame kind {value}"),
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct MuxFrame {
    pub kind: MuxFrameKind,
    pub stream_id: Uuid,
    pub payload: Bytes,
}

impl MuxFrame {
    pub fn new(kind: MuxFrameKind, stream_id: Uuid, payload: Bytes) -> Self {
        Self {
            kind,
            stream_id,
            payload,
        }
    }

    pub fn empty(kind: MuxFrameKind, stream_id: Uuid) -> Self {
        Self::new(kind, stream_id, Bytes::new())
    }

    pub fn window_update(stream_id: Uuid, credits: u32) -> Self {
        let mut payload = BytesMut::with_capacity(4);
        payload.put_u32(credits);
        Self::new(MuxFrameKind::WindowUpdate, stream_id, payload.freeze())
    }

    pub fn connection_hello(version: u8, frontend_server_id: Uuid) -> Self {
        let mut payload = BytesMut::with_capacity(CONNECTION_HELLO_LEN);
        payload.put_u8(version);
        payload.extend_from_slice(frontend_server_id.as_bytes());
        Self::new(MuxFrameKind::ConnectionHello, Uuid::nil(), payload.freeze())
    }

    pub fn connection_ready() -> Self {
        Self::empty(MuxFrameKind::ConnectionReady, Uuid::nil())
    }

    pub fn connection_identity(&self) -> io::Result<(u8, Uuid)> {
        if self.kind != MuxFrameKind::ConnectionHello || self.payload.len() != CONNECTION_HELLO_LEN
        {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "response mux connection hello must contain a version and frontend UUID",
            ));
        }
        let frontend_server_id = Uuid::from_slice(&self.payload[1..]).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid response mux frontend UUID: {err}"),
            )
        })?;
        Ok((self.payload[0], frontend_server_id))
    }

    pub fn encoded_len(&self) -> usize {
        MUX_HEADER_LEN + self.payload.len()
    }

    pub fn window_credits(&self) -> io::Result<u32> {
        if self.kind != MuxFrameKind::WindowUpdate || self.payload.len() != 4 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "credit update must contain exactly four payload bytes",
            ));
        }
        Ok(u32::from_be_bytes(self.payload[..4].try_into().unwrap()))
    }

    /// Split the wire representation into a small header and the original
    /// payload allocation so connection writers can coalesce headers while
    /// retaining large payloads as `Bytes` for vectored I/O.
    pub fn encode_parts(&self) -> io::Result<(Bytes, Bytes)> {
        let mut header = BytesMut::with_capacity(MUX_HEADER_LEN);
        MuxCodec::default().encode_header(self, &mut header)?;
        Ok((header.freeze(), self.payload.clone()))
    }

    fn validate(frame: Self) -> io::Result<Self> {
        let is_connection_frame = matches!(
            frame.kind,
            MuxFrameKind::ConnectionHello | MuxFrameKind::ConnectionReady
        );
        if frame.stream_id.is_nil() != is_connection_frame {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "only connection-level frames must use the nil stream UUID",
            ));
        }
        match frame.kind {
            MuxFrameKind::End | MuxFrameKind::Stop | MuxFrameKind::Kill | MuxFrameKind::Reset
                if !frame.payload.is_empty() =>
            {
                Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "control frame must not contain a payload",
                ))
            }
            MuxFrameKind::WindowUpdate => {
                frame.window_credits()?;
                Ok(frame)
            }
            MuxFrameKind::ConnectionHello => {
                frame.connection_identity()?;
                Ok(frame)
            }
            MuxFrameKind::ConnectionReady if !frame.payload.is_empty() => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "response mux connection ready must be empty",
            )),
            _ => Ok(frame),
        }
    }
}

/// Compact response-mux framing. Each frame is `payload_len:u32`, kind, UUID,
/// then payload.
#[derive(Clone, Debug)]
pub struct MuxCodec {
    max_message_size: usize,
}

impl Default for MuxCodec {
    fn default() -> Self {
        Self::new(crate::pipeline::network::get_tcp_max_message_size())
    }
}

impl MuxCodec {
    pub fn new(max_message_size: usize) -> Self {
        Self { max_message_size }
    }

    fn encode_header(&self, frame: &MuxFrame, dst: &mut BytesMut) -> io::Result<()> {
        let payload_len = u32::try_from(frame.payload.len()).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidInput,
                "response mux payload exceeds u32",
            )
        })?;
        let encoded_len = MUX_HEADER_LEN
            .checked_add(frame.payload.len())
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidInput, "frame size overflow"))?;
        if encoded_len > self.max_message_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "response mux frame size {encoded_len} exceeds maximum {}",
                    self.max_message_size
                ),
            ));
        }
        dst.reserve(MUX_HEADER_LEN);
        dst.put_u32(payload_len);
        dst.put_u8(frame.kind as u8);
        dst.extend_from_slice(frame.stream_id.as_bytes());
        Ok(())
    }
}

impl Encoder<MuxFrame> for MuxCodec {
    type Error = io::Error;

    fn encode(&mut self, frame: MuxFrame, dst: &mut BytesMut) -> io::Result<()> {
        self.encode_header(&frame, dst)?;
        dst.extend_from_slice(&frame.payload);
        Ok(())
    }
}

impl Decoder for MuxCodec {
    type Item = MuxFrame;
    type Error = io::Error;

    fn decode(&mut self, src: &mut BytesMut) -> io::Result<Option<MuxFrame>> {
        if src.len() < MUX_HEADER_LEN {
            return Ok(None);
        }
        let payload_len = u32::from_be_bytes(src[..4].try_into().unwrap()) as usize;
        let encoded_len = MUX_HEADER_LEN.checked_add(payload_len).ok_or_else(|| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "response mux frame size overflow",
            )
        })?;
        if encoded_len > self.max_message_size {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "response mux frame size {encoded_len} exceeds maximum {}",
                    self.max_message_size
                ),
            ));
        }
        if src.len() < encoded_len {
            src.reserve(encoded_len - src.len());
            return Ok(None);
        }
        let kind = MuxFrameKind::try_from(src[4])?;
        let stream_id = Uuid::from_slice(&src[5..21]).map_err(|err| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid response mux stream UUID: {err}"),
            )
        })?;
        let mut encoded = src.split_to(encoded_len);
        let payload = encoded.split_off(MUX_HEADER_LEN).freeze();
        MuxFrame::validate(MuxFrame::new(kind, stream_id, payload)).map(Some)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_util::codec::{Decoder, Encoder};

    fn encode_compact(frame: MuxFrame) -> BytesMut {
        let mut bytes = BytesMut::new();
        MuxCodec::default().encode(frame, &mut bytes).unwrap();
        bytes
    }

    #[test]
    fn mux_frame_round_trip_preserves_uuid_and_payload() {
        let stream_id = Uuid::new_v4();
        let frame = MuxFrame::new(
            MuxFrameKind::Data,
            stream_id,
            Bytes::from_static(b"payload"),
        );
        let mut encoded = encode_compact(frame.clone());
        assert_eq!(
            MuxCodec::default().decode(&mut encoded).unwrap(),
            Some(frame)
        );
    }

    #[test]
    fn window_update_requires_four_bytes() {
        let frame = MuxFrame::new(
            MuxFrameKind::WindowUpdate,
            Uuid::new_v4(),
            Bytes::from_static(&[1, 2]),
        );
        let mut encoded = BytesMut::new();
        MuxCodec::default().encode(frame, &mut encoded).unwrap();
        assert!(MuxCodec::default().decode(&mut encoded).is_err());
    }

    #[test]
    fn connection_hello_round_trips_identity() {
        let frontend_server_id = Uuid::new_v4();
        let frame = MuxFrame::connection_hello(RESPONSE_MUX_VERSION, frontend_server_id);
        let mut encoded = encode_compact(frame.clone());
        let decoded = MuxCodec::default().decode(&mut encoded).unwrap().unwrap();
        assert_eq!(decoded, frame);
        assert_eq!(
            decoded.connection_identity().unwrap(),
            (RESPONSE_MUX_VERSION, frontend_server_id)
        );

        let mut invalid = encode_compact(MuxFrame::new(
            MuxFrameKind::ConnectionReady,
            Uuid::new_v4(),
            Bytes::new(),
        ));
        assert!(MuxCodec::default().decode(&mut invalid).is_err());
    }

    #[test]
    fn unknown_kind_is_rejected() {
        let mut encoded = encode_compact(MuxFrame::empty(MuxFrameKind::End, Uuid::new_v4()));
        encoded[4] = 99;
        assert!(MuxCodec::default().decode(&mut encoded).is_err());
    }

    #[test]
    fn partial_compact_frame_waits_for_remaining_bytes() {
        let expected = MuxFrame::new(
            MuxFrameKind::Data,
            Uuid::new_v4(),
            Bytes::from_static(b"partial"),
        );
        let encoded = encode_compact(expected.clone());
        let split = encoded.len() / 2;
        let mut input = BytesMut::from(&encoded[..split]);
        let mut codec = MuxCodec::default();
        assert!(codec.decode(&mut input).unwrap().is_none());
        input.extend_from_slice(&encoded[split..]);
        let decoded = codec.decode(&mut input).unwrap().unwrap();
        assert_eq!(decoded, expected);
    }

    #[test]
    fn concatenated_frames_decode_independently_and_route_by_uuid() {
        let first = MuxFrame::new(
            MuxFrameKind::Data,
            Uuid::new_v4(),
            Bytes::from_static(b"one"),
        );
        let second = MuxFrame::empty(MuxFrameKind::End, Uuid::new_v4());
        let mut input = encode_compact(first.clone());
        input.extend_from_slice(&encode_compact(second.clone()));
        let mut codec = MuxCodec::default();
        let decoded_first = codec.decode(&mut input).unwrap().unwrap();
        let decoded_second = codec.decode(&mut input).unwrap().unwrap();
        assert_eq!(decoded_first, first);
        assert_eq!(decoded_second, second);
        assert_ne!(decoded_first.stream_id, decoded_second.stream_id);
    }

    #[test]
    fn maximum_compact_size_is_enforced() {
        let frame = MuxFrame::new(
            MuxFrameKind::Data,
            Uuid::new_v4(),
            Bytes::from_static(b"bounded"),
        );
        let exact_len = MUX_HEADER_LEN + frame.payload.len();
        let mut bytes = BytesMut::new();
        assert!(
            MuxCodec::new(exact_len)
                .encode(frame.clone(), &mut bytes)
                .is_ok()
        );
        assert!(
            MuxCodec::new(exact_len - 1)
                .encode(frame, &mut BytesMut::new())
                .is_err()
        );
    }

    fn config(values: &[(&str, &str)]) -> anyhow::Result<ResponseMuxConfig> {
        let values = values
            .iter()
            .map(|(key, value)| ((*key).to_string(), (*value).to_string()))
            .collect::<std::collections::HashMap<_, _>>();
        ResponseMuxConfig::parse_with(|name| values.get(name).cloned())
    }

    #[test]
    fn response_mux_config_defaults_to_one_ms() {
        let config = config(&[]).unwrap();
        assert!(!config.packet_metrics);
        assert_eq!(config.batch_interval, Duration::from_millis(1));
        assert_eq!(config.batch_max_bytes, 65_536);
        assert_eq!(config.batch_max_frames, 64);
        assert_eq!(config.stream_window_bytes, 262_144);
    }

    #[test]
    fn response_mux_config_accepts_zero_delay_and_valid_overrides() {
        use crate::config::environment_names::tcp_response_stream as env;
        let config = config(&[
            (env::DYN_TCP_RESPONSE_PACKET_METRICS, "true"),
            (env::DYN_TCP_RESPONSE_BATCH_INTERVAL_MS, "0"),
            (env::DYN_TCP_RESPONSE_BATCH_MAX_BYTES, "8192"),
            (env::DYN_TCP_RESPONSE_BATCH_MAX_FRAMES, "8"),
        ])
        .unwrap();
        assert!(config.packet_metrics);
        assert_eq!(config.batch_interval, Duration::ZERO);
        assert_eq!(config.batch_max_bytes, 8192);
        assert_eq!(config.batch_max_frames, 8);
    }

    #[test]
    fn response_mux_config_rejects_malformed_and_over_100_ms() {
        use crate::config::environment_names::tcp_response_stream as env;
        assert!(config(&[(env::DYN_TCP_RESPONSE_BATCH_INTERVAL_MS, "nope")]).is_err());
        assert!(config(&[(env::DYN_TCP_RESPONSE_BATCH_INTERVAL_MS, "101")]).is_err());
        assert!(config(&[(env::DYN_TCP_RESPONSE_BATCH_INTERVAL_MS, "100")]).is_ok());
        assert!(config(&[(env::DYN_TCP_RESPONSE_PACKET_METRICS, "maybe")]).is_err());
    }
}
