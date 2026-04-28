// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Raw protobuf wire-format encoding for the Perfetto trace format.
//!
//! Encodes Perfetto `Trace`, `TracePacket`, `TrackDescriptor`, `TrackEvent`,
//! `ClockSnapshot`, and `InternedData` messages without any protobuf library
//! dependency. Uses field numbers from the Perfetto proto definitions.

/// Protobuf wire types.
const WIRE_VARINT: u32 = 0;
const WIRE_64BIT: u32 = 1;
const WIRE_LEN: u32 = 2;

// ── Encoding helpers ────────────────────────────────────────────────────────

pub fn encode_varint(buf: &mut Vec<u8>, mut val: u64) {
    loop {
        let byte = (val & 0x7F) as u8;
        val >>= 7;
        if val == 0 {
            buf.push(byte);
            return;
        }
        buf.push(byte | 0x80);
    }
}

fn encode_tag(buf: &mut Vec<u8>, field: u32, wire_type: u32) {
    encode_varint(buf, ((field as u64) << 3) | wire_type as u64);
}

pub fn encode_varint_field(buf: &mut Vec<u8>, field: u32, val: u64) {
    if val == 0 {
        return;
    }
    encode_tag(buf, field, WIRE_VARINT);
    encode_varint(buf, val);
}

pub fn encode_sint64_field(buf: &mut Vec<u8>, field: u32, val: i64) {
    let zigzag = ((val << 1) ^ (val >> 63)) as u64;
    encode_varint_field(buf, field, zigzag);
}

pub fn encode_fixed64_field(buf: &mut Vec<u8>, field: u32, val: u64) {
    encode_tag(buf, field, WIRE_64BIT);
    buf.extend_from_slice(&val.to_le_bytes());
}

pub fn encode_double_field(buf: &mut Vec<u8>, field: u32, val: f64) {
    encode_fixed64_field(buf, field, val.to_bits());
}

pub fn encode_string_field(buf: &mut Vec<u8>, field: u32, val: &str) {
    if val.is_empty() {
        return;
    }
    encode_tag(buf, field, WIRE_LEN);
    encode_varint(buf, val.len() as u64);
    buf.extend_from_slice(val.as_bytes());
}

pub fn encode_bytes_field(buf: &mut Vec<u8>, field: u32, val: &[u8]) {
    if val.is_empty() {
        return;
    }
    encode_tag(buf, field, WIRE_LEN);
    encode_varint(buf, val.len() as u64);
    buf.extend_from_slice(val);
}

pub fn encode_submessage(buf: &mut Vec<u8>, field: u32, inner: &[u8]) {
    encode_tag(buf, field, WIRE_LEN);
    encode_varint(buf, inner.len() as u64);
    buf.extend_from_slice(inner);
}

// ── Decoding helpers ────────────────────────────────────────────────────────

pub fn read_varint(buf: &[u8], mut off: usize) -> Option<(u64, usize)> {
    let mut result: u64 = 0;
    let mut shift = 0;
    loop {
        if off >= buf.len() {
            return None;
        }
        let b = buf[off];
        off += 1;
        result |= ((b & 0x7F) as u64) << shift;
        if b & 0x80 == 0 {
            return Some((result, off));
        }
        shift += 7;
        if shift >= 64 {
            return None;
        }
    }
}

pub fn read_tag(buf: &[u8], off: usize) -> Option<(u32, u32, usize)> {
    let (tag, off) = read_varint(buf, off)?;
    Some(((tag >> 3) as u32, (tag & 0x07) as u32, off))
}

pub fn skip_field(buf: &[u8], off: usize, wire_type: u32) -> Option<usize> {
    match wire_type {
        0 => read_varint(buf, off).map(|(_, o)| o),
        1 => Some(off + 8),
        2 => {
            let (len, off) = read_varint(buf, off)?;
            Some(off + len as usize)
        }
        5 => Some(off + 4),
        _ => None,
    }
}

// ── Perfetto-specific message builders ──────────────────────────────────────

/// Perfetto TracePacket field numbers.
pub mod trace_packet {
    pub const TIMESTAMP: u32 = 8;
    pub const TIMESTAMP_CLOCK_ID: u32 = 58;
    pub const TRACK_DESCRIPTOR: u32 = 60;
    pub const TRACK_EVENT: u32 = 11;
    pub const CLOCK_SNAPSHOT: u32 = 6;
    pub const INTERNED_DATA: u32 = 12;
    pub const TRUSTED_PACKET_SEQUENCE_ID: u32 = 10;
    pub const SEQUENCE_FLAGS: u32 = 13;
    pub const INCREMENTAL_STATE_CLEARED: u32 = 1;
}

/// Perfetto TrackDescriptor field numbers.
pub mod track_descriptor {
    pub const UUID: u32 = 1;
    pub const PARENT_UUID: u32 = 5;
    pub const NAME: u32 = 2;
    pub const PROCESS: u32 = 3;
    pub const THREAD: u32 = 4;
}

/// Perfetto ProcessDescriptor field numbers.
pub mod process_descriptor {
    pub const PID: u32 = 1;
    pub const PROCESS_NAME: u32 = 6;
}

/// Perfetto ThreadDescriptor field numbers.
pub mod thread_descriptor {
    pub const PID: u32 = 1;
    pub const TID: u32 = 2;
    pub const THREAD_NAME: u32 = 5;
}

/// Perfetto TrackEvent field numbers.
pub mod track_event {
    pub const TRACK_UUID: u32 = 11;
    pub const NAME_IID: u32 = 10;
    pub const TYPE: u32 = 9;
    pub const DEBUG_ANNOTATIONS: u32 = 4;
    pub const FLOW_IDS: u32 = 47;

    pub const TYPE_SLICE_BEGIN: u64 = 1;
    pub const TYPE_SLICE_END: u64 = 2;
    pub const TYPE_INSTANT: u64 = 3;
    pub const TYPE_COUNTER: u64 = 4;
}

/// Perfetto DebugAnnotation field numbers.
pub mod debug_annotation {
    pub const NAME_IID: u32 = 1;
    pub const STRING_VALUE: u32 = 6;
    pub const INT_VALUE: u32 = 7;
    pub const DOUBLE_VALUE: u32 = 8;
    pub const NAME: u32 = 10;
}

/// Perfetto InternedData field numbers.
pub mod interned_data {
    pub const EVENT_NAMES: u32 = 2;
    pub const DEBUG_ANNOTATION_NAMES: u32 = 3;
}

/// Perfetto EventName / DebugAnnotationName field numbers.
pub mod interned_string {
    pub const IID: u32 = 1;
    pub const NAME: u32 = 2;
}

/// Perfetto ClockSnapshot field numbers.
pub mod clock_snapshot {
    pub const CLOCKS: u32 = 1;
}

/// Perfetto ClockSnapshot.Clock field numbers.
pub mod clock {
    pub const CLOCK_ID: u32 = 1;
    pub const TIMESTAMP: u32 = 2;
}

/// Build a TrackDescriptor for a process.
pub fn build_process_track(uuid: u64, pid: u32, name: &str) -> Vec<u8> {
    let mut process = Vec::new();
    encode_varint_field(&mut process, process_descriptor::PID, pid as u64);
    encode_string_field(&mut process, process_descriptor::PROCESS_NAME, name);

    let mut td = Vec::new();
    encode_varint_field(&mut td, track_descriptor::UUID, uuid);
    encode_string_field(&mut td, track_descriptor::NAME, name);
    encode_submessage(&mut td, track_descriptor::PROCESS, &process);
    td
}

/// Build a TrackDescriptor for a thread under a process track.
pub fn build_thread_track(uuid: u64, parent_uuid: u64, pid: u32, tid: u32, name: &str) -> Vec<u8> {
    let mut thread = Vec::new();
    encode_varint_field(&mut thread, thread_descriptor::PID, pid as u64);
    encode_varint_field(&mut thread, thread_descriptor::TID, tid as u64);
    encode_string_field(&mut thread, thread_descriptor::THREAD_NAME, name);

    let mut td = Vec::new();
    encode_varint_field(&mut td, track_descriptor::UUID, uuid);
    encode_varint_field(&mut td, track_descriptor::PARENT_UUID, parent_uuid);
    encode_string_field(&mut td, track_descriptor::NAME, name);
    encode_submessage(&mut td, track_descriptor::THREAD, &thread);
    td
}

/// Build a DebugAnnotation with a string value.
pub fn build_debug_annotation_str(name: &str, value: &str) -> Vec<u8> {
    let mut da = Vec::new();
    encode_string_field(&mut da, debug_annotation::NAME, name);
    encode_string_field(&mut da, debug_annotation::STRING_VALUE, value);
    da
}

/// Build a DebugAnnotation with an integer value.
pub fn build_debug_annotation_int(name: &str, value: i64) -> Vec<u8> {
    let mut da = Vec::new();
    encode_string_field(&mut da, debug_annotation::NAME, name);
    encode_sint64_field(&mut da, debug_annotation::INT_VALUE, value);
    da
}

/// Build a TrackEvent (SLICE_BEGIN or SLICE_END).
pub fn build_track_event(
    track_uuid: u64,
    event_type: u64,
    name_iid: u64,
    annotations: &[Vec<u8>],
) -> Vec<u8> {
    let mut te = Vec::new();
    encode_varint_field(&mut te, track_event::TRACK_UUID, track_uuid);
    encode_varint_field(&mut te, track_event::TYPE, event_type);
    if name_iid > 0 {
        encode_varint_field(&mut te, track_event::NAME_IID, name_iid);
    }
    for ann in annotations {
        encode_submessage(&mut te, track_event::DEBUG_ANNOTATIONS, ann);
    }
    te
}

/// Build a full TracePacket wrapping a TrackEvent.
pub fn build_trace_packet_event(
    timestamp_ns: u64,
    seq_id: u32,
    track_event_bytes: &[u8],
) -> Vec<u8> {
    let mut pkt = Vec::new();
    encode_varint_field(&mut pkt, trace_packet::TIMESTAMP, timestamp_ns);
    encode_varint_field(
        &mut pkt,
        trace_packet::TRUSTED_PACKET_SEQUENCE_ID,
        seq_id as u64,
    );
    encode_submessage(&mut pkt, trace_packet::TRACK_EVENT, track_event_bytes);
    pkt
}

/// Build a full TracePacket wrapping a TrackDescriptor.
pub fn build_trace_packet_descriptor(seq_id: u32, descriptor_bytes: &[u8]) -> Vec<u8> {
    let mut pkt = Vec::new();
    encode_varint_field(
        &mut pkt,
        trace_packet::TRUSTED_PACKET_SEQUENCE_ID,
        seq_id as u64,
    );
    encode_submessage(&mut pkt, trace_packet::TRACK_DESCRIPTOR, descriptor_bytes);
    pkt
}

/// Build a ClockSnapshot packet.
pub fn build_clock_snapshot(clocks: &[(u32, u64)]) -> Vec<u8> {
    let mut cs = Vec::new();
    for &(clock_id, timestamp) in clocks {
        let mut c = Vec::new();
        encode_varint_field(&mut c, clock::CLOCK_ID, clock_id as u64);
        encode_varint_field(&mut c, clock::TIMESTAMP, timestamp);
        encode_submessage(&mut cs, clock_snapshot::CLOCKS, &c);
    }

    let mut pkt = Vec::new();
    encode_submessage(&mut pkt, trace_packet::CLOCK_SNAPSHOT, &cs);
    pkt
}

/// Build an InternedData packet with event names.
pub fn build_interned_event_names(seq_id: u32, names: &[(u64, &str)]) -> Vec<u8> {
    let mut id = Vec::new();
    for &(iid, name) in names {
        let mut entry = Vec::new();
        encode_varint_field(&mut entry, interned_string::IID, iid);
        encode_string_field(&mut entry, interned_string::NAME, name);
        encode_submessage(&mut id, interned_data::EVENT_NAMES, &entry);
    }

    let mut pkt = Vec::new();
    encode_varint_field(
        &mut pkt,
        trace_packet::TRUSTED_PACKET_SEQUENCE_ID,
        seq_id as u64,
    );
    encode_submessage(&mut pkt, trace_packet::INTERNED_DATA, &id);
    encode_varint_field(
        &mut pkt,
        trace_packet::SEQUENCE_FLAGS,
        trace_packet::INCREMENTAL_STATE_CLEARED as u64,
    );
    pkt
}

/// Wrap raw packet bytes into a Trace.packet field (field 1 = LEN-delimited).
pub fn wrap_as_trace_packet(packet_bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    encode_submessage(&mut out, 1, packet_bytes);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_roundtrip() {
        for val in [0u64, 1, 127, 128, 300, 16384, u64::MAX] {
            let mut buf = Vec::new();
            encode_varint(&mut buf, val);
            let (decoded, end) = read_varint(&buf, 0).unwrap();
            assert_eq!(decoded, val);
            assert_eq!(end, buf.len());
        }
    }

    #[test]
    fn tag_roundtrip() {
        let mut buf = Vec::new();
        encode_tag(&mut buf, 11, WIRE_LEN);
        let (field, wire, _) = read_tag(&buf, 0).unwrap();
        assert_eq!(field, 11);
        assert_eq!(wire, WIRE_LEN);
    }

    #[test]
    fn process_track_descriptor() {
        let td = build_process_track(42, 1234, "engine-prefill-0");
        assert!(!td.is_empty());
        // Should be valid protobuf — verify we can skip all fields
        let mut off = 0;
        while off < td.len() {
            let (_, wire, new_off) = read_tag(&td, off).unwrap();
            off = skip_field(&td, new_off, wire).unwrap();
        }
    }

    #[test]
    fn trace_packet_event() {
        let te = build_track_event(1, track_event::TYPE_SLICE_BEGIN, 1, &[]);
        let pkt = build_trace_packet_event(1_000_000, 1, &te);
        let trace = wrap_as_trace_packet(&pkt);
        assert!(!trace.is_empty());
    }
}
