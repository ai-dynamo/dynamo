// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Trace reader for `.pftrace.gz` files produced by [`TraceWriter`].
//!
//! Parses the Perfetto protobuf wire format and extracts slices (begin/end
//! pairs), instants, track descriptors, and interned names. The output is a
//! flat list of [`Slice`] structs suitable for DAG construction and report
//! generation.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use crate::perfetto;

// ── Public types ──────────────────────────────────────────────────────────────

/// A resolved time-bounded range from a Perfetto trace.
#[derive(Debug, Clone)]
pub struct Slice {
    pub stage: String,
    pub traceparent: String,
    pub run_id: String,
    pub start_ns: u64,
    pub end_ns: u64,
    pub duration_ns: u64,
    pub track_uuid: u64,
    pub process_name: String,
    pub thread_name: String,
    pub pid: u32,
    pub tid: u32,
}

/// An instant event (no duration).
#[derive(Debug, Clone)]
pub struct Instant {
    pub stage: String,
    pub traceparent: String,
    pub timestamp_ns: u64,
    pub track_uuid: u64,
    pub process_name: String,
}

/// Track metadata recovered from TrackDescriptor packets.
#[derive(Debug, Clone, Default)]
pub struct TrackInfo {
    pub name: String,
    pub parent_uuid: u64,
    pub pid: u32,
    pub tid: u32,
    pub is_process: bool,
}

/// Complete parsed trace from one component file.
#[derive(Debug, Clone)]
pub struct ParsedTrace {
    pub source_file: String,
    pub slices: Vec<Slice>,
    pub instants: Vec<Instant>,
}

// ── Reader implementation ─────────────────────────────────────────────────────

/// Read and parse a `.pftrace.gz` file, returning all slices and instants.
pub fn read_trace(path: &Path) -> anyhow::Result<ParsedTrace> {
    let file = std::fs::File::open(path)?;
    let mut decoder = flate2::read::GzDecoder::new(file);
    let mut buf = Vec::new();
    match decoder.read_to_end(&mut buf) {
        Ok(_) => {}
        Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => {
            // Truncated gzip stream (writer never called finish()) — use whatever was decoded
        }
        Err(e) => return Err(e.into()),
    }

    let source = path
        .file_name()
        .map(|f| f.to_string_lossy().to_string())
        .unwrap_or_default();

    parse_trace_bytes(&buf, &source)
}

/// Parse raw (uncompressed) Perfetto trace bytes.
pub fn parse_trace_bytes(buf: &[u8], source: &str) -> anyhow::Result<ParsedTrace> {
    let mut tracks: HashMap<u64, TrackInfo> = HashMap::new();
    let mut interned_names: HashMap<(u32, u64), String> = HashMap::new(); // (seq_id, iid) -> name
    let mut open_slices: HashMap<u64, Vec<PendingSlice>> = HashMap::new(); // track_uuid -> stack
    let mut slices = Vec::new();
    let mut instants = Vec::new();

    // Walk the top-level Trace message: repeated field 1 = TracePacket
    let mut off = 0;
    while off < buf.len() {
        let (field, wire, new_off) = match perfetto::read_tag(buf, off) {
            Some(v) => v,
            None => break,
        };
        if field == 1 && wire == 2 {
            // LEN-delimited TracePacket
            let (len, data_off) = match perfetto::read_varint(buf, new_off) {
                Some(v) => v,
                None => break,
            };
            let pkt_end = data_off + len as usize;
            if pkt_end > buf.len() {
                break;
            }
            parse_trace_packet(
                &buf[data_off..pkt_end],
                &mut tracks,
                &mut interned_names,
                &mut open_slices,
                &mut slices,
                &mut instants,
            );
            off = pkt_end;
        } else {
            off = match perfetto::skip_field(buf, new_off, wire) {
                Some(o) => o,
                None => break,
            };
        }
    }

    // Resolve track names on slices and instants
    resolve_names(&mut slices, &mut instants, &tracks);

    Ok(ParsedTrace {
        source_file: source.to_string(),
        slices,
        instants,
    })
}

// ── Internal types ────────────────────────────────────────────────────────────

struct PendingSlice {
    stage: String,
    traceparent: String,
    run_id: String,
    start_ns: u64,
    #[allow(dead_code)]
    name_iid: u64,
    track_uuid: u64,
}

// ── Packet parsing ────────────────────────────────────────────────────────────

fn parse_trace_packet(
    pkt: &[u8],
    tracks: &mut HashMap<u64, TrackInfo>,
    interned_names: &mut HashMap<(u32, u64), String>,
    open_slices: &mut HashMap<u64, Vec<PendingSlice>>,
    slices: &mut Vec<Slice>,
    instants: &mut Vec<Instant>,
) {
    let mut timestamp: u64 = 0;
    let mut seq_id: u32 = 0;
    let mut track_descriptor: Option<&[u8]> = None;
    let mut track_event: Option<&[u8]> = None;
    let mut interned_data: Option<&[u8]> = None;

    let mut off = 0;
    while off < pkt.len() {
        let (field, wire, new_off) = match perfetto::read_tag(pkt, off) {
            Some(v) => v,
            None => break,
        };
        match (field, wire) {
            (perfetto::trace_packet::TIMESTAMP, 0) => {
                let (v, o) = perfetto::read_varint(pkt, new_off).unwrap_or((0, new_off));
                timestamp = v;
                off = o;
            }
            (perfetto::trace_packet::TRUSTED_PACKET_SEQUENCE_ID, 0) => {
                let (v, o) = perfetto::read_varint(pkt, new_off).unwrap_or((0, new_off));
                seq_id = v as u32;
                off = o;
            }
            (perfetto::trace_packet::TRACK_DESCRIPTOR, 2) => {
                let (len, data_off) = perfetto::read_varint(pkt, new_off).unwrap_or((0, new_off));
                track_descriptor = Some(&pkt[data_off..data_off + len as usize]);
                off = data_off + len as usize;
            }
            (perfetto::trace_packet::TRACK_EVENT, 2) => {
                let (len, data_off) = perfetto::read_varint(pkt, new_off).unwrap_or((0, new_off));
                track_event = Some(&pkt[data_off..data_off + len as usize]);
                off = data_off + len as usize;
            }
            (perfetto::trace_packet::INTERNED_DATA, 2) => {
                let (len, data_off) = perfetto::read_varint(pkt, new_off).unwrap_or((0, new_off));
                interned_data = Some(&pkt[data_off..data_off + len as usize]);
                off = data_off + len as usize;
            }
            _ => {
                off = perfetto::skip_field(pkt, new_off, wire).unwrap_or(pkt.len());
            }
        }
    }

    if let Some(td_bytes) = track_descriptor {
        parse_track_descriptor(td_bytes, tracks);
    }
    if let Some(id_bytes) = interned_data {
        parse_interned_data(id_bytes, seq_id, interned_names);
    }
    if let Some(te_bytes) = track_event {
        parse_track_event(
            te_bytes,
            timestamp,
            seq_id,
            interned_names,
            open_slices,
            slices,
            instants,
        );
    }
}

fn parse_track_descriptor(buf: &[u8], tracks: &mut HashMap<u64, TrackInfo>) {
    let mut uuid: u64 = 0;
    let mut parent_uuid: u64 = 0;
    let mut name = String::new();
    let mut pid: u32 = 0;
    let mut tid: u32 = 0;
    let mut is_process = false;

    let mut off = 0;
    while off < buf.len() {
        let (field, wire, new_off) = match perfetto::read_tag(buf, off) {
            Some(v) => v,
            None => break,
        };
        match (field, wire) {
            (perfetto::track_descriptor::UUID, 0) => {
                let (v, o) = perfetto::read_varint(buf, new_off).unwrap_or((0, new_off));
                uuid = v;
                off = o;
            }
            (perfetto::track_descriptor::PARENT_UUID, 0) => {
                let (v, o) = perfetto::read_varint(buf, new_off).unwrap_or((0, new_off));
                parent_uuid = v;
                off = o;
            }
            (perfetto::track_descriptor::NAME, 2) => {
                let (len, data_off) = perfetto::read_varint(buf, new_off).unwrap_or((0, new_off));
                name = String::from_utf8_lossy(&buf[data_off..data_off + len as usize]).to_string();
                off = data_off + len as usize;
            }
            (perfetto::track_descriptor::PROCESS, 2) => {
                let (len, data_off) = perfetto::read_varint(buf, new_off).unwrap_or((0, new_off));
                is_process = true;
                pid = parse_process_pid(&buf[data_off..data_off + len as usize]);
                off = data_off + len as usize;
            }
            (perfetto::track_descriptor::THREAD, 2) => {
                let (len, data_off) = perfetto::read_varint(buf, new_off).unwrap_or((0, new_off));
                let (p, t) = parse_thread_ids(&buf[data_off..data_off + len as usize]);
                pid = p;
                tid = t;
                off = data_off + len as usize;
            }
            _ => {
                off = perfetto::skip_field(buf, new_off, wire).unwrap_or(buf.len());
            }
        }
    }

    tracks.insert(
        uuid,
        TrackInfo {
            name,
            parent_uuid,
            pid,
            tid,
            is_process,
        },
    );
}

fn parse_process_pid(buf: &[u8]) -> u32 {
    let mut off = 0;
    while off < buf.len() {
        let (field, wire, new_off) = match perfetto::read_tag(buf, off) {
            Some(v) => v,
            None => break,
        };
        if field == perfetto::process_descriptor::PID && wire == 0 {
            return perfetto::read_varint(buf, new_off).map(|(v, _)| v as u32).unwrap_or(0);
        }
        off = perfetto::skip_field(buf, new_off, wire).unwrap_or(buf.len());
    }
    0
}

fn parse_thread_ids(buf: &[u8]) -> (u32, u32) {
    let mut pid: u32 = 0;
    let mut tid: u32 = 0;
    let mut off = 0;
    while off < buf.len() {
        let (field, wire, new_off) = match perfetto::read_tag(buf, off) {
            Some(v) => v,
            None => break,
        };
        match (field, wire) {
            (perfetto::thread_descriptor::PID, 0) => {
                pid = perfetto::read_varint(buf, new_off).map(|(v, _)| v as u32).unwrap_or(0);
                off = perfetto::read_varint(buf, new_off).map(|(_, o)| o).unwrap_or(buf.len());
            }
            (perfetto::thread_descriptor::TID, 0) => {
                tid = perfetto::read_varint(buf, new_off).map(|(v, _)| v as u32).unwrap_or(0);
                off = perfetto::read_varint(buf, new_off).map(|(_, o)| o).unwrap_or(buf.len());
            }
            _ => {
                off = perfetto::skip_field(buf, new_off, wire).unwrap_or(buf.len());
            }
        }
    }
    (pid, tid)
}

fn parse_interned_data(
    buf: &[u8],
    seq_id: u32,
    names: &mut HashMap<(u32, u64), String>,
) {
    let mut off = 0;
    while off < buf.len() {
        let (field, wire, new_off) = match perfetto::read_tag(buf, off) {
            Some(v) => v,
            None => break,
        };
        if field == perfetto::interned_data::EVENT_NAMES && wire == 2 {
            let (len, data_off) = perfetto::read_varint(buf, new_off).unwrap_or((0, new_off));
            let entry = &buf[data_off..data_off + len as usize];
            if let Some((iid, name)) = parse_interned_string(entry) {
                names.insert((seq_id, iid), name);
            }
            off = data_off + len as usize;
        } else {
            off = perfetto::skip_field(buf, new_off, wire).unwrap_or(buf.len());
        }
    }
}

fn parse_interned_string(buf: &[u8]) -> Option<(u64, String)> {
    let mut iid: u64 = 0;
    let mut name = String::new();
    let mut off = 0;
    while off < buf.len() {
        let (field, wire, new_off) = match perfetto::read_tag(buf, off) {
            Some(v) => v,
            None => break,
        };
        match (field, wire) {
            (perfetto::interned_string::IID, 0) => {
                let (v, o) = perfetto::read_varint(buf, new_off)?;
                iid = v;
                off = o;
            }
            (perfetto::interned_string::NAME, 2) => {
                let (len, data_off) = perfetto::read_varint(buf, new_off)?;
                name = String::from_utf8_lossy(&buf[data_off..data_off + len as usize]).to_string();
                off = data_off + len as usize;
            }
            _ => {
                off = perfetto::skip_field(buf, new_off, wire).unwrap_or(buf.len());
            }
        }
    }
    if iid > 0 { Some((iid, name)) } else { None }
}

fn parse_track_event(
    buf: &[u8],
    timestamp: u64,
    seq_id: u32,
    interned_names: &HashMap<(u32, u64), String>,
    open_slices: &mut HashMap<u64, Vec<PendingSlice>>,
    slices: &mut Vec<Slice>,
    instants: &mut Vec<Instant>,
) {
    let mut track_uuid: u64 = 0;
    let mut event_type: u64 = 0;
    let mut name_iid: u64 = 0;
    let mut traceparent = String::new();
    let mut run_id = String::new();

    let mut off = 0;
    while off < buf.len() {
        let (field, wire, new_off) = match perfetto::read_tag(buf, off) {
            Some(v) => v,
            None => break,
        };
        match (field, wire) {
            (perfetto::track_event::TRACK_UUID, 0) => {
                let (v, o) = perfetto::read_varint(buf, new_off).unwrap_or((0, new_off));
                track_uuid = v;
                off = o;
            }
            (perfetto::track_event::TYPE, 0) => {
                let (v, o) = perfetto::read_varint(buf, new_off).unwrap_or((0, new_off));
                event_type = v;
                off = o;
            }
            (perfetto::track_event::NAME_IID, 0) => {
                let (v, o) = perfetto::read_varint(buf, new_off).unwrap_or((0, new_off));
                name_iid = v;
                off = o;
            }
            (perfetto::track_event::DEBUG_ANNOTATIONS, 2) => {
                let (len, data_off) = perfetto::read_varint(buf, new_off).unwrap_or((0, new_off));
                let ann = &buf[data_off..data_off + len as usize];
                let (k, v) = parse_debug_annotation(ann);
                match k.as_str() {
                    "traceparent" => traceparent = v,
                    "dynamo.run_id" => run_id = v,
                    _ => {}
                }
                off = data_off + len as usize;
            }
            _ => {
                off = perfetto::skip_field(buf, new_off, wire).unwrap_or(buf.len());
            }
        }
    }

    let stage_name = interned_names
        .get(&(seq_id, name_iid))
        .cloned()
        .unwrap_or_default();

    match event_type {
        perfetto::track_event::TYPE_SLICE_BEGIN => {
            open_slices
                .entry(track_uuid)
                .or_default()
                .push(PendingSlice {
                    stage: stage_name,
                    traceparent,
                    run_id,
                    start_ns: timestamp,
                    name_iid,
                    track_uuid,
                });
        }
        perfetto::track_event::TYPE_SLICE_END => {
            if let Some(stack) = open_slices.get_mut(&track_uuid) {
                if let Some(pending) = stack.pop() {
                    let duration = timestamp.saturating_sub(pending.start_ns);
                    slices.push(Slice {
                        stage: pending.stage,
                        traceparent: pending.traceparent,
                        run_id: pending.run_id,
                        start_ns: pending.start_ns,
                        end_ns: timestamp,
                        duration_ns: duration,
                        track_uuid: pending.track_uuid,
                        process_name: String::new(),
                        thread_name: String::new(),
                        pid: 0,
                        tid: 0,
                    });
                }
            }
        }
        perfetto::track_event::TYPE_INSTANT => {
            instants.push(Instant {
                stage: stage_name,
                traceparent,
                timestamp_ns: timestamp,
                track_uuid,
                process_name: String::new(),
            });
        }
        _ => {}
    }
}

fn parse_debug_annotation(buf: &[u8]) -> (String, String) {
    let mut name = String::new();
    let mut str_val = String::new();
    let mut off = 0;
    while off < buf.len() {
        let (field, wire, new_off) = match perfetto::read_tag(buf, off) {
            Some(v) => v,
            None => break,
        };
        match (field, wire) {
            (perfetto::debug_annotation::NAME, 2) => {
                let (len, data_off) = perfetto::read_varint(buf, new_off).unwrap_or((0, new_off));
                name = String::from_utf8_lossy(&buf[data_off..data_off + len as usize]).to_string();
                off = data_off + len as usize;
            }
            (perfetto::debug_annotation::STRING_VALUE, 2) => {
                let (len, data_off) = perfetto::read_varint(buf, new_off).unwrap_or((0, new_off));
                str_val =
                    String::from_utf8_lossy(&buf[data_off..data_off + len as usize]).to_string();
                off = data_off + len as usize;
            }
            _ => {
                off = perfetto::skip_field(buf, new_off, wire).unwrap_or(buf.len());
            }
        }
    }
    (name, str_val)
}

fn resolve_names(
    slices: &mut [Slice],
    instants: &mut [Instant],
    tracks: &HashMap<u64, TrackInfo>,
) {
    for slice in slices.iter_mut() {
        if let Some(track) = tracks.get(&slice.track_uuid) {
            slice.thread_name = track.name.clone();
            slice.tid = track.tid;
            // Walk up to process track
            if let Some(parent) = tracks.get(&track.parent_uuid) {
                slice.process_name = parent.name.clone();
                slice.pid = parent.pid;
            }
        }
    }
    for inst in instants.iter_mut() {
        if let Some(track) = tracks.get(&inst.track_uuid) {
            if let Some(parent) = tracks.get(&track.parent_uuid) {
                inst.process_name = parent.name.clone();
            } else if track.is_process {
                inst.process_name = track.name.clone();
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::writer::TraceWriter;

    #[test]
    fn roundtrip_write_and_read() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.pftrace.gz");

        let writer =
            TraceWriter::new(path.clone(), "engine-prefill-0", "node-0", 1234, 1).unwrap();

        let iid_compute = writer.intern_name("dynamo.prefill.compute");
        let iid_recv = writer.intern_name("dynamo.frontend.recv");

        let tp = "00-deadbeef00000001-000000000000beef-01";
        let ann_tp = perfetto::build_debug_annotation_str("traceparent", tp);
        let ann_rid = perfetto::build_debug_annotation_str("dynamo.run_id", "test-run-1");

        // Write two nested slices
        let t0 = 1_000_000_000u64;
        writer.write_slice_begin(t0, iid_recv, &[ann_tp.clone(), ann_rid.clone()]);
        writer.write_slice_begin(t0 + 1_000_000, iid_compute, &[ann_tp.clone(), ann_rid.clone()]);
        writer.write_slice_end(t0 + 5_000_000);
        writer.write_slice_end(t0 + 6_000_000);

        // Write an instant
        let iid_first = writer.intern_name("dynamo.decode.first_token");
        let ann_tp2 = perfetto::build_debug_annotation_str("traceparent", tp);
        writer.write_instant(t0 + 7_000_000, iid_first, &[ann_tp2]);

        writer.finish().unwrap();

        // Read back
        let parsed = read_trace(&path).unwrap();
        assert_eq!(parsed.slices.len(), 2, "should have 2 slices");
        assert_eq!(parsed.instants.len(), 1, "should have 1 instant");

        // Check outer slice
        let recv = parsed
            .slices
            .iter()
            .find(|s| s.stage == "dynamo.frontend.recv")
            .expect("should find recv slice");
        assert_eq!(recv.traceparent, tp);
        assert_eq!(recv.duration_ns, 6_000_000);
        assert_eq!(recv.process_name, "engine-prefill-0@node-0");

        // Check inner slice
        let compute = parsed
            .slices
            .iter()
            .find(|s| s.stage == "dynamo.prefill.compute")
            .expect("should find compute slice");
        assert_eq!(compute.duration_ns, 4_000_000);

        // Check instant
        assert_eq!(parsed.instants[0].stage, "dynamo.decode.first_token");
    }
}
