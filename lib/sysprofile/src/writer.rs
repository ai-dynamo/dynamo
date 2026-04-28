// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Per-process Perfetto trace writer.
//!
//! Writes `.pftrace.gz` files containing TrackDescriptor and TrackEvent
//! packets. Thread-safe via interior locking. Each writer has a unique
//! `trusted_packet_sequence_id` so the Perfetto UI does not collapse
//! events from different writers.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use flate2::write::GzEncoder;
use flate2::Compression;

use crate::perfetto;

struct WriterInner {
    encoder: GzEncoder<File>,
    seq_id: u32,
    process_track_uuid: u64,
    interned_names: HashMap<String, u64>,
    next_iid: u64,
    next_track_uuid: u64,
    thread_tracks: HashMap<u64, u64>,
    packets_written: u64,
}

pub struct TraceWriter {
    inner: Mutex<WriterInner>,
}

impl TraceWriter {
    pub fn new(
        output_path: PathBuf,
        component_name: &str,
        host: &str,
        pid: u32,
        seq_id: u32,
    ) -> anyhow::Result<Self> {
        if let Some(parent) = output_path.parent() {
            std::fs::create_dir_all(parent)?;
        }

        let file = File::create(&output_path)?;
        let mut encoder = GzEncoder::new(file, Compression::fast());

        let process_track_uuid = hash_uuid(host, component_name, pid);

        let td = perfetto::build_process_track(
            process_track_uuid,
            pid,
            &format!("{component_name}@{host}"),
        );
        let pkt = perfetto::build_trace_packet_descriptor(seq_id, &td);
        let trace_bytes = perfetto::wrap_as_trace_packet(&pkt);
        encoder.write_all(&trace_bytes)?;

        let now_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        let cs = perfetto::build_clock_snapshot(&[
            (6, now_ns), // CLOCK_BOOTTIME (clock id 6)
        ]);
        let cs_trace = perfetto::wrap_as_trace_packet(&cs);
        encoder.write_all(&cs_trace)?;

        Ok(Self {
            inner: Mutex::new(WriterInner {
                encoder,
                seq_id,
                process_track_uuid,
                interned_names: HashMap::new(),
                next_iid: 1,
                next_track_uuid: process_track_uuid + 1,
                thread_tracks: HashMap::new(),
                packets_written: 2,
            }),
        })
    }

    pub fn intern_name(&self, name: &str) -> u64 {
        let mut inner = self.inner.lock().unwrap();
        if let Some(&iid) = inner.interned_names.get(name) {
            return iid;
        }
        let iid = inner.next_iid;
        inner.next_iid += 1;
        inner.interned_names.insert(name.to_string(), iid);

        let interned =
            perfetto::build_interned_event_names(inner.seq_id, &[(iid, name)]);
        let trace_bytes = perfetto::wrap_as_trace_packet(&interned);
        let _ = inner.encoder.write_all(&trace_bytes);
        inner.packets_written += 1;

        iid
    }

    fn ensure_thread_track(&self, inner: &mut WriterInner, tid: u64) -> u64 {
        if let Some(&uuid) = inner.thread_tracks.get(&tid) {
            return uuid;
        }
        let uuid = inner.next_track_uuid;
        inner.next_track_uuid += 1;
        inner.thread_tracks.insert(tid, uuid);

        let td = perfetto::build_thread_track(
            uuid,
            inner.process_track_uuid,
            std::process::id(),
            tid as u32,
            &format!("thread-{tid}"),
        );
        let pkt = perfetto::build_trace_packet_descriptor(inner.seq_id, &td);
        let trace_bytes = perfetto::wrap_as_trace_packet(&pkt);
        let _ = inner.encoder.write_all(&trace_bytes);
        inner.packets_written += 1;

        uuid
    }

    pub fn write_slice_begin(
        &self,
        timestamp_ns: u64,
        name_iid: u64,
        annotations: &[Vec<u8>],
    ) {
        let mut inner = self.inner.lock().unwrap();
        let tid = current_thread_id();
        let track_uuid = self.ensure_thread_track(&mut inner, tid);

        let te = perfetto::build_track_event(
            track_uuid,
            perfetto::track_event::TYPE_SLICE_BEGIN,
            name_iid,
            annotations,
        );
        let pkt = perfetto::build_trace_packet_event(timestamp_ns, inner.seq_id, &te);
        let trace_bytes = perfetto::wrap_as_trace_packet(&pkt);
        let _ = inner.encoder.write_all(&trace_bytes);
        inner.packets_written += 1;
    }

    pub fn write_slice_end(&self, timestamp_ns: u64) {
        let mut inner = self.inner.lock().unwrap();
        let tid = current_thread_id();
        let track_uuid = self.ensure_thread_track(&mut inner, tid);

        let te = perfetto::build_track_event(
            track_uuid,
            perfetto::track_event::TYPE_SLICE_END,
            0,
            &[],
        );
        let pkt = perfetto::build_trace_packet_event(timestamp_ns, inner.seq_id, &te);
        let trace_bytes = perfetto::wrap_as_trace_packet(&pkt);
        let _ = inner.encoder.write_all(&trace_bytes);
        inner.packets_written += 1;
    }

    pub fn write_instant(&self, timestamp_ns: u64, name_iid: u64, annotations: &[Vec<u8>]) {
        let mut inner = self.inner.lock().unwrap();
        let tid = current_thread_id();
        let track_uuid = self.ensure_thread_track(&mut inner, tid);

        let te = perfetto::build_track_event(
            track_uuid,
            perfetto::track_event::TYPE_INSTANT,
            name_iid,
            annotations,
        );
        let pkt = perfetto::build_trace_packet_event(timestamp_ns, inner.seq_id, &te);
        let trace_bytes = perfetto::wrap_as_trace_packet(&pkt);
        let _ = inner.encoder.write_all(&trace_bytes);
        inner.packets_written += 1;
    }

    pub fn flush(&self) -> anyhow::Result<()> {
        let mut inner = self.inner.lock().unwrap();
        inner.encoder.flush()?;
        Ok(())
    }

    pub fn finish(self) -> anyhow::Result<u64> {
        let inner = self.inner.into_inner().unwrap();
        let count = inner.packets_written;
        inner.encoder.finish()?;
        Ok(count)
    }
}

fn hash_uuid(host: &str, component: &str, pid: u32) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in host.as_bytes().iter().chain(b":").chain(component.as_bytes()) {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h ^= pid as u64;
    h = h.wrapping_mul(0x100000001b3);
    h
}

fn current_thread_id() -> u64 {
    #[cfg(target_os = "linux")]
    {
        unsafe { libc::syscall(libc::SYS_gettid) as u64 }
    }
    #[cfg(not(target_os = "linux"))]
    {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        std::thread::current().id().hash(&mut hasher);
        hasher.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_and_finish() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.pftrace.gz");

        let writer = TraceWriter::new(path.clone(), "engine-prefill-0", "node-1", 1234, 1).unwrap();

        let name_iid = writer.intern_name("dynamo.prefill.compute");
        let traceparent_ann =
            perfetto::build_debug_annotation_str("traceparent", "00-abc123-def456-01");

        let ts = 1_000_000_000u64;
        writer.write_slice_begin(ts, name_iid, &[traceparent_ann]);
        writer.write_slice_end(ts + 500_000);

        let count = writer.finish().unwrap();
        assert!(count >= 4); // process track + clock snapshot + interned name + begin + end
        assert!(path.exists());

        // Verify gzip is valid
        let data = std::fs::read(&path).unwrap();
        let mut decoder = flate2::read::GzDecoder::new(&data[..]);
        let mut decoded = Vec::new();
        std::io::Read::read_to_end(&mut decoder, &mut decoded).unwrap();
        assert!(!decoded.is_empty());
    }
}
