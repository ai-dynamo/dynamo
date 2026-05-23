// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Unit tests proving that each audit emit point fires correctly.
//!
//! Each test uses `tracing::dispatcher::with_default` to install a scoped subscriber
//! that captures `kvbm_consolidator_audit=info` log output into a per-test buffer.

use std::sync::{Arc, Mutex};

use dynamo_tokens::PositionalLineageHash;
use kvbm_consolidator::ingress::kvbm_bridge::audit_kvbm_event;
use kvbm_consolidator::ingress::zmq_subscriber::decode_and_audit;
use kvbm_consolidator::wire::vllm_in::{BlockHashValue, RawKvEvent};
use kvbm_logical::events::protocol::KvCacheEvent;
use serde::Serialize;
use tracing_subscriber::{EnvFilter, fmt};

// ─── subscriber helper ───────────────────────────────────────────────────────

/// Build a scoped dispatcher and a log buffer. Call the returned closure with the
/// code under test; all tracing emits inside will be captured.
fn scoped_buf() -> (tracing::Dispatch, Arc<Mutex<Vec<u8>>>) {
    let buf = Arc::new(Mutex::new(Vec::<u8>::new()));
    let buf2 = buf.clone();

    struct BufWriter(Arc<Mutex<Vec<u8>>>);
    impl std::io::Write for BufWriter {
        fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(data);
            Ok(data.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let subscriber = fmt::Subscriber::builder()
        .with_env_filter(EnvFilter::new(
            "kvbm_consolidator=debug,kvbm_consolidator_audit=info",
        ))
        .with_writer(move || BufWriter(buf2.clone()))
        .with_ansi(false)
        .finish();

    let dispatch = tracing::Dispatch::new(subscriber);
    (dispatch, buf)
}

fn buf_contains(buf: &Arc<Mutex<Vec<u8>>>, needle: &str) -> bool {
    let locked = buf.lock().unwrap();
    let s = std::str::from_utf8(&locked).unwrap_or("");
    s.contains(needle)
}

// ─── helpers ─────────────────────────────────────────────────────────────────

/// Wire-compatible batch: serializes as `[ts, events, dp_rank]` array (matches vLLM codec).
#[derive(Serialize)]
struct WireBatch(f64, Vec<RawKvEvent>, Option<i32>);

/// Build a minimal valid msgpack payload for a KvEventBatch with one BlockStored.
fn make_stored_payload(block_hash: u64, tokens: &[u32], block_size: usize) -> Vec<u8> {
    let batch = WireBatch(
        1.0,
        vec![RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(block_hash)],
            parent_block_hash: None,
            token_ids: tokens.to_vec(),
            block_size,
            lora_name: None,
            medium: None,
            block_mm_infos: None,
            is_eagle: None,
            group_idx: None,
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
        }],
        Some(2),
    );
    rmp_serde::to_vec(&batch).expect("serialize")
}

fn make_empty_payload() -> Vec<u8> {
    let batch = WireBatch(0.0, vec![], None);
    rmp_serde::to_vec(&batch).expect("serialize empty batch")
}

// ─── test A: ingress_zmq audit fires ─────────────────────────────────────────

/// decode_and_audit emits event="ingress_zmq" with num_events and first_block_hash.
#[test]
fn audit_ingress_zmq_fires() {
    let (dispatch, buf) = scoped_buf();

    let payload = make_stored_payload(0xDEADBEEF, &[1, 2, 3, 4], 4);

    tracing::dispatcher::with_default(&dispatch, || {
        let result = decode_and_audit(&payload, 42u64);
        assert!(result.is_some(), "decode_and_audit should succeed");
    });

    assert!(
        buf_contains(&buf, "kvbm_consolidator_audit"),
        "log must contain target marker 'kvbm_consolidator_audit'; buf={:?}",
        std::str::from_utf8(&buf.lock().unwrap()).unwrap_or("(invalid utf8)")
    );
    assert!(
        buf_contains(&buf, "event=\"ingress_zmq\""),
        "log must contain event=\"ingress_zmq\""
    );
    assert!(
        buf_contains(&buf, "num_events=1"),
        "log must contain num_events field"
    );
}

/// decode_and_audit must NOT emit for an empty batch.
#[test]
fn audit_ingress_zmq_skips_empty() {
    let (dispatch, buf) = scoped_buf();

    let payload = make_empty_payload();

    tracing::dispatcher::with_default(&dispatch, || {
        let result = decode_and_audit(&payload, 0u64);
        assert!(
            result.is_some(),
            "decode_and_audit should succeed for empty"
        );
    });

    // No audit emit for empty batch.
    assert!(
        !buf_contains(&buf, "ingress_zmq"),
        "empty batch must not emit ingress_zmq"
    );
}

// ─── test B: ingress_kvbm audit fires ─────────────────────────────────────────

/// audit_kvbm_event emits event="ingress_kvbm" with kind="store" for Create.
#[test]
fn audit_ingress_kvbm_store() {
    let (dispatch, buf) = scoped_buf();

    let plh = PositionalLineageHash::new(0xABCD_1234_5678_9ABCu64, None, 0);

    tracing::dispatcher::with_default(&dispatch, || {
        audit_kvbm_event(&KvCacheEvent::Create(plh));
    });

    assert!(
        buf_contains(&buf, "kvbm_consolidator_audit"),
        "must contain target marker; buf={:?}",
        std::str::from_utf8(&buf.lock().unwrap()).unwrap_or("(invalid utf8)")
    );
    assert!(
        buf_contains(&buf, "event=\"ingress_kvbm\""),
        "must contain event=\"ingress_kvbm\""
    );
    assert!(
        buf_contains(&buf, "kind=\"store\""),
        "Create maps to kind=\"store\""
    );
    assert!(
        buf_contains(&buf, "num_blocks=1"),
        "must contain num_blocks"
    );
}

/// audit_kvbm_event emits kind="remove" for Remove.
#[test]
fn audit_ingress_kvbm_remove() {
    let (dispatch, buf) = scoped_buf();

    let plh = PositionalLineageHash::new(0x1111_2222_3333_4444u64, None, 0);

    tracing::dispatcher::with_default(&dispatch, || {
        audit_kvbm_event(&KvCacheEvent::Remove(plh));
    });

    assert!(
        buf_contains(&buf, "event=\"ingress_kvbm\""),
        "must contain event=\"ingress_kvbm\""
    );
    assert!(
        buf_contains(&buf, "kind=\"remove\""),
        "Remove maps to kind=\"remove\""
    );
}

// ─── test C: egress audit fires via full consolidator round-trip ─────────────

/// Full round-trip: inject a vLLM batch via PUB, let the consolidator process
/// and emit, assert both ingress_zmq and egress audit lines fire.
///
/// The subscriber is installed globally (try_init) for this async test because
/// `with_default` does not cross await points. We accept sharing the global
/// subscriber with any concurrently running tests — the assertions look for
/// substrings that are unique to the audit target.
#[tokio::test]
async fn audit_egress_fires_roundtrip() {
    // Install globally once; subsequent calls from other tests are no-ops.
    let buf = Arc::new(Mutex::new(Vec::<u8>::new()));
    let buf2 = buf.clone();

    struct BufWriter(Arc<Mutex<Vec<u8>>>);
    impl std::io::Write for BufWriter {
        fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
            self.0.lock().unwrap().extend_from_slice(data);
            Ok(data.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }

    let subscriber = fmt::Subscriber::builder()
        .with_env_filter(EnvFilter::new(
            "kvbm_consolidator=debug,kvbm_consolidator_audit=info",
        ))
        .with_writer(move || BufWriter(buf2.clone()))
        .with_ansi(false)
        .finish();

    // try_init returns Err if a global subscriber was already installed by other tests;
    // that is fine — the egress audit will still fire and write to whatever subscriber
    // is active. We use a fresh buf only as a best-effort capture.
    let _ = tracing::dispatcher::set_global_default(tracing::Dispatch::new(subscriber));

    use kvbm_consolidator::{ConsolidatorBuilder, EventSource};
    use std::time::Duration;

    let port = portpicker::pick_unused_port().expect("no free port");
    let pub_ep = format!("tcp://127.0.0.1:{port}");
    let egress_port = portpicker::pick_unused_port().expect("no free port");
    let egress_ep = format!("tcp://127.0.0.1:{egress_port}");

    // Bind PUB before the consolidator connects SUB.
    let pub_sock = kvbm_consolidator::zmq_util::bind_pub_socket(&pub_ep)
        .await
        .expect("bind pub");

    let consolidator = ConsolidatorBuilder::new(&egress_ep, EventSource::Vllm)
        .zmq_in(&pub_ep)
        .poll_interval(Duration::from_millis(20))
        .build()
        .await
        .expect("build consolidator");

    // Give ZMQ slow-joiner a moment.
    tokio::time::sleep(Duration::from_millis(150)).await;

    // Send a BlockStored batch.
    let batch = WireBatch(
        1.0,
        vec![RawKvEvent::BlockStored {
            block_hashes: vec![BlockHashValue::Unsigned(0xCAFE_BABE)],
            parent_block_hash: None,
            token_ids: vec![1, 2, 3, 4],
            block_size: 4,
            lora_name: None,
            medium: None,
            block_mm_infos: None,
            is_eagle: None,
            group_idx: None,
            kv_cache_spec_kind: None,
            kv_cache_spec_sliding_window: None,
        }],
        None,
    );
    let payload = rmp_serde::to_vec(&batch).expect("serialize");
    let frames = vec![vec![], vec![0u8; 8], payload];
    kvbm_consolidator::zmq_util::send_multipart(&pub_sock, frames)
        .await
        .expect("send");

    // Wait for publisher to drain + emit audit (poll at 20ms, wait 300ms).
    tokio::time::sleep(Duration::from_millis(300)).await;
    consolidator.shutdown().await;

    let locked = buf.lock().unwrap();
    let log = std::str::from_utf8(&locked).unwrap_or("");

    // ingress_zmq must have fired.
    assert!(
        log.contains("ingress_zmq"),
        "ingress_zmq audit must fire; log={log}"
    );
    // egress must have fired.
    assert!(
        log.contains("event=\"egress\""),
        "egress audit must fire; log={log}"
    );
    assert!(
        log.contains("kvbm_consolidator_audit"),
        "audit target marker must appear in log; log={log}"
    );
    assert!(
        log.contains("num_events"),
        "egress must log num_events; log={log}"
    );
}
