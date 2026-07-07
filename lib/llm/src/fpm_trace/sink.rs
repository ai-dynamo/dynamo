// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::BTreeMap;
use std::io::Write as _;
use std::path::Path;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::Context as _;
use serde::Serialize;
use serde_json::{Map, Value};
use tokio::sync::broadcast;

use dynamo_runtime::utils::GracefulTaskGuard;

use crate::telemetry::jsonl_gz::{JsonlGzipSinkOptions, JsonlGzipWriter};

use super::config::{
    DEFAULT_JSONL_BUFFER_BYTES, DEFAULT_JSONL_FLUSH_INTERVAL_MS, FpmTraceMode, FpmTracePolicy,
};
use super::{FpmTraceEvent, FpmTraceInner};

static PROBE_SEQUENCE: AtomicU64 = AtomicU64::new(0);

#[derive(Clone, Debug, Eq, Hash, Ord, PartialEq, PartialOrd, Serialize)]
pub(super) struct FpmTraceSource {
    pub(super) namespace: String,
    pub(super) component: String,
    pub(super) producer_id: String,
}

#[derive(Clone, Debug, Serialize)]
struct FpmTraceRecord {
    schema: &'static str,
    source: FpmTraceSource,
    capture_mode: FpmTraceMode,
    observed_at_unix_ms: u64,
    fpm: Value,
}

// A producer can expose multiple Dynamo components through one shared trace
// worker. Keep their sampled state independent even when backend worker IDs
// and data-parallel ranks overlap.
type FpmKey = (FpmTraceSource, String, i64);

fn require_nonnegative_integer(object: &Map<String, Value>, field: &str) -> anyhow::Result<()> {
    let valid = object.get(field).is_some_and(|value| {
        value.as_u64().is_some() || value.as_i64().is_some_and(|number| number >= 0)
    });
    if !valid {
        anyhow::bail!("FPM field {field} must be a non-negative integer");
    }
    Ok(())
}

fn require_nonnegative_finite_number(
    object: &Map<String, Value>,
    field: &str,
) -> anyhow::Result<()> {
    let valid = object
        .get(field)
        .and_then(Value::as_f64)
        .is_some_and(|number| number.is_finite() && number >= 0.0);
    if !valid {
        anyhow::bail!("FPM field {field} must be a finite non-negative number");
    }
    Ok(())
}

fn validate_request_metrics(
    object: &Map<String, Value>,
    integer_fields: &[&str],
    variance_fields: &[&str],
) -> anyhow::Result<()> {
    for field in integer_fields {
        require_nonnegative_integer(object, field)?;
    }
    for field in variance_fields {
        require_nonnegative_finite_number(object, field)?;
    }
    Ok(())
}

fn validate_canonical_fpm(object: &Map<String, Value>) -> anyhow::Result<()> {
    if object.get("version").and_then(Value::as_i64) != Some(1) {
        anyhow::bail!("FPM payload has unsupported or missing version");
    }
    require_nonnegative_finite_number(object, "wall_time")?;

    let scheduled = object
        .get("scheduled_requests")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("FPM payload has no scheduled_requests map"))?;
    validate_request_metrics(
        scheduled,
        &[
            "num_prefill_requests",
            "sum_prefill_tokens",
            "sum_prefill_kv_tokens",
            "num_decode_requests",
            "sum_decode_kv_tokens",
        ],
        &["var_prefill_length", "var_decode_kv_tokens"],
    )?;

    let queued = object
        .get("queued_requests")
        .and_then(Value::as_object)
        .ok_or_else(|| anyhow::anyhow!("FPM payload has no queued_requests map"))?;
    validate_request_metrics(
        queued,
        &[
            "num_prefill_requests",
            "sum_prefill_tokens",
            "num_decode_requests",
            "sum_decode_kv_tokens",
        ],
        &["var_prefill_length", "var_decode_kv_tokens"],
    )?;
    Ok(())
}

fn decode_event(
    event: FpmTraceEvent,
    capture_mode: FpmTraceMode,
) -> anyhow::Result<(FpmKey, i64, FpmTraceRecord)> {
    let fpm: Value = rmp_serde::from_slice(&event.payload).context("decoding FPM msgpack")?;
    let object = fpm
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("FPM payload is not a msgpack map"))?;
    validate_canonical_fpm(object)?;
    let worker_id = object
        .get("worker_id")
        .and_then(Value::as_str)
        .ok_or_else(|| anyhow::anyhow!("FPM payload has no string worker_id"))?
        .to_string();
    let dp_rank = object
        .get("dp_rank")
        .and_then(Value::as_i64)
        .filter(|dp_rank| *dp_rank >= 0)
        .ok_or_else(|| anyhow::anyhow!("FPM payload has no non-negative integer dp_rank"))?;
    let counter_id = object
        .get("counter_id")
        .and_then(Value::as_i64)
        .filter(|counter_id| *counter_id >= 0)
        .ok_or_else(|| anyhow::anyhow!("FPM payload has no non-negative integer counter_id"))?;

    let source = (*event.source).clone();
    Ok((
        (source.clone(), worker_id, dp_rank),
        counter_id,
        FpmTraceRecord {
            schema: "dynamo.fpm.trace.v1",
            source,
            capture_mode,
            observed_at_unix_ms: event.observed_at_unix_ms,
            fpm,
        },
    ))
}

fn sanitize_producer_id(producer_id: &str) -> String {
    let sanitized: String = producer_id
        .chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect();
    if sanitized.is_empty() {
        "unknown".to_string()
    } else {
        sanitized
    }
}

pub(super) fn producer_output_path(base_path: &str, producer_id: &str) -> String {
    let prefix = base_path
        .strip_suffix(".jsonl.gz")
        .or_else(|| base_path.strip_suffix(".jsonl"))
        .unwrap_or(base_path);
    format!("{prefix}.{}", sanitize_producer_id(producer_id))
}

fn preflight_writable_parent(output_path: &str) -> anyhow::Result<()> {
    let output_path = Path::new(output_path);
    let parent = output_path
        .parent()
        .filter(|parent| !parent.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    std::fs::create_dir_all(parent)
        .with_context(|| format!("creating FPM trace directory {}", parent.display()))?;

    let sequence = PROBE_SEQUENCE.fetch_add(1, Ordering::Relaxed);
    let producer_prefix = output_path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("dynamo-fpm");
    let probe_path = parent.join(format!(
        ".{producer_prefix}.write-probe-{}-{sequence}",
        std::process::id()
    ));
    let mut probe = std::fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&probe_path)
        .with_context(|| format!("creating FPM trace write probe {}", probe_path.display()))?;

    let write_result = probe
        .write_all(b"dynamo-fpm-write-probe")
        .and_then(|_| probe.sync_all());
    drop(probe);
    if let Err(error) = write_result {
        let _ = std::fs::remove_file(&probe_path);
        return Err(error)
            .with_context(|| format!("writing FPM trace probe {}", probe_path.display()));
    }
    std::fs::remove_file(&probe_path)
        .with_context(|| format!("removing FPM trace write probe {}", probe_path.display()))?;
    Ok(())
}

pub(super) async fn spawn_worker(
    policy: FpmTracePolicy,
    receiver: broadcast::Receiver<FpmTraceEvent>,
    owner: Arc<FpmTraceInner>,
    graceful_guard: Option<GracefulTaskGuard>,
) -> anyhow::Result<()> {
    let output_path = producer_output_path(&policy.output_path, &owner.producer_id);
    let preflight_path = output_path.clone();
    tokio::task::spawn_blocking(move || preflight_writable_parent(&preflight_path))
        .await
        .context("joining FPM trace write preflight")??;
    let writer = JsonlGzipWriter::new(
        output_path.clone(),
        JsonlGzipSinkOptions {
            buffer_bytes: DEFAULT_JSONL_BUFFER_BYTES,
            flush_interval: Duration::from_millis(DEFAULT_JSONL_FLUSH_INTERVAL_MS),
            roll_uncompressed_bytes: policy.jsonl_gz_roll_bytes,
            roll_lines: None,
            max_segments: Some(policy.max_segments),
        },
    )
    .await
    .with_context(|| format!("opening FPM gzip JSONL trace at {output_path}"))?;
    let completion = MarkClosedOnDrop(owner.clone());
    tokio::spawn(async move {
        // The guard keeps runtime shutdown in phase 2 until this task has
        // drained the queue and awaited writer close.
        let _graceful_guard = graceful_guard;
        // Constructed outside the async block, so task abort before its first
        // poll still marks the producer closed and wakes registry waiters. It
        // is declared after the graceful guard so it drops first on return.
        let _completion = completion;
        // Keeping the owner alive keeps this producer discoverable in the
        // registry until its writer has finished closing.
        match policy.mode {
            FpmTraceMode::Full => run_full(receiver, writer, owner.clone()).await,
            FpmTraceMode::Sampled => {
                run_sampled(receiver, writer, owner.clone(), policy.sample_interval_ms).await
            }
        }
    });

    Ok(())
}

struct MarkClosedOnDrop(Arc<FpmTraceInner>);

impl Drop for MarkClosedOnDrop {
    fn drop(&mut self) {
        self.0.mark_closed();
    }
}

async fn send_decoded(writer: &JsonlGzipWriter<FpmTraceRecord>, event: FpmTraceEvent) {
    match decode_event(event, FpmTraceMode::Full) {
        Ok((_, _, record)) => {
            if writer.send(record).await.is_err() {
                tracing::warn!("FPM trace writer closed; dropping record");
            }
        }
        Err(error) => tracing::warn!(%error, "FPM trace dropped malformed payload"),
    }
}

fn report_lag(owner: &FpmTraceInner, dropped: u64, during_shutdown: bool) {
    let total_dropped = owner.record_dropped(dropped);
    tracing::warn!(
        producer_id = %owner.producer_id,
        dropped,
        total_dropped,
        during_shutdown,
        "FPM trace queue lagged; older records were overwritten"
    );
}

fn report_drop_summary(owner: &FpmTraceInner) {
    let dropped = owner.dropped.load(Ordering::Relaxed);
    if dropped > 0 {
        tracing::warn!(
            producer_id = %owner.producer_id,
            dropped,
            "FPM trace closed after dropping records"
        );
    }
}

async fn run_full(
    mut receiver: broadcast::Receiver<FpmTraceEvent>,
    writer: JsonlGzipWriter<FpmTraceRecord>,
    owner: Arc<FpmTraceInner>,
) {
    loop {
        tokio::select! {
            biased;
            _ = owner.shutdown.cancelled() => {
                // Stop new publishes and wait for publishers that crossed the
                // acceptance boundary. The subsequent try_recv loop is then a
                // stable drain of the bounded queue.
                owner.stop_accepting().await;
                loop {
                    match receiver.try_recv() {
                        Ok(event) => send_decoded(&writer, event).await,
                        Err(broadcast::error::TryRecvError::Lagged(dropped)) => {
                            report_lag(&owner, dropped, true);
                        }
                        Err(
                            broadcast::error::TryRecvError::Empty
                            | broadcast::error::TryRecvError::Closed
                        ) => break,
                    }
                }
                if let Err(error) = writer.close().await {
                    tracing::warn!(%error, "FPM trace writer failed to close cleanly");
                }
                report_drop_summary(&owner);
                return;
            }
            message = receiver.recv() => {
                match message {
                    Ok(event) => send_decoded(&writer, event).await,
                    Err(broadcast::error::RecvError::Lagged(dropped)) => {
                        report_lag(&owner, dropped, false);
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        owner.stop_accepting().await;
                        if let Err(error) = writer.close().await {
                            tracing::warn!(%error, "FPM trace writer failed to close cleanly");
                        }
                        report_drop_summary(&owner);
                        return;
                    }
                }
            }
        }
    }
}

struct PendingSample {
    counter_id: i64,
    record: FpmTraceRecord,
}

fn retain_latest(
    latest: &mut BTreeMap<FpmKey, PendingSample>,
    last_emitted: &BTreeMap<FpmKey, i64>,
    event: FpmTraceEvent,
) {
    match decode_event(event, FpmTraceMode::Sampled) {
        Ok((key, counter_id, record)) => {
            if last_emitted.get(&key) == Some(&counter_id) {
                return;
            }
            latest.insert(key, PendingSample { counter_id, record });
        }
        Err(error) => tracing::warn!(%error, "FPM trace dropped malformed payload"),
    }
}

async fn flush_latest(
    writer: &JsonlGzipWriter<FpmTraceRecord>,
    latest: &mut BTreeMap<FpmKey, PendingSample>,
    last_emitted: &mut BTreeMap<FpmKey, i64>,
) {
    for (key, pending) in std::mem::take(latest) {
        let counter_id = pending.counter_id;
        if writer.send(pending.record).await.is_err() {
            tracing::warn!("FPM trace writer closed; dropping sampled record");
            break;
        }
        last_emitted.insert(key, counter_id);
    }
}

async fn run_sampled(
    mut receiver: broadcast::Receiver<FpmTraceEvent>,
    writer: JsonlGzipWriter<FpmTraceRecord>,
    owner: Arc<FpmTraceInner>,
    sample_interval_ms: u64,
) {
    let interval = Duration::from_millis(sample_interval_ms.max(1));
    let mut flush_tick = tokio::time::interval_at(tokio::time::Instant::now() + interval, interval);
    flush_tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
    let mut latest = BTreeMap::new();
    let mut last_emitted = BTreeMap::new();

    loop {
        tokio::select! {
            biased;
            _ = owner.shutdown.cancelled() => {
                owner.stop_accepting().await;
                loop {
                    match receiver.try_recv() {
                        Ok(event) => retain_latest(&mut latest, &last_emitted, event),
                        Err(broadcast::error::TryRecvError::Lagged(dropped)) => {
                            report_lag(&owner, dropped, true);
                        }
                        Err(
                            broadcast::error::TryRecvError::Empty
                            | broadcast::error::TryRecvError::Closed
                        ) => break,
                    }
                }
                flush_latest(&writer, &mut latest, &mut last_emitted).await;
                if let Err(error) = writer.close().await {
                    tracing::warn!(%error, "FPM trace writer failed to close cleanly");
                }
                report_drop_summary(&owner);
                return;
            }
            _ = flush_tick.tick() => {
                flush_latest(&writer, &mut latest, &mut last_emitted).await;
            }
            message = receiver.recv() => {
                match message {
                    Ok(event) => retain_latest(&mut latest, &last_emitted, event),
                    Err(broadcast::error::RecvError::Lagged(dropped)) => {
                        report_lag(&owner, dropped, false);
                    }
                    Err(broadcast::error::RecvError::Closed) => {
                        owner.stop_accepting().await;
                        flush_latest(&writer, &mut latest, &mut last_emitted).await;
                        if let Err(error) = writer.close().await {
                            tracing::warn!(%error, "FPM trace writer failed to close cleanly");
                        }
                        report_drop_summary(&owner);
                        return;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use std::io::Read;
    use std::sync::atomic::{AtomicBool, AtomicUsize};

    use bytes::Bytes;
    use flate2::read::MultiGzDecoder;
    use tokio_util::sync::CancellationToken;

    use super::*;
    use crate::telemetry::jsonl_gz::segment_path;

    fn source() -> FpmTraceSource {
        FpmTraceSource {
            namespace: "dynamo".to_string(),
            component: "backend".to_string(),
            producer_id: "producer-1".to_string(),
        }
    }

    fn event_with_wall_time(
        worker_id: &str,
        dp_rank: i64,
        counter_id: i64,
        wall_time: f64,
    ) -> FpmTraceEvent {
        let active = wall_time > 0.0;
        let payload = rmp_serde::to_vec_named(&serde_json::json!({
            "version": 1,
            "worker_id": worker_id,
            "dp_rank": dp_rank,
            "counter_id": counter_id,
            "wall_time": wall_time,
            "scheduled_requests": {
                "num_prefill_requests": if active { 1 } else { 0 },
                "sum_prefill_tokens": if active { 16 } else { 0 },
                "var_prefill_length": 0.0,
                "sum_prefill_kv_tokens": 0,
                "num_decode_requests": if active { 2 } else { 0 },
                "sum_decode_kv_tokens": if active { 64 } else { 0 },
                "var_decode_kv_tokens": 0.0,
            },
            "queued_requests": {
                "num_prefill_requests": 0,
                "sum_prefill_tokens": 0,
                "var_prefill_length": 0.0,
                "num_decode_requests": 0,
                "sum_decode_kv_tokens": 0,
                "var_decode_kv_tokens": 0.0,
            },
        }))
        .unwrap();
        FpmTraceEvent {
            observed_at_unix_ms: counter_id as u64 * 100,
            payload: Bytes::from(payload),
            source: Arc::new(source()),
        }
    }

    fn event(worker_id: &str, dp_rank: i64, counter_id: i64) -> FpmTraceEvent {
        event_with_wall_time(worker_id, dp_rank, counter_id, 0.01)
    }

    fn key(worker_id: &str, dp_rank: i64) -> FpmKey {
        (source(), worker_id.to_string(), dp_rank)
    }

    async fn test_writer(path: &Path) -> JsonlGzipWriter<FpmTraceRecord> {
        JsonlGzipWriter::new(
            path.display().to_string(),
            JsonlGzipSinkOptions {
                buffer_bytes: 1,
                flush_interval: Duration::from_secs(60),
                roll_uncompressed_bytes: 1024 * 1024,
                roll_lines: None,
                max_segments: Some(4),
            },
        )
        .await
        .unwrap()
    }

    fn read_trace_records(path: &Path) -> Vec<Value> {
        let bytes = std::fs::read(segment_path(path, 0)).unwrap();
        let mut decoder = MultiGzDecoder::new(bytes.as_slice());
        let mut content = String::new();
        decoder.read_to_string(&mut content).unwrap();
        content
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect()
    }

    async fn yield_to_trace_worker() {
        for _ in 0..8 {
            tokio::task::yield_now().await;
        }
    }

    fn owner(
        sender: broadcast::Sender<FpmTraceEvent>,
        shutdown: CancellationToken,
    ) -> Arc<FpmTraceInner> {
        Arc::new(FpmTraceInner {
            producer_id: "producer-1".to_string(),
            sender,
            shutdown,
            accepting: AtomicBool::new(true),
            publishers_in_flight: AtomicUsize::new(0),
            publishers_idle: tokio::sync::Notify::new(),
            leases: AtomicUsize::new(1),
            dropped: AtomicU64::new(0),
            closed: AtomicBool::new(false),
            closed_notify: tokio::sync::Notify::new(),
        })
    }

    #[test]
    fn producer_id_is_sanitized_and_inserted_before_segment_suffix() {
        assert_eq!(
            producer_output_path("/tmp/dynamo-fpm.jsonl.gz", "worker/a:b"),
            "/tmp/dynamo-fpm.worker_a_b"
        );
        assert_eq!(
            producer_output_path("/tmp/dynamo-fpm", ""),
            "/tmp/dynamo-fpm.unknown"
        );
    }

    #[test]
    fn writable_parent_preflight_rejects_a_blocked_path() {
        let dir = tempfile::tempdir().unwrap();
        let blocker = dir.path().join("not-a-directory");
        std::fs::write(&blocker, b"file").unwrap();
        let output = blocker.join("dynamo-fpm");

        assert!(preflight_writable_parent(&output.to_string_lossy()).is_err());
    }

    #[test]
    fn writable_parent_preflight_removes_its_unique_probe() {
        let dir = tempfile::tempdir().unwrap();
        let output = dir.path().join("dynamo-fpm.producer-7");

        preflight_writable_parent(&output.to_string_lossy()).unwrap();

        assert_eq!(std::fs::read_dir(dir.path()).unwrap().count(), 0);
    }

    #[test]
    fn sampled_mode_keeps_latest_event_per_worker_and_rank() {
        let mut latest = BTreeMap::new();
        let last_emitted = BTreeMap::new();
        retain_latest(&mut latest, &last_emitted, event("worker-a", 0, 1));
        retain_latest(&mut latest, &last_emitted, event("worker-a", 0, 2));
        retain_latest(&mut latest, &last_emitted, event("worker-a", 1, 3));

        assert_eq!(latest.len(), 2);
        assert_eq!(latest[&key("worker-a", 0)].record.observed_at_unix_ms, 200);
        assert_eq!(latest[&key("worker-a", 1)].record.observed_at_unix_ms, 300);
        assert_eq!(latest[&key("worker-a", 0)].record.fpm["counter_id"], 2);
    }

    #[test]
    fn sampled_mode_keeps_components_with_overlapping_worker_ids_independent() {
        let mut latest = BTreeMap::new();
        let last_emitted = BTreeMap::new();
        let backend = event("worker-a", 0, 1);
        let mut prefill = event("worker-a", 0, 2);
        prefill.source = Arc::new(FpmTraceSource {
            component: "prefill".to_string(),
            ..source()
        });

        retain_latest(&mut latest, &last_emitted, backend);
        retain_latest(&mut latest, &last_emitted, prefill);

        assert_eq!(latest.len(), 2);
        assert_eq!(latest[&key("worker-a", 0)].counter_id, 1);
        let prefill_key = (
            FpmTraceSource {
                component: "prefill".to_string(),
                ..source()
            },
            "worker-a".to_string(),
            0,
        );
        assert_eq!(latest[&prefill_key].counter_id, 2);
    }

    #[test]
    fn sampled_state_stays_bounded_to_one_record_per_rank() {
        let mut latest = BTreeMap::new();
        let last_emitted = BTreeMap::new();

        for counter_id in 1..=32 {
            for dp_rank in 0..256 {
                retain_latest(
                    &mut latest,
                    &last_emitted,
                    event("worker-a", dp_rank, counter_id),
                );
            }
        }

        assert_eq!(latest.len(), 256);
        assert!(latest.values().all(|pending| pending.counter_id == 32));
    }

    #[test]
    fn sampled_mode_does_not_reemit_the_last_counter() {
        let mut latest = BTreeMap::new();
        let mut last_emitted = BTreeMap::new();
        let key = key("worker-a", 0);
        last_emitted.insert(key, 9);

        retain_latest(&mut latest, &last_emitted, event("worker-a", 0, 9));

        assert!(latest.is_empty());
    }

    #[test]
    fn malformed_payload_is_rejected() {
        let malformed = FpmTraceEvent {
            observed_at_unix_ms: 1,
            payload: Bytes::from_static(b"not-msgpack"),
            source: Arc::new(source()),
        };
        assert!(decode_event(malformed, FpmTraceMode::Full).is_err());
    }

    #[test]
    fn incomplete_fpm_map_is_rejected() {
        let payload = rmp_serde::to_vec_named(&serde_json::json!({
            "version": 1,
            "worker_id": "worker-a",
            "dp_rank": 0,
            "counter_id": 1,
            "wall_time": 0.01,
            "scheduled_requests": {},
            "queued_requests": {},
        }))
        .unwrap();
        let incomplete = FpmTraceEvent {
            observed_at_unix_ms: 1,
            payload: Bytes::from(payload),
            source: Arc::new(source()),
        };
        assert!(decode_event(incomplete, FpmTraceMode::Full).is_err());
    }

    #[test]
    fn trace_record_carries_stable_schema_source_and_mode() {
        let (_, _, record) = decode_event(event("worker-a", 2, 7), FpmTraceMode::Sampled).unwrap();
        let value = serde_json::to_value(record).unwrap();
        assert_eq!(value["schema"], "dynamo.fpm.trace.v1");
        assert_eq!(value["source"]["namespace"], "dynamo");
        assert_eq!(value["source"]["component"], "backend");
        assert_eq!(value["source"]["producer_id"], "producer-1");
        assert_eq!(value["capture_mode"], "sampled");
        assert_eq!(value["fpm"]["counter_id"], 7);
    }

    #[tokio::test]
    async fn full_mode_persists_active_and_idle_payloads_in_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("full-trace");
        let writer = test_writer(&path).await;
        let (sender, receiver) = broadcast::channel(8);
        let shutdown = CancellationToken::new();
        let owner = owner(sender.clone(), shutdown.clone());
        let task = tokio::spawn(run_full(receiver, writer, owner));

        sender
            .send(event_with_wall_time("worker-a", 0, 1, 0.01))
            .unwrap();
        sender
            .send(event_with_wall_time("worker-a", 0, 2, 0.0))
            .unwrap();
        sender
            .send(event_with_wall_time("worker-a", 0, 3, 0.02))
            .unwrap();
        shutdown.cancel();
        task.await.unwrap();

        let records = read_trace_records(&path);
        let counters: Vec<_> = records
            .iter()
            .map(|record| record["event"]["fpm"]["counter_id"].as_i64().unwrap())
            .collect();
        assert_eq!(counters, [1, 2, 3]);
        assert_eq!(records[1]["event"]["fpm"]["wall_time"], 0.0);
        assert!(
            records
                .iter()
                .all(|record| record["event"]["capture_mode"] == "full")
        );
    }

    #[tokio::test(start_paused = true)]
    async fn sampled_mode_is_per_rank_suppresses_unchanged_and_flushes_dirty_shutdown() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("sampled-trace");
        let writer = test_writer(&path).await;
        let (sender, receiver) = broadcast::channel(16);
        let shutdown = CancellationToken::new();
        let owner = owner(sender.clone(), shutdown.clone());
        let task = tokio::spawn(run_sampled(receiver, writer, owner, 100));

        sender.send(event("worker-a", 0, 1)).unwrap();
        sender.send(event("worker-a", 0, 2)).unwrap();
        sender.send(event("worker-a", 1, 1)).unwrap();
        yield_to_trace_worker().await;
        tokio::time::advance(Duration::from_millis(100)).await;
        yield_to_trace_worker().await;

        // Repeating the last emitted counters in a later interval must not
        // create duplicate samples.
        sender.send(event("worker-a", 0, 2)).unwrap();
        sender.send(event("worker-a", 1, 1)).unwrap();
        yield_to_trace_worker().await;
        tokio::time::advance(Duration::from_millis(100)).await;
        yield_to_trace_worker().await;

        // This sample is dirty but its interval has not elapsed; cancellation
        // must still flush it and await the gzip writer close.
        sender.send(event("worker-a", 0, 3)).unwrap();
        shutdown.cancel();
        task.await.unwrap();

        let records = read_trace_records(&path);
        let rank_counters: Vec<_> = records
            .iter()
            .map(|record| {
                (
                    record["event"]["fpm"]["dp_rank"].as_i64().unwrap(),
                    record["event"]["fpm"]["counter_id"].as_i64().unwrap(),
                )
            })
            .collect();
        assert_eq!(rank_counters, [(0, 2), (1, 1), (0, 3)]);
        assert!(
            records
                .iter()
                .all(|record| record["event"]["capture_mode"] == "sampled")
        );
    }

    #[tokio::test]
    async fn bounded_trace_queue_retains_newest_and_counts_overwritten_records() {
        let (sender, mut receiver) = broadcast::channel(2);
        let owner = owner(sender.clone(), CancellationToken::new());
        sender.send(event("worker-a", 0, 1)).unwrap();
        sender.send(event("worker-a", 0, 2)).unwrap();
        sender.send(event("worker-a", 0, 3)).unwrap();

        let dropped = match receiver.recv().await {
            Err(broadcast::error::RecvError::Lagged(dropped)) => dropped,
            result => panic!("expected lag, got {result:?}"),
        };
        report_lag(&owner, dropped, false);
        assert_eq!(owner.dropped.load(Ordering::Relaxed), 1);

        let second = receiver.recv().await.unwrap();
        let third = receiver.recv().await.unwrap();
        let (_, second_counter, _) = decode_event(second, FpmTraceMode::Full).unwrap();
        let (_, third_counter, _) = decode_event(third, FpmTraceMode::Full).unwrap();
        assert_eq!((second_counter, third_counter), (2, 3));
    }

    #[tokio::test]
    async fn worker_completion_notifies_registry_even_after_task_panic() {
        let (sender, _receiver) = broadcast::channel(1);
        let owner = owner(sender, CancellationToken::new());
        let completion = MarkClosedOnDrop(owner.clone());

        let task = tokio::spawn(async move {
            let _completion = completion;
            panic!("simulated trace worker panic");
        });
        assert!(task.await.unwrap_err().is_panic());
        owner.wait_closed().await;
        assert!(owner.closed.load(Ordering::Acquire));
        assert!(!owner.accepting.load(Ordering::Acquire));
    }
}
