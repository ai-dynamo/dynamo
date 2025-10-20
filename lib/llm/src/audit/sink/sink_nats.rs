// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_nats::jetstream;
use rand::Rng;
use std::time::{Duration, Instant};
use tokio::{sync::mpsc, time};
use tracing::{error, info, warn};

use crate::audit::handle::AuditRecord;
use crate::audit::sink::AuditSink;

/// Runtime configuration loaded from env.
#[derive(Clone)]
struct NatsCfg {
    subject: String,
    batch: usize,
    flush_every: Duration,
    backoff_base: Duration,
    queue_cap: usize,
    max_backoff: Duration,
    cb_threshold: u32,  // consecutive failure threshold to trip breaker
    cb_sleep: Duration, // how long we stay "open" before retrying
}

impl Default for NatsCfg {
    fn default() -> Self {
        Self {
            subject: "dynamo.audit.v1".into(),
            batch: 128,
            flush_every: Duration::from_millis(100),
            backoff_base: Duration::from_millis(250),
            queue_cap: 4096,                      // bounded; never blocks hot path
            max_backoff: Duration::from_secs(30), // cap backoff
            cb_threshold: 10,
            cb_sleep: Duration::from_secs(60),
        }
    }
}

impl NatsCfg {
    fn from_env() -> Self {
        let mut cfg = Self::default();
        if let Ok(v) = std::env::var("DYN_AUDIT_NATS_SUBJECT") {
            cfg.subject = v;
        }
        if let Ok(v) = std::env::var("DYN_AUDIT_NATS_BATCH") {
            cfg.batch = v.parse().unwrap_or(cfg.batch);
        }
        if let Ok(v) = std::env::var("DYN_AUDIT_NATS_FLUSH_MS") {
            cfg.flush_every = Duration::from_millis(v.parse().unwrap_or(100));
        }
        if let Ok(v) = std::env::var("DYN_AUDIT_NATS_BACKOFF_MS") {
            cfg.backoff_base = Duration::from_millis(v.parse().unwrap_or(250));
        }
        // keep queue_cap derived from batch unless explicitly provided later
        // (we intentionally do not add a separate env to keep scope small)
        cfg.queue_cap = cfg.queue_cap.max(cfg.batch * 32);
        cfg
    }
}

/// Internals run on a background task so `emit()` is fully non-blocking and cheap.
struct NatsWorker {
    js: jetstream::Context,
    subject: String,
    batch_size: usize,
    flush_every: Duration,
    backoff_base: Duration,
    max_backoff: Duration,
    cb_threshold: u32,
    cb_sleep: Duration,
    rx: mpsc::Receiver<Vec<u8>>,
}

impl NatsWorker {
    fn jittered(delay: Duration) -> Duration {
        let base_ms = delay.as_millis() as i64;
        let j = rand::rng().random_range(0..=base_ms.max(1)) as u64;
        Duration::from_millis(delay.as_millis() as u64 + j)
    }

    async fn flush(
        js: &jetstream::Context,
        subject: &str,
        buf: &mut Vec<Vec<u8>>,
    ) -> anyhow::Result<()> {
        // publish each record as an individual JetStream message with ack
        // This preserves per-record replay semantics and keeps consumers simple.
        for payload in buf.drain(..) {
            js.publish(subject.to_string(), payload.into()).await?;
        }
        Ok(())
    }

    async fn run(mut self) {
        info!("nats: using shared NATS connection from DistributedRuntime");

        // Batching state
        let mut buf: Vec<Vec<u8>> = Vec::with_capacity(self.batch_size);
        let mut ticker = time::interval(self.flush_every);
        ticker.set_missed_tick_behavior(time::MissedTickBehavior::Delay);

        // Failure / circuit-breaker state
        let mut consecutive_failures: u32 = 0;
        let mut breaker_open_until: Option<Instant> = None;

        loop {
            // If circuit breaker is open, sleep and drain rx to avoid backpressure.
            if let Some(until) = breaker_open_until {
                if Instant::now() < until {
                    // Drain without blocking; drop to protect latency & memory.
                    while let Ok(_msg) = self.rx.try_recv() { /* drop */ }
                    time::sleep(Duration::from_millis(100)).await;
                    continue;
                } else {
                    breaker_open_until = None;
                    // Trust the shared client's reconnection logic
                    info!(
                        "nats: circuit breaker reopening (relying on shared client reconnection)"
                    );
                    consecutive_failures = 0;
                }
            }

            tokio::select! {
                maybe = self.rx.recv() => {
                    match maybe {
                        Some(payload) => {
                            buf.push(payload);
                            if buf.len() >= self.batch_size {
                                // Try to flush immediately on size
                                if let Err(e) = Self::flush(&self.js, &self.subject, &mut buf).await {
                                    consecutive_failures += 1;
                                    warn!(error=%e, fails = consecutive_failures, "nats: flush(size) failed");
                                    // Put back the (not flushed) messages: leave in `buf`
                                    // Backoff and possibly trip breaker
                                    let mut backoff = self.backoff_base.saturating_mul(1 << (consecutive_failures.min(15)));
                                    if backoff > self.max_backoff { backoff = self.max_backoff; }
                                    time::sleep(Self::jittered(backoff)).await;
                                    if consecutive_failures >= self.cb_threshold {
                                        error!(fails = consecutive_failures, "nats: consecutive failures; opening circuit breaker");
                                        breaker_open_until = Some(Instant::now() + Self::jittered(self.cb_sleep));
                                        // drop buffered messages to avoid unbounded growth
                                        buf.clear();
                                    }
                                } else {
                                    consecutive_failures = 0;
                                }
                            }
                        }
                        None => {
                            // Channel closed: flush best-effort and exit task
                            if let Err(e) = Self::flush(&self.js, &self.subject, &mut buf).await {
                                warn!(error=%e, "nats: final flush failed on shutdown");
                            }
                            break;
                        }
                    }
                }
                _ = ticker.tick() => {
                    if buf.is_empty() { continue; }
                    if let Err(e) = Self::flush(&self.js, &self.subject, &mut buf).await {
                        consecutive_failures += 1;
                        warn!(error=%e, fails=consecutive_failures, "nats: flush(interval) failed");
                        let mut backoff = self.backoff_base.saturating_mul(1 << (consecutive_failures.min(15)));
                        if backoff > self.max_backoff { backoff = self.max_backoff; }
                        time::sleep(Self::jittered(backoff)).await;
                        if consecutive_failures >= self.cb_threshold {
                            error!(fails = consecutive_failures, "nats: consecutive failures; opening circuit breaker");
                            breaker_open_until = Some(Instant::now() + Self::jittered(self.cb_sleep));
                            buf.clear();
                        }
                    } else {
                        consecutive_failures = 0;
                    }
                }
            }
        }
    }
}

/// Public sink type – thin wrapper around a bounded queue to a background worker.
pub struct NatsSink {
    tx: mpsc::Sender<Vec<u8>>,
}

impl NatsSink {
    /// Create a new NatsSink using the shared NATS client from DistributedRuntime.
    /// Returns None if no NATS client is available.
    pub fn new(nats_client: Option<&dynamo_runtime::transports::nats::Client>) -> Option<Self> {
        let Some(client) = nats_client else {
            warn!(
                "NATS sink requested but no DistributedRuntime NATS client available; skipping NATS audit sink"
            );
            return None;
        };

        let cfg = NatsCfg::from_env();
        let (tx, rx) = mpsc::channel::<Vec<u8>>(cfg.queue_cap);

        // spawn background worker with shared connection
        let worker = NatsWorker {
            js: client.jetstream().clone(),
            subject: cfg.subject,
            batch_size: cfg.batch,
            flush_every: cfg.flush_every,
            backoff_base: cfg.backoff_base,
            max_backoff: cfg.max_backoff,
            cb_threshold: cfg.cb_threshold,
            cb_sleep: cfg.cb_sleep,
            rx,
        };
        tokio::spawn(async move {
            worker.run().await;
        });
        Some(NatsSink { tx })
    }
}

impl AuditSink for NatsSink {
    fn name(&self) -> &'static str {
        "nats"
    }

    fn emit(&self, rec: &AuditRecord) {
        // do not block; drop if internal queue is full
        match serde_json::to_vec(rec) {
            Ok(bytes) => {
                if let Err(e) = self.tx.try_send(bytes) {
                    warn!(err=%e, "nats: internal queue full; dropping audit record");
                }
            }
            Err(e) => warn!("nats: serialize failed: {e}"),
        }
    }
}
