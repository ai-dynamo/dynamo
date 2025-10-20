// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use async_nats::jetstream;
use async_nats::{Client, ConnectOptions};
use rand::Rng;
use std::time::{Duration, Instant};
use tokio::{sync::mpsc, time};
use tracing::{error, info, warn};

use crate::audit::handle::AuditRecord;
use crate::audit::sink::AuditSink;

/// Runtime configuration loaded from env.
#[derive(Clone)]
struct NatsCfg {
    url: String,
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
            url: "nats://127.0.0.1:4222".into(),
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
        if let Ok(v) = std::env::var("DYN_AUDIT_NATS_URL") {
            cfg.url = v;
        }
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
    cfg: NatsCfg,
    rx: mpsc::Receiver<Vec<u8>>,
}

impl NatsWorker {
    async fn connect(cfg: &NatsCfg) -> anyhow::Result<(Client, jetstream::Context)> {
        // Keep simple, no TLS/auth for initial scope; ConnectOptions available if needed.
        let client = ConnectOptions::default().connect(&cfg.url).await?;
        let js = jetstream::new(client.clone());
        Ok((client, js))
    }

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
        // Batching state
        let mut buf: Vec<Vec<u8>> = Vec::with_capacity(self.cfg.batch);
        let mut ticker = time::interval(self.cfg.flush_every);
        ticker.set_missed_tick_behavior(time::MissedTickBehavior::Delay);

        // Failure / circuit-breaker state
        let mut consecutive_failures: u32 = 0;
        let mut breaker_open_until: Option<Instant> = None;

        // Connection - retry until successful on startup
        let (mut client, mut js) = loop {
            match Self::connect(&self.cfg).await {
                Ok(x) => break x,
                Err(e) => {
                    warn!(error = %e, "nats: connect failed; retrying in 5s");
                    time::sleep(Duration::from_secs(5)).await;
                }
            }
        };

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
                    // Reconnect before resuming normal processing
                    match Self::connect(&self.cfg).await {
                        Ok((c, jsc)) => {
                            client = c;
                            js = jsc;
                            info!("nats: reconnected after breaker open");
                            consecutive_failures = 0;
                        }
                        Err(e) => {
                            warn!(error=%e, "nats: reconnect failed; keeping breaker open");
                            breaker_open_until =
                                Some(Instant::now() + Self::jittered(self.cfg.cb_sleep));
                            continue;
                        }
                    }
                }
            }

            tokio::select! {
                maybe = self.rx.recv() => {
                    match maybe {
                        Some(payload) => {
                            buf.push(payload);
                            if buf.len() >= self.cfg.batch {
                                // Try to flush immediately on size
                                if let Err(e) = Self::flush(&js, &self.cfg.subject, &mut buf).await {
                                    consecutive_failures += 1;
                                    warn!(error=%e, fails = consecutive_failures, "nats: flush(size) failed");
                                    // Put back the (not flushed) messages: leave in `buf`
                                    // Backoff and possibly trip breaker
                                    let mut backoff = self.cfg.backoff_base.saturating_mul(1 << (consecutive_failures.min(15)));
                                    if backoff > self.cfg.max_backoff { backoff = self.cfg.max_backoff; }
                                    time::sleep(Self::jittered(backoff)).await;
                                    if consecutive_failures >= self.cfg.cb_threshold {
                                        error!(fails = consecutive_failures, "nats: consecutive failures; opening circuit breaker");
                                        breaker_open_until = Some(Instant::now() + Self::jittered(self.cfg.cb_sleep));
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
                            if let Err(e) = Self::flush(&js, &self.cfg.subject, &mut buf).await {
                                warn!(error=%e, "nats: final flush failed on shutdown");
                            }
                            break;
                        }
                    }
                }
                _ = ticker.tick() => {
                    if buf.is_empty() { continue; }
                    if let Err(e) = Self::flush(&js, &self.cfg.subject, &mut buf).await {
                        consecutive_failures += 1;
                        warn!(error=%e, fails=consecutive_failures, "nats: flush(interval) failed");
                        let mut backoff = self.cfg.backoff_base.saturating_mul(1 << (consecutive_failures.min(15)));
                        if backoff > self.cfg.max_backoff { backoff = self.cfg.max_backoff; }
                        time::sleep(Self::jittered(backoff)).await;
                        if consecutive_failures >= self.cfg.cb_threshold {
                            error!(fails = consecutive_failures, "nats: consecutive failures; opening circuit breaker");
                            breaker_open_until = Some(Instant::now() + Self::jittered(self.cfg.cb_sleep));
                            buf.clear();
                        }
                    } else {
                        consecutive_failures = 0;
                    }
                }
            }
        }

        // avoid unused warnings when the first connect failed before initialization
        let _ = client;
    }
}

/// Public sink type – thin wrapper around a bounded queue to a background worker.
pub struct NatsSink {
    tx: mpsc::Sender<Vec<u8>>,
}

impl NatsSink {
    pub fn from_env() -> Self {
        let cfg = NatsCfg::from_env();
        let (tx, rx) = mpsc::channel::<Vec<u8>>(cfg.queue_cap);
        // spawn background worker
        tokio::spawn(async move {
            NatsWorker { cfg, rx }.run().await;
        });
        NatsSink { tx }
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
