// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A single Python process owns the engine; Rust owns RPCs and stream lifetimes.

use std::collections::HashMap;
use std::process::Stdio;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use anyhow::{Context, bail, ensure};
use parking_lot::Mutex;
use serde::Deserialize;
use serde_json::{Value, json};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tokio::process::Command;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;

#[derive(Deserialize)]
pub struct Ready {
    pub vocab_size: u32,
    pub eos_token_id: u32,
}

#[derive(Debug, Deserialize)]
pub struct Output {
    pub request_id: String,
    pub token_id: Option<u32>,
    #[serde(default)]
    pub finished: bool,
    #[serde(default)]
    pub finish_reason: String,
    pub logprob: Option<f64>,
    pub stop_token_id: Option<u32>,
    pub error: Option<String>,
}

#[derive(Clone)]
struct Pending {
    wire_id: String,
    sender: mpsc::Sender<Output>,
    cancelled: CancellationToken,
    completed: Arc<AtomicBool>,
    failure: Arc<Mutex<Option<String>>>,
}

pub struct Bridge {
    pub ready: Ready,
    pub failed: CancellationToken,
    commands: mpsc::UnboundedSender<Value>,
    pending: Arc<Mutex<HashMap<String, Pending>>>,
    next_id: AtomicU64,
    alive: Arc<AtomicBool>,
    limit: usize,
    engine_active: Arc<AtomicU32>,
}

/// The guard lives inside the gRPC response stream, including before first poll.
pub struct Session {
    bridge: Arc<Bridge>,
    id: String,
    pub receiver: mpsc::Receiver<Output>,
    pub cancelled: CancellationToken,
    pub failure: Arc<Mutex<Option<String>>>,
    completed: Arc<AtomicBool>,
}

impl Drop for Session {
    fn drop(&mut self) {
        self.bridge.pending.lock().remove(&self.id);
        if !self.completed.load(Ordering::Acquire) {
            // Internal IDs never repeat: a late disconnect cannot abort a new RPC.
            let _ = self
                .bridge
                .commands
                .send(json!({"type":"abort", "request_id":self.id}));
        }
    }
}

impl Bridge {
    pub async fn spawn(
        mut command: Command,
        limit: usize,
    ) -> anyhow::Result<(Arc<Self>, JoinHandle<()>)> {
        let mut child = command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .kill_on_drop(true)
            .spawn()
            .context("start DeepInfer engine worker")?;
        let mut input = child.stdin.take().context("worker stdin")?;
        let mut lines = BufReader::new(child.stdout.take().context("worker stdout")?).lines();
        let line = tokio::time::timeout(Duration::from_secs(120), lines.next_line())
            .await??
            .context("engine exited before readiness")?;
        let ready_value: Value = serde_json::from_str(&line).context("worker readiness JSON")?;
        ensure!(
            ready_value["type"] == "ready",
            "worker did not report ready: {ready_value}"
        );
        let ready: Ready = serde_json::from_value(ready_value)?;
        ensure!(
            ready.vocab_size > 0 && ready.eos_token_id < ready.vocab_size,
            "invalid tokenizer metadata"
        );
        let (commands, mut command_rx) = mpsc::unbounded_channel();
        let pending: Arc<Mutex<HashMap<String, Pending>>> = Arc::default();
        let alive = Arc::new(AtomicBool::new(true));
        let failed = CancellationToken::new();
        let engine_active = Arc::new(AtomicU32::new(0));
        let bridge = Arc::new(Self {
            ready,
            failed: failed.clone(),
            commands,
            pending: pending.clone(),
            next_id: AtomicU64::new(1),
            alive: alive.clone(),
            limit,
            engine_active: engine_active.clone(),
        });
        let task = tokio::spawn(async move {
            let result: anyhow::Result<()> = async {
                loop {
                    tokio::select! {
                        command = command_rx.recv() => {
                            let Some(command) = command else { break; };
                            let mut encoded = serde_json::to_vec(&command)?;
                            encoded.push(b'\n');
                            input.write_all(&encoded).await?;
                            if command["type"] == "shutdown" { break; }
                        }
                        line = lines.next_line() => {
                            let Some(line) = line? else { bail!("engine worker output closed"); };
                            let value: Value = serde_json::from_str(&line)?;
                            if value["type"] == "load" {
                                let count = value["running_requests"].as_u64().context("invalid engine load")?;
                                engine_active.store(u32::try_from(count)?, Ordering::Release);
                                continue;
                            }
                            ensure!(value["type"] == "output", "unexpected worker message: {value}");
                            let output: Output = serde_json::from_value(value)?;
                            let request = pending.lock().get(&output.request_id).cloned();
                            if let Some(request) = request {
                                let terminal = output.finished || output.error.is_some();
                                if request.sender.try_send(output).is_err() {
                                    *request.failure.lock() = Some("response consumer exceeded the bounded output queue".into());
                                    request.cancelled.cancel();
                                } else if terminal {
                                    request.completed.store(true, Ordering::Release);
                                }
                            }
                        }
                        status = child.wait() => { bail!("engine worker exited: {:?}", status?); }
                    }
                }
                Ok(())
            }.await;
            alive.store(false, Ordering::Release);
            if let Err(error) = result {
                tracing::error!(%error, "DeepInfer engine worker failed");
                for request in pending.lock().values() {
                    *request.failure.lock() = Some(format!("engine worker failed: {error}"));
                    request.cancelled.cancel();
                }
                failed.cancel();
            }
            drop(input);
            if tokio::time::timeout(Duration::from_secs(5), child.wait())
                .await
                .is_err()
            {
                let _ = child.kill().await;
                let _ = child.wait().await;
            }
        });
        Ok((bridge, task))
    }

    #[allow(clippy::result_large_err)]
    pub fn submit(
        self: &Arc<Self>,
        wire_id: &str,
        mut command: Value,
    ) -> Result<Session, tonic::Status> {
        if !self.alive() {
            return Err(tonic::Status::unavailable("engine worker is not alive"));
        }
        let mut pending = self.pending.lock();
        if pending.len() >= self.limit {
            return Err(tonic::Status::resource_exhausted(
                "engine request capacity reached",
            ));
        }
        if pending.values().any(|request| request.wire_id == wire_id) {
            return Err(tonic::Status::already_exists(
                "request_id is already active",
            ));
        }
        let id = self.next_id.fetch_add(1, Ordering::Relaxed).to_string();
        let (sender, receiver) = mpsc::channel(128);
        let request = Pending {
            wire_id: wire_id.to_owned(),
            sender,
            cancelled: CancellationToken::new(),
            completed: Arc::new(AtomicBool::new(false)),
            failure: Arc::default(),
        };
        let session = Session {
            bridge: self.clone(),
            id: id.clone(),
            receiver,
            cancelled: request.cancelled.clone(),
            failure: request.failure.clone(),
            completed: request.completed.clone(),
        };
        pending.insert(id.clone(), request);
        command["request_id"] = json!(id);
        if self.commands.send(command).is_err() {
            pending.remove(&id);
            // Release the map lock before the Session destructor reacquires it.
            drop(pending);
            return Err(tonic::Status::unavailable("engine command channel closed"));
        }
        Ok(session)
    }

    pub fn abort(&self, wire_id: Option<&str>) -> bool {
        let pending = self.pending.lock();
        let mut aborted = false;
        for (id, request) in pending.iter() {
            if wire_id.is_none_or(|wire_id| request.wire_id == wire_id)
                && !request.completed.load(Ordering::Acquire)
                && !request.cancelled.is_cancelled()
            {
                request.cancelled.cancel();
                let _ = self.commands.send(json!({"type":"abort", "request_id":id}));
                aborted = true;
            }
        }
        aborted
    }

    pub fn active(&self) -> u32 {
        self.engine_active.load(Ordering::Acquire)
    }
    pub fn alive(&self) -> bool {
        self.alive.load(Ordering::Acquire)
    }
    pub fn shutdown(&self) {
        let _ = self.commands.send(json!({"type":"shutdown"}));
    }
}
