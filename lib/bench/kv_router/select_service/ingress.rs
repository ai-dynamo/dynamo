// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Client for `select_service_ray_ingress.py`, which hosts an in-process
//! `SelectionService` the way a Ray Serve LLM ingress replica does.
//!
//! Frames are `u32 len | u8 type | body`, little endian; see the Python side for the
//! layout of each body. Selections are request/reply; lifecycle events are fire and
//! forget and are drained into batches, like the replica-to-ingress lifecycle RPCs.

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, bail, ensure};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::UnixStream;
use tokio::sync::{mpsc, oneshot};

const REGISTER: u8 = 1;
const REGISTER_DONE: u8 = 2;
const REGISTERED: u8 = 3;
const SELECT: u8 = 4;
const SELECT_REPLY: u8 = 5;
const LIFECYCLE_BATCH: u8 = 6;

const PREFILL_COMPLETE: u8 = 1;
const DECODE_PROGRESS: u8 = 2;
const COMPLETED: u8 = 3;

const MAX_LIFECYCLE_BATCH: usize = 4096;

pub enum LifecycleEvent {
    PrefillComplete(String),
    /// Cumulative output tokens generated so far.
    DecodeProgress(String, u32),
    Completed(String),
}

pub struct Selected {
    pub worker_index: u32,
    pub gpu_overlap_tokens: u32,
}

type Pending = Arc<Mutex<HashMap<String, oneshot::Sender<Result<Selected, String>>>>>;

pub struct IngressClient {
    frames: mpsc::UnboundedSender<Vec<u8>>,
    lifecycle: mpsc::UnboundedSender<LifecycleEvent>,
    pending: Pending,
    registered: tokio::sync::Mutex<Option<oneshot::Receiver<u32>>>,
}

pub struct WorkerRegistration {
    pub worker_id: u64,
    pub block_size: u32,
    pub max_num_batched_tokens: u32,
    pub kv_events_endpoint: String,
    pub replay_endpoint: Option<String>,
}

impl IngressClient {
    /// Connect to the ingress socket, waiting up to `timeout` for it to appear.
    pub async fn connect(path: &Path, timeout: Duration) -> anyhow::Result<Self> {
        let deadline = Instant::now() + timeout;
        let stream = loop {
            match UnixStream::connect(path).await {
                Ok(stream) => break stream,
                Err(error) if Instant::now() < deadline => {
                    let _ = error;
                    tokio::time::sleep(Duration::from_millis(200)).await;
                }
                Err(error) => {
                    return Err(error)
                        .with_context(|| format!("connect to ingress {}", path.display()));
                }
            }
        };
        let (mut read_half, mut write_half) = stream.into_split();
        let (frames, mut frame_rx) = mpsc::unbounded_channel::<Vec<u8>>();
        tokio::spawn(async move {
            while let Some(frame) = frame_rx.recv().await {
                if let Err(error) = write_half.write_all(&frame).await {
                    eprintln!("ingress write failed: {error}");
                    return;
                }
            }
        });

        let (lifecycle, mut lifecycle_rx) = mpsc::unbounded_channel::<LifecycleEvent>();
        let batch_frames = frames.clone();
        tokio::spawn(async move {
            while let Some(first) = lifecycle_rx.recv().await {
                let mut events = vec![first];
                while events.len() < MAX_LIFECYCLE_BATCH {
                    match lifecycle_rx.try_recv() {
                        Ok(event) => events.push(event),
                        Err(_) => break,
                    }
                }
                if batch_frames.send(encode_lifecycle(&events)).is_err() {
                    return;
                }
            }
        });

        let pending: Pending = Arc::default();
        let (registered_tx, registered_rx) = oneshot::channel();
        let reader_pending = Arc::clone(&pending);
        tokio::spawn(async move {
            let mut registered_tx = Some(registered_tx);
            loop {
                let mut header = [0u8; 5];
                if read_half.read_exact(&mut header).await.is_err() {
                    eprintln!("ingress connection closed");
                    return;
                }
                let len = u32::from_le_bytes(header[..4].try_into().expect("4 bytes")) as usize;
                let mut body = vec![0u8; len - 1];
                if read_half.read_exact(&mut body).await.is_err() {
                    eprintln!("ingress connection closed mid-frame");
                    return;
                }
                match header[4] {
                    SELECT_REPLY => {
                        let Some((id, reply)) = decode_select_reply(&body) else {
                            eprintln!("malformed ingress select reply");
                            continue;
                        };
                        let waiter = reader_pending.lock().unwrap().remove(&id);
                        if let Some(waiter) = waiter {
                            let _ = waiter.send(reply);
                        }
                    }
                    REGISTERED => {
                        let count = body
                            .get(..4)
                            .map(|b| u32::from_le_bytes(b.try_into().expect("4 bytes")))
                            .unwrap_or_default();
                        if let Some(tx) = registered_tx.take() {
                            let _ = tx.send(count);
                        }
                    }
                    other => eprintln!("unexpected ingress frame type {other}"),
                }
            }
        });

        Ok(Self {
            frames,
            lifecycle,
            pending,
            registered: tokio::sync::Mutex::new(Some(registered_rx)),
        })
    }

    /// Register every worker and wait until the ingress reports them schedulable.
    pub async fn register(&self, workers: &[WorkerRegistration]) -> anyhow::Result<()> {
        for worker in workers {
            let mut body = Vec::with_capacity(32 + worker.kv_events_endpoint.len());
            body.extend_from_slice(&worker.worker_id.to_le_bytes());
            body.extend_from_slice(&worker.block_size.to_le_bytes());
            body.extend_from_slice(&worker.max_num_batched_tokens.to_le_bytes());
            put_str(&mut body, &worker.kv_events_endpoint);
            put_str(
                &mut body,
                worker.replay_endpoint.as_deref().unwrap_or_default(),
            );
            self.send(REGISTER, body)?;
        }
        self.send(REGISTER_DONE, Vec::new())?;
        let receiver = self
            .registered
            .lock()
            .await
            .take()
            .context("workers already registered")?;
        let count = receiver
            .await
            .context("ingress closed before registering")?;
        ensure!(
            count as usize == workers.len(),
            "ingress registered {count} of {} workers",
            workers.len()
        );
        Ok(())
    }

    pub async fn select(
        &self,
        request_id: &str,
        session_id: Option<&str>,
        max_tokens: u32,
        tokens: &[u32],
    ) -> Result<Selected, String> {
        let mut body = Vec::with_capacity(64 + tokens.len() * 4);
        put_str(&mut body, request_id);
        put_str(&mut body, session_id.unwrap_or_default());
        body.extend_from_slice(&max_tokens.to_le_bytes());
        body.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
        for token in tokens {
            body.extend_from_slice(&token.to_le_bytes());
        }
        let (tx, rx) = oneshot::channel();
        self.pending
            .lock()
            .unwrap()
            .insert(request_id.to_string(), tx);
        if let Err(error) = self.send(SELECT, body) {
            self.pending.lock().unwrap().remove(request_id);
            return Err(error.to_string());
        }
        rx.await
            .unwrap_or_else(|_| Err("ingress dropped the selection".to_string()))
    }

    pub fn lifecycle(&self, event: LifecycleEvent) {
        let _ = self.lifecycle.send(event);
    }

    fn send(&self, kind: u8, body: Vec<u8>) -> anyhow::Result<()> {
        let mut frame = Vec::with_capacity(body.len() + 5);
        frame.extend_from_slice(&(body.len() as u32 + 1).to_le_bytes());
        frame.push(kind);
        frame.extend_from_slice(&body);
        if self.frames.send(frame).is_err() {
            bail!("ingress writer stopped");
        }
        Ok(())
    }
}

fn put_str(buf: &mut Vec<u8>, value: &str) {
    let len = u16::try_from(value.len()).expect("string fits a u16 length");
    buf.extend_from_slice(&len.to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
}

fn encode_lifecycle(events: &[LifecycleEvent]) -> Vec<u8> {
    let mut body = Vec::with_capacity(4 + events.len() * 44);
    body.extend_from_slice(&(events.len() as u32).to_le_bytes());
    for event in events {
        match event {
            LifecycleEvent::PrefillComplete(id) => {
                body.push(PREFILL_COMPLETE);
                put_str(&mut body, id);
            }
            LifecycleEvent::DecodeProgress(id, tokens) => {
                body.push(DECODE_PROGRESS);
                put_str(&mut body, id);
                body.extend_from_slice(&tokens.to_le_bytes());
            }
            LifecycleEvent::Completed(id) => {
                body.push(COMPLETED);
                put_str(&mut body, id);
            }
        }
    }
    let mut frame = Vec::with_capacity(body.len() + 5);
    frame.extend_from_slice(&(body.len() as u32 + 1).to_le_bytes());
    frame.push(LIFECYCLE_BATCH);
    frame.extend_from_slice(&body);
    frame
}

/// `u16 id_len | id | u8 ok | ok: u32 index, u32 gpu_overlap | err: u16 len, message`.
fn decode_select_reply(body: &[u8]) -> Option<(String, Result<Selected, String>)> {
    let id_len = u16::from_le_bytes(body.get(..2)?.try_into().ok()?) as usize;
    let id = String::from_utf8(body.get(2..2 + id_len)?.to_vec()).ok()?;
    let rest = body.get(2 + id_len..)?;
    let reply = if *rest.first()? == 1 {
        Ok(Selected {
            worker_index: u32::from_le_bytes(rest.get(1..5)?.try_into().ok()?),
            gpu_overlap_tokens: u32::from_le_bytes(rest.get(5..9)?.try_into().ok()?),
        })
    } else {
        let len = u16::from_le_bytes(rest.get(1..3)?.try_into().ok()?) as usize;
        Err(String::from_utf8_lossy(rest.get(3..3 + len)?).into_owned())
    };
    Some((id, reply))
}
