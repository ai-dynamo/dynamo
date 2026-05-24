// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Load-aware multi-prefill HTTP dispatcher.
//!
//! Selects a prefill worker per request using two signals tracked
//! per-worker:
//!
//! - **In-flight count** — capped at a configurable concurrency limit
//!   `N` (default 4). A worker at the cap is skipped during selection.
//! - **Net-new token load** — `sum(net_new)` across the worker's
//!   in-flight requests, where
//!   `net_new = token_ids.len() - sequence_hashes.len() * block_size`.
//!   The worker with the lowest sum is preferred.
//!
//! Total fleet concurrency is gated by a [`tokio::sync::Semaphore`] sized
//! `workers.len() * N`. The dispatcher loop blocks acquiring a permit
//! when the fleet is saturated — that propagates backpressure back to the
//! CD prefill queue without dropping requests.
//!
//! Workers are discovered at construction from a single snapshot of the
//! hub's CD registry. Live re-discovery (workers added/removed after
//! startup) is deliberately out of scope here; restart the hub when the
//! prefill fleet changes shape.

use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use futures::future::BoxFuture;
use parking_lot::Mutex;
use reqwest::StatusCode;
use serde_json::json;
use tokio::sync::Semaphore;

use super::dispatcher::{DispatchOutcome, PrefillRequestDispatcher};
use crate::protocol::{PrefillRequest, VllmHttpEndpoint};

/// One prefill worker the dispatcher can route to.
#[derive(Debug, Clone)]
struct WorkerSlot {
    base_url: String,
    model: String,
}

/// Mutable per-worker counters protected by [`LoadAwareHttpDispatcher::fleet`].
#[derive(Debug, Default, Clone, Copy)]
struct WorkerCounters {
    /// Number of requests currently in flight on this worker.
    inflight: u32,
    /// Sum of `net_new` tokens across the in-flight requests on this
    /// worker.
    load_net_new: u64,
}

/// Production dispatcher for the multi-prefill case.
///
/// Implements [`PrefillRequestDispatcher`]. Each `dispatch` call:
/// 1. acquires one of `workers.len() * per_worker_concurrency` permits
///    (blocks if the fleet is full),
/// 2. picks the worker with `inflight < N` and minimum `load_net_new`
///    (ties broken by lower `inflight`, then lower index),
/// 3. spawns the HTTP POST in a background task and returns
///    [`DispatchOutcome::Accepted`] immediately.
///
/// The spawned task decrements the counters and drops the permit on
/// completion (success or failure). Network/HTTP errors are logged but
/// do not propagate to the caller — the queue pump has already moved
/// past the request.
pub struct LoadAwareHttpDispatcher {
    client: reqwest::Client,
    workers: Vec<WorkerSlot>,
    block_size: usize,
    per_worker_concurrency: u32,
    fleet: Arc<Mutex<Vec<WorkerCounters>>>,
    capacity: Arc<Semaphore>,
}

impl std::fmt::Debug for LoadAwareHttpDispatcher {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LoadAwareHttpDispatcher")
            .field("worker_count", &self.workers.len())
            .field("per_worker_concurrency", &self.per_worker_concurrency)
            .field("block_size", &self.block_size)
            .finish()
    }
}

impl LoadAwareHttpDispatcher {
    /// Build a dispatcher from a registry snapshot. `workers` must be
    /// non-empty; `per_worker_concurrency` must be ≥ 1.
    pub fn new(
        workers: Vec<VllmHttpEndpoint>,
        block_size: usize,
        per_worker_concurrency: u32,
    ) -> Result<Arc<Self>> {
        anyhow::ensure!(
            !workers.is_empty(),
            "LoadAwareHttpDispatcher requires at least one prefill worker"
        );
        anyhow::ensure!(
            per_worker_concurrency >= 1,
            "per_worker_concurrency must be >= 1, got {per_worker_concurrency}"
        );
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(60))
            .build()
            .context("build reqwest client for LoadAwareHttpDispatcher")?;
        let total_permits = workers.len() * per_worker_concurrency as usize;
        let workers: Vec<WorkerSlot> = workers
            .into_iter()
            .map(|ep| WorkerSlot {
                base_url: ep.base_url.trim_end_matches('/').to_string(),
                model: ep.model,
            })
            .collect();
        let fleet = vec![WorkerCounters::default(); workers.len()];
        Ok(Arc::new(Self {
            client,
            workers,
            block_size,
            per_worker_concurrency,
            fleet: Arc::new(Mutex::new(fleet)),
            capacity: Arc::new(Semaphore::new(total_permits)),
        }))
    }

    /// Compute the worst-case net-new prefill token count for a
    /// request: `token_ids.len() - sequence_hashes.len() * block_size`,
    /// floored at zero.
    ///
    /// Each entry in `sequence_hashes` corresponds to one decode-side
    /// cached block of `block_size` tokens — those don't need to be
    /// recomputed by the prefill worker (it RDMA-pulls the KV state).
    pub fn net_new(req: &PrefillRequest, block_size: usize) -> u64 {
        let cached = (req.sequence_hashes.len() as u64).saturating_mul(block_size as u64);
        (req.token_ids.len() as u64).saturating_sub(cached)
    }

    /// Pick a worker index with free capacity. Caller must hold a
    /// fleet permit; the permit guarantees at least one worker has
    /// `inflight < per_worker_concurrency`.
    fn select(&self, fleet: &[WorkerCounters]) -> usize {
        (0..self.workers.len())
            .filter(|&i| fleet[i].inflight < self.per_worker_concurrency)
            .min_by_key(|&i| (fleet[i].load_net_new, fleet[i].inflight, i))
            .expect("a permit was acquired, so some worker must have capacity")
    }
}

impl PrefillRequestDispatcher for LoadAwareHttpDispatcher {
    fn dispatch(&self, request: PrefillRequest) -> BoxFuture<'_, Result<DispatchOutcome>> {
        Box::pin(async move {
            // Acquire a fleet permit before selection. This blocks when
            // every worker is at its concurrency cap — the right
            // behavior: the CD queue pump halts until a slot frees up.
            let permit = Arc::clone(&self.capacity)
                .acquire_owned()
                .await
                .context("LoadAwareHttpDispatcher capacity semaphore closed")?;

            let net_new = Self::net_new(&request, self.block_size);

            let (worker_idx, inflight_after, load_after) = {
                let mut fleet = self.fleet.lock();
                let idx = self.select(&fleet);
                fleet[idx].inflight = fleet[idx].inflight.saturating_add(1);
                fleet[idx].load_net_new = fleet[idx].load_net_new.saturating_add(net_new);
                (idx, fleet[idx].inflight, fleet[idx].load_net_new)
            };
            let worker = self.workers[worker_idx].clone();

            tracing::info!(
                worker_idx,
                worker_url = %worker.base_url,
                request_id = %request.request_id,
                net_new,
                inflight_after,
                load_after,
                "LoadAwareHttpDispatcher: dispatching"
            );

            let client = self.client.clone();
            let fleet = Arc::clone(&self.fleet);
            tokio::spawn(async move {
                let request_id = request.request_id.clone();
                let outcome = post_to_vllm(&client, &worker, request).await;
                let (inflight_after, load_after) = {
                    let mut fleet = fleet.lock();
                    let s = &mut fleet[worker_idx];
                    s.inflight = s.inflight.saturating_sub(1);
                    s.load_net_new = s.load_net_new.saturating_sub(net_new);
                    (s.inflight, s.load_net_new)
                };
                match &outcome {
                    DispatchOutcome::Accepted => {
                        tracing::info!(
                            worker_idx,
                            request_id = %request_id,
                            inflight_after,
                            load_after,
                            "LoadAwareHttpDispatcher: completed"
                        );
                    }
                    DispatchOutcome::Rejected { reason } => {
                        tracing::warn!(
                            worker_idx,
                            request_id = %request_id,
                            reason = %reason,
                            inflight_after,
                            load_after,
                            "LoadAwareHttpDispatcher: rejected"
                        );
                    }
                }
                drop(permit);
            });

            Ok(DispatchOutcome::Accepted)
        })
    }
}

/// POST a [`PrefillRequest`] to a specific vLLM frontend. Shape matches
/// the single-target [`super::dispatcher::HttpVllmDispatcher`] — same
/// `kv_transfer_params` blob the connector expects.
async fn post_to_vllm(
    client: &reqwest::Client,
    worker: &WorkerSlot,
    request: PrefillRequest,
) -> DispatchOutcome {
    let url = format!("{}/v1/completions", worker.base_url);
    let transfer_params =
        kvbm_protocols::disagg::TransferParams::remote_prefill(request.remote_prefill_params());
    let body = json!({
        "model": worker.model,
        "prompt": request.token_ids,
        "max_tokens": 1,
        "kv_transfer_params": transfer_params,
    });

    let resp = match client.post(&url).json(&body).send().await {
        Ok(r) => r,
        Err(err) => {
            return DispatchOutcome::Rejected {
                reason: format!("POST {url} failed: {err}"),
            };
        }
    };
    let status = resp.status();
    if status == StatusCode::OK {
        DispatchOutcome::Accepted
    } else {
        let body = resp.text().await.unwrap_or_else(|_| "<unreadable>".into());
        DispatchOutcome::Rejected {
            reason: format!("POST {url} returned {status}: {body}"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocol::PrefillRequest;
    use kvbm_common::SequenceHash;
    use kvbm_protocols::disagg::DISAGG_PROTOCOL_VERSION;
    use std::net::SocketAddr;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;
    use tokio::net::TcpListener;
    use velo_ext::InstanceId;

    fn make_request(id: &str, n_tokens: usize, n_hashes: usize) -> PrefillRequest {
        PrefillRequest {
            protocol_version: DISAGG_PROTOCOL_VERSION,
            request_id: id.to_string(),
            session_id: uuid::Uuid::new_v4(),
            initiator_instance_id: InstanceId::new_v4(),
            decode_endpoint: None,
            sequence_hashes: (0..n_hashes)
                .map(|i| SequenceHash::new(i as u64, None, i as u64))
                .collect(),
            token_ids: vec![0u32; n_tokens],
            num_computed_tokens: 0,
        }
    }

    fn ep(base_url: &str, model: &str) -> VllmHttpEndpoint {
        VllmHttpEndpoint {
            base_url: base_url.to_string(),
            model: model.to_string(),
        }
    }

    #[test]
    fn net_new_subtracts_cached_blocks() {
        let req = make_request("r", 100, 2);
        // 100 - 2 * 16 = 68
        assert_eq!(LoadAwareHttpDispatcher::net_new(&req, 16), 68);
    }

    #[test]
    fn net_new_floors_at_zero_when_cache_exceeds_tokens() {
        let req = make_request("r", 10, 5);
        // 10 - 5 * 16 -> saturating_sub -> 0
        assert_eq!(LoadAwareHttpDispatcher::net_new(&req, 16), 0);
    }

    #[tokio::test]
    async fn rejects_construction_with_no_workers() {
        assert!(LoadAwareHttpDispatcher::new(vec![], 16, 4).is_err());
    }

    #[tokio::test]
    async fn rejects_construction_with_zero_concurrency() {
        assert!(LoadAwareHttpDispatcher::new(vec![ep("http://x", "m")], 16, 0).is_err());
    }

    #[test]
    fn select_picks_lowest_load() {
        // Build a dispatcher (URLs don't matter — we only exercise select()).
        let d = LoadAwareHttpDispatcher::new(
            vec![
                ep("http://a", "m"),
                ep("http://b", "m"),
                ep("http://c", "m"),
            ],
            16,
            4,
        )
        .unwrap();
        let mut fleet = d.fleet.lock();
        fleet[0] = WorkerCounters {
            inflight: 2,
            load_net_new: 500,
        };
        fleet[1] = WorkerCounters {
            inflight: 1,
            load_net_new: 100,
        };
        fleet[2] = WorkerCounters {
            inflight: 1,
            load_net_new: 300,
        };
        assert_eq!(d.select(&fleet), 1);
    }

    #[test]
    fn select_skips_workers_at_concurrency_cap() {
        let d = LoadAwareHttpDispatcher::new(vec![ep("http://a", "m"), ep("http://b", "m")], 16, 2)
            .unwrap();
        let mut fleet = d.fleet.lock();
        // worker 0 has lowest load but is at the cap → skipped
        fleet[0] = WorkerCounters {
            inflight: 2,
            load_net_new: 100,
        };
        fleet[1] = WorkerCounters {
            inflight: 0,
            load_net_new: 999,
        };
        assert_eq!(d.select(&fleet), 1);
    }

    #[test]
    fn select_breaks_ties_by_inflight_then_index() {
        let d = LoadAwareHttpDispatcher::new(
            vec![
                ep("http://a", "m"),
                ep("http://b", "m"),
                ep("http://c", "m"),
            ],
            16,
            4,
        )
        .unwrap();
        let mut fleet = d.fleet.lock();
        // Equal load → lower inflight wins.
        fleet[0] = WorkerCounters {
            inflight: 2,
            load_net_new: 100,
        };
        fleet[1] = WorkerCounters {
            inflight: 1,
            load_net_new: 100,
        };
        fleet[2] = WorkerCounters {
            inflight: 1,
            load_net_new: 100,
        };
        // Equal load and inflight between 1 and 2 → lower index wins.
        assert_eq!(d.select(&fleet), 1);
    }

    /// Spin up an axum stub vLLM that records every POST body, with an
    /// optional pre-response delay so capacity tests can hold a worker
    /// busy.
    async fn spawn_stub_vllm(
        delay: Duration,
    ) -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>) {
        use axum::{Json, Router, http::StatusCode as AxumStatus, routing::post};

        let count = Arc::new(AtomicUsize::new(0));
        let count_state = Arc::clone(&count);

        let app = Router::new().route(
            "/v1/completions",
            post(move |Json(_payload): Json<serde_json::Value>| {
                let count = Arc::clone(&count_state);
                let delay = delay;
                async move {
                    if !delay.is_zero() {
                        tokio::time::sleep(delay).await;
                    }
                    count.fetch_add(1, Ordering::SeqCst);
                    (AxumStatus::OK, "{}".to_string())
                }
            }),
        );

        let addr: SocketAddr = "127.0.0.1:0".parse().unwrap();
        let listener = TcpListener::bind(addr).await.unwrap();
        let local = listener.local_addr().unwrap();
        let handle = tokio::spawn(async move {
            axum::serve(listener, app).await.unwrap();
        });
        (format!("http://{}", local), count, handle)
    }

    /// Wait until `predicate` returns true or `timeout` elapses, polling
    /// every 10ms. Returns `true` if the predicate fired in time.
    async fn poll_until<F: FnMut() -> bool>(mut predicate: F, timeout: Duration) -> bool {
        let deadline = std::time::Instant::now() + timeout;
        while std::time::Instant::now() < deadline {
            if predicate() {
                return true;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        predicate()
    }

    #[tokio::test]
    async fn dispatch_records_and_clears_inflight() {
        let (base_a, count_a, _ha) = spawn_stub_vllm(Duration::ZERO).await;
        let (base_b, count_b, _hb) = spawn_stub_vllm(Duration::ZERO).await;

        let dispatcher =
            LoadAwareHttpDispatcher::new(vec![ep(&base_a, "m"), ep(&base_b, "m")], 16, 2).unwrap();

        for i in 0..20 {
            let req = make_request(&format!("r{i}"), 64, 1);
            let out = dispatcher.dispatch(req).await.unwrap();
            assert_eq!(out, DispatchOutcome::Accepted);
        }

        let total_arrived = || count_a.load(Ordering::SeqCst) + count_b.load(Ordering::SeqCst);
        assert!(
            poll_until(|| total_arrived() == 20, Duration::from_secs(5)).await,
            "expected all 20 POSTs to arrive at stub workers; got {}",
            total_arrived()
        );

        // After completion, both workers should be back at zero.
        assert!(
            poll_until(
                || {
                    let fleet = dispatcher.fleet.lock();
                    fleet.iter().all(|c| c.inflight == 0 && c.load_net_new == 0)
                },
                Duration::from_secs(2)
            )
            .await,
            "expected all per-worker counters to drain to zero"
        );

        // Both stubs received some traffic — distribution isn't strict,
        // but neither should be starved when both can serve.
        assert!(count_a.load(Ordering::SeqCst) > 0);
        assert!(count_b.load(Ordering::SeqCst) > 0);
    }

    #[tokio::test]
    async fn capacity_semaphore_blocks_when_fleet_is_full() {
        // One worker, concurrency 1, so the second dispatch must block
        // until the first POST completes. We use a 300ms stub delay.
        let (base, count, _h) = spawn_stub_vllm(Duration::from_millis(300)).await;
        let dispatcher = LoadAwareHttpDispatcher::new(vec![ep(&base, "m")], 16, 1).unwrap();

        // First dispatch occupies the only permit.
        let t0 = std::time::Instant::now();
        let out = dispatcher
            .dispatch(make_request("r1", 32, 0))
            .await
            .unwrap();
        assert_eq!(out, DispatchOutcome::Accepted);
        assert!(
            t0.elapsed() < Duration::from_millis(100),
            "first dispatch should return immediately (just spawned)"
        );

        // Second dispatch must wait for the first POST to complete.
        let t1 = std::time::Instant::now();
        let out = dispatcher
            .dispatch(make_request("r2", 32, 0))
            .await
            .unwrap();
        assert_eq!(out, DispatchOutcome::Accepted);
        let waited = t1.elapsed();
        assert!(
            waited >= Duration::from_millis(200),
            "second dispatch should have waited for the first to free its permit; waited only {waited:?}"
        );

        // Eventually both arrive.
        assert!(
            poll_until(|| count.load(Ordering::SeqCst) == 2, Duration::from_secs(2),).await,
            "expected both POSTs to arrive"
        );
    }
}
