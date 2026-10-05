// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//! Native-only policy host for synchronous, virtual-time callers. No engine or
//! simulator owns the routing algorithm: selections and cache state are Dynamo's.

use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::{Arc, mpsc};
use std::time::Duration;

use anyhow::{Context, Result, bail, ensure};
use dynamo_kv_router::indexer::KvIndexerInterface;
use dynamo_kv_router::protocols::{
    KvCacheEvent, LocalBlockHash, RouterEvent, RoutingConstraints, WorkerAffinityTarget,
};
use dynamo_kv_router::scheduling::queue::SchedulerBookingDescriptor;
use dynamo_kv_router::sequences::ReplicaRequestLeaseObserver;
use dynamo_kv_router::services::indexer::{backend::Indexer, registry::WorkerRegistry};
use dynamo_kv_router::services::selection::affinity::{
    AcquireStep, AffinityLease, Hold, SessionAffinity, SessionAffinityConfig,
    subagent_group_affinity_id,
};
use dynamo_kv_router::services::selection::{
    HostCache, HostReplication, KvEventIngress, KvIndexSource, PromptRequest, SelectionAdmission,
    SelectionCore, SelectionHost, SelectionOperation, SelectionOutcome, SelectionService,
    SelectionServiceBuilder, SessionBinding, WorkerRequest,
};
use dynamo_kv_router::{KvRouterConfig, RoutingPartitionId, WorkerType};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::runtime::Runtime;

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RouterConfig {
    policy: String,
    affinity: Option<AffinityConfig>,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct AffinityConfig {
    mode: AffinityMode,
    #[serde(default = "default_ttl")]
    ttl_seconds: f64,
}
fn default_ttl() -> f64 {
    3600.0
}
#[derive(Clone, Copy, Deserialize)]
#[serde(rename_all = "snake_case")]
enum AffinityMode {
    Session,
    SiblingGroup,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct WorkerConfig {
    api_version: u32,
    block_size: u32,
    total_kv_blocks: u64,
    max_num_batched_tokens: u64,
    dp_size: u32,
    workers: Vec<Worker>,
    #[serde(default)]
    host_offload: bool,
    #[serde(default)]
    g3_offload: bool,
    #[serde(default)]
    capture_decisions: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Worker {
    worker_id: u64,
}
#[derive(Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct Identity {
    scope: Option<String>,
    session: Option<String>,
    root: Option<String>,
    parent: Option<String>,
    #[serde(default)]
    lineage_available: bool,
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Request {
    request_id: String,
    input_tokens: usize,
    output_tokens: u32,
    local_block_hashes: Vec<u64>,
    sequence_hashes: Vec<u64>,
    #[serde(default)]
    priority: f64,
    #[serde(default)]
    strict_priority: u32,
    #[serde(default)]
    identity: Identity,
    prompt_token_source: String,
    preferred_dp_rank: Option<u32>,
    preferred_prefill_dp_rank: Option<u32>,
    policy_class: Option<String>,
    authored_request_id: Option<String>,
    session_id: Option<String>,
}
#[derive(Serialize)]
struct Placement {
    request_id: String,
    worker_id: u64,
    dp_rank: u32,
    overlap_blocks: u32,
    best_available_overlap_blocks: u32,
    cached_tokens: usize,
    isl_blocks: u32,
}
// The payload is the existing native event schema, including external block and
// local token hashes. Only the worker envelope and device-tier validation are added.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PhysicalEvent {
    worker_id: u64,
    event: PhysicalEventPayload,
}
#[derive(Deserialize)]
struct PhysicalEventPayload {
    #[serde(flatten)]
    native: KvCacheEvent,
    #[serde(default = "device_tier")]
    tier: String,
}
fn device_tier() -> String {
    "device".into()
}

struct ReplayIngress;
#[async_trait::async_trait]
impl KvEventIngress for ReplayIngress {
    fn open(
        &self,
        registry: &WorkerRegistry,
        key: &RoutingPartitionId,
        block_size: u32,
    ) -> Indexer {
        registry.get_or_create_indexer(key.clone(), block_size)
    }
}
// The host's terminal callbacks, rather than wall-clock timeout sweepers, own bookings.
struct ReplayLeases;
impl ReplicaRequestLeaseObserver for ReplayLeases {
    fn admitted(&self, _: SchedulerBookingDescriptor) {}
    fn progressed(&self, _: &SchedulerBookingDescriptor) {}
    fn completed(&self, _: &SchedulerBookingDescriptor) {}
}
struct ActiveRequest {
    hold: Option<Hold>,
    lease: Option<AffinityLease>,
    target: WorkerAffinityTarget,
}

/// Synchronous virtual-time access to Dynamo's production selector and affinity.
///
/// A host supplies materialized prompt hashes, physical cache events, and explicit
/// dispatch/terminal callbacks. This object has no dependency on a simulator API.
/// The caller must serialize operations and must not call it from inside Tokio.
#[pyclass(unsendable)]
pub(crate) struct NativeReplayPolicy {
    runtime: Runtime,
    // A live blocking task inhibits Tokio's automatic paused-clock advancement.
    // Drop the sender before the runtime, including construction failures.
    stop_clock_guard: Option<mpsc::Sender<()>>,
    core: Arc<SelectionCore>,
    _service: SelectionService,
    key: RoutingPartitionId,
    affinity: Option<SessionAffinity>,
    config: RouterConfig,
    workers: HashSet<u64>,
    block_size: u32,
    dp_size: u32,
    active: HashMap<String, ActiveRequest>,
    pending: VecDeque<Request>,
    pending_ready: bool,
    now: Duration,
    epoch: tokio::time::Instant,
    now_ms: f64,
    role: String,
    capture_decisions: bool,
    decisions: Vec<Value>,
    decision_count: u64,
    physical_kv_events: u64,
}

fn python_error(error: impl std::fmt::Display) -> PyErr {
    PyValueError::new_err(error.to_string())
}
fn contract_value() -> Value {
    json!({
        "api_version": 1,
        "dynamo_version": env!("CARGO_PKG_VERSION"),
        "dynamo_revision": option_env!("DYNAMO_BUILD_REVISION"),
    })
}

#[pymethods]
impl NativeReplayPolicy {
    #[new]
    fn new(role: &str, router_config_json: &str, workers_json: &str) -> PyResult<Self> {
        Self::build(role, router_config_json, workers_json).map_err(python_error)
    }

    /// Return the wire contract version and available build provenance.
    #[staticmethod]
    fn contract(py: Python<'_>) -> PyResult<PyObject> {
        Ok(pythonize::pythonize(py, &contract_value())?.unbind())
    }

    /// Select using native KV/load scoring. A null decision waits for an affinity initializer.
    fn place(&mut self, request_json: &str, now_ms: f64) -> PyResult<String> {
        let mut run = || -> Result<String> {
            let request: Request = serde_json::from_str(request_json)?;
            ensure!(
                !self.active.contains_key(&request.request_id)
                    && !self
                        .pending
                        .iter()
                        .any(|p| p.request_id == request.request_id),
                "duplicate native policy request ID"
            );
            self.advance(now_ms)?;
            Ok(serde_json::to_string(&json!({
                "decision": self.select(request)?, "released": [],
            }))?)
        };
        run().map_err(python_error)
    }

    /// Apply actual native KV events; host-pinned/offloaded residency is unsupported.
    fn observe(&mut self, events_json: &str, now_ms: f64) -> PyResult<String> {
        let mut run = || -> Result<String> {
            let events: Vec<PhysicalEvent> = serde_json::from_str(events_json)?;
            for event in &events {
                ensure!(
                    event.event.tier == "device",
                    "native replay policy supports device KV only"
                );
                ensure!(
                    self.workers.contains(&event.worker_id)
                        && event.event.native.dp_rank < self.dp_size,
                    "KV event refers to an unavailable worker/DP"
                );
            }
            self.advance(now_ms)?;
            let partition = self
                .core
                .partition(&self.key)
                .context("missing routing partition")?;
            let count = events.len() as u64;
            self.runtime.block_on(async {
                for event in events {
                    partition
                        .indexer()
                        .try_apply_event(RouterEvent::new(event.worker_id, event.event.native))
                        .await?;
                }
                // Make every supplied physical event visible to the next selection.
                match partition.indexer() {
                    Indexer::Single { primary, .. } => {
                        primary.flush().await;
                    }
                    Indexer::Concurrent { primary, .. } => {
                        primary.flush().await;
                    }
                    _ => bail!("native replay policy requires a local physical indexer"),
                }
                Ok::<_, anyhow::Error>(())
            })?;
            self.physical_kv_events += count;
            Ok("[]".into())
        };
        run().map_err(python_error)
    }

    /// Advance both native load tracking and affinity TTL using host virtual time.
    fn advance_clock(&mut self, now_ms: f64) -> PyResult<String> {
        let mut run = || -> Result<String> {
            self.advance(now_ms)?;
            let released = if self.pending_ready {
                self.retry_pending()?
            } else {
                Vec::new()
            };
            Ok(serde_json::to_string(&released)?)
        };
        run().map_err(python_error)
    }

    /// Affinity waiters are retried only after the preceding dispatch commits or aborts.
    fn next_wakeup_ms(&self) -> Option<f64> {
        (self.pending_ready && !self.pending.is_empty()).then_some(self.now_ms)
    }

    fn pending_count(&self) -> usize {
        self.pending.len()
    }

    fn cancel_pending(&mut self, request_id: &str) -> bool {
        let before = self.pending.len();
        self.pending
            .retain(|request| request.request_id != request_id);
        self.pending.len() != before
    }

    /// Bind affinity only after the host successfully commits dispatch.
    fn dispatch_committed(&mut self, request_id: &str, now_ms: f64) -> PyResult<()> {
        let mut run = || -> Result<()> {
            self.advance(now_ms)?;
            let active = self
                .active
                .get_mut(request_id)
                .context("unknown dispatch request ID")?;
            if let (Some(table), Some(hold)) = (&self.affinity, active.hold.take()) {
                active.lease = Some(table.commit(hold, active.target)?);
            }
            self.pending_ready = true;
            Ok(())
        };
        run().map_err(python_error)
    }

    /// Abort a tentative selection, releasing its native reservation and affinity hold.
    fn dispatch_aborted(&mut self, request_id: &str, now_ms: f64) -> PyResult<()> {
        let mut run = || -> Result<()> {
            self.advance(now_ms)?;
            self.release(request_id)
        };
        run().map_err(python_error)
    }

    /// End a native prefill booking while keeping the decode reservation active.
    fn prefill_completed(&mut self, request_id: &str, now_ms: f64) -> PyResult<String> {
        let mut run = || -> Result<String> {
            self.advance(now_ms)?;
            if self.active.contains_key(request_id) {
                self.runtime
                    .block_on(self.core.prefill_complete(request_id))?;
            }
            Ok("[]".into())
        };
        run().map_err(python_error)
    }

    /// Complete or cancel a dispatched request, releasing its booking and affinity lease.
    fn request_terminal(&mut self, request_id: &str, now_ms: f64) -> PyResult<String> {
        let mut run = || -> Result<String> {
            self.advance(now_ms)?;
            self.release(request_id)?;
            Ok("[]".into())
        };
        run().map_err(python_error)
    }

    /// Return measured native decision and physical-event evidence for this role.
    fn evidence(&self) -> PyResult<String> {
        serde_json::to_string(&json!({
            "native_policy": "dynamo.SelectionCore",
            "dynamo_revision": option_env!("DYNAMO_BUILD_REVISION"),
            "physical_kv_events": self.physical_kv_events,
            "decision_count": self.decision_count,
            "decisions_captured": self.capture_decisions,
            "decisions": self.decisions,
        }))
        .map_err(python_error)
    }
}

impl NativeReplayPolicy {
    fn build(role: &str, router_json: &str, workers_json: &str) -> Result<Self> {
        ensure!(
            tokio::runtime::Handle::try_current().is_err(),
            "native replay policy must run outside an async Tokio runtime"
        );
        let config: RouterConfig = serde_json::from_str(router_json)?;
        ensure!(
            config.policy == "kv_router",
            "native replay policy supports router.policy: kv_router only"
        );
        if let Some(affinity) = &config.affinity {
            ensure!(
                affinity.ttl_seconds.is_finite()
                    && (1.0..=31_536_000.0).contains(&affinity.ttl_seconds),
                "affinity.ttl_seconds must be between 1 and 31536000"
            );
        }
        let workers: WorkerConfig = serde_json::from_str(workers_json)?;
        ensure!(
            workers.api_version == 1,
            "unsupported native replay policy wire API version"
        );
        ensure!(
            !workers.host_offload && !workers.g3_offload,
            "native replay policy supports device KV only, not host or G3 offload"
        );
        ensure!(
            workers.block_size > 0
                && workers.dp_size > 0
                && workers.total_kv_blocks > 0
                && workers.max_num_batched_tokens > 0
                && !workers.workers.is_empty(),
            "worker capacities and membership must be nonempty"
        );
        let ids: HashSet<_> = workers.workers.iter().map(|w| w.worker_id).collect();
        ensure!(ids.len() == workers.workers.len(), "duplicate worker ID");
        let worker_type = match role {
            "aggregated" => WorkerType::Aggregated,
            "prefill" => WorkerType::Prefill,
            "decode" => WorkerType::Decode,
            _ => bail!("unknown native replay role {role:?}"),
        };
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()?;
        let (stop, receiver) = mpsc::channel();
        let built = runtime.block_on(async {
            tokio::time::pause();
            tokio::task::spawn_blocking(move || { let _ = receiver.recv(); });
            // No admission threshold: waiting for future engine capacity inside
            // block_on would prevent the host from delivering that capacity.
            let router_config = KvRouterConfig {
                use_kv_events: true,
                router_queue_threshold: None,
                ..Default::default()
            };
            let host = SelectionHost {
                cache: HostCache { index: KvIndexSource::Owned(Arc::new(ReplayIngress)), ..Default::default() },
                replication: HostReplication { request_leases: Some(Arc::new(ReplayLeases)), ..Default::default() },
                ..Default::default()
            };
            let service = SelectionServiceBuilder::new(router_config, worker_type, dynamo_custom_policy_builtin::default_registry())
                .host(host).indexer_threads(1).build().await?;
            let core = service.core().clone();
            let key = RoutingPartitionId::new("native-replay", role);
            core.ensure_partition(key.clone(), workers.block_size, false)?;
            let epoch = tokio::time::Instant::now();
            let affinity = config.affinity.as_ref().map(|a| {
                SessionAffinity::with_manual_clock(SessionAffinityConfig::new(Duration::try_from_secs_f64(a.ttl_seconds)?), epoch).map_err(anyhow::Error::from)
            }).transpose()?;
            for worker in &workers.workers {
                let request: WorkerRequest = serde_json::from_value(json!({
                    "worker_id": worker.worker_id, "model_name": "native-replay", "routing_group": role,
                    "endpoint": format!("http://simulation-worker-{}:1", worker.worker_id),
                    "block_size": workers.block_size, "data_parallel_start_rank": 0, "data_parallel_size": workers.dp_size,
                    "max_num_batched_tokens": workers.max_num_batched_tokens, "total_kv_blocks": workers.total_kv_blocks,
                }))?;
                core.upsert_worker(request).await?;
            }
            Ok::<_, anyhow::Error>((core, key, affinity, service, epoch))
        });
        let (core, key, affinity, service, epoch) = match built {
            Ok(result) => result,
            Err(error) => {
                drop(stop);
                return Err(error);
            }
        };
        Ok(Self {
            runtime,
            stop_clock_guard: Some(stop),
            core,
            _service: service,
            key,
            affinity,
            config,
            workers: ids,
            block_size: workers.block_size,
            dp_size: workers.dp_size,
            active: HashMap::new(),
            pending: VecDeque::new(),
            pending_ready: false,
            now: Duration::ZERO,
            epoch,
            now_ms: 0.0,
            role: role.into(),
            capture_decisions: workers.capture_decisions,
            decisions: Vec::new(),
            decision_count: 0,
            physical_kv_events: 0,
        })
    }

    fn advance(&mut self, now_ms: f64) -> Result<()> {
        ensure!(
            now_ms.is_finite() && now_ms >= self.now_ms,
            "native replay clock must be finite, nonnegative and monotonic"
        );
        let next =
            Duration::try_from_secs_f64(now_ms / 1000.0).context("native replay clock overflow")?;
        let instant = self
            .epoch
            .checked_add(next)
            .context("native replay clock exceeds Instant range")?;
        if let Some(affinity) = &self.affinity {
            affinity.advance_clock(instant)?;
        }
        let delta = next - self.now;
        if !delta.is_zero() {
            self.runtime.block_on(tokio::time::advance(delta));
            self.now = next;
        }
        self.now_ms = now_ms;
        Ok(())
    }

    fn group_key(&self, identity: &Identity) -> Result<Option<String>> {
        let Some(affinity) = &self.config.affinity else {
            return Ok(None);
        };
        let session = identity
            .session
            .as_deref()
            .filter(|s| !s.is_empty())
            .context("conversation affinity requires nonempty session identity")?;
        let key = match affinity.mode {
            AffinityMode::Session => json!(["session", identity.scope, session]),
            AffinityMode::SiblingGroup => {
                ensure!(
                    identity.lineage_available,
                    "sibling_group affinity requires unambiguous conversation lineage"
                );
                let root = identity
                    .root
                    .as_deref()
                    .filter(|s| !s.is_empty())
                    .context("sibling_group affinity requires nonempty root identity")?;
                if let Some(parent) = &identity.parent {
                    ensure!(
                        !parent.is_empty(),
                        "sibling_group affinity requires nonempty parent identity"
                    );
                    return Ok(Some(subagent_group_affinity_id(&serde_json::to_string(
                        &json!(["siblings", identity.scope, root, parent]),
                    )?)));
                }
                json!(["root", identity.scope, root, session])
            }
        };
        Ok(Some(subagent_group_affinity_id(&serde_json::to_string(
            &key,
        )?)))
    }

    fn select(&mut self, request: Request) -> Result<Option<Placement>> {
        ensure!(
            !request.request_id.is_empty(),
            "request ID must not be empty"
        );
        ensure!(
            request.preferred_dp_rank.is_none() && request.preferred_prefill_dp_rank.is_none(),
            "native replay policy does not support authored DP pins"
        );
        ensure!(
            request.policy_class.is_none(),
            "native replay policy does not provide custom policy classes"
        );
        ensure!(
            request.prompt_token_source == "materialized",
            "KV routing requires materialized token identities, not length-only prompt placeholders"
        );
        ensure!(
            request.priority.is_finite(),
            "request priority must be finite"
        );
        ensure!(
            request.local_block_hashes.len() == request.sequence_hashes.len(),
            "local and sequence hash counts differ"
        );
        let group_key = self.group_key(&request.identity)?;
        let hold = match (&self.affinity, &group_key) {
            (Some(table), Some(key)) => match table.try_acquire(key, None)? {
                AcquireStep::Held(hold) => Some(hold),
                AcquireStep::Wait(_) => {
                    self.pending.push_back(request);
                    return Ok(None);
                }
            },
            _ => None,
        };
        let binding_reused = hold.as_ref().and_then(Hold::target).is_some();
        let prompt = PromptRequest {
            block_hashes: Some(
                request
                    .local_block_hashes
                    .iter()
                    .map(|&v| v as i64)
                    .collect(),
            ),
            sequence_hashes: Some(request.sequence_hashes.iter().map(|&v| v as i64).collect()),
            isl_tokens: Some(request.input_tokens),
            ..Default::default()
        };
        let partition = self
            .core
            .partition(&self.key)
            .context("missing routing partition")?;
        let (selected, best_overlap) = self.runtime.block_on(async {
            let overlaps = partition
                .indexer()
                .find_matches(
                    request
                        .local_block_hashes
                        .iter()
                        .copied()
                        .map(LocalBlockHash)
                        .collect(),
                )
                .await?;
            let best_overlap = overlaps
                .scores
                .iter()
                .filter(|(w, _)| self.workers.contains(&w.worker_id))
                .map(|(_, &count)| count)
                .max()
                .unwrap_or(0);
            let outcome = self
                .core
                .run_selection(SelectionOperation {
                    key: self.key.clone(),
                    prompt: prompt.view(),
                    router_config_override: None,
                    expected_output_tokens: Some(request.output_tokens),
                    priority_jump: request.priority,
                    strict_priority: request.strict_priority,
                    policy_class: None,
                    session_context: None,
                    session: SessionBinding::None,
                    affinity_target: hold.as_ref().and_then(Hold::target),
                    pinned_worker: None,
                    allowed_worker_ids: None,
                    routing_constraints: RoutingConstraints::default(),
                    admission: SelectionAdmission::Book {
                        selection_id: request.request_id.clone(),
                    },
                    track_active_blocks: true,
                    return_routing_hashes: false,
                    replay_id: None,
                })
                .await
                .result?;
            match outcome {
                SelectionOutcome::Selected(selected) => Ok((selected, best_overlap)),
                SelectionOutcome::QueueRejected { rejection } => {
                    bail!("native Dynamo policy rejected request: {rejection:?}")
                }
            }
        })?;
        let worker = selected.response.best_worker;
        self.active.insert(
            request.request_id.clone(),
            ActiveRequest {
                hold,
                lease: None,
                target: worker.into(),
            },
        );
        self.decision_count += 1;
        if self.capture_decisions {
            self.decisions.push(json!({
                "request_id": request.request_id, "authored_request_id": request.authored_request_id,
                "session_id": request.session_id, "group_key": group_key, "role": self.role,
                "binding_reused": binding_reused,
                "worker_id": worker.worker_id, "dp_rank": worker.dp_rank, "at_ms": self.now_ms,
                "overlap_blocks": selected.response.target_cached_prefix_blocks,
                "best_available_overlap_blocks": best_overlap, "native_policy": "dynamo.SelectionCore",
            }));
        }
        Ok(Some(Placement {
            request_id: request.request_id,
            worker_id: worker.worker_id,
            dp_rank: worker.dp_rank,
            overlap_blocks: selected.response.target_cached_prefix_blocks,
            best_available_overlap_blocks: best_overlap,
            cached_tokens: selected.response.cached_tokens,
            isl_blocks: (request.input_tokens / self.block_size as usize).try_into()?,
        }))
    }

    fn release(&mut self, request_id: &str) -> Result<()> {
        if let Some(active) = self.active.remove(request_id) {
            self.runtime
                .block_on(self.core.free_reservation(request_id))?;
            drop(active);
            self.pending_ready = true;
        }
        Ok(())
    }

    fn retry_pending(&mut self) -> Result<Vec<Placement>> {
        self.pending_ready = false;
        let mut released = Vec::new();
        for _ in 0..self.pending.len() {
            let request = self
                .pending
                .pop_front()
                .expect("bounded by original pending count");
            if let Some(placement) = self.select(request)? {
                released.push(placement);
            }
        }
        Ok(released)
    }
}

impl Drop for NativeReplayPolicy {
    fn drop(&mut self) {
        self.core.shutdown();
        self.stop_clock_guard.take();
    }
}
