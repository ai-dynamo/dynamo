// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Backend admission gates: one per backend worker, shared by every backend
//! request path that serves for that worker.
//!
//! Admission happens where a backend receives its typed requests, never in the
//! generic transport. The request planes keep routing serialized requests,
//! responses, cancellation and errors to registered endpoints; a backend's own
//! engine wrapper admits each request through its worker's gate —
//! [`BackendAdmissionGate::admit`], directly or through [`AdmissionEngine`] —
//! before handing it to its engine, and the stream that comes back holds the
//! request's capacity.
//!
//! ```text
//!                                      ┌─> native backend (PreprocessedRequest) ─┐
//! TCP / NATS request plane ─> Ingress ─┤                                         ├─> worker's gate ─> engine
//!                                      └─> Python backend (PythonPayload) ───────┘
//! ```
//!
//! Each backend worker owns its gate through a [`WorkerAdmissionGate`] held by
//! the object that is the worker — the backend-common `Worker`, a native
//! `EngineConfig::InProcessTokens`, or the Python `DistributedRuntime` that
//! serves a Python backend's endpoints — and every endpoint serving for that
//! worker is built with a clone of it. Endpoints of one worker therefore draw
//! on the same capacity, while independent workers, even in one process, share
//! none; nothing ever looks a gate up. Ordinary runtime endpoints — control,
//! indexer and other RPCs, and generic Python endpoints — never reach a gate,
//! and neither do a worker's local health-check canaries, which call its engine
//! directly.
//!
//! # Engine limits
//!
//! Two independent limits govern entry to the engine, and a request enters only
//! when both have room:
//!
//! - The **engine request limit** bounds admitted requests that have not
//!   finished. A request holds request capacity from admission until its
//!   response stream ends or is dropped, however long it then runs.
//! - The **engine request waiting limit** bounds admitted requests still
//!   waiting for their engine's output. A request holds waiting capacity from
//!   admission until the gate observes its first generated token or its
//!   completion, whichever comes first, on every worker role (see
//!   [`ResponseOutput::ends_wait`]). Interim responses — metadata or control
//!   only, such as a disaggregation bootstrap announcement or a cache report —
//!   leave it waiting. The backend classifies its own responses through
//!   [`AdmissionResponse`]; the gate never parses a payload. Obtaining the
//!   stream is not a response.
//!
//! Admission reserves both together, under one lock, so no request ever holds
//! one while it waits for the other. A request whose stream never produces a
//! response that ends its wait — `generate` fails, the stream ends or fails
//! first, or the request is cancelled or dropped — releases both on the way
//! out. A request whose wait has ended, even by a completion, keeps its request
//! capacity until its stream ends. Every release hands capacity to the queues,
//! so queued work proceeds as soon as it has room.
//!
//! # DP ranks
//!
//! A worker may serve several data-parallel ranks, which its model card reports
//! as a range of global rank IDs: start rank `A` and count `N` serve
//! `A..A + N`. Every limit applies to each rank, and the worker-wide limits
//! aggregate that same capacity, `N` times each per-rank limit, without adding
//! any. A request is admitted under the rank its own decoded body names, read
//! through [`AdmissionRequest`] from the field the backend also reads to choose
//! its engine rank, when that rank is one the worker serves: it then needs room
//! under both its rank's limits and the worker-wide ones, and counts at both.
//! Any other request — naming no rank, arriving before the worker has reported
//! its ranks, or naming a rank the worker does not serve — is admitted as
//! **unspecified**: it needs room under the worker-wide limits only and counts
//! only there, sharing, not adding to, the capacity of every rank. A request
//! keeps the rank it was classified under for its whole stay, however the
//! reported ranks later change, so every release returns exactly what it took.
//!
//! # Queues
//!
//! A request that cannot enter immediately waits in a bounded FIFO queue,
//! holding no engine capacity while it does. Each served rank has its own
//! queue, bounded by the queue limit, and unspecified requests share one more;
//! all of them together are bounded by the worker-wide queue limit. A new
//! arrival is admitted directly only when its own queue is empty, so it never
//! bypasses an older request in it. Whenever capacity is released, the oldest
//! request at the head of any queue that now has all the room it needs is
//! admitted, so a rank at its own limit never blocks another queue and no
//! eligible queue starves. When its queue is full the request is rejected before
//! it reaches the engine; a queue limit of zero disables queueing. Queue
//! residence has no age limit: a request leaves the queue only by admission or
//! cancellation.
//!
//! A rejection is [`ErrorType::WorkerRankOverloaded`] with the request's rank
//! when it is limited to that rank and the worker could still admit or queue
//! other work. Otherwise it is [`ErrorType::WorkerOverloaded`]. A rank-local
//! rejection does not make routing back off from the whole worker.
//!
//! Every queue operation is O(1) in the queue's length: the queues are doubly
//! linked lists threaded through one preallocated pool of slots, and a queued
//! request keeps a handle to its own slot, so a cancellation unlinks exactly
//! that entry without searching, and its slot is reused at once.
//!
//! Choosing the next request to admit scans no queues either. The gate keeps
//! the head of every queue whose rank has room for it in a schedule ordered by
//! arrival, and a transition reschedules only the queues it changed, each at
//! O(log Q) in the Q queues there are: one per rank with queued requests, served
//! or no longer, and one for unspecified work. A head stays scheduled while only
//! the worker-wide limits are full, so whatever frees worker-wide room hands it
//! to the oldest scheduled head at once. Only a change of limits or ranks
//! reschedules every queue.
//!
//! # Configuration
//!
//! Each setting is a per-rank limit, read once when a worker's gate is created,
//! from the first of its environment names that holds a valid value:
//!
//! | Setting | Environment names, in precedence order | Automatic value |
//! |---|---|---|
//! | Engine request limit | `DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT`, `DYN_ENGINE_REQUEST_LIMIT` | `max_num_seqs` |
//! | Engine request waiting limit | `DYN_BACKEND_ADMISSION_ENGINE_REQUEST_WAITING_LIMIT`, `DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT` | `max_num_seqs` |
//! | Request queue limit | `DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT`, `DYN_DYNAMO_REQUEST_QUEUE_LIMIT` | `2 * max_num_seqs` |
//!
//! Engine limits must be positive integers, and the queue limit a non-negative
//! one. A value that is set but invalid is warned about and passed over. A
//! Python worker that accepts `--engine-request-limit` applies its value to the
//! canonical engine request limit name before the gate is created, so an
//! explicit flag takes precedence over both environment names.
//!
//! A setting without an explicit value is automatic: it follows the
//! `max_num_seqs` the engine reports through its worker's model card, and
//! assumes [`FALLBACK_MAX_NUM_SEQS`] until then. Capacity and rank reports
//! update the automatic limits and the rank count at any time, without
//! resetting live occupancy; they never replace an explicit value. A lower
//! limit evicts nothing: it holds further admission back until occupancy falls
//! below it.
//!
//! TCP requests pass through the TCP work queue and worker pool before reaching
//! this gate, then keep their worker-pool permit while waiting here. The TCP
//! pool can therefore limit how many requests reach the gate at once. Its
//! `DYN_TCP_WORKER_POOL_SIZE` / `DYN_TCP_WORK_QUEUE_SIZE` controls remain
//! separately configured from this gate; NATS has no corresponding TCP-side
//! bound.

use std::collections::{BTreeMap, HashMap};
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, OnceLock};
use std::task::Poll;

use dynamo_runtime::component::Component;
use dynamo_runtime::engine::{
    AsyncEngine, AsyncEngineContext, AsyncEngineContextProvider, AsyncEngineStream, Data,
    EngineStream,
};
use dynamo_runtime::error::{DynamoError, ErrorType};
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::pipeline::{ManyOut, ServiceEngine, SingleIn};
use dynamo_runtime::protocols::annotated::Annotated;
use futures::Stream;
use parking_lot::Mutex;
use tokio::sync::oneshot;

use crate::model_card::ModelDeploymentCard;

mod metrics;

use metrics::{AdmissionSource, BackendAdmissionMetrics, DP_RANK_UNSPECIFIED, RankSeries};

/// Maximum admitted requests that have not finished, per DP rank.
/// `--engine-request-limit` reaches the gate through this name.
const DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT: &str =
    "DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT";

/// Legacy alias of [`DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT`].
const DYN_ENGINE_REQUEST_LIMIT: &str = "DYN_ENGINE_REQUEST_LIMIT";

/// Maximum admitted requests still waiting for the engine's output — a first
/// generated token or a completion, whichever comes first — per DP rank.
const DYN_BACKEND_ADMISSION_ENGINE_REQUEST_WAITING_LIMIT: &str =
    "DYN_BACKEND_ADMISSION_ENGINE_REQUEST_WAITING_LIMIT";

/// Former name of [`DYN_BACKEND_ADMISSION_ENGINE_REQUEST_WAITING_LIMIT`], still
/// accepted as its alias.
const DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT: &str = "DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT";

/// Maximum requests waiting in each DP rank's gate queue.
const DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT: &str = "DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT";

/// Legacy alias of [`DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT`].
const DYN_DYNAMO_REQUEST_QUEUE_LIMIT: &str = "DYN_DYNAMO_REQUEST_QUEUE_LIMIT";

/// The `max_num_seqs` the automatic limits assume until the engine reports its
/// own.
const FALLBACK_MAX_NUM_SEQS: usize = 10_000;

/// How far a limit may exceed the engine's reported capacity before the gate
/// warns that it is unlikely to hold any work back.
const CAPACITY_WARNING_FACTOR: usize = 10;

/// One configuration setting: its environment names in precedence order, the
/// smallest value it accepts, and its automatic value as a multiple of the
/// engine's `max_num_seqs`.
struct Setting {
    names: &'static [&'static str],
    min: usize,
    capacity_multiple: usize,
}

const ENGINE_REQUEST_LIMIT: Setting = Setting {
    names: &[
        DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT,
        DYN_ENGINE_REQUEST_LIMIT,
    ],
    min: 1,
    capacity_multiple: 1,
};

const ENGINE_REQUEST_WAITING_LIMIT: Setting = Setting {
    names: &[
        DYN_BACKEND_ADMISSION_ENGINE_REQUEST_WAITING_LIMIT,
        DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT,
    ],
    min: 1,
    capacity_multiple: 1,
};

/// Zero is valid: it disables queueing.
const REQUEST_QUEUE_LIMIT: Setting = Setting {
    names: &[
        DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT,
        DYN_DYNAMO_REQUEST_QUEUE_LIMIT,
    ],
    min: 0,
    capacity_multiple: 2,
};

impl Setting {
    /// The value of the first name `lookup` finds valid, and that name, or
    /// `None` when the setting is left automatic.
    ///
    /// A value that is set but is not an integer of at least `min` is warned
    /// about and passed over, so it cannot hide a valid value behind it.
    fn resolve(&self, lookup: &impl Fn(&str) -> Option<String>) -> Option<Explicit> {
        for &name in self.names {
            let Some(raw) = lookup(name) else {
                continue;
            };
            match raw.trim().parse::<usize>() {
                Ok(value) if value >= self.min => return Some(Explicit { value, name }),
                _ => tracing::warn!(
                    env = name,
                    value = %raw,
                    minimum = self.min,
                    "Ignoring invalid backend admission setting; expected an integer no less \
                     than the minimum"
                ),
            }
        }
        None
    }

    /// The per-rank limit in force: the explicit value, or the automatic
    /// multiple of `max_num_seqs`.
    fn value(&self, explicit: Option<Explicit>, max_num_seqs: usize) -> usize {
        explicit.map_or_else(
            || max_num_seqs.saturating_mul(self.capacity_multiple),
            |explicit| explicit.value,
        )
    }
}

/// A setting's explicit per-rank value, and the environment name it came from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Explicit {
    value: usize,
    name: &'static str,
}

/// The per-rank values configured explicitly, each `None` when left
/// automatic. Read once, when the gate is created; no capacity report ever
/// replaces one, including a queue limit of zero.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ExplicitLimits {
    engine_request_limit: Option<Explicit>,
    engine_request_waiting_limit: Option<Explicit>,
    request_queue_limit: Option<Explicit>,
}

impl ExplicitLimits {
    fn from_environment() -> Self {
        Self::resolve(|name| std::env::var(name).ok())
    }

    /// Resolve every setting through `lookup`. Pure, so the names, their
    /// precedence and their validation are testable without touching the
    /// process environment.
    fn resolve(lookup: impl Fn(&str) -> Option<String>) -> Self {
        Self {
            engine_request_limit: ENGINE_REQUEST_LIMIT.resolve(&lookup),
            engine_request_waiting_limit: ENGINE_REQUEST_WAITING_LIMIT.resolve(&lookup),
            request_queue_limit: REQUEST_QUEUE_LIMIT.resolve(&lookup),
        }
    }
}

/// The global DP ranks a worker serves, as its model card reports
/// them: `start` through `start + count - 1`, which map to local slots `0`
/// through `count - 1`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RankRange {
    start: u32,
    /// At least one.
    count: u32,
}

impl RankRange {
    fn contains(&self, rank: u32) -> bool {
        rank.checked_sub(self.start)
            .is_some_and(|slot| slot < self.count)
    }

    fn ranks(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.count).map_while(|slot| self.start.checked_add(slot))
    }
}

impl std::fmt::Display for RankRange {
    /// As the half-open range of global rank IDs, e.g. `[4, 6)`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let end = u64::from(self.start) + u64::from(self.count);
        write!(f, "[{}, {end})", self.start)
    }
}

/// The limits the gate enforces. Each is per DP rank; its worker-wide limit is
/// the same capacity aggregated over `rank_count` ranks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct AdmissionLimits {
    /// Admitted requests that have not finished.
    engine_request_limit: usize,
    /// Admitted requests still waiting for the engine's output.
    engine_request_waiting_limit: usize,
    /// Requests waiting in a queue.
    request_queue_limit: usize,
    /// The ranks the worker serves: one until it reports them.
    rank_count: usize,
}

impl AdmissionLimits {
    /// The limits for these explicit settings, reported `max_num_seqs` and
    /// rank count. Each setting resolves independently: an explicit value wins,
    /// and an automatic one derives from `max_num_seqs` alone, never from
    /// another setting.
    fn resolve(explicit: &ExplicitLimits, max_num_seqs: Option<usize>, rank_count: usize) -> Self {
        let max_num_seqs = max_num_seqs.unwrap_or(FALLBACK_MAX_NUM_SEQS);
        Self {
            engine_request_limit: ENGINE_REQUEST_LIMIT
                .value(explicit.engine_request_limit, max_num_seqs),
            engine_request_waiting_limit: ENGINE_REQUEST_WAITING_LIMIT
                .value(explicit.engine_request_waiting_limit, max_num_seqs),
            request_queue_limit: REQUEST_QUEUE_LIMIT
                .value(explicit.request_queue_limit, max_num_seqs),
            rank_count,
        }
    }

    /// The worker-wide limit for a per-rank one.
    fn aggregate(&self, per_rank_limit: usize) -> usize {
        per_rank_limit.saturating_mul(self.rank_count)
    }
}

/// The effective configuration, captured under the lock to be logged after it.
#[derive(Clone, Copy, Debug)]
struct LimitsSnapshot {
    limits: AdmissionLimits,
    explicit: ExplicitLimits,
    max_num_seqs: Option<usize>,
    served: Option<RankRange>,
}

impl LimitsSnapshot {
    /// Where a limit's value comes from: the environment name that set it, or
    /// the capacity its automatic value follows.
    fn source(&self, explicit: Option<Explicit>) -> &'static str {
        match (explicit, self.max_num_seqs) {
            (Some(explicit), _) => explicit.name,
            (None, Some(_)) => "reported max_num_seqs",
            (None, None) => "default until max_num_seqs is reported",
        }
    }

    /// Log the limits in force for `owner`'s worker, with each one's source and
    /// scope.
    fn log(&self, owner: &GateOwner, message: &str) {
        let limits = self.limits;
        let served = self
            .served
            .map_or_else(|| "unreported".to_string(), |served| served.to_string());
        tracing::info!(
            component = %owner.component,
            worker_id = %owner.worker_id,
            engine_request_limit = limits.engine_request_limit,
            engine_request_limit_source = self.source(self.explicit.engine_request_limit),
            engine_request_waiting_limit = limits.engine_request_waiting_limit,
            engine_request_waiting_limit_source =
                self.source(self.explicit.engine_request_waiting_limit),
            request_queue_limit = limits.request_queue_limit,
            request_queue_limit_source = self.source(self.explicit.request_queue_limit),
            limit_scope = "per DP rank",
            dp_rank_count = limits.rank_count,
            dp_ranks = %served,
            worker_engine_request_limit = limits.aggregate(limits.engine_request_limit),
            worker_engine_request_waiting_limit =
                limits.aggregate(limits.engine_request_waiting_limit),
            worker_request_queue_limit = limits.aggregate(limits.request_queue_limit),
            "{message}"
        );
    }
}

/// A nonzero limit more than [`CAPACITY_WARNING_FACTOR`] times the engine's
/// reported capacity, which is unlikely to hold any work back.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct CapacityWarning {
    setting: &'static str,
    per_rank_limit: usize,
    max_num_seqs: usize,
    rank_count: usize,
}

impl CapacityWarning {
    /// Every limit in `limits` that exceeds `max_num_seqs` by more than the
    /// warning factor. A per-rank limit is compared with per-rank capacity; the
    /// worker-wide one with the worker's capacity, which is the same ratio.
    fn check(limits: &AdmissionLimits, max_num_seqs: usize) -> Vec<Self> {
        let threshold = max_num_seqs.saturating_mul(CAPACITY_WARNING_FACTOR);
        [
            (
                DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT,
                limits.engine_request_limit,
            ),
            (
                DYN_BACKEND_ADMISSION_ENGINE_REQUEST_WAITING_LIMIT,
                limits.engine_request_waiting_limit,
            ),
            (
                DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT,
                limits.request_queue_limit,
            ),
        ]
        .into_iter()
        .filter(|&(_, limit)| limit > threshold)
        .map(|(setting, per_rank_limit)| Self {
            setting,
            per_rank_limit,
            max_num_seqs,
            rank_count: limits.rank_count,
        })
        .collect()
    }

    fn log(&self, owner: &GateOwner) {
        tracing::warn!(
            component = %owner.component,
            worker_id = %owner.worker_id,
            setting = self.setting,
            per_rank_limit = self.per_rank_limit,
            per_rank_capacity = self.max_num_seqs,
            worker_limit = self.per_rank_limit.saturating_mul(self.rank_count),
            worker_capacity = self.max_num_seqs.saturating_mul(self.rank_count),
            "Backend admission limit is more than {CAPACITY_WARNING_FACTOR}x the engine's \
             reported max_num_seqs, so it is unlikely to hold work back"
        );
    }
}

/// The message a shed request is refused with when the worker can take no more
/// work.
const OVERLOADED_MESSAGE: &str = "Server overloaded: worker at capacity";

/// The message a shed request is refused with when only its DP rank is full.
const RANK_OVERLOADED_MESSAGE: &str = "Server overloaded: DP rank at capacity";

/// The message a request cancelled while queued is refused with. It is
/// deliberately not an overload: the caller went away, which is not
/// backpressure.
const CANCELLED_MESSAGE: &str = "Request cancelled while queued for backend admission";

/// One backend worker's admission gate, held by the object that is the worker
/// and shared, by cloning this handle, with every endpoint and request path
/// that serves for it.
///
/// Clones share one gate; separately created handles never do. A request
/// reaches its gate only through the handle its serving path was built with,
/// so the gate lives as long as its worker and the paths serving for it.
#[derive(Clone, Default)]
pub struct WorkerAdmissionGate(Arc<OnceLock<Arc<BackendAdmissionGate>>>);

impl WorkerAdmissionGate {
    /// The worker's gate, created by the first call — as the worker first
    /// registers or serves as a model backend — with its settings read from
    /// the environment then, once.
    ///
    /// `component` is the one the worker serves. It only labels the gate's
    /// metric family and exposes it, once, on that component's registry when
    /// the gate is created (see [`metrics`]); it never selects a gate, and a
    /// later call through any endpoint of the worker returns the same one.
    pub fn get_or_create(&self, component: &Component) -> Arc<BackendAdmissionGate> {
        Arc::clone(
            self.0
                .get_or_init(|| BackendAdmissionGate::for_component(component)),
        )
    }
}

/// A backend request, which may name the global DP rank its router selected
/// for it.
///
/// Only backend request types implement this, at the boundary where their
/// backend admits them; the generic transport never asks. The gate admits a
/// request under this rank, so it must be the rank in the request's own body —
/// the field the backend reads to choose the engine rank that serves it — set
/// by routing for each dispatch. It is read once, before admission, so it must
/// be a constant-time read of the decoded request, never a parse.
pub trait AdmissionRequest {
    /// The global DP rank this request names, or `None` when it names none.
    fn dp_rank(&self) -> Option<u32>;
}

/// What one backend response carries, as far as its request's wait for the
/// engine is concerned: the backend's own reading of its response contract.
///
/// The two facts are independent. A response may carry both — a last token
/// with its finish reason — or neither: an interim response, metadata or
/// control only, such as a disaggregation bootstrap announcement, a cache
/// report or an annotation. A completion response may report an error finish
/// reason; an annotated error frame remains interim.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResponseOutput {
    /// It carries at least one generated token.
    pub generated_token: bool,
    /// It is the backend's completion response for the request, which need
    /// not carry a token.
    pub completion: bool,
}

impl ResponseOutput {
    /// Metadata or control only.
    pub const INTERIM: Self = Self {
        generated_token: false,
        completion: false,
    };

    /// Whether a response carrying this output ends its request's wait: it
    /// carries a generated token or is the completion. The gate applies this
    /// one rule to every worker role, and reads no classification after the
    /// first response that ends the wait, so a backend classifying its
    /// responses as it produces them can stop there too.
    #[inline]
    pub fn ends_wait(self) -> bool {
        self.generated_token || self.completion
    }
}

/// A backend response type, which classifies each of its items for admission.
///
/// The backend implements this from its own response contract, at its own
/// boundary; the gate only applies [`ResponseOutput::ends_wait`] to the
/// classification and never interprets a payload itself. It is asked about
/// each item while the request still holds waiting capacity, so it must be a
/// cheap read of the item, never a parse.
pub trait AdmissionResponse {
    fn admission_output(&self) -> ResponseOutput;
}

/// Only an annotated response's data can carry output: an error frame, or an
/// event or comment without data, is interim.
impl<T: AdmissionResponse> AdmissionResponse for Annotated<T> {
    fn admission_output(&self) -> ResponseOutput {
        match &self.data {
            Some(data) if !self.is_error() => data.admission_output(),
            _ => ResponseOutput::INTERIM,
        }
    }
}

/// A backend engine whose every request is admitted through its worker's gate
/// first: the typed wrapper a native backend serves on its network ingress.
///
/// Each request is admitted under the rank it names, and its stream holds
/// waiting capacity until the first response that ends the wait. Serve only
/// the network path through it: local callers and health-check canaries use
/// the inner engine, outside admission.
pub struct AdmissionEngine<Req: Data, Resp: Data> {
    gate: Arc<BackendAdmissionGate>,
    engine: ServiceEngine<SingleIn<Req>, ManyOut<Resp>>,
}

impl<Req: Data, Resp: Data> AdmissionEngine<Req, Resp> {
    /// Admit `engine`'s network requests through `gate`, its worker's.
    pub fn new(
        gate: Arc<BackendAdmissionGate>,
        engine: ServiceEngine<SingleIn<Req>, ManyOut<Resp>>,
    ) -> Arc<Self> {
        Arc::new(Self { gate, engine })
    }
}

#[async_trait::async_trait]
impl<Req, Resp> AsyncEngine<SingleIn<Req>, ManyOut<Resp>, anyhow::Error>
    for AdmissionEngine<Req, Resp>
where
    Req: Data + AdmissionRequest,
    Resp: Data + AdmissionResponse,
{
    async fn generate(&self, request: SingleIn<Req>) -> anyhow::Result<ManyOut<Resp>> {
        let context = request.context();
        let dp_rank = AdmissionRequest::dp_rank(&*request);
        self.gate
            .admit(&context, dp_rank, self.engine.generate(request))
            .await
    }
}

/// Why the gate refused a request. Private: callers see only the typed error
/// [`reject`] builds from it, so no admission state escapes this module.
enum Rejection {
    /// The request could not be admitted and its queue had no room for it.
    /// `dp_rank` is its DP rank when the worker could still take other work
    /// (see [`GateState::refusal_rank`]), or `None` for the whole worker.
    QueueFull { dp_rank: Option<u32> },
    /// The request was cancelled before it was admitted.
    Cancelled,
}

/// Trace a refusal against the request it refused, and build the error the
/// caller fails it with.
///
/// Shedding is backpressure on this one worker, so it carries
/// [`ErrorType::WorkerOverloaded`], or [`ErrorType::WorkerRankOverloaded`] when
/// it is limited to the request's DP rank. A cancellation is not backpressure: the
/// caller went away, so classifying it as an overload would misreport a worker
/// that never refused anything.
fn reject(rejection: Rejection, context: Option<&Arc<dyn AsyncEngineContext>>) -> DynamoError {
    let request_id = context.map(|context| context.id()).unwrap_or_default();
    let (error_type, message) = match rejection {
        Rejection::QueueFull { dp_rank: None } => {
            tracing::warn!(
                request_id,
                "Worker at capacity (an engine limit and the queue are both full), rejecting \
                 request"
            );
            (ErrorType::WorkerOverloaded, OVERLOADED_MESSAGE)
        }
        Rejection::QueueFull {
            dp_rank: Some(dp_rank),
        } => {
            tracing::warn!(
                request_id,
                dp_rank,
                "DP rank at capacity while the worker can still take other work, rejecting \
                 request"
            );
            (
                ErrorType::WorkerRankOverloaded(dp_rank),
                RANK_OVERLOADED_MESSAGE,
            )
        }
        Rejection::Cancelled => {
            tracing::debug!(request_id, "{CANCELLED_MESSAGE}");
            (ErrorType::Cancelled, CANCELLED_MESSAGE)
        }
    };
    DynamoError::builder()
        .error_type(error_type)
        .message(message)
        .build()
}

/// The occupancy a request is counted under for its whole stay at the gate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum AdmissionRank {
    /// A global DP rank the worker served when the request arrived.
    Rank(u32),
    /// No usable rank: none was supplied, the worker had not reported its
    /// ranks yet, or it does not serve the supplied one.
    Unspecified,
}

impl AdmissionRank {
    /// The `dp_rank` label of this rank's metric series.
    fn label(self) -> String {
        match self {
            AdmissionRank::Rank(rank) => rank.to_string(),
            AdmissionRank::Unspecified => DP_RANK_UNSPECIFIED.to_string(),
        }
    }
}

/// Why a request that supplied a rank is admitted as unspecified.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RankFallback {
    /// The worker has not reported which ranks it serves yet.
    RanksUnreported { supplied: u32 },
    /// The worker does not serve the supplied rank.
    OutOfRange { supplied: u32, served: RankRange },
}

impl RankFallback {
    fn warn(self, request_id: &str) {
        match self {
            RankFallback::RanksUnreported { supplied } => tracing::warn!(
                request_id,
                dp_rank = supplied,
                "DP rank setup is not reported yet; admitting the request as unspecified-rank \
                 work"
            ),
            RankFallback::OutOfRange { supplied, served } => tracing::warn!(
                request_id,
                dp_rank = supplied,
                served_dp_ranks = %served,
                "Request names a DP rank this worker does not serve; admitting it as \
                 unspecified-rank work"
            ),
        }
    }
}

/// Classify a request's supplied rank against the ranks the worker serves.
///
/// A rank inside the served range is used as is. Without a supplied rank the
/// request is unspecified, with nothing to warn about. A supplied rank the gate
/// cannot use — the ranks are not reported yet, or it is outside them — is
/// also unspecified, never rejected, and reported as a fallback to warn about.
/// The request's own routing fields are untouched either way.
fn classify(
    supplied: Option<u32>,
    served: Option<RankRange>,
) -> (AdmissionRank, Option<RankFallback>) {
    match (supplied, served) {
        (None, _) => (AdmissionRank::Unspecified, None),
        (Some(supplied), None) => (
            AdmissionRank::Unspecified,
            Some(RankFallback::RanksUnreported { supplied }),
        ),
        (Some(rank), Some(served)) if served.contains(rank) => (AdmissionRank::Rank(rank), None),
        (Some(supplied), Some(served)) => (
            AdmissionRank::Unspecified,
            Some(RankFallback::OutOfRange { supplied, served }),
        ),
    }
}

/// Occupancy of one admission measure across a worker's DP ranks.
///
/// The gate keeps three: engine requests, engine-waiting requests and queued
/// requests. Each owns its measure's limits and counts and nothing else — no
/// lock, no queue, no request lifetimes. The gate checks every measure a
/// transition needs before changing any of them, under its one mutex, so the
/// three always change together.
///
/// A request with a rank counts at that rank and in the worker total; an
/// unspecified one counts only in the total, through the unspecified count. The
/// total is therefore always the ranks' counts plus the unspecified count, and
/// both kinds draw on the one aggregate limit: unspecified requests have no
/// capacity of their own.
#[derive(Debug)]
struct DpRankCounts {
    /// Each rank's limit.
    per_rank_limit: usize,
    /// The worker-wide limit: the per-rank limit over every served rank.
    aggregate_limit: usize,
    /// Each rank's count, for ranks with any; a rank that drains is dropped.
    ranks: HashMap<u32, usize>,
    /// Requests counted without a rank.
    unspecified: usize,
    /// Every request counted, with a rank or without.
    total: usize,
}

impl DpRankCounts {
    fn new(per_rank_limit: usize, rank_count: usize) -> Self {
        Self {
            per_rank_limit,
            aggregate_limit: per_rank_limit.saturating_mul(rank_count),
            ranks: HashMap::new(),
            unspecified: 0,
            total: 0,
        }
    }

    /// Replace the limits, keeping every count: a lower limit evicts nothing,
    /// it only holds further acquisitions back until the counts fall below it.
    fn set_limits(&mut self, per_rank_limit: usize, rank_count: usize) {
        self.per_rank_limit = per_rank_limit;
        self.aggregate_limit = per_rank_limit.saturating_mul(rank_count);
    }

    /// Whether one more request fits under the worker-wide limit, whatever
    /// its rank.
    fn has_aggregate_room(&self) -> bool {
        self.total < self.aggregate_limit
    }

    /// Whether one more request of `rank` fits under its rank's own limit,
    /// whatever the worker-wide one: always, for an unspecified request, which
    /// has no rank limit.
    fn has_rank_room(&self, rank: AdmissionRank) -> bool {
        match rank {
            AdmissionRank::Rank(_) => self.count(rank) < self.per_rank_limit,
            AdmissionRank::Unspecified => true,
        }
    }

    /// Whether one more request of `rank` fits: under the worker-wide limit,
    /// and under its rank's limit when it has one.
    fn has_room(&self, rank: AdmissionRank) -> bool {
        self.has_aggregate_room() && self.has_rank_room(rank)
    }

    /// Count one request of `rank`. The caller has already checked
    /// [`Self::has_room`] for everything the transition needs.
    fn acquire(&mut self, rank: AdmissionRank) {
        match rank {
            AdmissionRank::Rank(rank) => *self.ranks.entry(rank).or_default() += 1,
            AdmissionRank::Unspecified => self.unspecified += 1,
        }
        self.total += 1;
    }

    /// Stop counting one request of `rank`, which it acquired under that same
    /// rank.
    fn release(&mut self, rank: AdmissionRank) {
        match rank {
            AdmissionRank::Rank(rank) => {
                let count = self
                    .ranks
                    .get_mut(&rank)
                    .expect("released a DP rank that holds no count");
                *count -= 1;
                if *count == 0 {
                    self.ranks.remove(&rank);
                }
            }
            AdmissionRank::Unspecified => {
                self.unspecified = self
                    .unspecified
                    .checked_sub(1)
                    .expect("released an unspecified count that was never acquired");
            }
        }
        self.total -= 1;
    }

    /// The count of `rank`: one rank's, or the unspecified count.
    fn count(&self, rank: AdmissionRank) -> usize {
        match rank {
            AdmissionRank::Rank(rank) => self.ranks.get(&rank).copied().unwrap_or_default(),
            AdmissionRank::Unspecified => self.unspecified,
        }
    }

    /// Every request counted, with a rank or without.
    #[cfg(test)]
    fn total(&self) -> usize {
        self.total
    }
}

/// No slot: the end of a list.
const NIL: u32 = u32::MAX;

/// One FIFO of queued requests: its two ends in the shared [`QueueSlots`]
/// pool. Each rank with queued requests has one, and so do unspecified
/// requests.
#[derive(Debug)]
struct Fifo {
    head: u32,
    tail: u32,
    /// The arrival its head is scheduled under in [`GateState::schedule`],
    /// while it is scheduled.
    scheduled: Option<u64>,
}

impl Fifo {
    fn new() -> Self {
        Self {
            head: NIL,
            tail: NIL,
            scheduled: None,
        }
    }

    fn is_empty(&self) -> bool {
        self.head == NIL
    }
}

/// Where a queued request sits: its slot, and the generation the slot had when
/// the request took it. Every release of a slot advances its generation, so a
/// handle that outlives its entry can never reach the request that reuses the
/// slot.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct QueueHandle {
    slot: u32,
    generation: u64,
}

/// A queued request, as the gate sees it.
struct Waiter {
    /// Sending on it hands the request engine capacity already reserved for
    /// it. Dropping it unsent tells the request the gate discarded it,
    /// cancelled, at its dequeue.
    grant: oneshot::Sender<()>,
    /// Arrival order across every queue, so capacity goes to the oldest
    /// request that can use it.
    arrival: u64,
    /// The caller's context, checked once more at the dequeue decision.
    context: Option<Arc<dyn AsyncEngineContext>>,
}

/// One entry of the slot pool: either queued, and linked into its FIFO, or
/// free, and linked into the free list through `next`.
struct Slot {
    prev: u32,
    next: u32,
    generation: u64,
    waiter: Option<Waiter>,
}

/// Storage for every gate queue: one pool of slots, threaded into doubly
/// linked FIFOs and a free list.
///
/// Every operation is O(1) in the queue's length — enqueue, inspecting and
/// removing the head, and removing any entry through its handle, which unlinks
/// exactly that slot from its neighbours with no search. A removed slot is
/// reused at once, so storage stays bounded by the most requests ever queued
/// at the same time. Slots are allocated only by [`Self::reserve`], when the
/// gate is created or its queue capacity grows, never by a queue operation.
struct QueueSlots {
    slots: Vec<Slot>,
    /// The first free slot, or [`NIL`] when every slot is taken.
    free: u32,
    /// Slot reads and writes, so tests can prove an operation does constant
    /// work however long the queue is.
    #[cfg(test)]
    steps: usize,
}

impl QueueSlots {
    /// The most slots the pool can index; [`NIL`] is reserved.
    const MAX_SLOTS: usize = NIL as usize;

    fn with_capacity(capacity: usize) -> Self {
        let mut slots = Self {
            slots: Vec::new(),
            free: NIL,
            #[cfg(test)]
            steps: 0,
        };
        slots.reserve(capacity);
        slots
    }

    /// Slots in the pool, free or queued.
    #[cfg(test)]
    fn capacity(&self) -> usize {
        self.slots.len()
    }

    /// Grow the pool to hold at least `capacity` requests at once. Never
    /// shrinks it, so every live handle stays valid. If the allocation fails,
    /// the pool keeps its size and the gate rejects what it cannot queue.
    fn reserve(&mut self, capacity: usize) {
        let capacity = capacity.min(Self::MAX_SLOTS);
        let additional = capacity.saturating_sub(self.slots.len());
        if additional == 0 {
            return;
        }
        if let Err(error) = self.slots.try_reserve_exact(additional) {
            tracing::error!(
                %error,
                capacity,
                "Cannot allocate backend admission queue storage; requests beyond the existing \
                 storage will be rejected"
            );
            return;
        }
        // New slots go to the front of the free list, each pointing at the one
        // after it and the last at the old head.
        let first = self.slots.len() as u32;
        let last = (self.slots.len() + additional - 1) as u32;
        for index in first..=last {
            let next = if index == last { self.free } else { index + 1 };
            self.slots.push(Slot {
                prev: NIL,
                next,
                generation: 0,
                waiter: None,
            });
        }
        self.free = first;
    }

    fn slot(&self, index: u32) -> &Slot {
        &self.slots[index as usize]
    }

    fn slot_mut(&mut self, index: u32) -> &mut Slot {
        #[cfg(test)]
        {
            self.steps += 1;
        }
        &mut self.slots[index as usize]
    }

    /// Whether the pool has a free slot, so a request can queue.
    fn has_free_slot(&self) -> bool {
        self.free != NIL
    }

    /// Queue `waiter` at the tail of `fifo`, or hand it back when the pool is
    /// full.
    fn push_back(&mut self, fifo: &mut Fifo, waiter: Waiter) -> Result<QueueHandle, Waiter> {
        let index = self.free;
        if index == NIL {
            return Err(waiter);
        }
        let tail = fifo.tail;
        let slot = self.slot_mut(index);
        let free = slot.next;
        slot.prev = tail;
        slot.next = NIL;
        slot.waiter = Some(waiter);
        let generation = slot.generation;
        self.free = free;
        if tail == NIL {
            fifo.head = index;
        } else {
            self.slot_mut(tail).next = index;
        }
        fifo.tail = index;
        Ok(QueueHandle {
            slot: index,
            generation,
        })
    }

    /// The request at the head of `fifo`.
    fn front(&self, fifo: &Fifo) -> Option<&Waiter> {
        if fifo.is_empty() {
            return None;
        }
        self.slot(fifo.head).waiter.as_ref()
    }

    /// Dequeue the request at the head of `fifo`.
    fn pop_front(&mut self, fifo: &mut Fifo) -> Option<Waiter> {
        if fifo.is_empty() {
            return None;
        }
        Some(self.unlink(fifo, fifo.head))
    }

    /// Dequeue the request `handle` names, wherever it is in `fifo`, or `None`
    /// when the handle no longer names a queued request: it was dequeued
    /// already, and its slot may since hold another request.
    fn remove(&mut self, fifo: &mut Fifo, handle: QueueHandle) -> Option<Waiter> {
        let slot = self.slots.get(handle.slot as usize)?;
        if slot.generation != handle.generation || slot.waiter.is_none() {
            return None;
        }
        Some(self.unlink(fifo, handle.slot))
    }

    /// Unlink the queued slot `index` from `fifo` by joining its neighbours —
    /// or moving the head or tail when it is at an end — then free it under a
    /// new generation.
    fn unlink(&mut self, fifo: &mut Fifo, index: u32) -> Waiter {
        let free = self.free;
        let slot = self.slot_mut(index);
        let (prev, next) = (slot.prev, slot.next);
        let waiter = slot
            .waiter
            .take()
            .expect("unlinked a slot that holds no queued request");
        slot.generation += 1;
        slot.prev = NIL;
        slot.next = free;
        self.free = index;
        if prev == NIL {
            fifo.head = next;
        } else {
            self.slot_mut(prev).next = next;
        }
        if next == NIL {
            fifo.tail = prev;
        } else {
            self.slot_mut(next).prev = prev;
        }
        waiter
    }
}

/// The gate's state, behind its one lock.
struct GateState {
    explicit: ExplicitLimits,
    /// The engine's `max_num_seqs` per rank, once a model card reports it.
    max_num_seqs: Option<usize>,
    /// The ranks the worker serves, once a model card reports them.
    served: Option<RankRange>,
    limits: AdmissionLimits,
    /// Admitted requests that have not finished, including those whose wait
    /// for the engine's output has already ended.
    engine_requests: DpRankCounts,
    /// Admitted requests still waiting for the engine's output; each is also
    /// counted in `engine_requests`.
    engine_requests_waiting: DpRankCounts,
    /// Requests waiting in a queue; they hold neither engine count.
    queued: DpRankCounts,
    /// The queue of each rank with queued requests, and of unspecified ones.
    /// Never empty: a queue is dropped as its last request leaves.
    queues: HashMap<AdmissionRank, Fifo>,
    /// The candidates for the next admission: the head of every queue whose
    /// rank's own limits have room for it, by arrival, oldest first. The
    /// worker-wide limits are the same for every queue, so they are checked
    /// apart, and a head stays scheduled while only they are full.
    schedule: BTreeMap<u64, AdmissionRank>,
    slots: QueueSlots,
    next_arrival: u64,
    metrics: BackendAdmissionMetrics,
    /// Each rank's series, resolved once, and whether the current transition
    /// changed its occupancy.
    series: HashMap<AdmissionRank, RankMetrics>,
    /// Ranks whose occupancy the current transition changed, each once, to
    /// publish.
    touched: Vec<AdmissionRank>,
    /// The limits and capacity the capacity warning was last checked for, so
    /// a repeated report does not repeat it.
    capacity_checked: Option<(AdmissionLimits, usize)>,
    /// Queues the scheduler examined, so tests can prove a transition examines
    /// only those it changed, however many there are.
    #[cfg(test)]
    schedule_steps: usize,
    /// Ranks published, so tests can prove a transition publishes each rank it
    /// changed exactly once, and no other.
    #[cfg(test)]
    publications: usize,
}

/// One rank's metric series, resolved once so the gate never formats a label
/// on its hot path, and whether the current transition has changed the rank's
/// occupancy, so it is listed for publication once however often it changes.
struct RankMetrics {
    series: Arc<RankSeries>,
    touched: bool,
}

impl GateState {
    fn new(explicit: ExplicitLimits, metrics: BackendAdmissionMetrics) -> Self {
        let limits = AdmissionLimits::resolve(&explicit, None, 1);
        let mut state = Self {
            explicit,
            max_num_seqs: None,
            served: None,
            limits,
            engine_requests: DpRankCounts::new(limits.engine_request_limit, limits.rank_count),
            engine_requests_waiting: DpRankCounts::new(
                limits.engine_request_waiting_limit,
                limits.rank_count,
            ),
            queued: DpRankCounts::new(limits.request_queue_limit, limits.rank_count),
            queues: HashMap::new(),
            schedule: BTreeMap::new(),
            slots: QueueSlots::with_capacity(limits.aggregate(limits.request_queue_limit)),
            next_arrival: 0,
            metrics,
            series: HashMap::new(),
            touched: Vec::new(),
            capacity_checked: None,
            #[cfg(test)]
            schedule_steps: 0,
            #[cfg(test)]
            publications: 0,
        };
        // Scrapeable at zero before the first request.
        state.series(AdmissionRank::Unspecified);
        state
    }

    fn snapshot(&self) -> LimitsSnapshot {
        LimitsSnapshot {
            limits: self.limits,
            explicit: self.explicit,
            max_num_seqs: self.max_num_seqs,
            served: self.served,
        }
    }

    /// The metrics of `rank`, its series created at zero on first use.
    fn rank_metrics(&mut self, rank: AdmissionRank) -> &mut RankMetrics {
        let metrics = &self.metrics;
        self.series.entry(rank).or_insert_with(|| RankMetrics {
            series: Arc::new(metrics.series(&rank.label())),
            touched: false,
        })
    }

    /// The metric series of `rank`, created at zero on first use.
    fn series(&mut self, rank: AdmissionRank) -> Arc<RankSeries> {
        Arc::clone(&self.rank_metrics(rank).series)
    }

    /// Note that `rank`'s occupancy changed, so the transition publishes it,
    /// once however often it changes: its own flag says whether it is listed
    /// already, so noting it is O(1) however many ranks the transition changes.
    fn touch(&mut self, rank: AdmissionRank) {
        if !std::mem::replace(&mut self.rank_metrics(rank).touched, true) {
            self.touched.push(rank);
        }
    }

    /// Publish the occupancy of every rank the transition changed, from the
    /// authoritative counts, and nothing else. The scratch list keeps its
    /// capacity, so a warm gate allocates nothing here.
    fn publish(&mut self) {
        let mut touched = std::mem::take(&mut self.touched);
        for rank in touched.drain(..) {
            let engine_requests = self.engine_requests.count(rank);
            let waiting = self.engine_requests_waiting.count(rank);
            let queued = self.queued.count(rank);
            let metrics = self
                .series
                .get_mut(&rank)
                .expect("a touched rank has its series");
            metrics.touched = false;
            metrics
                .series
                .set_occupancy(engine_requests, waiting, queued);
            #[cfg(test)]
            {
                self.publications += 1;
            }
        }
        self.touched = touched;
    }

    /// Whether the worker-wide limits of both engine measures have room for
    /// one more request, whatever its rank.
    fn has_aggregate_engine_room(&self) -> bool {
        self.engine_requests.has_aggregate_room()
            && self.engine_requests_waiting.has_aggregate_room()
    }

    /// Whether `rank`'s own limits of both engine measures have room for one
    /// more of its requests: always, for unspecified work.
    fn has_rank_engine_room(&self, rank: AdmissionRank) -> bool {
        self.engine_requests.has_rank_room(rank) && self.engine_requests_waiting.has_rank_room(rank)
    }

    /// Whether one more request of `rank` can enter the engine: both engine
    /// measures need room, at its rank and worker-wide.
    fn has_engine_room(&self, rank: AdmissionRank) -> bool {
        self.has_aggregate_engine_room() && self.has_rank_engine_room(rank)
    }

    /// The DP rank to limit the refusal of a request of `rank` to, when it
    /// could neither enter the engine nor queue: its own while the worker could
    /// still take other work, or `None` to refuse it for the whole worker.
    ///
    /// Whichever limit refused this request, the worker could take other work
    /// in either of two ways. An unspecified request could enter the engine at
    /// once while both engine measures have worker-wide room, since no
    /// unspecified request is left queued then; and it could queue while the
    /// worker-wide queue limit has room and the queue storage, which every
    /// queue shares, has a free slot. Only when neither holds is the refusal
    /// the whole worker's, as an unspecified request's refusal therefore
    /// always is.
    fn refusal_rank(&self, rank: AdmissionRank) -> Option<u32> {
        let AdmissionRank::Rank(rank) = rank else {
            return None;
        };
        let can_enter = self.has_aggregate_engine_room();
        let can_queue = self.queued.has_aggregate_room() && self.slots.has_free_slot();
        (can_enter || can_queue).then_some(rank)
    }

    /// Reserve both engine measures for one request of `rank` entering the
    /// engine. Always together, so no request holds one while it waits for the
    /// other.
    fn take_engine(&mut self, rank: AdmissionRank) {
        self.engine_requests.acquire(rank);
        self.engine_requests_waiting.acquire(rank);
        self.touch(rank);
        self.reschedule(rank);
    }

    /// Bring `rank`'s queue up to date in the schedule after its head, or its
    /// rank's room, changed: its head is scheduled under its own arrival
    /// exactly while its rank's limits have room for it. A queue left empty is
    /// dropped.
    ///
    /// Examines that one queue, at O(log Q) in the Q queues scheduled, so a
    /// transition never revisits a queue it did not change. A rank with no
    /// queue, the usual case, costs one lookup.
    fn reschedule(&mut self, rank: AdmissionRank) {
        let Some(fifo) = self.queues.get_mut(&rank) else {
            return;
        };
        #[cfg(test)]
        {
            self.schedule_steps += 1;
        }
        let head = self.slots.front(fifo).map(|waiter| waiter.arrival);
        // [`Self::has_rank_engine_room`], on the counts beside the queue.
        let scheduled = head.filter(|_| {
            self.engine_requests.has_rank_room(rank)
                && self.engine_requests_waiting.has_rank_room(rank)
        });
        if fifo.scheduled != scheduled {
            if let Some(arrival) = fifo.scheduled {
                self.schedule.remove(&arrival);
            }
            if let Some(arrival) = scheduled {
                self.schedule.insert(arrival, rank);
            }
            fifo.scheduled = scheduled;
        }
        if head.is_none() {
            self.queues.remove(&rank);
        }
    }

    /// Reschedule every queue at once, after a change of limits, which may
    /// change any rank's room: served ranks', retired ranks' still holding
    /// queued requests, and unspecified work's.
    fn reschedule_all(&mut self) {
        let Self {
            queues,
            schedule,
            slots,
            engine_requests,
            engine_requests_waiting,
            ..
        } = self;
        schedule.clear();
        for (&rank, fifo) in queues.iter_mut() {
            let has_room =
                engine_requests.has_rank_room(rank) && engine_requests_waiting.has_rank_room(rank);
            fifo.scheduled = slots
                .front(fifo)
                .map(|waiter| waiter.arrival)
                .filter(|_| has_room);
            if let Some(arrival) = fifo.scheduled {
                schedule.insert(arrival, rank);
            }
        }
        #[cfg(test)]
        {
            self.schedule_steps += self.queues.len();
        }
    }

    /// Queue one request of `rank` at the tail of its rank's FIFO, counting it
    /// there. `None` when the queue storage could not be allocated.
    fn enqueue(
        &mut self,
        rank: AdmissionRank,
        context: Option<Arc<dyn AsyncEngineContext>>,
    ) -> Option<(QueueHandle, oneshot::Receiver<()>)> {
        let (grant, granted) = oneshot::channel();
        let waiter = Waiter {
            grant,
            arrival: self.next_arrival,
            context,
        };
        let fifo = self.queues.entry(rank).or_insert_with(Fifo::new);
        let handle = match self.slots.push_back(fifo, waiter) {
            Ok(handle) => handle,
            Err(_) => {
                if fifo.is_empty() {
                    self.queues.remove(&rank);
                }
                return None;
            }
        };
        self.next_arrival += 1;
        self.queued.acquire(rank);
        self.touch(rank);
        // The request heads its queue when the queue was empty.
        self.reschedule(rank);
        Some((handle, granted))
    }

    /// Dequeue the request `handle` names from `rank`'s FIFO, if it is still
    /// queued there, releasing its queue place.
    fn remove_queued(&mut self, rank: AdmissionRank, handle: QueueHandle) -> Option<Waiter> {
        let fifo = self.queues.get_mut(&rank)?;
        let waiter = self.slots.remove(fifo, handle)?;
        self.queued.release(rank);
        self.touch(rank);
        // A removed head hands the schedule to the request behind it, if any.
        self.reschedule(rank);
        Some(waiter)
    }

    /// Hand engine capacity to queued requests for as long as the worker-wide
    /// limits have room and a queue's head is scheduled, oldest arrival first.
    /// Each admission reads the schedule's oldest head and reschedules only that
    /// head's queue, so it never scans the other queues.
    ///
    /// The head is checked for cancellation once more at this dequeue
    /// decision: a request whose caller has gone is discarded without engine
    /// capacity, counted as cancelled here, and dropping its grant unsent tells
    /// it so; processing continues with the next. A grant the request then
    /// refuses is refunded by its ticket, which calls back here.
    fn admit_queued(&mut self) {
        while self.has_aggregate_engine_room() {
            let Some((_, &rank)) = self.schedule.first_key_value() else {
                break;
            };
            #[cfg(test)]
            {
                self.schedule_steps += 1;
            }
            let fifo = self
                .queues
                .get_mut(&rank)
                .expect("a scheduled queue exists");
            let waiter = self
                .slots
                .pop_front(fifo)
                .expect("a scheduled queue has a head");
            self.queued.release(rank);
            self.touch(rank);
            if waiter
                .context
                .as_ref()
                .is_some_and(|context| context.is_stopped())
            {
                self.series(rank).cancelled();
                self.reschedule(rank);
                continue;
            }
            // A successful send means the ticket is still there to receive the
            // offer, not that it will take it: a request cancelled while queued
            // may still refuse the capacity, refunding it as it settles. The
            // capacity moves here because the ticket now owns it; the admission
            // is counted only once the request actually reaches the engine.
            if waiter.grant.send(()).is_ok() {
                // Reschedules the queue for its new head and its rank's room.
                self.take_engine(rank);
            } else {
                // Its ticket was dropped without leaving the queue first, which
                // no ticket does; account for it here rather than lose it.
                self.series(rank).cancelled();
                self.reschedule(rank);
            }
        }
    }

    /// Release the waiting measure of a request of `rank` whose wait has
    /// ended, and pass the capacity on.
    fn release_waiting(&mut self, rank: AdmissionRank) {
        self.engine_requests_waiting.release(rank);
        self.touch(rank);
        self.reschedule(rank);
        self.admit_queued();
    }

    /// Release the request measure of a request of `rank` that finished,
    /// failed or went away, or of an offer a queued request refused — with its
    /// waiting measure too when it still holds that — and pass the capacity
    /// on.
    fn release_engine(&mut self, rank: AdmissionRank, holds_waiting: bool) {
        self.engine_requests.release(rank);
        if holds_waiting {
            self.engine_requests_waiting.release(rank);
        }
        self.touch(rank);
        self.reschedule(rank);
        self.admit_queued();
    }

    /// Apply a model card's report, returning what changed for logging.
    fn apply_report(
        &mut self,
        max_num_seqs: Option<usize>,
        served: Option<RankRange>,
    ) -> ReportOutcome {
        if max_num_seqs.is_some() {
            self.max_num_seqs = max_num_seqs;
        }
        if served.is_some() {
            self.served = served;
        }
        let rank_count = self.served.map_or(1, |served| served.count as usize);
        let limits = AdmissionLimits::resolve(&self.explicit, self.max_num_seqs, rank_count);
        let changed = limits != self.limits;
        if changed {
            self.set_limits(limits);
        }
        if let Some(served) = self.served {
            // Every served rank is scrapeable at zero before its first request.
            for rank in served.ranks() {
                self.series(AdmissionRank::Rank(rank));
            }
        }
        let warnings = match self.max_num_seqs {
            Some(max_num_seqs) if self.capacity_checked != Some((limits, max_num_seqs)) => {
                self.capacity_checked = Some((limits, max_num_seqs));
                CapacityWarning::check(&limits, max_num_seqs)
            }
            _ => Vec::new(),
        };
        ReportOutcome {
            changed: changed.then(|| self.snapshot()),
            warnings,
        }
    }

    /// Put new limits in force without resetting any occupancy, queue handle
    /// or permit: queue storage only grows, and a raised limit admits whatever
    /// it now fits.
    fn set_limits(&mut self, limits: AdmissionLimits) {
        self.limits = limits;
        self.engine_requests
            .set_limits(limits.engine_request_limit, limits.rank_count);
        self.engine_requests_waiting
            .set_limits(limits.engine_request_waiting_limit, limits.rank_count);
        self.queued
            .set_limits(limits.request_queue_limit, limits.rank_count);
        self.slots
            .reserve(limits.aggregate(limits.request_queue_limit));
        self.reschedule_all();
        self.admit_queued();
    }
}

/// What a model card's report changed, logged once the lock is released.
struct ReportOutcome {
    /// The new limits, when they changed.
    changed: Option<LimitsSnapshot>,
    warnings: Vec<CapacityWarning>,
}

/// Which worker a gate admits for, as its logs name it.
#[derive(Debug, Default)]
struct GateOwner {
    /// The component the worker serves, as `namespace.component`.
    component: String,
    /// The worker's runtime connection ID, in hex as its metrics label it.
    worker_id: String,
}

/// One worker's two engine limits plus bounded FIFO queues, per DP rank,
/// shared by every endpoint serving for that worker. Reached only through the
/// [`WorkerAdmissionGate`] its worker holds.
pub struct BackendAdmissionGate {
    owner: GateOwner,
    state: Mutex<GateState>,
}

impl BackendAdmissionGate {
    /// The gate of a worker serving `component`: its settings read once from
    /// the environment, and its metric family created, once, on `component`,
    /// as the component's own metrics are.
    fn for_component(component: &Component) -> Arc<Self> {
        let metrics = BackendAdmissionMetrics::new(component);
        let owner = GateOwner {
            component: component.to_string(),
            worker_id: component
                .connection_id()
                .map(|worker_id| format!("{worker_id:x}"))
                .unwrap_or_default(),
        };
        let gate = Self::with_metrics(owner, ExplicitLimits::from_environment(), metrics);
        // Logged once per worker, as the effective configuration: the limits
        // are not exported as metrics.
        gate.state
            .lock()
            .snapshot()
            .log(&gate.owner, "Backend admission gate created");
        gate
    }

    /// Build a standalone gate whose metrics are exposed nowhere, so tests
    /// never contend for any worker's capacity.
    #[cfg(test)]
    fn new(explicit: ExplicitLimits) -> Arc<Self> {
        Self::with_metrics(
            GateOwner::default(),
            explicit,
            BackendAdmissionMetrics::new(&metrics::TestHierarchy::standalone()),
        )
    }

    fn with_metrics(
        owner: GateOwner,
        explicit: ExplicitLimits,
        metrics: BackendAdmissionMetrics,
    ) -> Arc<Self> {
        Arc::new(Self {
            owner,
            state: Mutex::new(GateState::new(explicit, metrics)),
        })
    }

    /// Run `generate` under engine capacity reserved for one backend request
    /// of this gate's worker, under the DP rank it names, and return the
    /// engine's own stream, which holds waiting capacity until the first
    /// response that ends the wait (see [`ResponseOutput::ends_wait`]) and
    /// request capacity until the stream ends.
    ///
    /// This is a backend boundary's whole admission interface. Call it inside
    /// the backend's typed `generate`, with the request's own context, before
    /// the request reaches the engine: capacity is reserved *before* `generate`
    /// is polled, so a refused request never reaches the engine, and the
    /// refusal is this call's error — [`ErrorType::WorkerOverloaded`] when the
    /// gate is full, or [`ErrorType::WorkerRankOverloaded`] while the worker
    /// could still take other work — before any response stream exists. The gate watches
    /// `context` while the request is queued, and refuses it as
    /// [`ErrorType::Cancelled`] once stopped. An engine failure keeps its own
    /// error untouched.
    pub async fn admit<R, F>(
        self: &Arc<Self>,
        context: &Arc<dyn AsyncEngineContext>,
        dp_rank: Option<u32>,
        generate: F,
    ) -> anyhow::Result<EngineStream<R>>
    where
        R: Data + AdmissionResponse,
        F: Future<Output = anyhow::Result<EngineStream<R>>>,
    {
        self.admit_inner(Some(context), dp_rank, generate).await
    }

    /// Accept the capacity and DP ranks a model card this worker registered
    /// reports, once its registration has succeeded. A LoRA adapter's card
    /// carries no capacity of its own, so it changes nothing.
    pub fn record_model_card(&self, card: &ModelDeploymentCard) {
        if card.lora.is_some() {
            return;
        }
        self.record_engine_capacity(
            card.runtime_config.max_num_seqs,
            card.runtime_config.data_parallel_start_rank,
            card.runtime_config.data_parallel_size,
        );
    }

    /// Apply one state transition under the lock, then publish the occupancy
    /// it changed from the authoritative counts. Every mutation goes through
    /// here, so no transition can drift a gauge, and nothing is awaited while
    /// the lock is held.
    fn transition<T>(&self, apply: impl FnOnce(&mut GateState) -> T) -> T {
        let mut state = self.state.lock();
        let result = apply(&mut state);
        state.publish();
        result
    }

    /// Run `generate` under engine capacity reserved by both limits, for the
    /// DP rank the request names (see [`AdmissionRequest`]), if any, watching
    /// `context` when there is one: [`Self::admit`], which tests also drive
    /// without a context.
    ///
    /// The result is exactly [`AsyncEngine::generate`]'s own result type:
    /// capacity is reserved *before* `generate` is polled, so a refused request
    /// never reaches the engine, and what comes back is the engine's stream —
    /// no permit, no admission outcome, nothing to match on. The stream holds
    /// request capacity until it ends or is dropped, and waiting capacity until
    /// the first response that ends the wait; a `generate` that fails after
    /// admission releases both at once.
    ///
    /// Every request entering here is counted as received, and one that
    /// reaches `generate` as admitted, under the rank it is admitted as. A
    /// refusal is traced and classified here and surfaces as the standardized
    /// [`DynamoError`]; an engine failure keeps its own error untouched.
    async fn admit_inner<R, F>(
        self: &Arc<Self>,
        context: Option<&Arc<dyn AsyncEngineContext>>,
        dp_rank: Option<u32>,
        generate: F,
    ) -> anyhow::Result<EngineStream<R>>
    where
        R: Data + AdmissionResponse,
        F: Future<Output = anyhow::Result<EngineStream<R>>>,
    {
        let (permit, source) = self
            .acquire(context, dp_rank)
            .await
            .map_err(anyhow::Error::new)?;
        // Polling `generate` is the engine handoff, and nothing can intervene
        // between here and that first poll, so this — not reserving capacity —
        // is where the request is counted as admitted, exactly once. Whatever
        // the engine then does with it does not undo that.
        permit.series.admitted(source);
        let stream = generate.await?;
        Ok(Box::pin(AdmittedStream {
            inner: stream,
            permit: Some(permit),
        }))
    }

    /// Classify the request's rank and reserve engine capacity for it under
    /// both limits, waiting in its rank's FIFO when it cannot enter yet, and
    /// report whether it had to queue.
    ///
    /// The lock is never held across an await: the decision is made under the
    /// lock, and only the handoff is awaited.
    async fn acquire(
        self: &Arc<Self>,
        context: Option<&Arc<dyn AsyncEngineContext>>,
        dp_rank: Option<u32>,
    ) -> Result<(EnginePermit, AdmissionSource), DynamoError> {
        enum Decision {
            Cancelled,
            Admitted,
            Queued(QueueHandle, oneshot::Receiver<()>),
            Rejected(Option<u32>),
        }

        let (rank, fallback, series, decision) = self.transition(|state| {
            let (rank, fallback) = classify(dp_rank, state.served);
            let series = state.series(rank);
            series.received();
            // A caller that has already gone away must neither queue nor enter
            // the engine. Direct admission never awaits, so this is the only
            // point at which that can be caught before `generate` is polled;
            // the queued path checks again while it waits and at its dequeue.
            let decision = if context.is_some_and(|context| context.is_stopped()) {
                // Counted as a cancellation, which a rejection never is.
                series.cancelled();
                Decision::Cancelled
            } else if !state.queues.contains_key(&rank) && state.has_engine_room(rank) {
                // Direct admission only when nothing of its rank is waiting, so
                // a new request never bypasses an older one in its queue.
                state.take_engine(rank);
                Decision::Admitted
            } else if let Some((handle, granted)) = state
                .queued
                .has_room(rank)
                .then(|| state.enqueue(rank, context.cloned()))
                .flatten()
            {
                Decision::Queued(handle, granted)
            } else {
                series.rejected();
                // Scoped under the same lock, from the state that refused it.
                Decision::Rejected(state.refusal_rank(rank))
            };
            (rank, fallback, series, decision)
        });

        if let Some(fallback) = fallback {
            fallback.warn(context.map(|context| context.id()).unwrap_or_default());
        }
        match decision {
            Decision::Cancelled => Err(reject(Rejection::Cancelled, context)),
            Decision::Admitted => Ok((
                EnginePermit::new(Arc::clone(self), rank, series),
                AdmissionSource::Direct,
            )),
            Decision::Rejected(dp_rank) => Err(reject(Rejection::QueueFull { dp_rank }, context)),
            Decision::Queued(handle, granted) => {
                let permit = AdmissionTicket {
                    gate: Arc::clone(self),
                    rank,
                    series,
                    handle,
                    granted,
                    settled: false,
                }
                .wait(context)
                .await?;
                Ok((permit, AdmissionSource::Queue))
            }
        }
    }

    /// Release the waiting capacity of a request whose wait has ended, and
    /// pass it on to the queues.
    fn release_waiting(&self, rank: AdmissionRank) {
        self.transition(|state| state.release_waiting(rank));
    }

    /// Release the request capacity of a request that finished, failed or went
    /// away, or of an offer a queued request refused — with its waiting
    /// capacity too when it still holds that — and pass it on to the queues.
    fn release_request(&self, rank: AdmissionRank, holds_waiting: bool) {
        self.transition(|state| state.release_engine(rank, holds_waiting));
    }

    /// Remove an abandoned request from its queue in O(1) through its handle,
    /// freeing its queue place at once and counting its cancellation, if it is
    /// still queued. `false` when the gate dequeued it first.
    fn remove_queued(&self, rank: AdmissionRank, handle: QueueHandle, series: &RankSeries) -> bool {
        let removed = self.transition(|state| {
            let removed = state.remove_queued(rank, handle);
            if removed.is_some() {
                series.cancelled();
            }
            removed
        });
        // Its context and grant are dropped outside the lock.
        removed.is_some()
    }

    /// Accept the capacity and DP ranks this worker's engine reports:
    /// `max_num_seqs` per rank, and the ranks `data_parallel_start_rank`
    /// through `data_parallel_start_rank + data_parallel_size - 1`.
    ///
    /// Updates the automatic limits and the rank count the worker-wide limits
    /// aggregate over, whenever a report arrives or changes. Explicit settings,
    /// live occupancy, queued requests and outstanding permits are kept; a
    /// request already classified keeps its rank.
    pub fn record_engine_capacity(
        &self,
        max_num_seqs: Option<u64>,
        data_parallel_start_rank: u32,
        data_parallel_size: u32,
    ) {
        let owner = &self.owner;
        let max_num_seqs = match max_num_seqs {
            Some(0) => {
                tracing::warn!(
                    component = %owner.component,
                    worker_id = %owner.worker_id,
                    "Ignoring a reported max_num_seqs of 0 for backend admission sizing"
                );
                None
            }
            max_num_seqs => max_num_seqs.map(|value| usize::try_from(value).unwrap_or(usize::MAX)),
        };
        let served = if data_parallel_size == 0 {
            tracing::warn!(
                component = %owner.component,
                worker_id = %owner.worker_id,
                data_parallel_start_rank,
                "Ignoring a reported data_parallel_size of 0 for backend admission"
            );
            None
        } else {
            Some(RankRange {
                start: data_parallel_start_rank,
                count: data_parallel_size,
            })
        };
        let outcome = self.transition(|state| state.apply_report(max_num_seqs, served));
        if let Some(snapshot) = outcome.changed {
            snapshot.log(
                owner,
                "Backend admission limits updated from the reported engine capacity",
            );
        }
        for warning in outcome.warnings {
            warning.log(owner);
        }
    }
}

/// Engine capacity held by one admitted request: request capacity until it
/// finishes, and waiting capacity until the response that ends its wait, both
/// under the rank it was admitted as. Private, and never named in any signature
/// a caller can reach: only `admit` and the [`AdmittedStream`] it returns ever
/// own one, so no caller can hold, forget or forge one.
struct EnginePermit {
    gate: Arc<BackendAdmissionGate>,
    rank: AdmissionRank,
    series: Arc<RankSeries>,
    /// Still waiting for the engine's output, so still holding waiting
    /// capacity.
    holds_waiting: bool,
}

impl EnginePermit {
    fn new(gate: Arc<BackendAdmissionGate>, rank: AdmissionRank, series: Arc<RankSeries>) -> Self {
        Self {
            gate,
            rank,
            series,
            holds_waiting: true,
        }
    }

    /// The response that ends the request's wait: release waiting capacity,
    /// exactly once, and keep request capacity while the request continues.
    fn end_waiting(&mut self) {
        if std::mem::take(&mut self.holds_waiting) {
            self.gate.release_waiting(self.rank);
        }
    }
}

impl Drop for EnginePermit {
    fn drop(&mut self) {
        // The request finished, failed or went away: release everything it
        // still holds.
        self.gate.release_request(self.rank, self.holds_waiting);
    }
}

/// The engine's own response stream, holding the capacity it was admitted on.
///
/// It is an [`EngineStream`] like any other — item type, items and context are
/// all the engine's — so the caller sees no admission type at all.
struct AdmittedStream<R: Data> {
    inner: EngineStream<R>,
    /// Taken when the stream ends, so its request capacity is released exactly
    /// once; `Drop` covers a stream abandoned before that.
    permit: Option<EnginePermit>,
}

impl<R: Data + AdmissionResponse> Stream for AdmittedStream<R> {
    type Item = R;

    #[inline]
    fn poll_next(
        mut self: Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Option<Self::Item>> {
        let polled = self.inner.as_mut().poll_next(cx);
        match &polled {
            // A response the backend classified. While the request still
            // waits, the first one that ends the wait releases waiting capacity
            // to the queues before the item is forwarded, unchanged; an interim
            // one, and every one after the wait has ended, releases nothing.
            Poll::Ready(Some(item)) => {
                if let Some(permit) = self.permit.as_mut()
                    && permit.holds_waiting
                    && item.admission_output().ends_wait()
                {
                    permit.end_waiting();
                }
            }
            // The stream finished: release everything this request still holds,
            // including waiting capacity if no response ever ended its wait.
            Poll::Ready(None) => drop(self.permit.take()),
            Poll::Pending => {}
        }
        polled
    }
}

impl<R: Data + AdmissionResponse> AsyncEngineContextProvider for AdmittedStream<R> {
    fn context(&self) -> Arc<dyn AsyncEngineContext> {
        self.inner.context()
    }
}

impl<R: Data + AdmissionResponse> AsyncEngineStream<R> for AdmittedStream<R> {}

impl<R: Data> std::fmt::Debug for AdmittedStream<R> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AdmittedStream")
            .field("inner", &self.inner)
            .finish()
    }
}

/// A place in a FIFO. However the request leaves without becoming an
/// admission — its caller's cancellation is observed, or the whole future is
/// dropped — the ticket settles it exactly once: still queued, it is removed
/// through its handle and counted as cancelled; already offered capacity, the
/// offer is refunded and counted; already discarded at its dequeue, the gate
/// has counted it.
struct AdmissionTicket {
    gate: Arc<BackendAdmissionGate>,
    rank: AdmissionRank,
    series: Arc<RankSeries>,
    handle: QueueHandle,
    granted: oneshot::Receiver<()>,
    /// Set once the request's outcome has been accounted for — converted into
    /// an [`EnginePermit`], or settled as a cancellation — so `Drop` has
    /// nothing left to do.
    settled: bool,
}

/// How a queued request's wait ended.
enum WaitOutcome {
    /// The gate reserved engine capacity for it.
    Granted,
    /// The gate found it cancelled at its dequeue, discarded it and counted it.
    Discarded,
    /// Its caller's context stopped.
    Stopped,
}

impl AdmissionTicket {
    async fn wait(
        mut self,
        context: Option<&Arc<dyn AsyncEngineContext>>,
    ) -> Result<EnginePermit, DynamoError> {
        // Cancellation while queued must be prompt, and it must win a
        // simultaneous offer: one can be sent to this request after its context
        // stopped but before this future is polled again, and admitting then
        // would run a request the caller already abandoned. The precheck plus
        // the biased ordering give the stop strictly higher priority; an offer
        // that had already arrived is then refused in `leave`.
        let outcome = match context {
            Some(context) if context.is_stopped() => WaitOutcome::Stopped,
            Some(context) => tokio::select! {
                biased;
                _ = context.stopped() => WaitOutcome::Stopped,
                granted = &mut self.granted => match granted {
                    Ok(()) => WaitOutcome::Granted,
                    Err(_) => WaitOutcome::Discarded,
                },
            },
            None => match (&mut self.granted).await {
                Ok(()) => WaitOutcome::Granted,
                Err(_) => WaitOutcome::Discarded,
            },
        };
        match outcome {
            // The one place a queued request takes the capacity it was offered:
            // `admit` counts the admission when it hands the request to the
            // engine.
            WaitOutcome::Granted => {
                self.settled = true;
                Ok(EnginePermit::new(
                    Arc::clone(&self.gate),
                    self.rank,
                    Arc::clone(&self.series),
                ))
            }
            WaitOutcome::Discarded => {
                self.settled = true;
                Err(reject(Rejection::Cancelled, context))
            }
            // A cancellation is the caller leaving: never a rejection, and
            // never an admission either.
            WaitOutcome::Stopped => {
                self.leave();
                Err(reject(Rejection::Cancelled, context))
            }
        }
    }

    /// Settle a request leaving without admission, exactly once.
    ///
    /// Removal and dequeue both run under the gate lock, so exactly one of them
    /// takes the request off its queue. If the removal wins, it frees the queue
    /// place and counts the cancellation. Otherwise the gate already dequeued
    /// it: either it sent capacity, which closing the channel first makes this
    /// function's to take and refund — cancellation wins an offer that is
    /// already there, because honoring it would admit a caller that has gone
    /// and hold capacity ahead of live requests — or it discarded the request
    /// as cancelled and counted it, leaving nothing here.
    fn leave(&mut self) {
        self.settled = true;
        if self
            .gate
            .remove_queued(self.rank, self.handle, &self.series)
        {
            return;
        }
        self.granted.close();
        if self.granted.try_recv().is_ok() {
            self.series.cancelled();
            self.gate.release_request(self.rank, true);
        }
    }
}

impl Drop for AdmissionTicket {
    fn drop(&mut self) {
        // Reaching here unsettled means the whole future was dropped while the
        // request waited.
        if !self.settled {
            self.leave();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::{FutureExt, StreamExt};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use dynamo_runtime::pipeline::context::Controller;

    /// An explicit per-rank value, as if read from the environment.
    fn explicit(value: usize) -> Option<Explicit> {
        Some(Explicit {
            value,
            name: "test",
        })
    }

    /// A gate with these explicit per-rank limits, whose process has not
    /// reported its ranks: one rank's worth of capacity, every request
    /// unspecified.
    fn gate(
        engine_request_limit: usize,
        engine_request_waiting_limit: usize,
        request_queue_limit: usize,
    ) -> Arc<BackendAdmissionGate> {
        BackendAdmissionGate::new(ExplicitLimits {
            engine_request_limit: explicit(engine_request_limit),
            engine_request_waiting_limit: explicit(engine_request_waiting_limit),
            request_queue_limit: explicit(request_queue_limit),
        })
    }

    /// [`gate`] for a process serving ranks `start..start + count`.
    fn ranked_gate(
        limits: (usize, usize, usize),
        start: u32,
        count: u32,
    ) -> Arc<BackendAdmissionGate> {
        let gate = gate(limits.0, limits.1, limits.2);
        gate.record_engine_capacity(None, start, count);
        gate
    }

    const UNSPECIFIED: AdmissionRank = AdmissionRank::Unspecified;

    fn rank(rank: u32) -> AdmissionRank {
        AdmissionRank::Rank(rank)
    }

    /// Published process totals as (engine requests, engine-waiting requests,
    /// queued): the sum over every `dp_rank` series, which is the whole
    /// process.
    fn published(gate: &Arc<BackendAdmissionGate>) -> (i64, i64, i64) {
        let state = gate.state.lock();
        state.series.values().fold((0, 0, 0), |sum, metrics| {
            let (requests, waiting, queued) = metrics.series.published();
            (sum.0 + requests, sum.1 + waiting, sum.2 + queued)
        })
    }

    /// Published occupancy of one `dp_rank` series.
    fn published_at(gate: &Arc<BackendAdmissionGate>, at: AdmissionRank) -> (i64, i64, i64) {
        gate.state.lock().series(at).published()
    }

    /// Outcome counters of one `dp_rank` series as (received, admitted direct,
    /// admitted from the queue, rejected, cancelled).
    fn outcomes_at(
        gate: &Arc<BackendAdmissionGate>,
        at: AdmissionRank,
    ) -> (u64, u64, u64, u64, u64) {
        gate.state.lock().series(at).outcomes()
    }

    /// Outcome counters summed over every `dp_rank` series.
    fn outcomes(gate: &Arc<BackendAdmissionGate>) -> (u64, u64, u64, u64, u64) {
        let state = gate.state.lock();
        state.series.values().fold((0, 0, 0, 0, 0), |sum, metrics| {
            let o = metrics.series.outcomes();
            (
                sum.0 + o.0,
                sum.1 + o.1,
                sum.2 + o.2,
                sum.3 + o.3,
                sum.4 + o.4,
            )
        })
    }

    fn receives(gate: &Arc<BackendAdmissionGate>) -> u64 {
        outcomes(gate).0
    }

    /// Requests passed into the engine as (direct, queue).
    fn admissions(gate: &Arc<BackendAdmissionGate>) -> (u64, u64) {
        let o = outcomes(gate);
        (o.1, o.2)
    }

    fn rejections(gate: &Arc<BackendAdmissionGate>) -> u64 {
        outcomes(gate).3
    }

    fn cancellations(gate: &Arc<BackendAdmissionGate>) -> u64 {
        outcomes(gate).4
    }

    /// Requests queued process-wide.
    fn queued(gate: &Arc<BackendAdmissionGate>) -> usize {
        gate.state.lock().queued.total()
    }

    /// Authoritative occupancy of one rank as (engine requests, engine-waiting
    /// requests, queued).
    fn occupancy(gate: &Arc<BackendAdmissionGate>, at: AdmissionRank) -> (usize, usize, usize) {
        let state = gate.state.lock();
        (
            state.engine_requests.count(at),
            state.engine_requests_waiting.count(at),
            state.queued.count(at),
        )
    }

    /// The scheduling tests drive [`BackendAdmissionGate::acquire`] directly:
    /// FIFO order, refunds and both limits are decided before an engine is ever
    /// involved, and a permit is what those transitions move around. Calling
    /// [`EnginePermit::end_waiting`] and dropping the permit stand in for the
    /// response that ends a request's wait and the end of its stream. The
    /// stream-returning [`BackendAdmissionGate::admit`], which does both from
    /// the engine's own stream, is covered separately.
    type Acquired = Result<(EnginePermit, AdmissionSource), DynamoError>;

    fn permit(acquired: Acquired) -> EnginePermit {
        acquired.expect("expected admission").0
    }

    /// [`BackendAdmissionGate::acquire`], counting the admission at the
    /// handoff as [`BackendAdmissionGate::admit`] does, so the outcome counters
    /// read as they would for a real engine.
    async fn handoff(
        gate: &Arc<BackendAdmissionGate>,
        context: Option<&Arc<dyn AsyncEngineContext>>,
        dp_rank: Option<u32>,
    ) -> Acquired {
        let acquired = gate.acquire(context, dp_rank).await;
        if let Ok((permit, source)) = &acquired {
            permit.series.admitted(*source);
        }
        acquired
    }

    /// Yield, never sleep, until the queue holds `expected` requests.
    async fn await_queued(gate: &Arc<BackendAdmissionGate>, expected: usize) {
        for _ in 0..1_000 {
            if queued(gate) == expected {
                return;
            }
            tokio::task::yield_now().await;
        }
        panic!(
            "the queue holds {} requests, never {expected}",
            queued(gate)
        );
    }

    /// Spawn an acquisition for `dp_rank` — it only resolves once the gate
    /// decides, so a queueing request can never be awaited on the test task —
    /// and return once it has joined a FIFO.
    async fn spawn_queued_at(
        gate: &Arc<BackendAdmissionGate>,
        dp_rank: Option<u32>,
        context: Option<Arc<dyn AsyncEngineContext>>,
    ) -> tokio::task::JoinHandle<Acquired> {
        let before = queued(gate);
        let handle = tokio::spawn({
            let gate = Arc::clone(gate);
            async move { handoff(&gate, context.as_ref(), dp_rank).await }
        });
        await_queued(gate, before + 1).await;
        handle
    }

    /// [`spawn_queued_at`] without a rank or a context.
    async fn spawn_queued_admit(
        gate: &Arc<BackendAdmissionGate>,
    ) -> tokio::task::JoinHandle<Acquired> {
        spawn_queued_at(gate, None, None).await
    }

    /// The outcome of a spawned request, failing rather than hanging if the gate
    /// never decides it.
    async fn outcome<T>(handle: tokio::task::JoinHandle<T>) -> T {
        tokio::time::timeout(Duration::from_secs(5), handle)
            .await
            .expect("the gate must decide promptly")
            .expect("the request task completes")
    }

    /// The `(type, message)` pair a refusal carries, or `None` once admitted.
    fn refusal(acquired: &Acquired) -> Option<(ErrorType, &str)> {
        acquired
            .as_ref()
            .err()
            .map(|error| (error.error_type(), error.message()))
    }

    /// Refused as an overload of the whole worker.
    fn is_queue_full(acquired: &Acquired) -> bool {
        refusal(acquired) == Some((ErrorType::WorkerOverloaded, OVERLOADED_MESSAGE))
    }

    /// Refused as an overload of DP rank `dp_rank` alone.
    fn is_rank_full(acquired: &Acquired, dp_rank: u32) -> bool {
        refusal(acquired)
            == Some((
                ErrorType::WorkerRankOverloaded(dp_rank),
                RANK_OVERLOADED_MESSAGE,
            ))
    }

    fn is_cancelled(acquired: &Acquired) -> bool {
        refusal(acquired) == Some((ErrorType::Cancelled, CANCELLED_MESSAGE))
    }

    fn stoppable() -> (Arc<Controller>, Arc<dyn AsyncEngineContext>) {
        let controller = Arc::new(Controller::default());
        let context: Arc<dyn AsyncEngineContext> = controller.clone();
        (controller, context)
    }

    ///////////////////// CONFIGURATION /////////////////////

    /// Explicit settings resolved from exactly these environment values.
    fn explicit_from(env: &[(&str, &str)]) -> ExplicitLimits {
        ExplicitLimits::resolve(|name| {
            env.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        })
    }

    /// The explicit values alone, without the names that set them.
    fn values(explicit: ExplicitLimits) -> (Option<usize>, Option<usize>, Option<usize>) {
        let value = |explicit: Option<Explicit>| explicit.map(|explicit| explicit.value);
        (
            value(explicit.engine_request_limit),
            value(explicit.engine_request_waiting_limit),
            value(explicit.request_queue_limit),
        )
    }

    #[test]
    fn unset_settings_are_left_automatic() {
        assert_eq!(explicit_from(&[]), ExplicitLimits::default());
    }

    /// The canonical name outranks its legacy alias, and a value that is not a
    /// positive integer is passed over rather than hiding the next source. It
    /// configures the full-request limit and nothing else.
    #[test]
    fn the_engine_request_limit_reads_the_canonical_name_then_its_legacy_alias() {
        const CANONICAL: &str = DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT;
        const LEGACY: &str = DYN_ENGINE_REQUEST_LIMIT;
        for (env, expected) in [
            (vec![(CANONICAL, "7")], Some((7, CANONICAL))),
            (vec![(LEGACY, "5")], Some((5, LEGACY))),
            (vec![(CANONICAL, "7"), (LEGACY, "5")], Some((7, CANONICAL))),
            (
                vec![(CANONICAL, "7"), (LEGACY, "invalid")],
                Some((7, CANONICAL)),
            ),
            (vec![(CANONICAL, "7"), (LEGACY, "0")], Some((7, CANONICAL))),
            (vec![(CANONICAL, " 7 ")], Some((7, CANONICAL))),
            (vec![(CANONICAL, "0"), (LEGACY, "5")], Some((5, LEGACY))),
            (vec![(CANONICAL, "-1"), (LEGACY, "5")], Some((5, LEGACY))),
            (
                vec![(CANONICAL, "invalid"), (LEGACY, "5")],
                Some((5, LEGACY)),
            ),
            (vec![(CANONICAL, ""), (LEGACY, "5")], Some((5, LEGACY))),
            (vec![(CANONICAL, "0"), (LEGACY, "0")], None),
            (vec![(CANONICAL, "99999999999999999999999999999999")], None),
        ] {
            let resolved = explicit_from(&env);
            assert_eq!(
                resolved.engine_request_limit,
                expected.map(|(value, name)| Explicit { value, name }),
                "{env:?}"
            );
            assert_eq!(
                (
                    resolved.engine_request_waiting_limit,
                    resolved.request_queue_limit
                ),
                (None, None),
                "{env:?} configures the full-request limit alone"
            );
        }
    }

    /// The waiting limit's canonical name outranks the name it was renamed
    /// from, which still configures it as an alias. Neither name of the
    /// full-request limit reaches it.
    #[test]
    fn the_engine_request_waiting_limit_reads_the_canonical_name_then_its_former_name() {
        const CANONICAL: &str = DYN_BACKEND_ADMISSION_ENGINE_REQUEST_WAITING_LIMIT;
        const FORMER: &str = DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT;
        for (env, expected) in [
            (vec![(CANONICAL, "3")], Some((3, CANONICAL))),
            (vec![(CANONICAL, " 3 ")], Some((3, CANONICAL))),
            (vec![(FORMER, "4")], Some((4, FORMER))),
            (vec![(CANONICAL, "3"), (FORMER, "4")], Some((3, CANONICAL))),
            (vec![(CANONICAL, "0"), (FORMER, "4")], Some((4, FORMER))),
            (
                vec![(CANONICAL, "invalid"), (FORMER, "4")],
                Some((4, FORMER)),
            ),
            (vec![(CANONICAL, "0")], None),
            (vec![(CANONICAL, "-1")], None),
            (vec![(FORMER, "invalid")], None),
        ] {
            assert_eq!(
                explicit_from(&env).engine_request_waiting_limit,
                expected.map(|(value, name)| Explicit { value, name }),
                "{env:?}"
            );
        }
        for name in [
            DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT,
            DYN_ENGINE_REQUEST_LIMIT,
        ] {
            assert_eq!(
                explicit_from(&[(name, "5")]).engine_request_waiting_limit,
                None,
                "{name}"
            );
        }
    }

    /// Zero is a valid queue limit under either name: it disables queueing.
    #[test]
    fn the_queue_limit_accepts_zero_and_reads_the_canonical_name_then_its_legacy_alias() {
        const CANONICAL: &str = DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT;
        const LEGACY: &str = DYN_DYNAMO_REQUEST_QUEUE_LIMIT;
        for (env, expected) in [
            (vec![(CANONICAL, "17")], Some((17, CANONICAL))),
            (vec![(LEGACY, "9")], Some((9, LEGACY))),
            (
                vec![(CANONICAL, "17"), (LEGACY, "9")],
                Some((17, CANONICAL)),
            ),
            (
                vec![(CANONICAL, "17"), (LEGACY, "invalid")],
                Some((17, CANONICAL)),
            ),
            (vec![(CANONICAL, "0")], Some((0, CANONICAL))),
            (vec![(LEGACY, "0")], Some((0, LEGACY))),
            (
                vec![(CANONICAL, " 0 "), (LEGACY, "9")],
                Some((0, CANONICAL)),
            ),
            (vec![(CANONICAL, "0"), (LEGACY, "-1")], Some((0, CANONICAL))),
            (vec![(CANONICAL, "-1"), (LEGACY, "9")], Some((9, LEGACY))),
            (vec![(CANONICAL, "invalid")], None),
            (vec![(CANONICAL, "")], None),
        ] {
            assert_eq!(
                explicit_from(&env).request_queue_limit,
                expected.map(|(value, name)| Explicit { value, name }),
                "{env:?}"
            );
        }
    }

    /// A valid canonical value ends the search: the alias it shadows is never
    /// read, so an invalid alias can neither replace it nor be warned about.
    #[test]
    fn a_valid_canonical_value_never_reads_its_shadowed_alias() {
        let env = [
            (DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT, "7"),
            (DYN_ENGINE_REQUEST_LIMIT, "invalid"),
            (DYN_BACKEND_ADMISSION_ENGINE_REQUEST_WAITING_LIMIT, "3"),
            (DYN_BACKEND_ADMISSION_ENGINE_WAIT_LIMIT, "invalid"),
            (DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT, "0"),
            (DYN_DYNAMO_REQUEST_QUEUE_LIMIT, "-1"),
        ];
        let read = std::cell::RefCell::new(Vec::new());
        let resolved = ExplicitLimits::resolve(|name| {
            read.borrow_mut().push(name.to_string());
            env.iter()
                .find(|(key, _)| *key == name)
                .map(|(_, value)| value.to_string())
        });
        assert_eq!(values(resolved), (Some(7), Some(3), Some(0)));
        assert_eq!(
            *read.borrow(),
            [
                DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT,
                DYN_BACKEND_ADMISSION_ENGINE_REQUEST_WAITING_LIMIT,
                DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT,
            ]
        );
    }

    /// Before any report the gate starts at `10000` / `10000` / `20000` with
    /// one rank; a reported `max_num_seqs` `S` makes them `S` / `S` / `2S`, and
    /// every process-wide limit is the per-rank one times the reported rank
    /// count, multiplied once.
    #[test]
    fn automatic_limits_follow_the_reported_capacity_and_rank_count() {
        let automatic = ExplicitLimits::default();
        assert_eq!(
            AdmissionLimits::resolve(&automatic, None, 1),
            AdmissionLimits {
                engine_request_limit: 10_000,
                engine_request_waiting_limit: 10_000,
                request_queue_limit: 20_000,
                rank_count: 1,
            }
        );
        let reported = AdmissionLimits::resolve(&automatic, Some(256), 4);
        assert_eq!(
            reported,
            AdmissionLimits {
                engine_request_limit: 256,
                engine_request_waiting_limit: 256,
                request_queue_limit: 512,
                rank_count: 4,
            }
        );
        assert_eq!(
            (
                reported.aggregate(reported.engine_request_limit),
                reported.aggregate(reported.engine_request_waiting_limit),
                reported.aggregate(reported.request_queue_limit),
            ),
            (1_024, 1_024, 2_048)
        );

        // The live gate applies the same, from its own reports.
        let gate = BackendAdmissionGate::new(automatic);
        assert_eq!(
            gate.state.lock().limits,
            AdmissionLimits::resolve(&automatic, None, 1)
        );
        gate.record_engine_capacity(Some(256), 4, 4);
        let state = gate.state.lock();
        assert_eq!(state.limits, reported);
        assert_eq!(
            (
                state.engine_requests.aggregate_limit,
                state.engine_requests_waiting.aggregate_limit,
                state.queued.aggregate_limit,
            ),
            (1_024, 1_024, 2_048)
        );
        assert!(
            state.slots.capacity() >= 2_048,
            "queue storage holds the whole aggregate queue"
        );
    }

    /// Overriding one setting leaves the others automatic; in particular the
    /// queue's automatic value follows `max_num_seqs`, not an overridden engine
    /// limit. Explicit values stay per-rank, and a queue of zero or one deeper
    /// than twice the capacity are kept.
    #[test]
    fn each_setting_resolves_independently() {
        let engine_only = ExplicitLimits {
            engine_request_limit: explicit(5),
            ..ExplicitLimits::default()
        };
        assert_eq!(
            AdmissionLimits::resolve(&engine_only, Some(64), 2),
            AdmissionLimits {
                engine_request_limit: 5,
                engine_request_waiting_limit: 64,
                request_queue_limit: 128,
                rank_count: 2,
            }
        );
        for queue in [0, 1_000] {
            let queue_only = ExplicitLimits {
                request_queue_limit: explicit(queue),
                ..ExplicitLimits::default()
            };
            let limits = AdmissionLimits::resolve(&queue_only, Some(64), 2);
            assert_eq!(
                (
                    limits.request_queue_limit,
                    limits.aggregate(limits.request_queue_limit)
                ),
                (queue, queue * 2)
            );
            assert_eq!(limits.engine_request_limit, 64);
        }
    }

    /// Reports update the automatic limits as they arrive or change, but never
    /// replace an explicit value — a queue limit of zero included.
    #[test]
    fn reports_never_replace_explicit_settings() {
        let gate = BackendAdmissionGate::new(ExplicitLimits {
            engine_request_waiting_limit: explicit(3),
            request_queue_limit: explicit(0),
            ..ExplicitLimits::default()
        });
        for (max_num_seqs, start, count) in [(Some(8), 0, 1), (Some(16), 4, 2), (None, 4, 3)] {
            gate.record_engine_capacity(max_num_seqs, start, count);
        }
        assert_eq!(
            gate.state.lock().limits,
            AdmissionLimits {
                engine_request_limit: 16,
                engine_request_waiting_limit: 3,
                request_queue_limit: 0,
                rank_count: 3,
            },
            "a report without max_num_seqs keeps the last one"
        );
    }

    /// The warning compares each nonzero limit with the reported capacity, per
    /// rank and process-wide, and fires only beyond ten times it. Automatic
    /// limits never trigger it.
    #[test]
    fn capacity_warnings_name_every_limit_more_than_ten_times_the_capacity() {
        let limits = AdmissionLimits {
            engine_request_limit: 2_561,
            engine_request_waiting_limit: 2_560,
            request_queue_limit: 40_000,
            rank_count: 2,
        };
        assert_eq!(
            CapacityWarning::check(&limits, 256),
            [
                CapacityWarning {
                    setting: DYN_BACKEND_ADMISSION_ENGINE_REQUEST_LIMIT,
                    per_rank_limit: 2_561,
                    max_num_seqs: 256,
                    rank_count: 2,
                },
                CapacityWarning {
                    setting: DYN_BACKEND_ADMISSION_REQUEST_QUEUE_LIMIT,
                    per_rank_limit: 40_000,
                    max_num_seqs: 256,
                    rank_count: 2,
                },
            ]
        );
        let automatic = AdmissionLimits::resolve(&ExplicitLimits::default(), Some(256), 2);
        assert!(CapacityWarning::check(&automatic, 256).is_empty());
        let disabled_queue = AdmissionLimits {
            request_queue_limit: 0,
            ..automatic
        };
        assert!(CapacityWarning::check(&disabled_queue, 256).is_empty());
    }

    /// The warning is evaluated when capacity first arrives, never guessed
    /// before it, and not repeated by a report that changes nothing.
    #[test]
    fn the_capacity_warning_is_evaluated_once_per_change() {
        let gate = BackendAdmissionGate::new(ExplicitLimits {
            engine_request_limit: explicit(10_000),
            ..ExplicitLimits::default()
        });
        let report = |max_num_seqs, count| {
            gate.state
                .lock()
                .apply_report(max_num_seqs, Some(RankRange { start: 0, count }))
                .warnings
                .len()
        };
        assert_eq!(report(None, 1), 0, "no capacity, no guess");
        assert_eq!(report(Some(256), 1), 1, "evaluated when capacity arrives");
        assert_eq!(report(Some(256), 1), 0, "not repeated");
        assert_eq!(
            report(Some(256), 2),
            1,
            "re-evaluated when the limits change"
        );
        assert_eq!(report(Some(2_000), 2), 0, "and when capacity changes");
    }

    ///////////////////// RANK CLASSIFICATION /////////////////////

    /// Start rank 4 serving two ranks serves 4 and 5. Both boundaries are
    /// served; a rank below the start, at the exclusive end or beyond it falls
    /// back to unspecified with a warning naming the rank and the range. A
    /// missing rank is unspecified without a warning.
    #[test]
    fn supplied_ranks_are_used_only_inside_the_reported_range() {
        let served = RankRange { start: 4, count: 2 };
        assert_eq!(served.to_string(), "[4, 6)");
        assert_eq!(served.ranks().collect::<Vec<_>>(), [4, 5]);
        for supplied in [4, 5] {
            assert_eq!(
                classify(Some(supplied), Some(served)),
                (rank(supplied), None)
            );
        }
        for supplied in [0, 3, 6, 7, u32::MAX] {
            assert_eq!(
                classify(Some(supplied), Some(served)),
                (
                    UNSPECIFIED,
                    Some(RankFallback::OutOfRange { supplied, served })
                ),
                "{supplied}"
            );
        }
        assert_eq!(classify(None, Some(served)), (UNSPECIFIED, None));
    }

    /// Before the process reports its ranks every request is unspecified; a
    /// supplied rank — even zero — warns that rank setup is unavailable rather
    /// than establishing rank zero.
    #[test]
    fn every_request_is_unspecified_before_ranks_are_reported() {
        for supplied in [0, 3] {
            assert_eq!(
                classify(Some(supplied), None),
                (
                    UNSPECIFIED,
                    Some(RankFallback::RanksUnreported { supplied })
                )
            );
        }
        assert_eq!(classify(None, None), (UNSPECIFIED, None));
    }

    /// A single-rank process serves its one rank; any other supplied rank
    /// falls back to unspecified with a warning, and a missing rank is
    /// unspecified without one — never attributed to rank zero.
    #[test]
    fn a_single_rank_process_treats_other_ranks_as_unspecified() {
        let served = RankRange { start: 0, count: 1 };
        assert_eq!(classify(Some(0), Some(served)), (rank(0), None));
        assert_eq!(
            classify(Some(1), Some(served)),
            (
                UNSPECIFIED,
                Some(RankFallback::OutOfRange {
                    supplied: 1,
                    served
                })
            )
        );
        assert_eq!(classify(None, Some(served)), (UNSPECIFIED, None));
    }

    /// Everything the gate logs at warning level or above while `run` runs.
    fn warnings_during(run: impl FnOnce()) -> String {
        #[derive(Clone, Default)]
        struct Captured(Arc<std::sync::Mutex<Vec<u8>>>);

        impl std::io::Write for Captured {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0.lock().unwrap().extend_from_slice(buf);
                Ok(buf.len())
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }

        let captured = Captured::default();
        let writer = captured.clone();
        let subscriber = tracing_subscriber::fmt()
            .with_writer(move || writer.clone())
            .with_ansi(false)
            .with_max_level(tracing::Level::WARN)
            .finish();
        tracing::subscriber::with_default(subscriber, run);
        String::from_utf8(captured.0.lock().unwrap().clone()).unwrap()
    }

    /// A supplied rank the gate cannot use warns with the rank, and the range
    /// once it is known — before the ranks are reported, and outside them on a
    /// single-rank process — while a request without a rank warns about
    /// nothing. Neither is rejected.
    #[test]
    fn only_an_unusable_supplied_rank_warns() {
        let gate = gate(4, 4, 0);
        let admit = |dp_rank| {
            let acquired = gate.acquire(None, dp_rank).now_or_never();
            permit(acquired.expect("admitted without waiting"))
        };

        let unreported = warnings_during(|| drop(admit(Some(0))));
        assert!(unreported.contains("WARN"), "{unreported}");
        assert!(unreported.contains("dp_rank=0"), "{unreported}");
        assert!(unreported.contains("not reported"), "{unreported}");

        gate.record_engine_capacity(None, 0, 1);
        let out_of_range = warnings_during(|| drop(admit(Some(1))));
        assert!(out_of_range.contains("dp_rank=1"), "{out_of_range}");
        assert!(
            out_of_range.contains("served_dp_ranks=[0, 1)"),
            "{out_of_range}"
        );

        assert_eq!(warnings_during(|| drop(admit(None))), "");
        assert_eq!(warnings_during(|| drop(admit(Some(0)))), "");
        assert_eq!(outcomes_at(&gate, UNSPECIFIED).3, 0, "never rejected");
    }

    /// The gate classifies with the ranks reported when the request arrives,
    /// counts the fallback under `unspecified`, and admits it rather than
    /// rejecting it for its rank.
    #[tokio::test]
    async fn the_gate_admits_fallback_ranks_as_unspecified() {
        let gate = gate(4, 4, 0);
        let before_ranks = permit(handoff(&gate, None, Some(4)).await);
        assert_eq!(before_ranks.rank, UNSPECIFIED);

        gate.record_engine_capacity(None, 4, 2);
        let served = permit(handoff(&gate, None, Some(5)).await);
        let below = permit(handoff(&gate, None, Some(3)).await);
        let at_end = permit(handoff(&gate, None, Some(6)).await);
        assert_eq!(
            (served.rank, below.rank, at_end.rank),
            (rank(5), UNSPECIFIED, UNSPECIFIED)
        );
        assert_eq!(occupancy(&gate, UNSPECIFIED), (3, 3, 0));
        assert_eq!(occupancy(&gate, rank(5)), (1, 1, 0));
        assert_eq!(outcomes_at(&gate, UNSPECIFIED), (3, 3, 0, 0, 0));
        assert_eq!(outcomes_at(&gate, rank(5)), (1, 1, 0, 0, 0));
    }

    ///////////////////// DP-RANK COUNTING /////////////////////

    impl DpRankCounts {
        /// `total` is always the ranks' counts plus the unspecified count.
        fn assert_consistent(&self) {
            assert_eq!(
                self.total,
                self.ranks.values().sum::<usize>() + self.unspecified,
                "{self:?}"
            );
        }
    }

    /// A specified request counts at its rank and in the total; an unspecified
    /// one only in the total; release undoes exactly what was counted.
    #[test]
    fn specified_and_unspecified_requests_count_at_their_own_scopes() {
        let mut counts = DpRankCounts::new(2, 3);
        counts.acquire(rank(1));
        counts.acquire(UNSPECIFIED);
        assert_eq!(
            (
                counts.count(rank(1)),
                counts.count(UNSPECIFIED),
                counts.total()
            ),
            (1, 1, 2)
        );
        assert_eq!(counts.count(rank(0)), 0);
        counts.assert_consistent();

        counts.release(UNSPECIFIED);
        assert_eq!(
            (
                counts.count(rank(1)),
                counts.count(UNSPECIFIED),
                counts.total()
            ),
            (1, 0, 1)
        );
        counts.release(rank(1));
        assert_eq!(counts.total(), 0);
        assert!(counts.ranks.is_empty(), "a drained rank is dropped");
        counts.assert_consistent();
    }

    /// The plan's example: three ranks with limit 2, one request per rank, then
    /// an unspecified request and another rank-2 request give rank counts
    /// `[1, 1, 2]`, unspecified `1`, and a process count of `5` of `6`.
    #[test]
    fn three_ranks_with_limit_two_share_their_aggregate_with_unspecified_work() {
        let mut counts = DpRankCounts::new(2, 3);
        for rank_id in 0..3 {
            counts.acquire(rank(rank_id));
        }
        counts.acquire(UNSPECIFIED);
        assert!(counts.has_room(rank(2)));
        counts.acquire(rank(2));
        assert_eq!(
            (0..3).map(|r| counts.count(rank(r))).collect::<Vec<_>>(),
            [1, 1, 2]
        );
        assert_eq!(counts.count(UNSPECIFIED), 1);
        assert_eq!((counts.total(), counts.aggregate_limit), (5, 6));
        counts.assert_consistent();

        // Rank 2 is at its own limit; the others and unspecified work still fit.
        assert!(!counts.has_room(rank(2)));
        assert!(counts.has_room(rank(0)));
        assert!(counts.has_room(UNSPECIFIED));
        // One more of anything exhausts the aggregate for everyone.
        counts.acquire(UNSPECIFIED);
        assert_eq!(counts.total(), 6);
        for at in [rank(0), rank(1), rank(2), UNSPECIFIED] {
            assert!(!counts.has_room(at), "{at:?}");
        }
        counts.assert_consistent();
    }

    /// The per-rank and aggregate boundaries are exact, and unspecified work
    /// has no capacity beyond the aggregate.
    #[test]
    fn per_rank_and_aggregate_limits_bound_exactly() {
        let mut counts = DpRankCounts::new(2, 2);
        counts.acquire(rank(7));
        assert!(counts.has_room(rank(7)));
        counts.acquire(rank(7));
        assert!(!counts.has_room(rank(7)), "rank 7 at its limit");
        assert!(counts.has_room(rank(8)));
        assert!(counts.has_room(UNSPECIFIED));

        counts.acquire(UNSPECIFIED);
        counts.acquire(UNSPECIFIED);
        assert_eq!(counts.total(), 4);
        assert!(
            !counts.has_room(rank(8)),
            "unspecified work used rank 8's share"
        );
        assert!(
            !counts.has_room(UNSPECIFIED),
            "and has nothing of its own beyond it"
        );

        counts.release(rank(7));
        assert!(counts.has_room(rank(7)) && counts.has_room(rank(8)));
        counts.assert_consistent();
    }

    /// New limits keep every count; a lower one only blocks until the counts
    /// fall below it.
    #[test]
    fn changing_limits_keeps_every_count() {
        let mut counts = DpRankCounts::new(4, 2);
        for _ in 0..3 {
            counts.acquire(rank(0));
        }
        counts.acquire(UNSPECIFIED);
        counts.set_limits(1, 1);
        assert_eq!(
            (
                counts.count(rank(0)),
                counts.count(UNSPECIFIED),
                counts.total()
            ),
            (3, 1, 4)
        );
        assert!(!counts.has_room(rank(0)) && !counts.has_room(UNSPECIFIED));
        for _ in 0..3 {
            counts.release(rank(0));
        }
        assert!(
            !counts.has_room(UNSPECIFIED),
            "still at the lower aggregate"
        );
        counts.release(UNSPECIFIED);
        assert!(counts.has_room(rank(0)));
        counts.assert_consistent();
    }

    ///////////////////// QUEUE STORAGE /////////////////////

    /// A waiter whose arrival stands in for its identity.
    fn waiter(arrival: u64) -> Waiter {
        Waiter {
            grant: oneshot::channel().0,
            arrival,
            context: None,
        }
    }

    impl QueueSlots {
        /// Arrivals in `fifo` from head to tail, checking every link both ways.
        fn arrivals(&self, fifo: &Fifo) -> Vec<u64> {
            let mut arrivals = Vec::new();
            let (mut prev, mut index) = (NIL, fifo.head);
            while index != NIL {
                let slot = self.slot(index);
                assert_eq!(slot.prev, prev, "backward link of slot {index}");
                arrivals.push(
                    slot.waiter
                        .as_ref()
                        .expect("a linked slot is queued")
                        .arrival,
                );
                prev = index;
                index = slot.next;
            }
            assert_eq!(fifo.tail, prev, "tail");
            arrivals
        }

        /// Slots on the free list.
        fn free_slots(&self) -> usize {
            let (mut count, mut index) = (0, self.free);
            while index != NIL {
                assert!(self.slot(index).waiter.is_none());
                count += 1;
                index = self.slot(index).next;
            }
            count
        }
    }

    fn filled(slots: &mut QueueSlots, fifo: &mut Fifo, count: u64) -> Vec<QueueHandle> {
        (0..count)
            .map(|arrival| {
                slots
                    .push_back(fifo, waiter(arrival))
                    .unwrap_or_else(|_| panic!("room for {arrival}"))
            })
            .collect()
    }

    /// `A ↔ B ↔ C ↔ D`: removing `C` joins `B` and `D`; removing the head or
    /// tail moves that end. The survivors keep FIFO order with intact links.
    #[test]
    fn removal_unlinks_exactly_one_entry_at_the_head_middle_or_tail() {
        let mut slots = QueueSlots::with_capacity(4);
        let mut fifo = Fifo::new();
        let handles = filled(&mut slots, &mut fifo, 4);

        assert_eq!(
            slots.remove(&mut fifo, handles[2]).map(|w| w.arrival),
            Some(2)
        );
        assert_eq!(slots.arrivals(&fifo), [0, 1, 3]);
        assert_eq!(
            slots.remove(&mut fifo, handles[0]).map(|w| w.arrival),
            Some(0)
        );
        assert_eq!(slots.arrivals(&fifo), [1, 3]);
        assert_eq!(
            slots.remove(&mut fifo, handles[3]).map(|w| w.arrival),
            Some(3)
        );
        assert_eq!(slots.arrivals(&fifo), [1]);
        assert_eq!(slots.pop_front(&mut fifo).map(|w| w.arrival), Some(1));
        assert!(fifo.is_empty() && slots.arrivals(&fifo).is_empty());
        assert_eq!(slots.free_slots(), 4, "every slot is reusable");
    }

    /// A handle is good for exactly one removal: once its entry leaves, by
    /// removal or dequeue, it reaches nothing — not even the request that
    /// reuses its slot.
    #[test]
    fn a_stale_handle_never_reaches_the_request_reusing_its_slot() {
        let mut slots = QueueSlots::with_capacity(1);
        let mut fifo = Fifo::new();
        let stale = slots.push_back(&mut fifo, waiter(0)).ok().unwrap();
        assert_eq!(slots.pop_front(&mut fifo).map(|w| w.arrival), Some(0));
        let reused = slots.push_back(&mut fifo, waiter(1)).ok().unwrap();
        assert_eq!(reused.slot, stale.slot, "the slot was reused");

        assert!(slots.remove(&mut fifo, stale).is_none());
        assert_eq!(slots.arrivals(&fifo), [1], "the newer request is untouched");
        assert_eq!(slots.remove(&mut fifo, reused).map(|w| w.arrival), Some(1));
        assert!(slots.remove(&mut fifo, reused).is_none(), "and only once");
    }

    /// Several FIFOs share the pool without touching each other's links.
    #[test]
    fn fifos_sharing_the_pool_stay_independent() {
        let mut slots = QueueSlots::with_capacity(6);
        let (mut even, mut odd) = (Fifo::new(), Fifo::new());
        let mut handles = Vec::new();
        for arrival in 0..6 {
            let fifo = if arrival % 2 == 0 {
                &mut even
            } else {
                &mut odd
            };
            handles.push(slots.push_back(fifo, waiter(arrival)).ok().unwrap());
        }
        assert!(
            slots.push_back(&mut even, waiter(6)).is_err(),
            "the pool is full"
        );
        assert_eq!(
            slots.remove(&mut odd, handles[3]).map(|w| w.arrival),
            Some(3)
        );
        assert_eq!(slots.arrivals(&even), [0, 2, 4]);
        assert_eq!(slots.arrivals(&odd), [1, 5]);
        // The freed slot goes to whichever queue needs it next.
        assert!(slots.push_back(&mut even, waiter(6)).is_ok());
        assert_eq!(slots.arrivals(&even), [0, 2, 4, 6]);
    }

    /// Every operation touches at most a fixed number of slots — itself and its
    /// two neighbours — at the head, middle or tail of a queue of any length:
    /// nothing scans.
    #[test]
    fn every_queue_operation_does_constant_work_at_any_length() {
        let cost = |length: u64, position: usize| {
            let mut slots = QueueSlots::with_capacity(length as usize);
            let mut fifo = Fifo::new();
            let handles = filled(&mut slots, &mut fifo, length);
            let before = slots.steps;
            slots.remove(&mut fifo, handles[position]).expect("queued");
            let removal = slots.steps - before;
            let before = slots.steps;
            slots.pop_front(&mut fifo).expect("queued");
            let dequeue = slots.steps - before;
            let before = slots.steps;
            slots.push_back(&mut fifo, waiter(u64::MAX)).ok().unwrap();
            (removal, dequeue, slots.steps - before)
        };
        for length in [8, 1_000, 100_000] {
            for position in [0, length as usize / 2, length as usize - 1] {
                let (removal, dequeue, enqueue) = cost(length, position);
                assert!(removal <= 3, "removal at {position} of {length}: {removal}");
                assert!(dequeue <= 2, "dequeue from {length}: {dequeue}");
                assert!(enqueue <= 2, "enqueue onto {length}: {enqueue}");
            }
        }
    }

    /// Repeated cancellation and replacement reuses the same storage, at
    /// constant work per operation, and cancelling `k` entries costs O(k) in
    /// total, whatever order they go in.
    #[test]
    fn cancellation_reuses_bounded_storage_with_linear_total_work() {
        const QUEUED: u64 = 4_096;
        let mut slots = QueueSlots::with_capacity(QUEUED as usize);
        let mut fifo = Fifo::new();
        let mut handles = filled(&mut slots, &mut fifo, QUEUED);

        // Churn: cancel one entry and queue its replacement, many times over.
        let before = slots.steps;
        for (round, arrival) in (0..10_000usize).zip(QUEUED..) {
            let index = (round * 7_919) % handles.len();
            slots.remove(&mut fifo, handles[index]).expect("queued");
            handles[index] = slots
                .push_back(&mut fifo, waiter(arrival))
                .ok()
                .expect("the freed slot is reused");
        }
        assert!(
            slots.steps - before <= 10_000 * 5,
            "constant work per churn"
        );
        assert_eq!(slots.capacity(), QUEUED as usize, "storage never grew");
        assert_eq!(slots.arrivals(&fifo).len(), QUEUED as usize);

        // Mass cancellation, in an order unrelated to the queue's.
        let before = slots.steps;
        for index in (0..handles.len()).map(|i| (i * 2_053) % handles.len()) {
            slots.remove(&mut fifo, handles[index]).expect("queued");
        }
        assert!(fifo.is_empty());
        assert!(
            slots.steps - before <= 3 * QUEUED as usize,
            "linear total work: {}",
            slots.steps - before
        );
        assert_eq!(slots.free_slots(), QUEUED as usize);
    }

    /// Growing the pool keeps every live handle and the queue's order.
    #[test]
    fn growing_the_pool_preserves_live_entries() {
        let mut slots = QueueSlots::with_capacity(2);
        let mut fifo = Fifo::new();
        let handles = filled(&mut slots, &mut fifo, 2);
        slots.reserve(5);
        slots.reserve(3);
        assert_eq!(slots.capacity(), 5, "never shrinks");
        let more: Vec<_> = (2..5)
            .map(|arrival| slots.push_back(&mut fifo, waiter(arrival)).ok().unwrap())
            .collect();
        assert_eq!(slots.arrivals(&fifo), [0, 1, 2, 3, 4]);
        assert_eq!(
            slots.remove(&mut fifo, handles[1]).map(|w| w.arrival),
            Some(1)
        );
        assert_eq!(slots.remove(&mut fifo, more[1]).map(|w| w.arrival), Some(3));
        assert_eq!(slots.arrivals(&fifo), [0, 2, 4]);
    }

    ///////////////////// ENGINE LIMITS /////////////////////

    /// Request capacity is free, but the one request still waiting holds all
    /// the waiting capacity, so the next request queues until its wait ends.
    /// The request that answered keeps its request capacity.
    #[tokio::test]
    async fn the_waiting_limit_alone_holds_admission_until_a_wait_ends() {
        let gate = gate(10, 1, 4);
        let mut waiting = permit(gate.acquire(None, None).await);
        let queued = spawn_queued_admit(&gate).await;
        assert_eq!(published(&gate), (1, 1, 1));

        waiting.end_waiting();
        let (admitted, source) = outcome(queued).await.expect("admitted from the queue");
        assert_eq!(source, AdmissionSource::Queue);
        assert_eq!(
            published(&gate),
            (2, 1, 0),
            "the answering request still holds request capacity"
        );

        drop((admitted, waiting));
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// Waiting capacity is free, because the one admitted request's wait has
    /// ended, but it keeps its request capacity until it finishes.
    #[tokio::test]
    async fn the_request_limit_alone_holds_admission_until_a_request_finishes() {
        let gate = gate(1, 10, 4);
        let mut running = permit(gate.acquire(None, None).await);
        running.end_waiting();
        let queued = spawn_queued_admit(&gate).await;
        assert_eq!(published(&gate), (1, 0, 1));

        drop(running);
        let (admitted, source) = outcome(queued).await.expect("admitted from the queue");
        assert_eq!(source, AdmissionSource::Queue);
        assert_eq!(published(&gate), (1, 1, 0));

        drop(admitted);
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// With both limits exhausted, an ended wait frees waiting capacity but
    /// admits nothing while request capacity is still full.
    #[tokio::test]
    async fn an_ended_wait_admits_only_when_the_request_limit_has_room() {
        let gate = gate(2, 1, 4);
        let mut running = permit(gate.acquire(None, None).await);
        running.end_waiting();
        let mut waiting = permit(gate.acquire(None, None).await);
        let queued = spawn_queued_admit(&gate).await;
        assert_eq!(published(&gate), (2, 1, 1), "both limits are exhausted");

        waiting.end_waiting();
        assert_eq!(
            published(&gate),
            (2, 0, 1),
            "waiting capacity alone admits nothing"
        );

        drop(running);
        let (admitted, _) = outcome(queued).await.expect("admitted from the queue");
        assert_eq!(published(&gate), (2, 1, 0));

        drop((admitted, waiting));
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// The mirror case: a request that finishes frees request capacity but
    /// admits nothing while waiting capacity is still full.
    #[tokio::test]
    async fn a_completion_admits_only_when_the_waiting_limit_has_room() {
        let gate = gate(2, 1, 4);
        let mut running = permit(gate.acquire(None, None).await);
        running.end_waiting();
        let mut waiting = permit(gate.acquire(None, None).await);
        let queued = spawn_queued_admit(&gate).await;
        assert_eq!(published(&gate), (2, 1, 1), "both limits are exhausted");

        drop(running);
        assert_eq!(
            published(&gate),
            (1, 1, 1),
            "request capacity alone admits nothing"
        );

        waiting.end_waiting();
        let (admitted, _) = outcome(queued).await.expect("admitted from the queue");
        assert_eq!(published(&gate), (2, 1, 0));

        drop((admitted, waiting));
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// Waiting capacity is released once, when the wait ends, however often
    /// that is reported; request capacity is released once, at the end. A
    /// second request holds both kinds throughout, so a repeated release would
    /// show as a count below it rather than vanish at zero.
    #[tokio::test]
    async fn each_kind_of_capacity_is_released_exactly_once() {
        let gate = gate(10, 10, 0);
        let _held = permit(gate.acquire(None, None).await);
        let mut permit = permit(gate.acquire(None, None).await);
        assert_eq!(published(&gate), (2, 2, 0));

        permit.end_waiting();
        permit.end_waiting();
        assert_eq!(published(&gate), (2, 1, 0));

        drop(permit);
        assert_eq!(published(&gate), (1, 1, 0));
    }

    ///////////////////// QUEUE /////////////////////

    #[tokio::test]
    async fn zero_queue_rejects_without_capacity_and_admits_after_release() {
        let gate = gate(1, 1, 0);
        let held = permit(gate.acquire(None, None).await);
        let overflow = gate
            .acquire(None, None)
            .now_or_never()
            .expect("must not wait");
        assert!(is_queue_full(&overflow));
        assert_eq!(published(&gate), (1, 1, 0));
        assert_eq!(rejections(&gate), 1);

        drop(held);
        let next = permit(
            gate.acquire(None, None)
                .now_or_never()
                .expect("capacity is free"),
        );
        assert_eq!(published(&gate), (1, 1, 0));
        drop(next);
        assert_eq!(published(&gate), (0, 0, 0));
    }

    #[tokio::test]
    async fn n_direct_admissions_then_exactly_q_queued_then_reject() {
        let gate = gate(2, 2, 3);
        let held = vec![
            permit(gate.acquire(None, None).await),
            permit(gate.acquire(None, None).await),
        ];
        let mut waiters = Vec::new();
        for _ in 0..3 {
            waiters.push(spawn_queued_admit(&gate).await);
        }
        assert_eq!(
            published(&gate),
            (2, 2, 3),
            "queued requests hold no engine capacity"
        );

        assert!(
            is_queue_full(&gate.acquire(None, None).await),
            "the Q+1th request must be shed"
        );
        assert_eq!(queued(&gate), 3, "a rejection must not consume queue space");

        drop(held);
        for waiter in waiters {
            drop(permit(outcome(waiter).await));
        }
        assert_eq!(published(&gate), (0, 0, 0));
    }

    #[tokio::test]
    async fn queue_is_fifo_and_a_new_request_never_bypasses_an_older_waiter() {
        let gate = gate(1, 1, 8);
        let held = permit(gate.acquire(None, None).await);

        let order = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        // Each arrival joins behind the waiters already queued, so the last one
        // in must also be the last one served.
        for index in 0..5usize {
            let before = queued(&gate);
            handles.push(tokio::spawn({
                let gate = Arc::clone(&gate);
                let order = Arc::clone(&order);
                async move {
                    let permit = permit(gate.acquire(None, None).await);
                    order.lock().push(index);
                    drop(permit);
                }
            }));
            await_queued(&gate, before + 1).await;
        }

        drop(held);
        for handle in handles {
            outcome(handle).await;
        }
        assert_eq!(
            *order.lock(),
            vec![0, 1, 2, 3, 4],
            "queued requests must be admitted oldest first"
        );
        assert_eq!(published(&gate), (0, 0, 0));
    }

    ///////////////////// DP RANKS /////////////////////

    /// A rank at its own limit queues its next request but never blocks
    /// another rank, which is admitted directly while the aggregate has room.
    #[tokio::test]
    async fn a_full_rank_never_blocks_another_rank() {
        let gate = ranked_gate((1, 1, 2), 4, 2);
        let four = permit(gate.acquire(None, Some(4)).await);
        let queued_four = spawn_queued_at(&gate, Some(4), None).await;
        assert_eq!(occupancy(&gate, rank(4)), (1, 1, 1));

        let five = gate
            .acquire(None, Some(5))
            .now_or_never()
            .expect("rank 5 has room and must not wait");
        let (five, source) = five.expect("admitted");
        assert_eq!((five.rank, source), (rank(5), AdmissionSource::Direct));
        assert_eq!(occupancy(&gate, rank(5)), (1, 1, 0));

        drop(four);
        let (queued_four, source) = outcome(queued_four).await.expect("admitted");
        assert_eq!(
            (queued_four.rank, source),
            (rank(4), AdmissionSource::Queue)
        );
        drop((queued_four, five));
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// Each rank's queue is FIFO, and its queue limit is its own: a full queue
    /// on one rank rejects there, and a release on one rank admits that rank's
    /// oldest request and nothing from another rank's queue.
    #[tokio::test]
    async fn each_rank_has_its_own_fifo_and_queue_limit() {
        let gate = ranked_gate((1, 1, 2), 0, 2);
        let zero = permit(gate.acquire(None, Some(0)).await);
        let one = permit(gate.acquire(None, Some(1)).await);
        let zero_a = spawn_queued_at(&gate, Some(0), None).await;
        let one_a = spawn_queued_at(&gate, Some(1), None).await;
        let zero_b = spawn_queued_at(&gate, Some(0), None).await;
        let one_b = spawn_queued_at(&gate, Some(1), None).await;
        assert_eq!(occupancy(&gate, rank(0)), (1, 1, 2));
        assert_eq!(occupancy(&gate, rank(1)), (1, 1, 2));
        assert!(
            is_queue_full(&gate.acquire(None, Some(0)).await),
            "rank 0's queue is full"
        );
        assert_eq!(outcomes_at(&gate, rank(0)).3, 1, "rejected under rank 0");
        assert_eq!(outcomes_at(&gate, rank(1)).3, 0);

        // Releasing rank 0 admits rank 0's oldest request, and only that one.
        drop(zero);
        let zero_a = permit(outcome(zero_a).await);
        assert_eq!(zero_a.rank, rank(0));
        assert_eq!(occupancy(&gate, rank(0)), (1, 1, 1), "0b still waits");
        assert_eq!(occupancy(&gate, rank(1)), (1, 1, 2), "rank 1 is untouched");

        drop(one);
        let one_a = permit(outcome(one_a).await);
        assert_eq!(one_a.rank, rank(1));
        assert_eq!(occupancy(&gate, rank(1)), (1, 1, 1), "1b still waits");

        drop(zero_a);
        assert_eq!(permit(outcome(zero_b).await).rank, rank(0));
        drop(one_a);
        assert_eq!(permit(outcome(one_b).await).rank, rank(1));
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// Unspecified work shares the aggregate with every rank and has no
    /// capacity of its own, so it interferes both ways: it can use a rank's
    /// unused share, and a rank can use what unspecified work leaves.
    #[tokio::test]
    async fn unspecified_and_ranked_work_share_the_aggregate_both_ways() {
        let gate = ranked_gate((2, 2, 0), 0, 2);
        // Unspecified work fills the aggregate of 4 by itself.
        let unspecified: Vec<_> =
            futures::future::join_all((0..4).map(|_| gate.acquire(None, None)))
                .await
                .into_iter()
                .map(permit)
                .collect();
        assert_eq!(occupancy(&gate, UNSPECIFIED), (4, 4, 0));
        assert!(
            is_queue_full(&gate.acquire(None, Some(0)).await),
            "rank 0 is empty, but unspecified work used its share"
        );
        assert!(is_queue_full(&gate.acquire(None, None).await));

        // Freed aggregate capacity goes to ranked work, up to the rank's own limit.
        drop(unspecified);
        let ranked: Vec<_> = futures::future::join_all((0..2).map(|_| gate.acquire(None, Some(1))))
            .await
            .into_iter()
            .map(permit)
            .collect();
        assert!(
            is_rank_full(&gate.acquire(None, Some(1)).await, 1),
            "rank 1 is at its own limit, but the aggregate has room for other work"
        );
        let more_unspecified: Vec<_> =
            futures::future::join_all((0..2).map(|_| gate.acquire(None, None)))
                .await
                .into_iter()
                .map(permit)
                .collect();
        assert!(
            is_queue_full(&gate.acquire(None, Some(0)).await),
            "rank 1 and unspecified work together fill the aggregate"
        );
        assert_eq!(gate.state.lock().engine_requests.total(), 4);
        drop((ranked, more_unspecified));
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// Unspecified requests queue in their own FIFO against the aggregate queue
    /// limit only, consuming the same capacity as every rank's queue.
    #[tokio::test]
    async fn unspecified_requests_queue_against_the_aggregate_queue_limit() {
        let gate = ranked_gate((1, 1, 1), 0, 2);
        let held: Vec<_> =
            futures::future::join_all([Some(0), Some(1)].map(|r| gate.acquire(None, r)))
                .await
                .into_iter()
                .map(permit)
                .collect();
        // Aggregate queue limit 2: two unspecified requests fill it.
        let first = spawn_queued_at(&gate, None, None).await;
        let second = spawn_queued_at(&gate, None, None).await;
        assert_eq!(occupancy(&gate, UNSPECIFIED).2, 2);
        assert!(
            is_queue_full(&gate.acquire(None, Some(0)).await),
            "rank 0's queue is empty, but the aggregate queue is full"
        );
        drop(held);
        for waiter in [first, second] {
            assert_eq!(permit(outcome(waiter).await).rank, UNSPECIFIED);
        }
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// A refusal names its rank exactly while the worker could still take
    /// other work, by either path: entering the engine, or queueing. Once it
    /// could do neither, every refusal is the whole worker's, as an
    /// unspecified request's always is — including one naming a rank the
    /// worker does not serve.
    #[tokio::test]
    async fn a_refusal_names_its_rank_only_while_the_worker_can_take_other_work() {
        // Without queues the aggregate queue is always full, yet rank 5 can
        // still enter the engine while rank 4 is at its own limit.
        let gate = ranked_gate((1, 1, 0), 4, 2);
        let four = permit(gate.acquire(None, Some(4)).await);
        assert!(is_rank_full(&gate.acquire(None, Some(4)).await, 4));
        let five = permit(
            gate.acquire(None, Some(5))
                .now_or_never()
                .expect("rank 5 enters"),
        );
        for dp_rank in [Some(4), Some(5), None] {
            assert!(
                is_queue_full(&gate.acquire(None, dp_rank).await),
                "{dp_rank:?}"
            );
        }
        drop((four, five));

        let gate = ranked_gate((1, 1, 1), 4, 2);
        let four = permit(gate.acquire(None, Some(4)).await);
        let queued_four = spawn_queued_at(&gate, Some(4), None).await;
        assert!(is_rank_full(&gate.acquire(None, Some(4)).await, 4));
        let five = permit(
            gate.acquire(None, Some(5))
                .now_or_never()
                .expect("rank 5 enters"),
        );
        // The aggregate engine limits are full now, but other work can queue.
        assert!(is_rank_full(&gate.acquire(None, Some(4)).await, 4));
        let queued_five = spawn_queued_at(&gate, Some(5), None).await;
        for dp_rank in [Some(4), Some(5), None, Some(6)] {
            assert!(
                is_queue_full(&gate.acquire(None, dp_rank).await),
                "{dp_rank:?}"
            );
        }
        assert_eq!(
            outcomes_at(&gate, rank(4)).3,
            3,
            "each counted under rank 4"
        );
        assert_eq!(outcomes_at(&gate, rank(5)).3, 1);
        assert_eq!(outcomes_at(&gate, UNSPECIFIED).3, 2);

        drop(four);
        let queued_four = permit(outcome(queued_four).await);
        drop(five);
        drop((queued_four, permit(outcome(queued_five).await)));
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// Queue storage that could not grow refuses requests the queue limits
    /// would still take, and blocks queueing for every rank alike, not just
    /// the refused request's: such a refusal names its rank only while other
    /// work can still enter the engine.
    #[tokio::test]
    async fn a_refusal_for_want_of_queue_storage_names_its_rank_only_while_other_work_can_enter() {
        let gate = ranked_gate((1, 1, 2), 4, 2);
        // As if growing it had failed: one slot, under a worker-wide queue
        // limit of four.
        gate.state.lock().slots = QueueSlots::with_capacity(1);
        let four = permit(gate.acquire(None, Some(4)).await);
        let queued_four = spawn_queued_at(&gate, Some(4), None).await;
        assert!(is_rank_full(&gate.acquire(None, Some(4)).await, 4));
        let five = permit(
            gate.acquire(None, Some(5))
                .now_or_never()
                .expect("rank 5 enters"),
        );
        for dp_rank in [Some(5), None] {
            assert!(
                is_queue_full(&gate.acquire(None, dp_rank).await),
                "{dp_rank:?}"
            );
        }
        assert_eq!(queued(&gate), 1);

        drop(four);
        let queued_four = permit(outcome(queued_four).await);
        drop((five, queued_four));
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// Released capacity goes to the oldest request that can use it, across
    /// queues: an older unspecified request goes before a younger ranked one.
    #[tokio::test]
    async fn released_capacity_goes_to_the_oldest_eligible_request_across_queues() {
        let gate = ranked_gate((2, 2, 4), 0, 1);
        let held: Vec<_> = futures::future::join_all((0..2).map(|_| gate.acquire(None, Some(0))))
            .await
            .into_iter()
            .map(permit)
            .collect();
        let older_unspecified = spawn_queued_at(&gate, None, None).await;
        let younger_ranked = spawn_queued_at(&gate, Some(0), None).await;

        let mut held = held.into_iter();
        drop(held.next());
        let admitted = permit(outcome(older_unspecified).await);
        assert_eq!(admitted.rank, UNSPECIFIED);
        assert_eq!(
            occupancy(&gate, rank(0)).2,
            1,
            "the younger request still waits"
        );
        drop(admitted);
        assert_eq!(permit(outcome(younger_ranked).await).rank, rank(0));
    }

    /// For each occupancy gauge, every served rank and `unspecified` have their
    /// own series, and the series sum to the process occupancy without double
    /// counting.
    #[tokio::test]
    async fn each_gauge_has_a_series_per_served_rank_and_unspecified() {
        let gate = ranked_gate((1, 1, 2), 4, 2);
        let four = permit(gate.acquire(None, Some(4)).await);
        let mut five = permit(gate.acquire(None, Some(5)).await);
        five.end_waiting();
        let queued_four = spawn_queued_at(&gate, Some(4), None).await;
        let unspecified_queued = spawn_queued_at(&gate, None, None).await;

        assert_eq!(published_at(&gate, rank(4)), (1, 1, 1));
        assert_eq!(published_at(&gate, rank(5)), (1, 0, 0));
        assert_eq!(published_at(&gate, UNSPECIFIED), (0, 0, 1));
        let state_totals = {
            let state = gate.state.lock();
            (
                state.engine_requests.total() as i64,
                state.engine_requests_waiting.total() as i64,
                state.queued.total() as i64,
            )
        };
        assert_eq!(published(&gate), state_totals);
        assert_eq!(published(&gate), (2, 1, 2));
        let labels: std::collections::BTreeSet<String> = gate
            .state
            .lock()
            .series
            .keys()
            .map(|rank| rank.label())
            .collect();
        assert_eq!(
            labels,
            ["4", "5", "unspecified"].map(String::from).into(),
            "served ranks and unspecified, and no aggregate series"
        );

        drop((four, five));
        drop(permit(outcome(queued_four).await));
        drop(permit(outcome(unspecified_queued).await));
        assert_eq!(published(&gate), (0, 0, 0));
        for at in [rank(4), rank(5), UNSPECIFIED] {
            assert_eq!(published_at(&gate, at), (0, 0, 0), "{at:?}");
        }
    }

    ///////////////////// REPORTS WITH LIVE REQUESTS /////////////////////

    /// Unspecified requests admitted or queued before ranks are reported keep
    /// their unspecified accounting and label once ranks arrive, and release
    /// exactly what they took.
    #[tokio::test]
    async fn requests_keep_their_classification_when_ranks_arrive() {
        let gate = gate(1, 1, 2);
        let early = permit(gate.acquire(None, Some(0)).await);
        let early_queued = spawn_queued_at(&gate, Some(0), None).await;
        assert_eq!(occupancy(&gate, UNSPECIFIED), (1, 1, 1));

        // Two ranks double the aggregate, which admits the queued request at
        // once — still as unspecified work, not reassigned to rank 0.
        gate.record_engine_capacity(Some(1), 0, 2);
        let early_queued = permit(outcome(early_queued).await);
        assert_eq!(early_queued.rank, UNSPECIFIED);
        assert_eq!(occupancy(&gate, UNSPECIFIED), (2, 2, 0));
        assert_eq!(occupancy(&gate, rank(0)), (0, 0, 0));
        assert_eq!(published_at(&gate, UNSPECIFIED), (2, 2, 0));
        assert_eq!(published_at(&gate, rank(0)), (0, 0, 0));

        // Unspecified work now fills the aggregate, so ranked work waits for
        // it, and enters once unspecified work releases exactly what it held.
        let ranked = spawn_queued_at(&gate, Some(0), None).await;
        drop(early);
        let ranked = permit(outcome(ranked).await);
        assert_eq!(ranked.rank, rank(0));
        assert_eq!(occupancy(&gate, UNSPECIFIED), (1, 1, 0));
        assert_eq!(occupancy(&gate, rank(0)), (1, 1, 0));
        drop((early_queued, ranked));
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// A late capacity report raises automatic limits and admits queued work
    /// at once; a lower one evicts nothing and admits again only once
    /// occupancy falls below it. Queue handles and permits survive both.
    #[tokio::test]
    async fn capacity_changes_never_disturb_live_requests() {
        let gate = BackendAdmissionGate::new(ExplicitLimits::default());
        gate.record_engine_capacity(Some(1), 0, 1);
        let held = permit(gate.acquire(None, Some(0)).await);
        let first = spawn_queued_at(&gate, Some(0), None).await;
        let second = spawn_queued_at(&gate, Some(0), None).await;
        assert_eq!(occupancy(&gate, rank(0)), (1, 1, 2));

        // Capacity increase: both queued requests now fit.
        gate.record_engine_capacity(Some(3), 0, 1);
        let first = permit(outcome(first).await);
        let second = permit(outcome(second).await);
        assert_eq!(occupancy(&gate, rank(0)), (3, 3, 0));

        // Capacity decrease below occupancy: nothing is evicted, and a new
        // request queues until occupancy is below the new limit.
        gate.record_engine_capacity(Some(1), 0, 1);
        assert_eq!(occupancy(&gate, rank(0)), (3, 3, 0));
        let third = spawn_queued_at(&gate, Some(0), None).await;
        drop(held);
        drop(first);
        assert_eq!(
            occupancy(&gate, rank(0)),
            (1, 1, 1),
            "still at the lower limit"
        );
        drop(second);
        let third = permit(outcome(third).await);
        assert_eq!(occupancy(&gate, rank(0)), (1, 1, 0));
        drop(third);
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// Changing the rank count rescales the aggregate once; a rank no longer
    /// served keeps its live requests, admitted and queued, under its own label
    /// until they finish, while new requests naming it fall back to
    /// unspecified.
    #[tokio::test]
    async fn rank_changes_keep_requests_under_the_rank_they_were_admitted_as() {
        let gate = ranked_gate((1, 1, 1), 0, 3);
        assert_eq!(gate.state.lock().engine_requests.aggregate_limit, 3);
        let two = permit(gate.acquire(None, Some(2)).await);
        let queued_two = spawn_queued_at(&gate, Some(2), None).await;

        gate.record_engine_capacity(None, 0, 2);
        assert_eq!(gate.state.lock().engine_requests.aggregate_limit, 2);
        assert_eq!(occupancy(&gate, rank(2)), (1, 1, 1));
        let fallback = permit(gate.acquire(None, Some(2)).await);
        assert_eq!(fallback.rank, UNSPECIFIED, "rank 2 is no longer served");

        drop(two);
        let queued_two = permit(outcome(queued_two).await);
        assert_eq!(queued_two.rank, rank(2));
        assert_eq!(published_at(&gate, rank(2)), (1, 1, 0));
        drop((queued_two, fallback));
        assert_eq!(published(&gate), (0, 0, 0));
    }

    ///////////////////// SCHEDULE /////////////////////

    impl GateState {
        /// The schedule holds exactly the head of every queue whose rank's own
        /// limits have room, under that head's arrival, and each queue records
        /// the arrival it is scheduled under; no queue is empty or out of FIFO
        /// order; and no scheduled head is left waiting while the process-wide
        /// limits have room.
        fn assert_schedule_exact(&self) {
            let mut expected = BTreeMap::new();
            for (&rank, fifo) in &self.queues {
                let arrivals = self.slots.arrivals(fifo);
                assert!(!arrivals.is_empty(), "{rank:?}: an empty queue is dropped");
                assert!(arrivals.is_sorted(), "{rank:?}: FIFO order {arrivals:?}");
                let scheduled = self.has_rank_engine_room(rank).then_some(arrivals[0]);
                assert_eq!(fifo.scheduled, scheduled, "{rank:?}");
                expected.extend(scheduled.map(|arrival| (arrival, rank)));
            }
            assert_eq!(self.schedule, expected);
            assert!(
                self.schedule.is_empty() || !self.has_aggregate_engine_room(),
                "a scheduled head is left waiting while the gate has room: {:?}",
                self.schedule
            );
        }
    }

    /// The schedule, oldest first, once it is checked exact.
    fn scheduled(gate: &Arc<BackendAdmissionGate>) -> Vec<(u64, AdmissionRank)> {
        let state = gate.state.lock();
        state.assert_schedule_exact();
        state
            .schedule
            .iter()
            .map(|(&arrival, &rank)| (arrival, rank))
            .collect()
    }

    /// Queues the scheduler has examined so far.
    fn schedule_steps(gate: &Arc<BackendAdmissionGate>) -> usize {
        gate.state.lock().schedule_steps
    }

    /// A request waiting in its FIFO, polled only when a test asks.
    type QueuedRequest = Pin<Box<dyn Future<Output = Acquired>>>;

    /// A request for `dp_rank` that joins its FIFO: polled exactly once, so it
    /// registers there, and never again until a test asks, so the gate's
    /// decisions about it can be read one at a time without spawning.
    fn queue_at(
        gate: &Arc<BackendAdmissionGate>,
        dp_rank: Option<u32>,
        context: Option<Arc<dyn AsyncEngineContext>>,
    ) -> QueuedRequest {
        let gate = Arc::clone(gate);
        let mut request: QueuedRequest =
            Box::pin(async move { gate.acquire(context.as_ref(), dp_rank).await });
        assert!(
            (&mut request).now_or_never().is_none(),
            "the request must queue"
        );
        request
    }

    /// The gate's decision on a queued request, if it has made one.
    fn decided(request: &mut QueuedRequest) -> Option<Acquired> {
        request.now_or_never()
    }

    /// While only the process-wide limits are full, the head of every queue
    /// whose rank has room stays scheduled, so process-wide room freed by any
    /// request — unspecified, or of another rank — goes straight to the oldest
    /// of them, across rank and unspecified queues alike, while an older head
    /// whose own rank is full waits for that rank to free room. A head admitted
    /// hands its queue to the request behind it, under its own arrival.
    #[tokio::test]
    async fn freed_process_wide_room_goes_to_the_oldest_scheduled_head() {
        let gate = ranked_gate((1, 1, 4), 0, 3);
        let zero = permit(handoff(&gate, None, Some(0)).await);
        let unspecified = permit(handoff(&gate, None, None).await);
        let two = permit(handoff(&gate, None, Some(2)).await);
        // Oldest first: rank 0's request, whose rank is full; rank 1's;
        // unspecified; and rank 1's again, behind its first.
        let mut queued_zero = queue_at(&gate, Some(0), None);
        let mut queued_one = queue_at(&gate, Some(1), None);
        let mut queued_unspecified = queue_at(&gate, None, None);
        let mut queued_one_again = queue_at(&gate, Some(1), None);
        assert_eq!(
            scheduled(&gate),
            [(1, rank(1)), (2, UNSPECIFIED)],
            "heads stay scheduled while only the process is full"
        );

        drop(unspecified);
        let one = permit(decided(&mut queued_one).expect("unspecified work's room"));
        assert_eq!(one.rank, rank(1));
        assert!(decided(&mut queued_zero).is_none());
        assert!(decided(&mut queued_unspecified).is_none());
        assert_eq!(
            scheduled(&gate),
            [(2, UNSPECIFIED)],
            "rank 1 is full, so the head behind its first waits unscheduled"
        );

        drop(two);
        let unspecified = permit(decided(&mut queued_unspecified).expect("rank 2's room"));
        assert_eq!(unspecified.rank, UNSPECIFIED);
        assert!(scheduled(&gate).is_empty(), "ranks 0 and 1 are full");

        drop(zero);
        let zero = permit(decided(&mut queued_zero).expect("rank 0's own room"));
        assert_eq!(zero.rank, rank(0));
        drop(one);
        let one = permit(decided(&mut queued_one_again).expect("rank 1's own room"));
        assert_eq!(one.rank, rank(1));
        assert!(scheduled(&gate).is_empty());

        drop((zero, one, unspecified));
        assert_eq!(published(&gate), (0, 0, 0));
        assert!(gate.state.lock().queues.is_empty());
    }

    /// Cancelling a queued head hands the schedule to the request behind it,
    /// under that request's own arrival, and cancelling one behind the head
    /// changes nothing; an offer the scheduled head refuses is refunded to the
    /// oldest scheduled head of any queue.
    #[tokio::test]
    async fn cancelled_heads_and_refused_offers_keep_the_schedule_exact() {
        let gate = ranked_gate((2, 2, 4), 0, 2);
        // Rank 1 and unspecified work fill the process; rank 0 has room of its
        // own.
        let mut held: Vec<_> = futures::future::join_all(
            [Some(1), Some(1), None, None].map(|dp_rank| handoff(&gate, None, dp_rank)),
        )
        .await
        .into_iter()
        .map(permit)
        .collect();
        let controllers: Vec<_> = (0..3).map(|_| stoppable()).collect();
        let mut zero: Vec<_> = controllers
            .iter()
            .map(|(_, context)| queue_at(&gate, Some(0), Some(Arc::clone(context))))
            .collect();
        let mut later_unspecified = queue_at(&gate, None, None);
        assert_eq!(scheduled(&gate), [(0, rank(0)), (3, UNSPECIFIED)]);

        controllers[1].0.stop_generating();
        assert!(is_cancelled(&decided(&mut zero[1]).expect("observed")));
        assert_eq!(
            scheduled(&gate),
            [(0, rank(0)), (3, UNSPECIFIED)],
            "a request behind the head changes nothing"
        );
        controllers[0].0.stop_generating();
        assert!(is_cancelled(&decided(&mut zero[0]).expect("observed")));
        assert_eq!(
            scheduled(&gate),
            [(2, rank(0)), (3, UNSPECIFIED)],
            "the request behind the head takes its place, under its own arrival"
        );

        // Unspecified work frees room, offered to rank 0's head, whose caller
        // then leaves: the refund goes to the oldest scheduled head left.
        drop(held.pop());
        assert_eq!(occupancy(&gate, rank(0)), (1, 1, 0), "offered");
        controllers[2].0.stop_generating();
        assert!(is_cancelled(&decided(&mut zero[2]).expect("observed")));
        let later = permit(decided(&mut later_unspecified).expect("refunded room"));
        assert_eq!(later.rank, UNSPECIFIED);
        assert!(scheduled(&gate).is_empty());
        assert_eq!(cancellations(&gate), 3);

        drop((held, later));
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// A change of limits or ranks reschedules every queue, including the queue
    /// of a rank no longer served, whose requests keep their rank. A higher
    /// limit admits the oldest heads it makes room for; a lower one evicts
    /// nothing and unschedules the heads it leaves without room.
    #[tokio::test]
    async fn limit_and_rank_changes_reschedule_every_queue_including_retired_ranks() {
        let gate = BackendAdmissionGate::new(ExplicitLimits::default());
        gate.record_engine_capacity(Some(1), 0, 3);
        let held: Vec<_> =
            futures::future::join_all((0..3).map(|dp_rank| handoff(&gate, None, Some(dp_rank))))
                .await
                .into_iter()
                .map(permit)
                .collect();
        let mut two = queue_at(&gate, Some(2), None);
        let mut zero = queue_at(&gate, Some(0), None);
        let mut two_again = queue_at(&gate, Some(2), None);
        assert!(scheduled(&gate).is_empty(), "every rank is full");

        // Rank 2 is retired; its queue stays, under its own rank.
        gate.record_engine_capacity(Some(1), 0, 2);
        assert!(scheduled(&gate).is_empty());
        assert_eq!(occupancy(&gate, rank(2)), (1, 1, 2));

        // Twice the capacity per rank: every rank has room again, retired rank
        // 2 included, and the process has room for one more, which goes to the
        // oldest head: retired rank 2's, under its own rank.
        gate.record_engine_capacity(Some(2), 0, 2);
        let admitted = permit(decided(&mut two).expect("the oldest scheduled head"));
        assert_eq!(admitted.rank, rank(2));
        assert_eq!(
            scheduled(&gate),
            [(1, rank(0))],
            "rank 2 is full again; rank 0's head waits for process-wide room"
        );

        // Back below occupancy: nothing is evicted, and no head has room.
        gate.record_engine_capacity(Some(1), 0, 2);
        assert!(scheduled(&gate).is_empty());
        assert!(decided(&mut zero).is_none() && decided(&mut two_again).is_none());
        assert_eq!(occupancy(&gate, rank(2)), (2, 2, 1));

        drop((zero, two_again));
        drop((held, admitted));
        assert_eq!(published(&gate), (0, 0, 0));
        assert!(scheduled(&gate).is_empty());
    }

    /// Admitting from the queues examines only the queues the transition
    /// changes, however many wait: freeing process-wide room examines just the
    /// admitted head's queue, picked and then rescheduled, with 8 queues
    /// waiting or 1,024. Only a change of limits or ranks examines every queue,
    /// once.
    #[tokio::test]
    async fn a_transition_examines_only_the_queues_it_changes() {
        for ranks in [8u32, 1_024] {
            let gate = ranked_gate((1, 1, 1), 0, ranks);
            // Unspecified work fills the process, so every rank's request
            // queues with room of its own, and every head is scheduled.
            let mut held: Vec<_> = (0..ranks)
                .map(|_| permit(gate.acquire(None, None).now_or_never().expect("room")))
                .collect();
            // Queued from the highest rank down, so the oldest head is not the
            // lowest rank.
            let mut queued: Vec<_> = (0..ranks)
                .rev()
                .map(|dp_rank| queue_at(&gate, Some(dp_rank), None))
                .collect();
            assert_eq!(scheduled(&gate).len(), ranks as usize);

            for (index, request) in queued.iter_mut().enumerate().take(3) {
                let before = schedule_steps(&gate);
                drop(held.pop());
                assert_eq!(
                    schedule_steps(&gate) - before,
                    2,
                    "{ranks} queues: only the admitted head's queue is examined"
                );
                let admitted = permit(decided(request).expect("the oldest head"));
                assert_eq!(admitted.rank, rank(ranks - 1 - index as u32));
                held.push(admitted);
            }

            let queues = gate.state.lock().queues.len();
            let before = schedule_steps(&gate);
            gate.record_engine_capacity(None, 0, ranks + 1);
            assert_eq!(
                schedule_steps(&gate) - before,
                queues + 2,
                "{ranks} queues: a change of ranks examines every queue once, then admits \
                 the one head it made room for"
            );
            assert_eq!(scheduled(&gate).len(), queues - 1);

            drop(queued);
            drop(held);
            assert_eq!(published(&gate), (0, 0, 0));
        }
    }

    /// The schedule stays exact through any sequence of transitions: arrivals
    /// with and without a usable rank, admissions, ended waits, completions,
    /// cancellations observed while queued or first seen at the dequeue,
    /// refused offers, abandoned waiters, and reports that raise, lower and
    /// retire capacity and ranks. Every request is accounted for once at the
    /// end.
    #[tokio::test]
    async fn the_schedule_stays_exact_through_every_kind_of_transition() {
        let gate = BackendAdmissionGate::new(ExplicitLimits::default());
        let mut seed = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = |bound: usize| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % bound as u64) as usize
        };
        let mut permits: Vec<EnginePermit> = Vec::new();
        let mut waiting: Vec<(QueuedRequest, Arc<Controller>)> = Vec::new();
        for _ in 0..5_000 {
            match next(8) {
                0 | 1 => {
                    let dp_rank = match next(6) {
                        5 => None,
                        dp_rank => Some(dp_rank as u32),
                    };
                    let (controller, context) = stoppable();
                    let arriving = Arc::clone(&gate);
                    let mut request: QueuedRequest =
                        Box::pin(async move { handoff(&arriving, Some(&context), dp_rank).await });
                    match (&mut request).now_or_never() {
                        Some(Ok((permit, _))) => permits.push(permit),
                        Some(Err(_)) => {}
                        None => waiting.push((request, controller)),
                    }
                }
                2 if !permits.is_empty() => {
                    permits.swap_remove(next(permits.len()));
                }
                3 if !permits.is_empty() => {
                    let index = next(permits.len());
                    permits[index].end_waiting();
                }
                4 if !waiting.is_empty() => {
                    let index = next(waiting.len());
                    waiting[index].1.stop_generating();
                }
                5 if !waiting.is_empty() => {
                    // Abandoned: the whole waiting future is dropped.
                    drop(waiting.swap_remove(next(waiting.len())));
                }
                6 => waiting.retain_mut(|(request, _)| match decided(request) {
                    Some(Ok((permit, _))) => {
                        permits.push(permit);
                        false
                    }
                    Some(Err(_)) => false,
                    None => true,
                }),
                7 => {
                    let max_num_seqs = 1 + next(3) as u64;
                    let rank_count = 1 + next(4) as u32;
                    gate.record_engine_capacity(Some(max_num_seqs), 0, rank_count);
                }
                _ => {}
            }
            let state = gate.state.lock();
            state.assert_schedule_exact();
            for counts in [
                &state.engine_requests,
                &state.engine_requests_waiting,
                &state.queued,
            ] {
                counts.assert_consistent();
            }
        }

        drop(waiting);
        drop(permits);
        assert_eq!(published(&gate), (0, 0, 0));
        assert!(scheduled(&gate).is_empty());
        let (received, direct, queue, rejected, cancelled) = outcomes(&gate);
        assert_eq!(received, direct + queue + rejected + cancelled);
        assert!(direct > 0 && queue > 0 && rejected > 0 && cancelled > 0);
    }

    ///////////////////// CANCELLATION AND REFUND /////////////////////

    /// An already-cancelled arrival neither queues nor enters the engine, and
    /// counts one cancellation under its rank — not a rejection.
    #[tokio::test]
    async fn an_already_cancelled_arrival_is_neither_queued_nor_admitted() {
        let gate = ranked_gate((1, 1, 4), 0, 1);
        let (controller, context) = stoppable();
        controller.stop_generating();
        let refused = gate.acquire(Some(&context), Some(0)).await;
        assert!(is_cancelled(&refused));
        assert_eq!(published(&gate), (0, 0, 0));
        assert_eq!(outcomes_at(&gate, rank(0)), (1, 0, 0, 0, 1));

        // Also when the gate is full and it would otherwise have queued.
        let _held = permit(handoff(&gate, None, Some(0)).await);
        assert!(is_cancelled(&gate.acquire(Some(&context), Some(0)).await));
        assert_eq!(queued(&gate), 0);
        assert_eq!(outcomes_at(&gate, rank(0)), (3, 1, 0, 0, 2));
    }

    #[tokio::test]
    async fn a_dropped_queued_waiter_frees_its_queue_place_immediately() {
        let gate = gate(1, 1, 1);
        let held = permit(gate.acquire(None, None).await);

        let waiter = spawn_queued_admit(&gate).await;
        assert!(
            is_queue_full(&gate.acquire(None, None).await),
            "queue is full"
        );

        waiter.abort();
        let _ = waiter.await;
        await_queued(&gate, 0).await;
        assert_eq!(cancellations(&gate), 1);

        // The freed queue place is reusable: this request queues rather than
        // being shed.
        let reuse = spawn_queued_admit(&gate).await;
        drop(held);
        drop(permit(outcome(reuse).await));
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// Cancellation observed in a queued request's context removes it at once
    /// — from the head, middle or tail, without waiting to reach the head or
    /// for capacity — frees its place for the next arrival, lowers the gauge,
    /// counts once, and leaves the survivors in FIFO order.
    #[tokio::test]
    async fn an_observed_queued_cancellation_frees_its_place_at_once() {
        for cancelled in 0..3usize {
            let gate = ranked_gate((1, 1, 3), 0, 1);
            let held = permit(handoff(&gate, None, Some(0)).await);
            let mut controllers = Vec::new();
            let mut waiters = Vec::new();
            let order = Arc::new(Mutex::new(Vec::new()));
            for index in 0..3usize {
                let (controller, context) = stoppable();
                controllers.push(controller);
                let before = queued(&gate);
                waiters.push(tokio::spawn({
                    let gate = Arc::clone(&gate);
                    let order = Arc::clone(&order);
                    async move {
                        let acquired = handoff(&gate, Some(&context), Some(0)).await;
                        if acquired.is_ok() {
                            order.lock().push(index);
                        }
                        acquired.map(|(permit, _)| drop(permit))
                    }
                }));
                await_queued(&gate, before + 1).await;
            }
            assert_eq!(published_at(&gate, rank(0)), (1, 1, 3));

            controllers[cancelled].stop_generating();
            let refused = outcome(waiters.remove(cancelled)).await;
            assert_eq!(
                refused.err().map(|error| error.error_type()),
                Some(ErrorType::Cancelled)
            );
            assert_eq!(published_at(&gate, rank(0)), (1, 1, 2), "removed at once");
            assert_eq!(outcomes_at(&gate, rank(0)).4, 1);
            // The freed place is reusable straight away.
            let replacement = spawn_queued_at(&gate, Some(0), None).await;

            drop(held);
            for waiter in waiters {
                outcome(waiter).await.expect("survivor admitted");
            }
            drop(permit(outcome(replacement).await));
            let survivors: Vec<_> = (0..3).filter(|index| *index != cancelled).collect();
            assert_eq!(*order.lock(), survivors, "cancelled {cancelled}");
            assert_eq!(outcomes_at(&gate, rank(0)), (5, 1, 3, 0, 1));
            assert_eq!(published(&gate), (0, 0, 0));
        }
    }

    /// The race the dequeue check cannot see: capacity is offered while the
    /// caller is still there, then the caller leaves before taking it, so both
    /// outcomes are ready when the request is next polled. Cancellation must
    /// win the offer that is already there, refund both kinds to the next
    /// request in FIFO order, and count once.
    #[tokio::test]
    async fn cancellation_beats_a_simultaneous_offer_and_refunds_both_kinds() {
        let gate = gate(1, 1, 4);
        let held = permit(handoff(&gate, None, None).await);

        // Waiter A is cancellable; poll it exactly once so it registers in the
        // FIFO and then stops being polled.
        let (controller, context) = stoppable();
        let mut doomed = Box::pin(gate.acquire(Some(&context), None));
        assert!(futures::poll!(&mut doomed).is_pending());

        // Waiter B queues behind it and is the one that must inherit the
        // capacity.
        let survivor = spawn_queued_admit(&gate).await;

        // The released capacity is offered to A while its caller is still
        // there, and then the caller stops, before A is polled again.
        drop(held);
        assert_eq!(
            published(&gate),
            (1, 1, 1),
            "the capacity was offered to the live head"
        );
        controller.stop_generating();
        assert!(
            is_cancelled(&doomed.await),
            "a cancelled request must not be admitted by a racing offer"
        );

        // The refund covers both kinds, so the next live waiter is admitted.
        let survivor = permit(outcome(survivor).await);
        assert_eq!(published(&gate), (1, 1, 0));
        drop(survivor);
        assert_eq!(published(&gate), (0, 0, 0));
        assert_eq!(cancellations(&gate), 1);
        assert_eq!(
            admissions(&gate),
            (1, 1),
            "only the holder and the survivor"
        );
        assert_eq!(
            Arc::strong_count(&gate),
            1,
            "tickets and permits must not retain the gate"
        );
    }

    /// A request whose cancellation is first observed at its dequeue decision
    /// is discarded there: no engine capacity, one cancellation, and FIFO
    /// processing continues with the next request.
    #[tokio::test]
    async fn a_cancellation_first_seen_at_the_dequeue_decision_is_discarded() {
        let gate = gate(1, 1, 4);
        let held = permit(handoff(&gate, None, None).await);
        let (controller, context) = stoppable();
        // Queued, then never polled again, so only the dequeue check can see
        // its stop.
        let mut doomed = Box::pin(gate.acquire(Some(&context), None));
        assert!(futures::poll!(&mut doomed).is_pending());
        let next = spawn_queued_admit(&gate).await;
        controller.stop_generating();

        drop(held);
        let next = permit(outcome(next).await);
        assert_eq!(
            published(&gate),
            (1, 1, 0),
            "the next request got the capacity"
        );
        assert_eq!(cancellations(&gate), 1, "counted at the dequeue");
        assert!(
            is_cancelled(&doomed.await),
            "and the discarded request learns it"
        );
        assert_eq!(cancellations(&gate), 1, "without counting again");
        assert_eq!(admissions(&gate), (1, 1));
        drop(next);
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// A queued request that leaves without being admitted is a cancellation
    /// exactly once, whether its own future observed the stop or the whole task
    /// was dropped first. Neither shape is a rejection.
    #[tokio::test]
    async fn a_queued_cancellation_counts_once_whether_observed_or_dropped() {
        let gate = gate(1, 1, 4);
        let _held = permit(gate.acquire(None, None).await);

        // Observed: the waiting request is polled again and sees the stop.
        let (controller, context) = stoppable();
        let waiting = spawn_queued_at(&gate, None, Some(context)).await;
        controller.stop_generating();
        assert!(is_cancelled(&outcome(waiting).await));
        assert_eq!(queued(&gate), 0, "cancellation frees the queue place");
        assert_eq!(cancellations(&gate), 1, "counted where observed");

        // Dropped: the task is aborted, so only the ticket's own `Drop` can
        // account for it.
        let abandoned = spawn_queued_admit(&gate).await;
        abandoned.abort();
        let _ = abandoned.await;
        await_queued(&gate, 0).await;
        assert_eq!(cancellations(&gate), 2);

        assert_eq!(rejections(&gate), 0);
        assert_eq!(published(&gate), (1, 1, 0));
    }

    /// Cancelling every queued request — in any order, from every queue —
    /// releases each place once, rank and aggregate, and counts each once.
    #[tokio::test]
    async fn mass_cancellation_releases_every_place_once() {
        let gate = ranked_gate((1, 1, 50), 0, 2);
        let held: Vec<_> =
            futures::future::join_all([Some(0), Some(1)].map(|r| gate.acquire(None, r)))
                .await
                .into_iter()
                .map(permit)
                .collect();
        let mut controllers = Vec::new();
        let mut waiters = Vec::new();
        for index in 0..100u32 {
            let (controller, context) = stoppable();
            controllers.push(controller);
            let dp_rank = (index % 3 != 2).then_some(index % 2);
            waiters.push(spawn_queued_at(&gate, dp_rank, Some(context)).await);
        }
        assert_eq!(queued(&gate), 100);
        for index in (0..100).map(|i| (i * 37) % 100) {
            controllers[index].stop_generating();
        }
        for waiter in waiters {
            assert!(is_cancelled(&outcome(waiter).await));
        }
        assert_eq!(queued(&gate), 0);
        assert!(gate.state.lock().queues.is_empty());
        assert_eq!(cancellations(&gate), 100);
        assert_eq!(published(&gate), (2, 2, 0));
        drop(held);
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// Cancellation after admission is not the gate's: the request stays an
    /// admission, and its permit releases normally.
    #[tokio::test]
    async fn a_cancellation_after_admission_is_not_a_gate_cancellation() {
        let gate = gate(1, 1, 4);
        let held = permit(gate.acquire(None, None).await);
        let (controller, context) = stoppable();
        let queued = spawn_queued_at(&gate, None, Some(Arc::clone(&context))).await;
        drop(held);
        let (admitted, source) = outcome(queued).await.expect("admitted");
        assert_eq!(source, AdmissionSource::Queue);

        controller.stop_generating();
        assert_eq!(cancellations(&gate), 0);
        drop(admitted);
        assert_eq!(published(&gate), (0, 0, 0));
        assert_eq!(cancellations(&gate), 0);
    }

    #[test]
    fn shedding_is_a_worker_scoped_overload() {
        // Backpressure on this one worker. A pool-scoped `ResourceExhausted`
        // would claim the whole pool is out of room, and a `Backend` error
        // would report a fault that did not happen.
        let error = reject(Rejection::QueueFull { dp_rank: None }, None);
        assert_eq!(error.error_type(), ErrorType::WorkerOverloaded);
        assert_eq!(error.message(), OVERLOADED_MESSAGE);

        // The same overload, limited to one DP rank.
        let error = reject(Rejection::QueueFull { dp_rank: Some(4) }, None);
        assert_eq!(error.error_type(), ErrorType::WorkerRankOverloaded(4));
        assert_eq!(error.message(), RANK_OVERLOADED_MESSAGE);
    }

    #[test]
    fn a_cancellation_is_not_an_overload() {
        // The caller went away. Reporting that as backpressure would migrate or
        // shed against a worker that never refused the request.
        let error = reject(Rejection::Cancelled, None);
        assert_eq!(error.error_type(), ErrorType::Cancelled);
        assert_eq!(error.message(), CANCELLED_MESSAGE);
    }

    ///////////////////// THE ADMITTED STREAM /////////////////////

    /// A stand-in engine response: a sequence number, and what its backend says
    /// it carries.
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Item {
        n: usize,
        output: ResponseOutput,
    }

    impl AdmissionResponse for Item {
        fn admission_output(&self) -> ResponseOutput {
            self.output
        }
    }

    /// Metadata or control only: a bootstrap announcement, a cache report.
    fn interim(n: usize) -> Item {
        Item {
            n,
            output: ResponseOutput::INTERIM,
        }
    }

    /// A generated token that completes nothing.
    fn token(n: usize) -> Item {
        Item {
            n,
            output: ResponseOutput {
                generated_token: true,
                completion: false,
            },
        }
    }

    /// The backend's completion response, carrying no token.
    fn completion(n: usize) -> Item {
        Item {
            n,
            output: ResponseOutput {
                generated_token: false,
                completion: true,
            },
        }
    }

    /// A last token, carried by the completion response itself.
    fn last_token(n: usize) -> Item {
        Item {
            n,
            output: ResponseOutput {
                generated_token: true,
                completion: true,
            },
        }
    }

    /// A stand-in engine: it records that it ran, then returns the two tokens
    /// its stream yields, on its own context.
    async fn generate(
        ran: Arc<AtomicUsize>,
        context: Arc<dyn AsyncEngineContext>,
    ) -> anyhow::Result<EngineStream<Item>> {
        ran.fetch_add(1, Ordering::SeqCst);
        Ok(dynamo_runtime::engine::ResponseStream::new(
            Box::pin(futures::stream::iter([token(1), token(2)])),
            context,
        ))
    }

    /// A stand-in engine whose stream yields exactly `items`, then ends.
    async fn generate_items(items: Vec<Item>) -> anyhow::Result<EngineStream<Item>> {
        Ok(dynamo_runtime::engine::ResponseStream::new(
            Box::pin(futures::stream::iter(items)),
            engine_context(),
        ))
    }

    /// A stand-in engine whose stream yields `items`, then stays open without
    /// another response until it is dropped.
    async fn generate_then_hold(items: Vec<Item>) -> anyhow::Result<EngineStream<Item>> {
        Ok(dynamo_runtime::engine::ResponseStream::new(
            Box::pin(futures::stream::iter(items).chain(futures::stream::pending())),
            engine_context(),
        ))
    }

    /// A stand-in engine whose stream ends without a single response.
    async fn generate_nothing() -> anyhow::Result<EngineStream<Item>> {
        generate_items(Vec::new()).await
    }

    fn engine_context() -> Arc<dyn AsyncEngineContext> {
        Arc::new(Controller::default())
    }

    async fn admit_two(
        gate: &Arc<BackendAdmissionGate>,
        ran: &Arc<AtomicUsize>,
    ) -> EngineStream<Item> {
        gate.admit_inner(None, None, generate(Arc::clone(ran), engine_context()))
            .await
            .expect("admitted")
    }

    /// Admit a stream of `items`, alongside a stream that holds both kinds of
    /// capacity throughout, so a repeated release would show as a count below
    /// it rather than vanish at zero.
    async fn admit_items_beside_a_held_one(
        gate: &Arc<BackendAdmissionGate>,
        items: Vec<Item>,
    ) -> (EngineStream<Item>, EngineStream<Item>) {
        let held = gate
            .admit_inner(None, None, generate_then_hold(Vec::new()))
            .await
            .expect("admitted");
        let stream = gate
            .admit_inner(None, None, generate_items(items))
            .await
            .expect("admitted");
        (held, stream)
    }

    /// The stream is the engine's own. It holds waiting capacity until its
    /// first token and request capacity until it ends, each released exactly
    /// once: a second stream holds both kinds throughout, so a repeated release
    /// would show as a count below it.
    #[tokio::test]
    async fn an_admitted_stream_holds_each_kind_of_capacity_for_its_own_lifetime() {
        let gate = gate(10, 10, 0);
        let ran = Arc::new(AtomicUsize::new(0));
        let _held = admit_two(&gate, &ran).await;
        let context = engine_context();
        let mut stream = gate
            .admit_inner(None, None, generate(Arc::clone(&ran), Arc::clone(&context)))
            .await
            .expect("admitted");
        assert_eq!(ran.load(Ordering::SeqCst), 2);
        assert_eq!(
            published(&gate),
            (2, 2, 0),
            "obtaining the stream is not a response"
        );
        assert!(
            Arc::ptr_eq(&stream.context(), &context),
            "the engine's own context must be delegated, not replaced"
        );

        // The items are the engine's, unchanged.
        assert_eq!(stream.next().await, Some(token(1)));
        assert_eq!(
            published(&gate),
            (2, 1, 0),
            "the first token releases waiting capacity only"
        );
        assert_eq!(stream.next().await, Some(token(2)));
        assert_eq!(
            published(&gate),
            (2, 1, 0),
            "a later token releases nothing"
        );
        assert_eq!(stream.next().await, None);
        assert_eq!(
            published(&gate),
            (1, 1, 0),
            "the end of the stream releases request capacity"
        );
        drop(stream);
        assert_eq!(
            published(&gate),
            (1, 1, 0),
            "and dropping it releases nothing more"
        );
    }

    /// A first token admits the next queued request while the stream that
    /// answered keeps running and keeps its request capacity until it ends.
    #[tokio::test]
    async fn the_first_token_admits_a_queued_request_while_its_stream_continues() {
        let gate = gate(2, 1, 4);
        let ran = Arc::new(AtomicUsize::new(0));
        let mut first = admit_two(&gate, &ran).await;
        let queued = spawn_queued_admit(&gate).await;
        assert_eq!(
            published(&gate),
            (1, 1, 1),
            "no token yet, so the next waits"
        );

        assert_eq!(first.next().await, Some(token(1)));
        let (admitted, source) = outcome(queued).await.expect("admitted from the queue");
        assert_eq!(source, AdmissionSource::Queue);
        assert_eq!(
            published(&gate),
            (2, 1, 0),
            "the queued request waits for its own first token"
        );

        assert_eq!(first.next().await, Some(token(2)));
        assert_eq!(first.next().await, None);
        assert_eq!(published(&gate), (1, 1, 0));
        drop(admitted);
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// Interim responses — a bootstrap announcement, a cache report — leave the
    /// request waiting; its first token ends the wait, exactly once, and every
    /// response reaches the caller unchanged.
    #[tokio::test]
    async fn interim_responses_leave_a_request_waiting_until_its_first_token() {
        let gate = gate(10, 10, 0);
        let items = vec![interim(1), interim(2), token(3), token(4), last_token(5)];
        let (_held, mut stream) = admit_items_beside_a_held_one(&gate, items.clone()).await;
        assert_eq!(published(&gate), (2, 2, 0));

        let mut seen = Vec::new();
        for (index, expected) in [(2, 2, 0), (2, 2, 0), (2, 1, 0), (2, 1, 0), (2, 1, 0)]
            .into_iter()
            .enumerate()
        {
            seen.push(stream.next().await.expect("the engine's response"));
            assert_eq!(published(&gate), expected, "after response {index}");
        }
        assert_eq!(seen, items, "responses pass through unchanged");
        assert_eq!(stream.next().await, None);
        assert_eq!(
            published(&gate),
            (1, 1, 0),
            "the end releases request capacity once"
        );
        drop(stream);
        assert_eq!(published(&gate), (1, 1, 0));
    }

    /// A completion that carries no token ends its request's wait at once,
    /// while its stream stays open: the freed waiting capacity admits the next
    /// request then, not when the stream finally ends, and the open stream
    /// keeps its request capacity. A token after it releases nothing more.
    #[tokio::test]
    async fn a_tokenless_completion_ends_the_wait_while_its_stream_stays_open() {
        let gate = gate(10, 1, 4);
        let mut stream = gate
            .admit_inner(
                None,
                None,
                generate_then_hold(vec![interim(1), completion(2), token(3)]),
            )
            .await
            .expect("admitted");
        let queued = spawn_queued_admit(&gate).await;
        assert_eq!(published(&gate), (1, 1, 1));

        assert_eq!(stream.next().await, Some(interim(1)));
        assert_eq!(
            published(&gate),
            (1, 1, 1),
            "an interim response ends no wait"
        );
        assert_eq!(stream.next().await, Some(completion(2)));
        let (next, _) = outcome(queued).await.expect("admitted from the queue");
        assert_eq!(
            published(&gate),
            (2, 1, 0),
            "the completion ended the wait; its stream is still open"
        );
        assert_eq!(stream.next().await, Some(token(3)));
        assert_eq!(
            published(&gate),
            (2, 1, 0),
            "a token after the completion releases nothing more"
        );
        assert!(
            futures::poll!(stream.next()).is_pending(),
            "the stream is held open"
        );

        drop(stream);
        assert_eq!(
            published(&gate),
            (1, 1, 0),
            "dropping the open stream releases its request capacity only"
        );
        drop(next);
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// A token ends the wait even when the completion comes later, as a
    /// prefill's may: neither the completion nor anything else after the token
    /// releases more, and the end releases request capacity once.
    #[tokio::test]
    async fn a_token_before_the_completion_ends_the_wait_at_the_token() {
        let gate = gate(10, 10, 0);
        let items = vec![interim(1), token(2), interim(3), completion(4), interim(5)];
        let (_held, mut stream) = admit_items_beside_a_held_one(&gate, items.clone()).await;

        let mut seen = Vec::new();
        for (index, expected) in [(2, 2, 0), (2, 1, 0), (2, 1, 0), (2, 1, 0), (2, 1, 0)]
            .into_iter()
            .enumerate()
        {
            seen.push(stream.next().await.expect("the engine's response"));
            assert_eq!(published(&gate), expected, "after response {index}");
        }
        assert_eq!(seen, items, "responses pass through unchanged");
        assert_eq!(stream.next().await, None);
        assert_eq!(published(&gate), (1, 1, 0));
    }

    /// A completion response that also carries the last token ends the wait
    /// once, however many qualifying responses follow.
    #[tokio::test]
    async fn a_completion_carrying_a_token_ends_the_wait_once() {
        let gate = gate(10, 10, 0);
        let (_held, mut stream) = admit_items_beside_a_held_one(
            &gate,
            vec![last_token(1), last_token(2), completion(3), token(4)],
        )
        .await;

        assert_eq!(stream.next().await, Some(last_token(1)));
        assert_eq!(published(&gate), (2, 1, 0));
        while stream.next().await.is_some() {
            assert_eq!(published(&gate), (2, 1, 0));
        }
        assert_eq!(published(&gate), (1, 1, 0));
    }

    /// The gate reads a response's classification only while its request still
    /// waits, so never after the first response that ends the wait — a token
    /// or a completion, whichever comes first — however many follow: what lets
    /// a backend stop classifying its responses there, and leave every later
    /// one interim.
    #[tokio::test]
    async fn the_gate_reads_no_classification_after_the_wait_ends() {
        /// An [`Item`] that counts how often the gate reads it.
        #[derive(Debug)]
        struct Read {
            item: Item,
            reads: Arc<AtomicUsize>,
        }

        impl AdmissionResponse for Read {
            fn admission_output(&self) -> ResponseOutput {
                self.reads.fetch_add(1, Ordering::SeqCst);
                self.item.output
            }
        }

        // Each sequence, with the index of the response that ends its wait.
        let token_first = vec![interim(1), token(2), completion(3), token(4), last_token(5)];
        let completion_first = vec![interim(1), interim(2), completion(3), token(4), token(5)];
        let both_at_once = vec![last_token(1), completion(2), token(3)];
        for (items, decisive) in [(token_first, 1), (completion_first, 2), (both_at_once, 0)] {
            let gate = gate(10, 10, 0);
            let reads = Arc::new(AtomicUsize::new(0));
            let responses: Vec<_> = items
                .iter()
                .map(|&item| Read {
                    item,
                    reads: Arc::clone(&reads),
                })
                .collect();
            let mut stream = gate
                .admit_inner(None, None, async {
                    let stream: EngineStream<Read> = dynamo_runtime::engine::ResponseStream::new(
                        Box::pin(futures::stream::iter(responses)),
                        engine_context(),
                    );
                    Ok(stream)
                })
                .await
                .expect("admitted");
            for (index, item) in items.iter().enumerate() {
                assert_eq!(stream.next().await.map(|read| read.item), Some(*item));
                assert_eq!(
                    reads.load(Ordering::SeqCst),
                    index.min(decisive) + 1,
                    "response {decisive} ends the wait"
                );
            }
            assert!(stream.next().await.is_none());
            assert_eq!(published(&gate), (0, 0, 0));
        }
    }

    /// A stream of interim responses only — a prefill that announces its
    /// handoff and then simply ends — releases both kinds at its end, once.
    #[tokio::test]
    async fn a_stream_of_interim_responses_releases_both_kinds_once_at_its_end() {
        let gate = gate(10, 10, 0);
        let (_held, mut stream) =
            admit_items_beside_a_held_one(&gate, vec![interim(1), interim(2)]).await;
        assert_eq!(stream.next().await, Some(interim(1)));
        assert_eq!(stream.next().await, Some(interim(2)));
        assert_eq!(published(&gate), (2, 2, 0));
        assert_eq!(stream.next().await, None);
        assert_eq!(published(&gate), (1, 1, 0));
        drop(stream);
        assert_eq!(published(&gate), (1, 1, 0));
    }

    /// Only an annotated response's data can carry output: an error frame, and
    /// an event or comment without data, are interim whatever else they hold.
    #[test]
    fn annotations_carry_output_only_in_their_data() {
        let data = Annotated::from_data(last_token(1));
        assert_eq!(data.admission_output(), last_token(1).output);
        assert_eq!(
            Annotated::<Item>::from_error("engine failed").admission_output(),
            ResponseOutput::INTERIM
        );
        let mut error_with_data = Annotated::from_data(token(1));
        error_with_data.event = Some("error".to_string());
        assert_eq!(error_with_data.admission_output(), ResponseOutput::INTERIM);
        let event_only = Annotated::<Item> {
            data: None,
            id: None,
            event: Some("worker_instance_id".to_string()),
            comment: Some(vec!["7".to_string()]),
            error: None,
        };
        assert_eq!(event_only.admission_output(), ResponseOutput::INTERIM);
    }

    /// One rule, whatever the worker's role: a response ends its request's
    /// wait when it carries a generated token, is the completion, or both; an
    /// interim response never does.
    #[test]
    fn a_token_or_a_completion_ends_the_wait() {
        assert!(!ResponseOutput::INTERIM.ends_wait());
        for item in [token(1), completion(1), last_token(1)] {
            assert!(item.output.ends_wait(), "{item:?}");
        }
    }

    #[tokio::test]
    async fn a_stream_that_ends_without_a_response_releases_both_kinds_once() {
        let gate = gate(10, 10, 0);
        let ran = Arc::new(AtomicUsize::new(0));
        let _held = admit_two(&gate, &ran).await;
        let mut empty = gate
            .admit_inner(None, None, generate_nothing())
            .await
            .expect("admitted");
        assert_eq!(published(&gate), (2, 2, 0));

        assert_eq!(
            empty.next().await,
            None,
            "the engine ended without answering"
        );
        assert_eq!(
            published(&gate),
            (1, 1, 0),
            "an empty end releases both kinds"
        );
        drop(empty);
        assert_eq!(
            published(&gate),
            (1, 1, 0),
            "and dropping the ended stream releases nothing more"
        );
    }

    /// Cancellation, task abort and a client that goes away all end in a drop.
    #[tokio::test]
    async fn a_stream_dropped_before_its_wait_ends_releases_both_kinds() {
        let gate = gate(10, 10, 0);
        let (_held, mut stream) =
            admit_items_beside_a_held_one(&gate, vec![interim(1), token(2)]).await;
        assert_eq!(stream.next().await, Some(interim(1)));
        assert_eq!(published(&gate), (2, 2, 0));

        drop(stream);
        assert_eq!(published(&gate), (1, 1, 0));
    }

    #[tokio::test]
    async fn a_stream_dropped_after_its_wait_ends_releases_only_request_capacity() {
        let gate = gate(10, 10, 0);
        let ran = Arc::new(AtomicUsize::new(0));
        let _held = admit_two(&gate, &ran).await;
        let mut stream = admit_two(&gate, &ran).await;
        assert_eq!(stream.next().await, Some(token(1)));
        assert_eq!(published(&gate), (2, 1, 0));

        drop(stream);
        assert_eq!(
            published(&gate),
            (1, 1, 0),
            "waiting capacity is not released a second time"
        );
    }

    #[tokio::test]
    async fn aborting_admit_while_generate_is_pending_releases_both_kinds() {
        // Capacity is reserved before `generate` is polled, so the window
        // between admission and a stream existing has no wrapper to release it
        // — only dropping the `admit` future itself can.
        let gate = gate(1, 1, 4);
        let running = Arc::new(AtomicUsize::new(0));
        let task = tokio::spawn({
            let gate = Arc::clone(&gate);
            let running = Arc::clone(&running);
            async move {
                gate.admit_inner(None, None, async move {
                    running.fetch_add(1, Ordering::SeqCst);
                    std::future::pending::<anyhow::Result<EngineStream<Item>>>().await
                })
                .await
            }
        });
        while running.load(Ordering::SeqCst) == 0 {
            tokio::task::yield_now().await;
        }
        assert_eq!(published(&gate), (1, 1, 0), "held while generate runs");

        task.abort();
        let _ = task.await;
        assert_eq!(
            published(&gate),
            (0, 0, 0),
            "aborting mid-generate releases both"
        );
    }

    #[tokio::test]
    async fn a_failed_generate_releases_both_kinds_and_keeps_its_own_error() {
        let gate = gate(1, 1, 0);

        let error = gate
            .admit_inner(None, None, async {
                Err::<EngineStream<Item>, _>(anyhow::anyhow!("engine failed to start"))
            })
            .await
            .expect_err("the engine failed");
        assert_eq!(error.to_string(), "engine failed to start");
        assert!(
            error.downcast_ref::<DynamoError>().is_none(),
            "an engine failure must pass through untouched"
        );
        assert_eq!(
            published(&gate),
            (0, 0, 0),
            "a failed generate frees both at once"
        );
    }

    #[tokio::test]
    async fn a_refused_request_never_reaches_the_engine() {
        let gate = gate(1, 1, 0);
        let ran = Arc::new(AtomicUsize::new(0));
        let _held = admit_two(&gate, &ran).await;

        let error = gate
            .admit_inner(None, None, generate(Arc::clone(&ran), engine_context()))
            .await
            .expect_err("the second request is refused");
        assert_eq!(
            ran.load(Ordering::SeqCst),
            1,
            "a refused request must not run the engine"
        );

        // The refusal reaches the caller as the standardized error, through the
        // engine's own result type.
        let rejection = error
            .downcast_ref::<DynamoError>()
            .expect("a refusal is a DynamoError");
        assert_eq!(rejection.error_type(), ErrorType::WorkerOverloaded);
        assert_eq!(rejection.message(), OVERLOADED_MESSAGE);
    }

    /// An already-cancelled request must not reach the engine even when the gate
    /// is idle. Direct admission never awaits, so nothing downstream of the
    /// decision would ever notice the caller had gone.
    #[tokio::test]
    async fn an_already_cancelled_request_never_reaches_the_engine() {
        let gate = gate(1, 1, 4);
        let ran = Arc::new(AtomicUsize::new(0));
        let (controller, context) = stoppable();
        controller.stop_generating();

        let error = gate
            .admit_inner(
                Some(&context),
                None,
                generate(Arc::clone(&ran), engine_context()),
            )
            .await
            .expect_err("a cancelled request must be refused");

        assert_eq!(
            ran.load(Ordering::SeqCst),
            0,
            "generate must never be polled"
        );
        let rejection = error
            .downcast_ref::<DynamoError>()
            .expect("a refusal is a DynamoError");
        assert_eq!(rejection.error_type(), ErrorType::Cancelled);
        assert_eq!(rejection.message(), CANCELLED_MESSAGE);
        assert_eq!(
            published(&gate),
            (0, 0, 0),
            "nothing was reserved or queued"
        );
    }

    ///////////////////// METRICS /////////////////////

    /// [`spawn_queued_admit`] through the public interface, with the
    /// [`generate`] stand-in as the engine.
    async fn spawn_queued_generate(
        gate: &Arc<BackendAdmissionGate>,
        ran: &Arc<AtomicUsize>,
        context: Option<Arc<dyn AsyncEngineContext>>,
    ) -> tokio::task::JoinHandle<anyhow::Result<EngineStream<Item>>> {
        let before = queued(gate);
        let handle = tokio::spawn({
            let gate = Arc::clone(gate);
            let ran = Arc::clone(ran);
            async move {
                gate.admit_inner(context.as_ref(), None, generate(ran, engine_context()))
                    .await
            }
        });
        await_queued(gate, before + 1).await;
        handle
    }

    /// The typed refusal an `admit` error carries.
    fn admit_refusal(admitted: &anyhow::Result<EngineStream<Item>>) -> Option<(ErrorType, &str)> {
        admitted
            .as_ref()
            .err()
            .and_then(|error| error.downcast_ref::<DynamoError>())
            .map(|error| (error.error_type(), error.message()))
    }

    /// The gauges come from the gate's own counts, so they follow every
    /// transition: direct admission, enqueue, an ended wait with a queued
    /// handoff, completion, and drain.
    #[tokio::test]
    async fn the_gauges_follow_every_transition() {
        let gate = gate(2, 1, 4);
        assert_eq!(published(&gate), (0, 0, 0));
        let mut held = permit(gate.acquire(None, None).await);
        assert_eq!(published(&gate), (1, 1, 0), "direct admission");
        let waiter = spawn_queued_admit(&gate).await;
        assert_eq!(published(&gate), (1, 1, 1), "enqueue");
        held.end_waiting();
        let mut granted = permit(outcome(waiter).await);
        assert_eq!(published(&gate), (2, 1, 0), "ended wait and handoff");
        drop(held);
        assert_eq!(published(&gate), (1, 1, 0), "completion");
        granted.end_waiting();
        assert_eq!(
            published(&gate),
            (1, 0, 0),
            "the handed-off request answers"
        );
        drop(granted);
        assert_eq!(published(&gate), (0, 0, 0), "drain");
    }

    /// Ranks published so far.
    fn publications(gate: &Arc<BackendAdmissionGate>) -> usize {
        gate.state.lock().publications
    }

    /// One transition can change many ranks, each several times — here a
    /// capacity report admits a queued request on each of 512 ranks, which
    /// changes both its queue and its engine counts — and publishes each of
    /// them exactly once, from the authoritative counts, leaving none listed
    /// for the next transition, which publishes only the rank it changes.
    #[tokio::test]
    async fn a_transition_publishes_each_rank_it_changes_once() {
        const RANKS: u32 = 512;
        let gate = BackendAdmissionGate::new(ExplicitLimits::default());
        gate.record_engine_capacity(Some(1), 0, RANKS);
        let mut held: Vec<_> = (0..RANKS)
            .map(|dp_rank| {
                permit(
                    gate.acquire(None, Some(dp_rank))
                        .now_or_never()
                        .expect("room"),
                )
            })
            .collect();
        let queued: Vec<_> = (0..RANKS)
            .map(|dp_rank| queue_at(&gate, Some(dp_rank), None))
            .collect();

        let before = publications(&gate);
        gate.record_engine_capacity(Some(2), 0, RANKS);
        assert_eq!(publications(&gate) - before, RANKS as usize);
        for dp_rank in 0..RANKS {
            assert_eq!(occupancy(&gate, rank(dp_rank)), (2, 2, 0), "{dp_rank}");
            assert_eq!(published_at(&gate, rank(dp_rank)), (2, 2, 0), "{dp_rank}");
        }
        {
            let state = gate.state.lock();
            assert!(state.touched.is_empty());
            assert!(state.series.values().all(|metrics| !metrics.touched));
        }

        let before = publications(&gate);
        drop(held.pop());
        assert_eq!(publications(&gate) - before, 1, "only the rank it changes");
        assert_eq!(published_at(&gate, rank(RANKS - 1)), (1, 1, 0));

        drop((queued, held));
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// A transition never republishes a rank it does not change: a value
    /// written straight into another rank's series survives it, until a
    /// transition of that rank publishes its counts.
    #[tokio::test]
    async fn a_transition_leaves_the_ranks_it_does_not_change_alone() {
        let gate = ranked_gate((2, 2, 0), 0, 2);
        gate.state.lock().series(rank(1)).set_occupancy(7, 7, 7);
        let zero = permit(gate.acquire(None, Some(0)).await);
        assert_eq!(published_at(&gate, rank(1)), (7, 7, 7));
        let one = permit(gate.acquire(None, Some(1)).await);
        assert_eq!(published_at(&gate, rank(1)), (1, 1, 0));
        drop((zero, one));
        assert_eq!(published(&gate), (0, 0, 0));
    }

    /// Every request the gate receives is counted once on entry, and once more
    /// as admitted only when it is handed to the engine — directly, or after
    /// waiting in the queue. Waiting is not an admission, a shed or
    /// already-cancelled request is never one, and an engine that fails after
    /// the handoff does not undo one.
    #[tokio::test]
    async fn every_request_is_received_once_and_admitted_only_at_the_engine_handoff() {
        let gate = gate(2, 1, 1);
        let ran = Arc::new(AtomicUsize::new(0));
        assert_eq!((receives(&gate), admissions(&gate)), (0, (0, 0)));

        let mut held = admit_two(&gate, &ran).await;
        assert_eq!(
            (receives(&gate), admissions(&gate)),
            (1, (1, 0)),
            "free capacity is a direct admission"
        );

        let queued = spawn_queued_generate(&gate, &ran, None).await;
        assert_eq!(
            (receives(&gate), admissions(&gate)),
            (2, (1, 0)),
            "waiting in the queue is not an admission"
        );

        let shed = gate
            .admit_inner(None, None, generate(Arc::clone(&ran), engine_context()))
            .await;
        assert_eq!(
            admit_refusal(&shed),
            Some((ErrorType::WorkerOverloaded, OVERLOADED_MESSAGE))
        );
        assert_eq!(
            (receives(&gate), admissions(&gate)),
            (3, (1, 0)),
            "a shed request is received but never admitted"
        );
        assert_eq!(rejections(&gate), 1);

        let (controller, context) = stoppable();
        controller.stop_generating();
        let cancelled = gate
            .admit_inner(
                Some(&context),
                None,
                generate(Arc::clone(&ran), engine_context()),
            )
            .await;
        assert_eq!(
            admit_refusal(&cancelled),
            Some((ErrorType::Cancelled, CANCELLED_MESSAGE))
        );
        assert_eq!(
            (receives(&gate), admissions(&gate)),
            (4, (1, 0)),
            "nor is one already cancelled on arrival"
        );
        assert_eq!(cancellations(&gate), 1);
        assert_eq!(rejections(&gate), 1, "which is not a rejection");

        // The holder's first token hands waiting capacity to the queued
        // request, which is admitted as it is handed to the engine.
        assert_eq!(held.next().await, Some(token(1)));
        let queued = outcome(queued).await.expect("admitted from the queue");
        assert_eq!(
            (receives(&gate), admissions(&gate)),
            (4, (1, 1)),
            "a queued request is admitted at the handoff"
        );
        assert_eq!(
            ran.load(Ordering::SeqCst),
            2,
            "only the admitted requests ran"
        );

        // Never answered, so dropping it frees both kinds for the next one,
        // whose engine then fails.
        drop(queued);
        let failed = gate
            .admit_inner(None, None, async {
                Err::<EngineStream<Item>, _>(anyhow::anyhow!("engine failed to start"))
            })
            .await;
        assert!(failed.is_err());
        assert_eq!(
            (receives(&gate), admissions(&gate)),
            (5, (2, 1)),
            "an engine failure after the handoff stays an admission"
        );

        // Every request received is accounted for exactly once.
        let (direct, queue) = admissions(&gate);
        assert_eq!(
            receives(&gate),
            direct + queue + rejections(&gate) + cancellations(&gate)
        );
    }

    /// The settled identity `receive = admit_direct + admit_queue + rejection +
    /// cancellation` holds for each `dp_rank` series as well as in total.
    #[tokio::test]
    async fn the_settled_accounting_identity_holds_per_rank() {
        let gate = ranked_gate((1, 1, 1), 0, 2);
        let zero = permit(handoff(&gate, None, Some(0)).await);
        let queued_zero = spawn_queued_at(&gate, Some(0), None).await;
        assert!(is_rank_full(&gate.acquire(None, Some(0)).await, 0));
        let (controller, context) = stoppable();
        let queued_one_cancelled = {
            let _one = permit(handoff(&gate, None, Some(1)).await);
            let waiter = spawn_queued_at(&gate, Some(1), Some(context)).await;
            controller.stop_generating();
            outcome(waiter).await
        };
        assert!(is_cancelled(&queued_one_cancelled));
        // Rank 1 has gone quiet, which leaves aggregate room for this one.
        let unspecified = permit(
            handoff(&gate, None, None)
                .now_or_never()
                .expect("aggregate room is free"),
        );
        drop(zero);
        drop(permit(outcome(queued_zero).await));
        drop(unspecified);

        for at in [rank(0), rank(1), UNSPECIFIED] {
            let (received, direct, queue, rejected, cancelled) = outcomes_at(&gate, at);
            assert_eq!(received, direct + queue + rejected + cancelled, "{at:?}");
        }
        assert_eq!(outcomes_at(&gate, rank(0)), (3, 1, 1, 1, 0));
        assert_eq!(outcomes_at(&gate, rank(1)), (2, 1, 0, 0, 1));
        assert_eq!(outcomes_at(&gate, UNSPECIFIED), (1, 1, 0, 0, 0));
    }

    /// A request cancelled while queued, whether its own future observed the
    /// stop or the whole task was dropped, is received and cancelled but never
    /// admitted, and never reaches the engine.
    #[tokio::test]
    async fn a_request_cancelled_while_queued_is_received_but_never_admitted() {
        let gate = gate(1, 1, 4);
        let ran = Arc::new(AtomicUsize::new(0));
        let _held = admit_two(&gate, &ran).await;

        // Observed: the waiting request is polled again and sees the stop.
        let (controller, context) = stoppable();
        let waiting = spawn_queued_generate(&gate, &ran, Some(context)).await;
        controller.stop_generating();
        assert_eq!(
            admit_refusal(&outcome(waiting).await),
            Some((ErrorType::Cancelled, CANCELLED_MESSAGE))
        );

        // Dropped: the task is aborted while it waits.
        let abandoned = spawn_queued_generate(&gate, &ran, None).await;
        abandoned.abort();
        let _ = abandoned.await;
        await_queued(&gate, 0).await;

        assert_eq!(
            ran.load(Ordering::SeqCst),
            1,
            "only the holder reached the engine"
        );
        assert_eq!(receives(&gate), 3, "both cancelled requests were received");
        assert_eq!(
            admissions(&gate),
            (1, 0),
            "neither cancelled request was admitted"
        );
        assert_eq!(cancellations(&gate), 2);
        assert_eq!(rejections(&gate), 0);
    }

    /// A candidate whose caller has already gone is absent, not admitted. Its
    /// receiver stays open, so the gate's offer succeeds and only the ticket can
    /// refuse it: the capacity must not count as an admission or reach the
    /// engine, and must move on to the next request in FIFO order.
    #[tokio::test]
    async fn a_cancelled_candidate_is_never_admitted_and_its_capacity_moves_on() {
        let gate = gate(10, 1, 8);
        let ran = Arc::new(AtomicUsize::new(0));
        let mut held = admit_two(&gate, &ran).await;

        let (controller, context) = stoppable();
        let mut doomed = Box::pin(gate.admit_inner(
            Some(&context),
            None,
            generate(Arc::clone(&ran), engine_context()),
        ));
        assert!(futures::poll!(&mut doomed).is_pending());
        let next = spawn_queued_generate(&gate, &ran, None).await;
        assert_eq!(published(&gate), (1, 1, 2));

        // Offered while its caller is still there, then stopped before it is
        // polled again.
        assert_eq!(held.next().await, Some(token(1)));
        assert_eq!(
            published(&gate),
            (2, 1, 1),
            "the capacity was offered to the candidate"
        );
        controller.stop_generating();

        assert_eq!(
            admit_refusal(&doomed.await),
            Some((ErrorType::Cancelled, CANCELLED_MESSAGE)),
            "cancellation wins the offer"
        );

        let _next = outcome(next).await.expect("admitted from the queue");
        assert_eq!(
            published(&gate),
            (2, 1, 0),
            "the refund went to the next request in FIFO order"
        );
        assert_eq!(
            ran.load(Ordering::SeqCst),
            2,
            "the candidate never reached the engine"
        );
        assert_eq!(
            admissions(&gate),
            (1, 1),
            "the candidate was never admitted"
        );
        assert_eq!(cancellations(&gate), 1);
        assert_eq!(rejections(&gate), 0);
        assert_eq!(receives(&gate), 3);
    }

    /// Collectors are per gate: one gate's counts are invisible to another. The
    /// family a gate registers is covered in
    /// [`metrics`].
    #[tokio::test]
    async fn each_gate_owns_its_collectors() {
        let one = gate(1, 1, 0);
        let two = gate(1, 1, 0);
        let _held = permit(one.acquire(None, None).await);
        assert!(is_queue_full(&one.acquire(None, None).await));
        assert_eq!((rejections(&one), rejections(&two)), (1, 0));
        assert_eq!((published(&one), published(&two)), ((1, 1, 0), (0, 0, 0)));
    }
}
