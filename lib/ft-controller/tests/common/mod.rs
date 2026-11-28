// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A fake vLLM world for the integration tests: one axum front end per rank
//! serving vLLM 0.34.0's fault-tolerance routes, plus a controller harness.
//!
//! The fake follows vLLM 0.34.0 (`core_client.py`, `engine_core_sentinel.py`):
//! - A busy engine notices a dead peer by itself; an idle one only when a
//!   request runs (a wake). Noticing goes `diagnosing`, then `unhealthy`.
//! - The front end keeps `ft_state`, `last_ft_request_id` and `ft_error` for
//!   the last order. Every status the engine pushes replaces that entry and
//!   drops them, so after a successful round only `healthy` is left.
//! - An order to an engine that is not `unhealthy` fails (`ft_state` failed,
//!   reason in `ft_error`). An order while another one is in flight at the
//!   same front end is rejected without any change to the status.
//! - A round is one instruction and request id at every survivor. Each must
//!   check in within the check-in timeout (vLLM: 10 s); check-ins are never
//!   deleted. A check-in timeout, an invalid order or a store that cannot be
//!   rebuilt fails the round and leaves the engine `unhealthy`. Only a failed
//!   vote after check-in ends a scale_down with the engine `dead`.
//! - Only a scale_down that removes the store's host (the lowest live rank)
//!   rebuilds the store, on the round's `dp_master_ip` and `dp_store_port`.
//!
//! Every controller mistake the fake can see (a reused request id, an order
//! to a rank that is not a survivor or is gone, orders that differ between
//! ranks, a missing store address, an order while another is in flight) is
//! recorded in [`FakeWorld::violations`].

#![allow(dead_code)]

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use anyhow::Result;
use async_trait::async_trait;
use axum::extract::State;
use axum::http::StatusCode;
use axum::routing::{get, post};
use axum::{Json, Router};
use dynamo_ft_controller::controller::{Controller, Event, Outcome, Rank, Recovery, Settings};
use dynamo_ft_controller::decision::Policy;
use dynamo_ft_controller::engine::{HttpEngine, http_client};
use dynamo_ft_controller::escalate::Escalator;
use dynamo_ft_controller::round::{FileRoundStore, RoundRecord, RoundStore};
use serde_json::{Value, json};
use tokio::net::TcpListener;
use tokio::sync::{mpsc, oneshot};
use tokio::task::JoinHandle;

/// First store port the test controllers use.
pub const STORE_PORT_BASE: u16 = 40000;

pub fn ms(millis: u64) -> Duration {
    Duration::from_millis(millis)
}

/// Waits (at most 10 s, polling every 10 ms) until `done` is true.
pub async fn wait_until(mut done: impl FnMut() -> bool) {
    let reached = tokio::time::timeout(Duration::from_secs(10), async {
        while !done() {
            tokio::time::sleep(ms(10)).await;
        }
    })
    .await;
    assert!(reached.is_ok(), "the condition was not reached within 10 s");
}

/// Fake engine timing, scaled down from vLLM's.
#[derive(Clone, Copy, Debug)]
pub struct Timing {
    /// vLLM's `_ROUND_CHECKIN_TIMEOUT_SEC` (10 s).
    pub checkin_timeout: Duration,
    /// Wait before survivors are healthy again (vLLM: at least the NIXL EP
    /// GPU kernel timeout, 30 s).
    pub recovery_delay: Duration,
    /// How long `diagnosing` lasts once an engine notices a fault.
    pub detect_delay: Duration,
}

impl Default for Timing {
    fn default() -> Self {
        Self {
            checkin_timeout: ms(300),
            recovery_delay: ms(50),
            detect_delay: ms(20),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Status {
    Healthy,
    Diagnosing,
    Unhealthy,
    Dead,
}

impl Status {
    fn as_str(self) -> &'static str {
        match self {
            Self::Healthy => "healthy",
            Self::Diagnosing => "diagnosing",
            Self::Unhealthy => "unhealthy",
            Self::Dead => "dead",
        }
    }
}

/// An order as one front end received it.
#[derive(Clone, Debug)]
pub struct Order {
    pub rank: u32,
    pub request_id: String,
    pub instruction: String,
    pub params: Value,
    pub at: Instant,
    /// Lost on the way: it never reached the engine.
    pub dropped: bool,
}

/// A wake request as one front end received it.
#[derive(Clone, Debug)]
pub struct Wake {
    pub rank: u32,
    pub at: Instant,
}

/// The front end's record of the last order it handled.
#[derive(Clone, Debug)]
struct Ft {
    state: &'static str,
    request_id: String,
    error: Option<String>,
}

struct Engine {
    status: Status,
    /// Kept by the front end; dropped by every status push.
    ft: Option<Ft>,
    /// Serves no traffic, so it notices a dead peer only when woken.
    idle: bool,
    /// The front end does not answer (process gone, or the controller's
    /// network cannot reach it).
    unreachable: bool,
    /// Dies when the next order arrives, before checking in.
    die_on_order: bool,
    /// The next order is lost before it reaches the engine.
    drop_next_order: bool,
    /// Every request id this rank received, across world restarts.
    request_ids: BTreeSet<String>,
    polls: u64,
}

impl Engine {
    fn new() -> Self {
        Self {
            status: Status::Healthy,
            ft: None,
            idle: false,
            unreachable: false,
            die_on_order: false,
            drop_next_order: false,
            request_ids: BTreeSet::new(),
            polls: 0,
        }
    }

    /// The engine pushes a status: the front end replaces its entry.
    fn push(&mut self, status: Status) {
        self.status = status;
        self.ft = None;
    }

    fn set_ft(&mut self, state: &'static str, request_id: &str, error: Option<String>) {
        self.ft = Some(Ft {
            state,
            request_id: request_id.to_string(),
            error,
        });
    }

    fn in_flight(&self) -> Option<&str> {
        self.ft
            .as_ref()
            .filter(|ft| ft.state == "recovering")
            .map(|ft| ft.request_id.as_str())
    }
}

struct World {
    timing: Timing,
    ranks: u32,
    /// Ranks the engines consider part of the world (vLLM's committed view).
    members: BTreeSet<u32>,
    engines: BTreeMap<u32, Engine>,
    /// A fault is active: a request at a healthy engine makes it notice.
    faulted: bool,
    /// Check-ins per request id. Never deleted, like vLLM's keys.
    checkins: HashMap<String, BTreeSet<u32>>,
    first_order: HashMap<String, (String, Value)>,
    orders: Vec<Order>,
    wakes: Vec<Wake>,
    violations: Vec<String>,
    /// Some round carried a store address.
    store_carried: bool,
    /// Ports a store was rebuilt on, with the round that took each.
    store_ports: BTreeMap<u64, String>,
    refused_ports: BTreeSet<u64>,
}

type Shared = Arc<Mutex<World>>;

#[derive(Clone)]
struct Front {
    world: Shared,
    rank: u32,
}

/// The fake world and its front ends.
#[derive(Clone)]
pub struct FakeWorld {
    world: Shared,
    urls: BTreeMap<u32, String>,
}

impl FakeWorld {
    pub async fn start(ranks: u32, timing: Timing) -> Self {
        let world = Arc::new(Mutex::new(World {
            timing,
            ranks,
            members: (0..ranks).collect(),
            engines: (0..ranks).map(|rank| (rank, Engine::new())).collect(),
            faulted: false,
            checkins: HashMap::new(),
            first_order: HashMap::new(),
            orders: Vec::new(),
            wakes: Vec::new(),
            violations: Vec::new(),
            store_carried: false,
            store_ports: BTreeMap::new(),
            refused_ports: BTreeSet::new(),
        }));
        let mut urls = BTreeMap::new();
        for rank in 0..ranks {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            urls.insert(rank, format!("http://{}", listener.local_addr().unwrap()));
            let app = Router::new()
                .route("/v1/fault_tolerance/status", get(status))
                .route("/v1/fault_tolerance/apply", post(apply))
                .route("/v1/models", get(models))
                .route("/v1/completions", post(completions))
                .with_state(Front {
                    world: world.clone(),
                    rank,
                });
            tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
        }
        Self { world, urls }
    }

    fn lock(&self) -> MutexGuard<'_, World> {
        self.world.lock().unwrap()
    }

    fn engine(&self, rank: u32, change: impl FnOnce(&mut Engine)) {
        change(self.lock().engines.get_mut(&rank).unwrap());
    }

    pub fn url(&self, rank: u32) -> &str {
        &self.urls[&rank]
    }

    /// The rank's process dies: its front end stops answering, and the
    /// survivors run into the fault.
    pub fn kill(&self, rank: u32) {
        {
            let mut world = self.lock();
            world.engines.get_mut(&rank).unwrap().unreachable = true;
            world.faulted = true;
        }
        self.busy_engines_notice();
    }

    /// The controller loses contact with the rank, which keeps working: no
    /// engine sees a fault.
    pub fn partition(&self, rank: u32) {
        self.engine(rank, |engine| engine.unreachable = true);
    }

    /// The controller reaches the rank again.
    pub fn heal(&self, rank: u32) {
        self.engine(rank, |engine| engine.unreachable = false);
    }

    /// The rank's engine dies; its front end reports `dead`.
    pub fn engine_dies(&self, rank: u32) {
        {
            let mut world = self.lock();
            world.engines.get_mut(&rank).unwrap().push(Status::Dead);
            world.faulted = true;
        }
        self.busy_engines_notice();
    }

    /// A fault every engine sees while nobody dies.
    pub fn transient_fault(&self) {
        self.lock().faulted = true;
        self.busy_engines_notice();
    }

    /// The supervisor restarts the whole world: every rank is back, healthy,
    /// on its original store.
    pub fn restart_world(&self) {
        let mut world = self.lock();
        world.members = (0..world.ranks).collect();
        world.faulted = false;
        world.store_carried = false;
        world.store_ports.clear();
        for engine in world.engines.values_mut() {
            engine.push(Status::Healthy);
            engine.unreachable = false;
            engine.die_on_order = false;
            engine.drop_next_order = false;
        }
    }

    pub fn set_idle(&self, rank: u32, idle: bool) {
        self.engine(rank, |engine| engine.idle = idle);
    }

    /// Rebuilding the store on `port` fails (say, the port is taken).
    pub fn refuse_store_port(&self, port: u16) {
        self.lock().refused_ports.insert(u64::from(port));
    }

    /// The rank dies when its next order arrives, before checking in.
    pub fn die_on_order(&self, rank: u32) {
        self.engine(rank, |engine| engine.die_on_order = true);
    }

    /// The rank's next order is lost on the way: its front end answers, but
    /// the engine never sees the order.
    pub fn drop_next_order(&self, rank: u32) {
        self.engine(rank, |engine| engine.drop_next_order = true);
    }

    pub fn orders(&self) -> Vec<Order> {
        self.lock().orders.clone()
    }

    pub fn wakes(&self) -> Vec<Wake> {
        self.lock().wakes.clone()
    }

    pub fn wakes_of(&self, rank: u32) -> usize {
        self.lock()
            .wakes
            .iter()
            .filter(|wake| wake.rank == rank)
            .count()
    }

    pub fn violations(&self) -> Vec<String> {
        self.lock().violations.clone()
    }

    pub fn members(&self) -> BTreeSet<u32> {
        self.lock().members.clone()
    }

    pub fn status(&self, rank: u32) -> Status {
        self.lock().engines[&rank].status
    }

    pub fn polls(&self, rank: u32) -> u64 {
        self.lock().engines[&rank].polls
    }

    fn busy_engines_notice(&self) {
        let busy: Vec<u32> = {
            let world = self.lock();
            world
                .members
                .iter()
                .copied()
                .filter(|rank| {
                    let engine = &world.engines[rank];
                    !engine.idle && !engine.unreachable && engine.status == Status::Healthy
                })
                .collect()
        };
        for rank in busy {
            notice(&self.world, rank);
        }
    }
}

/// The engine runs a step that touches the dead peer: `diagnosing`, then
/// `unhealthy`.
fn notice(world: &Shared, rank: u32) {
    let world = world.clone();
    tokio::spawn(async move {
        let delay = {
            let mut world = world.lock().unwrap();
            let delay = world.timing.detect_delay;
            let engine = world.engines.get_mut(&rank).unwrap();
            if engine.status != Status::Healthy || engine.unreachable {
                return;
            }
            engine.push(Status::Diagnosing);
            delay
        };
        tokio::time::sleep(delay).await;
        let mut world = world.lock().unwrap();
        let engine = world.engines.get_mut(&rank).unwrap();
        if engine.status == Status::Diagnosing {
            engine.push(Status::Unhealthy);
        }
    });
}

async fn hang_if_unreachable(front: &Front) {
    let unreachable = front.world.lock().unwrap().engines[&front.rank].unreachable;
    if unreachable {
        std::future::pending::<()>().await;
    }
}

async fn status(State(front): State<Front>) -> Json<Value> {
    hang_if_unreachable(&front).await;
    let mut world = front.world.lock().unwrap();
    let engine = world.engines.get_mut(&front.rank).unwrap();
    engine.polls += 1;
    let mut entry = json!({"id": front.rank, "status": engine.status.as_str()});
    if engine.status == Status::Unhealthy {
        entry["fault_info"] = json!("EPPeerFailure");
    }
    if let Some(ft) = &engine.ft {
        entry["ft_state"] = json!(ft.state);
        entry["last_ft_request_id"] = json!(ft.request_id);
        if let Some(error) = &ft.error {
            entry["ft_error"] = json!(error);
        }
    }
    Json(json!({"schema_version": 1, "total_engines": 1, "engines": [entry]}))
}

async fn models(State(front): State<Front>) -> Json<Value> {
    hang_if_unreachable(&front).await;
    Json(json!({"object": "list", "data": [{"id": "fake-model", "object": "model"}]}))
}

async fn completions(State(front): State<Front>, Json(body): Json<Value>) -> Json<Value> {
    hang_if_unreachable(&front).await;
    let faulted = {
        let mut world = front.world.lock().unwrap();
        if body["model"] != "fake-model" || body["max_tokens"] != 1 {
            world
                .violations
                .push(format!("unexpected wake request {body}"));
        }
        world.wakes.push(Wake {
            rank: front.rank,
            at: Instant::now(),
        });
        world.faulted
    };
    if faulted {
        notice(&front.world, front.rank);
    }
    Json(json!({
        "id": "cmpl-fake",
        "object": "text_completion",
        "model": "fake-model",
        "choices": [{"index": 0, "text": "pong", "finish_reason": "length"}],
    }))
}

async fn apply(State(front): State<Front>, Json(body): Json<Value>) -> StatusCode {
    let rank = front.rank;
    let request_id = body["request_id"].as_str().unwrap_or_default().to_string();
    let instruction = body["instruction"].as_str().unwrap_or_default().to_string();
    let params = body["params"].clone();
    {
        let mut world = front.world.lock().unwrap();
        if world.engines[&rank].unreachable {
            world.violations.push(format!(
                "order {request_id} sent to unreachable rank {rank}"
            ));
        }
    }
    hang_if_unreachable(&front).await;
    if request_id.is_empty()
        || !matches!(instruction.as_str(), "retry" | "scale_down")
        || !params.is_object()
    {
        return StatusCode::BAD_REQUEST;
    }
    let accepted = front
        .world
        .lock()
        .unwrap()
        .receive(rank, &request_id, &instruction, &params);
    if accepted {
        tokio::spawn(run_round(
            front.world.clone(),
            rank,
            request_id,
            instruction,
            params,
        ));
    }
    StatusCode::ACCEPTED
}

fn removed_ranks(params: &Value) -> BTreeSet<u32> {
    params["removed_dp_ranks"]
        .as_array()
        .map(|ranks| {
            ranks
                .iter()
                .map(|rank| u32::try_from(rank.as_u64().unwrap()).unwrap())
                .collect()
        })
        .unwrap_or_default()
}

fn store_port(params: &Value) -> Option<u64> {
    match (params.get("dp_master_ip"), params.get("dp_store_port")) {
        (None, None) => None,
        (Some(Value::String(_)), Some(port)) => Some(port.as_u64().unwrap()),
        other => panic!("malformed store address {other:?}"),
    }
}

impl World {
    fn is_gone(&self, rank: u32) -> bool {
        let engine = &self.engines[&rank];
        engine.unreachable || engine.status == Status::Dead
    }

    /// The store's host: the lowest live rank.
    fn store_host(&self) -> Option<u32> {
        self.members.first().copied()
    }

    /// Records an order, checks it for controller mistakes and applies the
    /// front end's rules. True when the engine starts working on it.
    fn receive(&mut self, rank: u32, id: &str, instruction: &str, params: &Value) -> bool {
        let dropped = std::mem::take(&mut self.engines.get_mut(&rank).unwrap().drop_next_order);
        self.orders.push(Order {
            rank,
            request_id: id.to_string(),
            instruction: instruction.to_string(),
            params: params.clone(),
            at: Instant::now(),
            dropped,
        });

        // What the controller must never do.
        let removed = removed_ranks(params);
        if !self.members.contains(&rank) || removed.contains(&rank) {
            self.violations.push(format!(
                "order {id} sent to rank {rank}, which is not a survivor"
            ));
        }
        for &gone in &removed {
            if !self.is_gone(gone) {
                self.violations
                    .push(format!("order {id} removes rank {gone}, which is alive"));
            }
        }
        match self.first_order.get(id) {
            Some((first, first_params)) if first != instruction || first_params != params => {
                self.violations
                    .push(format!("order {id} differs between ranks"));
            }
            Some(_) => {}
            None => {
                self.first_order
                    .insert(id.to_string(), (instruction.to_string(), params.clone()));
            }
        }
        if store_port(params).is_some() {
            self.store_carried = true;
        } else if self.store_carried {
            self.violations.push(format!(
                "round {id} carries no store address after a store round"
            ));
        }
        if !self
            .engines
            .get_mut(&rank)
            .unwrap()
            .request_ids
            .insert(id.to_string())
        {
            self.violations
                .push(format!("request id {id} reused at rank {rank}"));
        }
        let in_flight = self.engines[&rank].in_flight().map(str::to_string);
        if let Some(other) = in_flight {
            // vLLM rejects it without touching the status.
            self.violations.push(format!(
                "order {id} sent to rank {rank} while round {other} is in flight there"
            ));
            return false;
        }
        if dropped {
            return false;
        }

        // vLLM's rules: the front end marks the order in flight, then the
        // engine takes it only from `unhealthy`.
        let engine = self.engines.get_mut(&rank).unwrap();
        if engine.die_on_order {
            engine.push(Status::Dead);
            engine.set_ft(
                "failed",
                id,
                Some("engine core died\nTraceback (most recent call last): ...".to_string()),
            );
            return false;
        }
        if engine.status != Status::Unhealthy {
            let error = format!(
                "[FT] Rejecting {instruction} on engine {rank}: status is {}",
                engine.status.as_str().to_ascii_uppercase()
            );
            engine.set_ft("failed", id, Some(error));
            return false;
        }
        engine.set_ft("recovering", id, None);
        true
    }

    /// The order failed before the vote: the engine stays as it is.
    fn fail(&mut self, rank: u32, id: &str, error: String) {
        self.engines
            .get_mut(&rank)
            .unwrap()
            .set_ft("failed", id, Some(error));
    }
}

/// How an engine's part of a round starts.
enum Start {
    CheckIn,
    /// The order is invalid or the store cannot be rebuilt.
    Fail(String),
}

/// One engine's part of a round.
async fn run_round(world: Shared, rank: u32, id: String, instruction: String, params: Value) {
    let removed = removed_ranks(&params);
    let scale_down = instruction == "scale_down";
    let (timing, expected) = {
        let mut world = world.lock().unwrap();
        let expected: BTreeSet<u32> = world.members.difference(&removed).copied().collect();
        let host = world.store_host();
        let newly_dead = removed.iter().any(|rank| world.members.contains(rank));
        let start = if scale_down
            && (removed.is_empty()
                || removed.iter().any(|&gone| gone >= world.ranks)
                || removed.contains(&rank)
                || !newly_dead
                || expected.is_empty())
        {
            Start::Fail(format!(
                "Invalid removed_dp_ranks {removed:?} for engine {rank}"
            ))
        } else if let Some(host) = host.filter(|host| scale_down && removed.contains(host)) {
            match store_port(&params) {
                None => {
                    world.violations.push(format!(
                        "round {id} removes store host {host} without a new store"
                    ));
                    Start::Fail(format!(
                        "dp_store_port and dp_master_ip required when the store master (rank {host}) is removed"
                    ))
                }
                Some(port) if world.refused_ports.contains(&port) => {
                    Start::Fail(format!("cannot bind the coordination store on port {port}"))
                }
                Some(port) => match world.store_ports.get(&port) {
                    Some(owner) if *owner != id => Start::Fail(format!(
                        "cannot bind the coordination store on port {port}: round {owner} took it"
                    )),
                    _ => {
                        world.store_ports.insert(port, id.clone());
                        Start::CheckIn
                    }
                },
            }
        } else {
            // vLLM ignores a store address unless the round removes the
            // store's host.
            Start::CheckIn
        };
        if let Start::Fail(error) = start {
            world.fail(rank, &id, error);
            return;
        }
        world.checkins.entry(id.clone()).or_default().insert(rank);
        (world.timing, expected)
    };

    let deadline = Instant::now() + timing.checkin_timeout;
    loop {
        let complete = world
            .lock()
            .unwrap()
            .checkins
            .get(&id)
            .is_some_and(|checked_in| expected.is_subset(checked_in));
        if complete {
            break;
        }
        if Instant::now() >= deadline {
            let error = format!(
                "[FT] Not all engines joined recovery round '{id}' within {:?}",
                timing.checkin_timeout
            );
            world.lock().unwrap().fail(rank, &id, error);
            return;
        }
        tokio::time::sleep(ms(5)).await;
    }

    tokio::time::sleep(timing.recovery_delay).await;
    let mut world = world.lock().unwrap();
    // Every survivor votes on the outcome over the rebuilt group.
    if expected.iter().any(|&member| world.is_gone(member)) {
        if scale_down {
            // Terminal: vLLM requires a full restart.
            world.engines.get_mut(&rank).unwrap().push(Status::Dead);
        }
        world.fail(rank, &id, "[FT] Recovery vote failed".to_string());
        return;
    }
    if scale_down {
        world.members.retain(|member| !removed.contains(member));
    }
    world.engines.get_mut(&rank).unwrap().push(Status::Healthy);
    let all_healthy = world
        .members
        .iter()
        .all(|member| world.engines[member].status == Status::Healthy);
    if all_healthy {
        world.faulted = false;
    }
}

/// Controller settings scaled for tests: 20 ms polls, dead after 400 ms.
pub fn settings(world: &str) -> Settings {
    Settings {
        world: world.to_string(),
        poll_interval: ms(20),
        policy: Policy {
            dead_after: ms(400),
            wake_after: ms(100),
            wake_max_gap: ms(400),
            checkin_timeout: ms(300),
            max_rounds: 3,
            recovery_deadline: Duration::from_secs(5),
        },
        round_timeout: None,
        retry_abort: false,
        store_port_base: STORE_PORT_BASE,
        dry_run: false,
        remind_every: ms(200),
    }
}

/// The store address rank `rank` advertises in the tests.
pub fn store_host(rank: u32) -> String {
    format!("10.0.0.{rank}")
}

pub fn file_store(dir: &Path, world: &str) -> Arc<dyn RoundStore> {
    Arc::new(FileRoundStore::new(dir, world))
}

/// Records escalations instead of running a command.
#[derive(Default)]
pub struct RecordingEscalator {
    pub calls: Mutex<Vec<(String, String)>>,
}

#[async_trait]
impl Escalator for RecordingEscalator {
    async fn escalate(&self, world: &str, reason: &str) -> Result<()> {
        self.calls
            .lock()
            .unwrap()
            .push((world.to_string(), reason.to_string()));
        Ok(())
    }
}

/// A controller running against a fake world.
pub struct Running {
    pub events: mpsc::UnboundedReceiver<Event>,
    pub escalator: Arc<RecordingEscalator>,
    stop: Option<oneshot::Sender<()>>,
    task: JoinHandle<Result<Outcome>>,
}

impl Running {
    /// Starts a controller for the fake's current members over real HTTP.
    pub fn start(fake: &FakeWorld, settings: Settings, rounds: Arc<dyn RoundStore>) -> Self {
        let client = http_client().unwrap();
        let ranks = fake
            .members()
            .into_iter()
            .map(|rank| {
                let url = fake.url(rank).parse().unwrap();
                let engine = HttpEngine::new(client.clone(), &url, ms(250), ms(250)).unwrap();
                let member = Rank {
                    engine: Arc::new(engine),
                    store_host: store_host(rank),
                };
                (rank, member)
            })
            .collect();
        let escalator = Arc::new(RecordingEscalator::default());
        let (events_tx, events) = mpsc::unbounded_channel();
        let (stop, stopped) = oneshot::channel::<()>();
        let controller = Controller::new(settings, ranks, rounds, escalator.clone())
            .unwrap()
            .with_events(events_tx);
        let task = tokio::spawn(controller.run(async move {
            // Dropping the sender stops the controller too.
            let _ = stopped.await;
        }));
        Self {
            events,
            escalator,
            stop: Some(stop),
            task,
        }
    }

    pub async fn next_event(&mut self) -> Event {
        let next = tokio::time::timeout(Duration::from_secs(10), self.events.recv())
            .await
            .expect("timed out waiting for a controller event");
        match next {
            Some(event) => event,
            None => panic!("the controller ended: {:?}", (&mut self.task).await),
        }
    }

    /// Waits for the first event `pick` returns something for.
    pub async fn next_matching<T>(&mut self, mut pick: impl FnMut(Event) -> Option<T>) -> T {
        loop {
            if let Some(found) = pick(self.next_event().await) {
                return found;
            }
        }
    }

    /// Waits until the controller watches the world.
    pub async fn formed(&mut self) {
        self.next_matching(|event| matches!(event, Event::Formed).then_some(()))
            .await;
    }

    /// Waits for the next recovery; returns the rounds sent on the way.
    pub async fn recovered(&mut self) -> (Vec<RoundRecord>, Recovery) {
        let mut rounds = Vec::new();
        loop {
            match self.next_event().await {
                Event::RoundSent(record) => rounds.push(record),
                Event::Recovered(recovery) => return (rounds, recovery),
                Event::WouldEscalate(reason) => panic!("would escalate: {reason}"),
                _ => {}
            }
        }
    }

    /// Waits until the fault clears without any round.
    pub async fn cleared(&mut self) {
        self.next_matching(|event| match event {
            Event::Cleared => Some(()),
            Event::RoundSent(record) => panic!("round {} sent", record.request_id),
            Event::WouldEscalate(reason) => panic!("would escalate: {reason}"),
            _ => None,
        })
        .await;
    }

    /// Dry run: waits until the controller would escalate.
    pub async fn would_escalate(&mut self) -> String {
        self.next_matching(|event| match event {
            Event::WouldEscalate(reason) => Some(reason),
            Event::RoundSent(record) => panic!("dry run sent round {}", record.request_id),
            _ => None,
        })
        .await
    }

    /// Waits for the controller to end by itself; returns what `run` did.
    pub async fn result(mut self) -> Result<Outcome> {
        tokio::time::timeout(Duration::from_secs(10), &mut self.task)
            .await
            .expect("the controller did not end")
            .expect("the controller task panicked")
    }

    /// Waits for the controller to end by itself (an escalation).
    pub async fn outcome(self) -> Outcome {
        self.result().await.expect("the controller failed")
    }

    pub async fn stop(mut self) -> Outcome {
        if let Some(stop) = self.stop.take() {
            stop.send(()).unwrap();
        }
        self.outcome().await
    }
}
