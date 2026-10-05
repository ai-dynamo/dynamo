// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Dispatch ownership for the native session-affinity table. The existing
//! selector and PolicyQueue still own worker selection and scheduling.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use aisimulate_core::replay::AGENTIC_CONVERSATION_LINEAGE_SCHEMA_V1;
use anyhow::{Context, Result, anyhow, ensure};
use dynamo_kv_router::protocols::{WorkerAffinityTarget, WorkerId};
use dynamo_kv_router::services::selection::affinity::{
    AcquireStep, AffinityLease, Hold, SessionAffinity, SessionAffinityConfig,
    subagent_group_affinity_id,
};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use tokio::time::Instant;
use uuid::Uuid;

use super::{PendingRequest, ReplayWorkerConfig};
use crate::common::protocols::DirectRequest;

#[derive(Clone, Copy, Debug, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ReplayAffinityMode {
    Session,
    SiblingGroup,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ReplayAffinityConfig {
    pub mode: ReplayAffinityMode,
    #[serde(default = "default_ttl_seconds")]
    pub ttl_seconds: f64,
}

fn default_ttl_seconds() -> f64 {
    3600.0
}

impl ReplayAffinityConfig {
    pub(super) fn group_key(
        &self,
        request: &DirectRequest,
        session: Option<&str>,
    ) -> Result<String> {
        let context = request.replay_context.as_ref();
        let agentic = context.and_then(|context| context.agentic.as_ref());
        let session = agentic
            .map(|agentic| agentic.conversation_id.as_str())
            .or(session)
            .or_else(|| context.and_then(|context| context.session_id.as_deref()))
            .filter(|session| !session.is_empty())
            .context("conversation affinity requires nonempty session identity")?;
        let scope = agentic.map(|agentic| agentic.play_id.as_str());
        let identity = match self.mode {
            ReplayAffinityMode::Session => json!(["session", scope, session]),
            ReplayAffinityMode::SiblingGroup => {
                let lineage = agentic
                    .and_then(|agentic| agentic.lineage.as_ref())
                    .filter(|lineage| lineage.schema == AGENTIC_CONVERSATION_LINEAGE_SCHEMA_V1)
                    .context("sibling_group affinity requires unambiguous conversation lineage")?;
                ensure!(
                    !lineage.root_conversation_id.is_empty(),
                    "empty root conversation identity"
                );
                match &lineage.parent_conversation_id {
                    Some(parent) => {
                        ensure!(!parent.is_empty(), "empty parent conversation identity");
                        json!(["siblings", scope, lineage.root_conversation_id, parent])
                    }
                    None => json!(["root", scope, lineage.root_conversation_id, session]),
                }
            }
        };
        // Namespace each play/root before using Dynamo's native group-key helper.
        Ok(subagent_group_affinity_id(&identity.to_string()))
    }
}

#[derive(Clone, Default)]
pub(in crate::replay) struct RoutingEvidence(pub Arc<Mutex<serde_json::Map<String, Value>>>);

impl RoutingEvidence {
    pub fn snapshot(&self) -> Result<Value> {
        Ok(
            json!({"api_version": 1, "roles": &*self.0.lock().map_err(|_| anyhow!("routing evidence lock poisoned"))?}),
        )
    }
}

pub(super) struct ReplayAffinity {
    pub config: ReplayAffinityConfig,
    table: SessionAffinity,
    epoch: Instant,
    pub now_ms: f64,
    staged: HashMap<Uuid, (Hold, WorkerAffinityTarget)>,
    active: HashMap<Uuid, AffinityLease>,
    waiting: Vec<PendingRequest>,
    pub ready: bool,
    role: &'static str,
    capture: bool,
    pub(super) evidence: RoutingEvidence,
}

impl ReplayAffinity {
    pub fn new(
        config: ReplayAffinityConfig,
        epoch: Instant,
        role: &'static str,
        capture: bool,
        evidence: RoutingEvidence,
    ) -> Result<Self> {
        ensure!(
            config.ttl_seconds.is_finite() && (1.0..=31_536_000.0).contains(&config.ttl_seconds),
            "affinity.ttl_seconds must be between 1 and 31536000"
        );
        let table = SessionAffinity::with_manual_clock(
            SessionAffinityConfig::new(Duration::try_from_secs_f64(config.ttl_seconds)?),
            epoch,
        )?;
        evidence.0.lock().map_err(|_| anyhow!("routing evidence lock poisoned"))?.insert(role.into(), json!({
            "native_policy": "dynamo.DefaultWorkerSelector", "dynamo_revision": option_env!("DYNAMO_BUILD_REVISION"),
            "physical_kv_events": 0, "decision_count": 0, "decisions_captured": capture, "decisions": [],
            "post_dispatch_checks": 0, "dispatch_aborts": 0,
        }));
        Ok(Self {
            config,
            table,
            epoch,
            now_ms: 0.0,
            staged: HashMap::new(),
            active: HashMap::new(),
            waiting: Vec::new(),
            ready: false,
            role,
            capture,
            evidence,
        })
    }

    pub fn advance(&mut self, now_ms: f64) -> Result<()> {
        ensure!(
            now_ms.is_finite() && now_ms >= self.now_ms,
            "invalid or backwards affinity time {now_ms}"
        );
        let delta = Duration::try_from_secs_f64(now_ms / 1000.0)
            .context("affinity time exceeds duration range")?;
        let now = self
            .epoch
            .checked_add(delta)
            .context("affinity time exceeds clock range")?;
        self.table.advance_clock(now)?;
        self.now_ms = now_ms;
        Ok(())
    }

    pub fn acquire(
        &self,
        request: &mut PendingRequest,
        workers: &HashMap<WorkerId, ReplayWorkerConfig>,
    ) -> Result<bool> {
        let key = request
            .group_key
            .as_deref()
            .context("missing affinity group key")?;
        for _ in 0..2 {
            if request.affinity_hold.is_none() {
                match self.table.try_acquire(key, None)? {
                    AcquireStep::Held(hold) => request.affinity_hold = Some(hold),
                    AcquireStep::Wait(_) => return Ok(false),
                }
            }
            if request
                .affinity_hold
                .as_ref()
                .and_then(Hold::target)
                .is_some_and(|target| !workers.contains_key(&target.worker_id))
            {
                request
                    .affinity_hold
                    .take()
                    .expect("checked above")
                    .invalidate();
            } else {
                return Ok(true);
            }
        }
        Err(anyhow!("could not acquire a routable affinity target"))
    }

    pub fn wait(&mut self, request: PendingRequest) {
        self.waiting.push(request);
    }
    pub fn waiting_count(&self) -> usize {
        self.waiting.len()
    }
    pub fn cancel_waiter(&mut self, id: Uuid) -> bool {
        let before = self.waiting.len();
        self.waiting.retain(|request| request.uuid != id);
        before != self.waiting.len()
    }
    pub fn take_ready(
        &mut self,
        workers: &HashMap<WorkerId, ReplayWorkerConfig>,
    ) -> Result<Vec<PendingRequest>> {
        if !std::mem::take(&mut self.ready) {
            return Ok(Vec::new());
        }
        let mut ready = Vec::new();
        for mut request in std::mem::take(&mut self.waiting) {
            if self.acquire(&mut request, workers)? {
                ready.push(request);
            } else {
                self.waiting.push(request);
            }
        }
        Ok(ready)
    }

    pub fn stage(
        &mut self,
        id: Uuid,
        hold: Hold,
        target: WorkerAffinityTarget,
        group: &str,
        overlap_blocks: u32,
        best_available_overlap_blocks: u32,
    ) -> Result<()> {
        let reused = hold.target().is_some();
        ensure!(
            !self.staged.contains_key(&id),
            "duplicate staged affinity dispatch {id}"
        );
        self.staged.insert(id, (hold, target));
        let mut evidence = self
            .evidence
            .0
            .lock()
            .map_err(|_| anyhow!("routing evidence lock poisoned"))?;
        let evidence = &mut evidence[self.role];
        evidence["decision_count"] = json!(evidence["decision_count"].as_u64().unwrap() + 1);
        if self.capture {
            evidence["decisions"].as_array_mut().unwrap().push(json!({"request_id": id, "role": self.role, "native_policy": "dynamo.DefaultWorkerSelector", "worker_id": target.worker_id, "dp_rank": target.dp_rank, "group_key": group, "binding_reused": reused, "overlap_blocks": overlap_blocks, "best_available_overlap_blocks": best_available_overlap_blocks}));
        }
        Ok(())
    }
    pub fn record_events(&self, count: usize) -> Result<()> {
        self.count("physical_kv_events", count as u64)
    }
    fn count(&self, field: &str, count: u64) -> Result<()> {
        let mut evidence = self
            .evidence
            .0
            .lock()
            .map_err(|_| anyhow!("routing evidence lock poisoned"))?;
        let counter = &mut evidence[self.role][field];
        *counter = json!(counter.as_u64().unwrap() + count);
        Ok(())
    }
    pub fn commit(&mut self, id: Uuid) -> Result<()> {
        let (hold, target) = self
            .staged
            .remove(&id)
            .context("affinity dispatch has no staged admission")?;
        self.active.insert(id, self.table.commit(hold, target)?);
        self.ready = true;
        self.count("post_dispatch_checks", 1)
    }
    pub fn release(&mut self, id: Uuid, aborted: bool) -> Result<()> {
        self.staged.remove(&id);
        self.active.remove(&id);
        self.ready = true;
        if aborted {
            self.count("dispatch_aborts", 1)?;
        }
        Ok(())
    }
}
