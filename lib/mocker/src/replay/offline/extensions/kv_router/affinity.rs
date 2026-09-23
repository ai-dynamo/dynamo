// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Replay clock and dispatch ownership for the native session-affinity table.
//! Selection, table eviction, lease renewal, and initialization remain Router-owned.

use std::collections::HashMap;
use std::sync::{Arc, mpsc};
use std::time::Duration;

use aisimulate_core::replay::AGENTIC_CONVERSATION_LINEAGE_SCHEMA_V1;
use anyhow::{Result, anyhow, ensure};
use dynamo_kv_router::protocols::{WorkerAffinityTarget, WorkerId};
use dynamo_kv_router::services::selection::affinity::{
    AcquireStep, AffinityLease, Hold, SessionAffinity, SessionAffinityConfig,
};
use serde::{Deserialize, Serialize};
use tokio::runtime::Runtime;
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
        session_id: Option<&str>,
    ) -> Result<String> {
        let context = request.replay_context.as_ref();
        let identity = context.and_then(|context| context.agentic.as_ref());
        let identity = match (self.mode, identity) {
            (ReplayAffinityMode::Session, Some(identity)) => {
                serde_json::json!(["session", identity.play_id, identity.conversation_id])
            }
            (ReplayAffinityMode::Session, None) => {
                let session = session_id
                    .or_else(|| context.and_then(|context| context.session_id.as_deref()))
                    .filter(|value| !value.is_empty())
                    .ok_or_else(|| anyhow!("session affinity requires a session identity"))?;
                serde_json::json!(["session", session])
            }
            (ReplayAffinityMode::SiblingGroup, Some(identity)) => {
                let lineage = identity.lineage.as_ref().ok_or_else(|| {
                    anyhow!("sibling_group affinity requires conversation lineage")
                })?;
                ensure!(
                    lineage.schema == AGENTIC_CONVERSATION_LINEAGE_SCHEMA_V1,
                    "unsupported conversation lineage schema {:?}",
                    lineage.schema
                );
                // Roots form their own groups. Children with the same parent share
                // a group, including when that parent precedes a replay snapshot.
                serde_json::json!([
                    "sibling_group",
                    identity.play_id,
                    lineage.root_conversation_id,
                    lineage
                        .parent_conversation_id
                        .as_deref()
                        .unwrap_or(&identity.conversation_id),
                    lineage.parent_conversation_id.is_some()
                ])
            }
            (ReplayAffinityMode::SiblingGroup, None) => {
                return Err(anyhow!(
                    "sibling_group affinity requires an Agentic identity"
                ));
            }
        };
        // Bound arbitrary authored IDs without ambiguities or the live API's ID-size limit.
        Ok(blake3::hash(identity.to_string().as_bytes())
            .to_hex()
            .to_string())
    }
}

pub(super) struct ReplayClock {
    pub runtime: Runtime,
    epoch: Instant,
    keep_running: Option<mpsc::Sender<()>>,
}

impl ReplayClock {
    fn new() -> Result<Self> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .start_paused(true)
            .build()?;
        // A live blocking task prevents Tokio from auto-advancing to the reaper's
        // next timer when replay is otherwise idle. Only replay time may advance TTL.
        let (keep_running, stop) = mpsc::channel();
        runtime.spawn_blocking(move || {
            let _ = stop.recv();
        });
        let epoch = {
            let _entered = runtime.enter();
            Instant::now()
        };
        Ok(Self {
            runtime,
            epoch,
            keep_running: Some(keep_running),
        })
    }

    pub fn advance(&self, now_ms: f64) -> Result<()> {
        ensure!(
            now_ms.is_finite() && now_ms >= 0.0,
            "invalid replay affinity time {now_ms}"
        );
        let target = self.epoch + Duration::from_secs_f64(now_ms / 1000.0);
        let _entered = self.runtime.enter();
        let now = Instant::now();
        ensure!(target >= now, "replay affinity clock moved backwards");
        if target > now {
            self.runtime.block_on(tokio::time::advance(target - now));
        }
        Ok(())
    }

    pub fn now_ms(&self) -> f64 {
        let _entered = self.runtime.enter();
        (Instant::now() - self.epoch).as_secs_f64() * 1000.0
    }
}

impl Drop for ReplayClock {
    fn drop(&mut self) {
        self.keep_running.take();
    }
}

pub(super) struct ReplayAffinity {
    pub config: ReplayAffinityConfig,
    pub clock: Arc<ReplayClock>,
    table: SessionAffinity,
    staged: HashMap<Uuid, (Hold, WorkerAffinityTarget)>,
    active: HashMap<Uuid, AffinityLease>,
}

impl ReplayAffinity {
    pub fn new(config: ReplayAffinityConfig) -> Result<Self> {
        ensure!(
            config.ttl_seconds.is_finite() && (1.0..=31_536_000.0).contains(&config.ttl_seconds),
            "affinity.ttl_seconds must be between 1 and 31536000"
        );
        let clock = Arc::new(ReplayClock::new()?);
        let table = {
            let _entered = clock.runtime.enter();
            SessionAffinity::with_config(SessionAffinityConfig::new(Duration::from_secs_f64(
                config.ttl_seconds,
            )))?
        };
        Ok(Self {
            config,
            clock,
            table,
            staged: HashMap::new(),
            active: HashMap::new(),
        })
    }

    pub fn acquire(
        &self,
        request: &PendingRequest,
        workers: &HashMap<WorkerId, ReplayWorkerConfig>,
    ) -> Result<bool> {
        let key = request
            .group_key
            .as_deref()
            .expect("affinity request has a group key");
        let mut held = request.affinity_hold.borrow_mut();
        for _ in 0..2 {
            if held.is_none() {
                match self.table.try_acquire(key, None)? {
                    AcquireStep::Held(hold) => *held = Some(hold),
                    AcquireStep::Wait(_) => return Ok(false),
                }
            }
            if held
                .as_ref()
                .and_then(Hold::target)
                .is_some_and(|target| !workers.contains_key(&target.worker_id))
            {
                held.take().expect("checked above").invalidate();
            } else {
                return Ok(true);
            }
        }
        Err(anyhow!(
            "could not acquire a routable replay affinity target"
        ))
    }

    pub fn stage(&mut self, id: Uuid, hold: Hold, target: WorkerAffinityTarget) {
        self.staged.insert(id, (hold, target));
    }

    pub fn commit(&mut self, id: Uuid) -> Result<()> {
        let (hold, target) = self
            .staged
            .remove(&id)
            .ok_or_else(|| anyhow!("affinity dispatch has no staged admission for {id}"))?;
        self.active.insert(id, self.table.commit(hold, target)?);
        Ok(())
    }

    pub fn release(&mut self, id: Uuid) {
        self.staged.remove(&id);
        self.active.remove(&id);
    }
}

impl Drop for ReplayAffinity {
    fn drop(&mut self) {
        let _entered = self.clock.runtime.enter();
        self.staged.clear();
        self.active.clear();
    }
}
