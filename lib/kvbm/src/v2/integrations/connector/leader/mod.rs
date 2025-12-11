// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

mod control;
mod request;
mod slot;

use super::worker::ConnectorWorkerClient;
use crate::distributed::leader::{
    FindMatchesOptions, FindMatchesResult, Leader, OnboardingStatus, StagingMode,
};
use crate::distributed::worker::NovaWorkerClient;
use crate::v2::distributed::leader::InstanceLeader;
use crate::v2::distributed::offload::OffloadEngine;
use crate::v2::distributed::worker::SerializedLayout;
use crate::{BlockId, InstanceId, KvbmRuntime};

use anyhow::{Result, anyhow, bail};
use dashmap::DashMap;
use parking_lot::Mutex;
use std::collections::HashSet;
use std::ops::Deref;
use std::sync::{Arc, OnceLock};

use scheduler::{DefaultOracle, Oracle};
use slot::{MatchCheckOutcome, RequestSlot, TransactionState};

pub use request::Request;
pub use scheduler::{CachedRequestData, NewRequestData, SchedulerOutput};
pub use slot::FinishedStatus;

pub trait ConnectorLeaderInterface: Send + Sync {}

pub struct ConnectorLeader {
    pub(crate) runtime: Arc<KvbmRuntime>,
    block_size: usize,
    init: Arc<Mutex<WorkerClients>>,
    workers: OnceLock<Arc<WorkerClients>>,
    instance_leader: OnceLock<InstanceLeader>,
    slots: DashMap<String, Arc<Mutex<RequestSlot>>>,
    #[allow(dead_code)] // Will be used for scheduling decisions
    oracle: Arc<dyn Oracle>,
    /// Offload engine for G1→G2→G3 transfers (initialized in initialize_async)
    offload_engine: OnceLock<OffloadEngine>,
    /// Control server shutdown handle (initialized in initialize_async)
    #[allow(dead_code)] // Kept for RAII-based shutdown on drop
    control_server_shutdown: OnceLock<oneshot::Sender<()>>,
}

#[derive(Default, Clone)]
struct WorkerClients {
    worker_instance_ids: Vec<InstanceId>,
    worker_connector_clients: Vec<ConnectorWorkerClient>,
    worker_transfer_clients: Vec<NovaWorkerClient>,
    worker_metadata: Vec<SerializedLayout>,
}

// Connector leader implementation extensions
mod init;

/// Implementation of the request_finished function.
mod finish;

/// Calls to coordinator workers.
mod clients;

// Implementation of search tools for the get_num_new_matched_tokens function.
mod search;

// Implementation of onboarding tools for the update_state_after_alloc function.
mod onboard;

// Implementation of offloading engine which will be triggered by the build_connector_metadata function.
mod offload;

// Implementation of the [`scheduler::SchedulerOutput`] struct.
pub mod scheduler;

impl ConnectorLeader {
    pub fn new(runtime: Arc<KvbmRuntime>, block_size: usize) -> Self {
        Self {
            runtime,
            block_size,
            init: Arc::new(Mutex::new(WorkerClients::default())),
            workers: OnceLock::new(),
            instance_leader: OnceLock::new(),
            slots: DashMap::new(),
            oracle: Arc::new(DefaultOracle::default()),
            offload_engine: OnceLock::new(),
            control_server_shutdown: OnceLock::new(),
        }
    }

    /// Get the block size.
    pub fn block_size(&self) -> usize {
        self.block_size
    }

    /// Access the InstanceLeader (available after initialize_workers()).
    pub(crate) fn instance_leader(&self) -> Option<&InstanceLeader> {
        self.instance_leader.get()
    }

    /// Set the InstanceLeader (called by test infrastructure after worker initialization).
    ///
    /// This is typically called by ConnectorInstance after workers are initialized
    /// and we have access to their DirectWorker instances.
    pub(crate) fn set_instance_leader(&self, leader: InstanceLeader) -> Result<()> {
        self.instance_leader
            .set(leader)
            .map_err(|_| anyhow!("InstanceLeader already set"))
    }

    /// Get the offload engine.
    ///
    /// Returns `None` if `initialize_async()` has not been called.
    pub fn offload_engine(&self) -> Option<&OffloadEngine> {
        self.offload_engine.get()
    }

    /// Check if a slot exists for the given request ID.
    pub fn has_slot(&self, request_id: &str) -> bool {
        self.slots.contains_key(request_id)
    }

    /// Get a slot for the given request ID.
    pub fn get_slot(&self, request_id: &str) -> Result<Arc<Mutex<RequestSlot>>> {
        self.slots
            .get(request_id)
            .map(|slot| slot.clone())
            .ok_or_else(|| anyhow!("Slot not found for request ID: {}", request_id))
    }

    /// Create a new slot for the given request ID, tokens and salt hash.
    pub fn create_slot(&self, request: Request) -> Result<()> {
        let request_id = request.request_id.clone();
        if self.has_slot(&request_id) {
            bail!("Slot already exists for request ID: {}", request_id);
        }
        let slot = RequestSlot::new(request, self.block_size)?;
        self.slots.insert(request_id, Arc::new(Mutex::new(slot)));
        Ok(())
    }

    /// Get the number of new tokens that can be loaded from external KV cache.
    ///
    /// This implements the vLLM KVConnector interface for `get_num_new_matched_tokens`:
    /// - Returns `(None, false)` while the find operation is still in progress
    /// - Returns `(Some(0), false)` if no external blocks are found
    /// - Returns `(Some(n), true)` if n tokens worth of blocks can be loaded asynchronously
    ///
    /// The first call for a request starts the find operation. Subsequent calls check
    /// the status of the operation and return results when complete.
    #[tracing::instrument(level = "debug", skip(self), fields(?request_id), ret)]
    pub fn get_num_new_matched_tokens(
        &self,
        request_id: &str,
        num_computed_tokens: usize,
    ) -> Result<(Option<usize>, bool)> {
        let shared_slot = self
            .slots
            .get(request_id)
            .map(|slot| slot.clone())
            .expect("slot should exist");

        let mut slot = shared_slot.lock();

        if slot.match_requires_reset() {
            todo!("reset the slot");
        }

        // Determine the match outcome
        let outcome = self.process_match(&mut slot, num_computed_tokens);

        // Single point for state transition
        match slot.finalize_match_check(outcome) {
            Ok(ok) => Ok(ok),
            Err(e) => {
                self.recover_from_match_error(&mut slot);
                if cfg!(debug_assertions) {
                    // If we are in debug mode, we want to fail the request so we can find and diagnose errors.
                    // Often times, errors will result in a misalignment in the understanding of the frameworks policy
                    // and how it calls the connnector api. Notably, these policies can change subtly across versions.
                    Err(e)
                } else {
                    // If we are in release mode, we want to ensure the request can still be processed normally,
                    // albeit without the benefits of getting an external kv cache match.
                    Ok((None, false))
                }
            }
        }
    }

    /// If this is called with `num_external_tokens` > 0, it will be called with all the blocks upto the block(s)
    /// that need to be onboarded, this included any matched G1 blocks.
    ///
    /// In this case, we compute the start block to onboard by scanning back from the end of the block lists by
    /// the `num_external_tokens/block_size`.
    ///
    /// If this is called with `num_external_tokens` == 0, we will be given the remainder of the blocks destined
    /// for prefill.
    #[tracing::instrument(level = "debug", skip(self), fields(?request_id))]
    pub fn update_state_after_alloc(
        self: &Arc<Self>,
        request_id: &str,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> Result<()> {
        let block_size = self.block_size;
        self.get_slot(request_id)?
            .lock()
            .set_match_requires_reset(true);

        if num_external_tokens == 0 {
            return Ok(());
        }

        if !num_external_tokens.is_multiple_of(block_size) {
            bail!(
                "num_external_tokens {} is not a multiple of block size {}",
                num_external_tokens,
                block_size
            );
        }

        let result = self.start_onboarding(request_id, block_ids, num_external_tokens);

        match result {
            Ok(()) => Ok(()),
            Err(e) => {
                tracing::error!("Failed to start onboarding: {}", e);
                todo!("clean up session and free resources")
            }
        }
    }

    #[tracing::instrument(level = "debug", skip_all, fields(iteration = output.iteration))]
    pub fn build_connector_meta(
        &self,
        output: scheduler::SchedulerOutput,
    ) -> Result<scheduler::KvConnectorMetadata> {
        self.process_scheduler_output(output)
    }
}

impl Deref for ConnectorLeader {
    type Target = dyn Leader;

    fn deref(&self) -> &Self::Target {
        self.instance_leader.get().expect("InstanceLeader not set")
    }
}
