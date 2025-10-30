// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod recorder;
pub mod slot;

use super::*;
use dynamo_llm::block_manager::metrics_kvbm::{KvbmMetrics, KvbmMetricsRegistry};
use dynamo_runtime::DistributedRuntime;
use crate::llm::block_manager::vllm::connector::leader::slot::{
    ConnectorSlotManager, SlotManager, SlotState,
};

use crate::DistributedRuntime as PyDistributedRuntime;
use crate::llm::block_manager::BlockManagerBuilder;
use crate::llm::block_manager::{
    VllmBlockManager, distributed::KvbmLeader as PyKvbmLeader, vllm::KvbmRequest,
    vllm::connector::leader::slot::VllmConnectorSlot,
};

use dynamo_llm::block_manager::{
    BasicMetadata, DiskStorage, ImmutableBlock, PinnedStorage,
    block::{
        data::logical::distributed_leader_worker::DistributedLeaderWorkerResources,
        locality::Logical,
    },
    connector::*,
};
use dynamo_llm::tokens::{SaltHash, TokenBlockSequence, Tokens};
use std::sync::{Arc, OnceLock};
use std::{collections::HashSet, sync::Mutex};
use tokio;
use tokio::runtime::Handle;
use tokio::sync::mpsc;
use tokio::sync::oneshot;

type VllmLocality = Logical<DistributedLeaderWorkerResources>;

impl From<SlotError> for PyErr {
    fn from(err: SlotError) -> Self {
        to_pyerr(err)
    }
}
use anyhow;
use dynamo_llm::recorder::Recorder;
use tokio_util::sync::CancellationToken;

pub trait Leader: Send + Sync + std::fmt::Debug {
    fn get_num_new_matched_tokens(
        &self,
        token_ids: Vec<u32>,
        num_computed_tokens: usize,
    ) -> anyhow::Result<usize>;

    fn update_state_after_alloc(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> anyhow::Result<()>;

    fn build_connector_metadata(
        &mut self,
        scheduler_output: SchedulerOutput,
    ) -> anyhow::Result<Vec<u8>>;

    fn request_finished(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
    ) -> anyhow::Result<bool>;

    fn has_slot(&self, request_id: String) -> bool;

    fn create_slot(&mut self, request: KvbmRequest, tokens: Vec<u32>) -> anyhow::Result<()>;

    fn slot_manager(&self) -> &ConnectorSlotManager<String>;
}

#[derive(Debug)]
pub struct KvConnectorLeader {
    slot_manager: Arc<OnceLock<ConnectorSlotManager<String>>>,
    block_manager: Arc<OnceLock<VllmBlockManager>>,
    block_size: usize,
    inflight_requests: HashSet<String>,
    onboarding_slots: HashSet<String>,
    iteration_counter: u64,
    kvbm_metrics: KvbmMetrics,
}

impl KvConnectorLeader {
    fn new(
        worker_id: String,
        drt: PyDistributedRuntime,
        page_size: usize,
        leader_py: PyKvbmLeader,
    ) -> Self {
        tracing::info!(
            "KvConnectorLeader initialized with worker_id: {}",
            worker_id
        );

        let leader = leader_py.get_inner().clone();
        let drt = drt.inner().clone();
        let handle: Handle = drt.runtime().primary();

        let kvbm_metrics = KvbmMetrics::new(
            &KvbmMetricsRegistry::default(),
            kvbm_metrics_endpoint_enabled(),
            parse_kvbm_metrics_port(),
        );
        let kvbm_metrics_clone = kvbm_metrics.clone();

        let slot_manager_cell = Arc::new(OnceLock::new());
        let block_manager_cell = Arc::new(OnceLock::new());
        let (leader_ready_tx, leader_ready_rx) = oneshot::channel::<String>();

        {
            let slot_manager_cell = slot_manager_cell.clone();
            let block_manager_cell = block_manager_cell.clone();

            handle.spawn(async move {
                let ready = leader.wait_worker_sync_ready().await;
                if !ready {
                    tracing::error!(
                        "KvConnectorLeader init aborted: leader worker barrier not ready!",
                    );
                    return;
                }

                let block_manager = match BlockManagerBuilder::new()
                    .worker_id(0)
                    .leader(leader_py)
                    .page_size(page_size)
                    .disable_device_pool(false)
                    .kvbm_metrics(kvbm_metrics_clone.clone())
                    .build()
                    .await
                {
                    Ok(bm) => bm,
                    Err(e) => {
                        tracing::error!("Failed to build BlockManager: {}", e);
                        return;
                    }
                };

                let inner_block_manager = block_manager.get_block_manager().clone();

                // Store the block manager reference
                let _ = block_manager_cell.set(inner_block_manager.clone());

                // Create the slot manager now that everything is ready
                let sm = ConnectorSlotManager::new(
                    inner_block_manager,
                    leader.clone(),
                    drt.clone(),
                    kvbm_metrics_clone.clone(),
                );

                let _ = slot_manager_cell.set(sm);

                if leader_ready_tx.send("finished".to_string()).is_err() {
                    tracing::error!("main routine receiver dropped before result was sent");
                }
            });
        }

        tokio::task::block_in_place(|| {
            handle.block_on(async {
                match leader_ready_rx.await {
                    Ok(_) => tracing::info!("KvConnectorLeader init complete."),
                    Err(_) => tracing::warn!("KvConnectorLeader init channel dropped"),
                }
            });
        });

        Self {
            slot_manager: slot_manager_cell,
            block_manager: block_manager_cell,
            block_size: page_size,
            inflight_requests: HashSet::new(),
            onboarding_slots: HashSet::new(),
            iteration_counter: 0,
            kvbm_metrics,
        }
    }
}

impl Leader for KvConnectorLeader {
    #[inline]
    fn slot_manager(&self) -> &ConnectorSlotManager<String> {
        self.slot_manager
            .get()
            .expect("slot_manager not initialized")
    }

    #[inline]
    fn block_manager(&self) -> &VllmBlockManager {
        self.block_manager
            .get()
            .expect("block_manager not initialized")
    }

    /// Match the tokens in the request with the available block pools.
    /// For SGLang, we receive token_ids directly without a request_id.
    /// We search the block manager directly to find cached blocks.
    ///
    /// Returns the number of tokens that can be matched from the cache.
    #[tracing::instrument(level = "debug", skip(self, token_ids, num_computed_tokens))]
    fn get_num_new_matched_tokens(
        &self,
        token_ids: Vec<u32>,
        num_computed_tokens: usize,
    ) -> anyhow::Result<usize> {
        tracing::debug!(
            "num_tokens: {}; num_computed_tokens: {num_computed_tokens}",
            token_ids.len()
        );

        // the num_computed_tokens must be a multiple of the block size
        debug_assert!(
            num_computed_tokens % self.block_size == 0,
            "num_computed_tokens must be a multiple of the block size"
        );

        let total_tokens = token_ids.len();

        // early exit if we cannot match a full block
        if (total_tokens - num_computed_tokens) < self.block_size {
            tracing::debug!("not enough tokens to match a full block");
            return Ok(0);
        }

        // Create a temporary TokenBlockSequence to compute sequence hashes
        let sequence = TokenBlockSequence::new(
            token_ids.into(),
            self.block_size as u32,
            None, // TODO(ziqif): pass salt hash from sglang here
        );

        let sequence_hashes: Vec<_> = sequence
            .blocks()
            .iter()
            .map(|b| b.sequence_hash())
            .collect();

        let num_computed_blocks = num_computed_tokens / self.block_size;
        let search_offset = num_computed_blocks;

        tracing::debug!(
            "searching {} block hashes (offset: {})",
            sequence_hashes.len() - search_offset,
            search_offset
        );

        // Search host storage for matches
        let host_blocks = self
            .block_manager()
            .host()
            .map(|host| host.match_sequence_hashes_blocking(&sequence_hashes[search_offset..]))
            .transpose()?
            .unwrap_or_default();

        let num_matched_host_blocks = host_blocks.len();
        let search_offset_disk = search_offset + num_matched_host_blocks;

        // Search disk storage for matches
        let disk_blocks = self
            .block_manager()
            .disk()
            .map(|disk| disk.match_sequence_hashes_blocking(&sequence_hashes[search_offset_disk..]))
            .transpose()?
            .unwrap_or_default();

        let num_matched_disk_blocks = disk_blocks.len();
        let total_matched_blocks = num_matched_host_blocks + num_matched_disk_blocks;

        tracing::debug!(
            "matched {} host blocks and {} disk blocks",
            num_matched_host_blocks,
            num_matched_disk_blocks
        );

        if total_matched_blocks == 0 {
            return Ok(0);
        }

        let mut num_matched_tokens = total_matched_blocks * self.block_size;

        // If we're on a block boundary (all tokens matched), we need to throw away the last block
        // because it's incomplete and can't be used for caching
        if (num_computed_tokens + num_matched_tokens) == sequence.total_tokens() {
            tracing::debug!("on block boundary, discarding last incomplete block");
            num_matched_tokens = num_matched_tokens.saturating_sub(self.block_size);
        }

        tracing::debug!("returning {} matched tokens", num_matched_tokens);
        self.kvbm_metrics
            .matched_tokens
            .inc_by(num_matched_tokens as u64);

        Ok(num_matched_tokens)
    }


}

#[pyclass]
pub struct PySglangKvConnectorLeader {
    connector_leader: Box<dyn Leader>,
}

#[pymethods]
impl PySglangKvConnectorLeader {
    #[new]
    #[pyo3(signature = (worker_id, drt, page_size, leader))]
    pub fn new(
        worker_id: String,
        drt: PyDistributedRuntime,
        page_size: usize,
        leader: PyKvbmLeader,
    ) -> Self {
        let enable_kvbm_record = std::env::var("ENABLE_KVBM_RECORD")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let connector_leader: Box<dyn Leader> = if enable_kvbm_record {
            Box::new(recorder::KvConnectorLeaderRecorder::new(
                worker_id, drt, page_size, leader,
            ))
        } else {
            Box::new(KvConnectorLeader::new(worker_id, drt, page_size, leader))
        };
        Self { connector_leader }
    }

    fn get_num_new_matched_tokens(
        &self,
        token_ids: Vec<u32>,
        num_computed_tokens: usize,
    ) -> PyResult<usize> {
        self.connector_leader
            .get_num_new_matched_tokens(token_ids, num_computed_tokens)
            .map_err(to_pyerr)
    }


}

pub fn kvbm_metrics_endpoint_enabled() -> bool {
    std::env::var("DYN_KVBM_METRICS")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
}

pub fn parse_kvbm_metrics_port() -> u16 {
    match std::env::var("DYN_KVBM_METRICS_PORT") {
        Ok(val) => match val.trim().parse::<u16>() {
            Ok(port) => port,
            Err(_) => {
                tracing::warn!(
                    "[kvbm] Invalid DYN_KVBM_METRICS_PORT='{}', falling back to 6880",
                    val
                );
                6880
            }
        },
        Err(_) => {
            tracing::warn!(
                "DYN_KVBM_METRICS_PORT not present or couldn’t be interpreted, falling back to 6880"
            );
            6880
        }
    }
}
