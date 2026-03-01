// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod recorder;
pub mod slot;

use super::*;
use anyhow;
use dynamo_runtime::config::environment_names::kvbm as env_kvbm;
use dynamo_llm::block_manager::config::{
    cpu_cache_lookup_dirty, cpu_cache_lookup_disabled, set_cpu_cache_lookup_disabled,
};
use dynamo_llm::block_manager::distributed::vllm::{
    ConnectorSlotManager, KvConnectorLeaderCore, SlotManager,
    create_distributed_registry_client, is_dev_mode, kvbm_metrics_endpoint_enabled,
    parse_kvbm_metrics_port,
};
use dynamo_llm::block_manager::kv_consolidator::EventSource;
use dynamo_llm::block_manager::metrics_kvbm::{KvbmMetrics, KvbmMetricsRegistry};
use std::sync::{Arc, OnceLock};
use tokio::runtime::Handle;
use tokio::sync::oneshot;

use crate::block_manager::BlockManagerBuilder;
use crate::block_manager::{distributed::KvbmLeader as PyKvbmLeader, vllm::KvbmRequest};
use crate::get_current_tokio_handle;

pub trait Leader: Send + Sync + std::fmt::Debug {
    fn get_num_new_matched_tokens(
        &self,
        request_id: String,
        request_num_tokens: usize,
        num_computed_tokens: usize,
    ) -> anyhow::Result<(Option<usize>, bool)>;

    fn update_state_after_alloc(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> anyhow::Result<()>;

    fn build_connector_metadata(&mut self, scheduler_output: SchedulerOutput) -> anyhow::Result<Vec<u8>>;

    fn request_finished(&mut self, request_id: String, block_ids: Vec<BlockId>) -> anyhow::Result<bool>;

    fn has_slot(&self, request_id: String) -> bool;

    fn create_slot(&mut self, request: KvbmRequest, tokens: Vec<u32>) -> anyhow::Result<()>;

    fn slot_manager(&self) -> &ConnectorSlotManager<String>;

    fn clear_pool(&mut self, pool: String) -> anyhow::Result<()>;
}

#[derive(Debug)]
pub struct KvConnectorLeader {
    core: KvConnectorLeaderCore,
}

impl KvConnectorLeader {
    pub(crate) fn from_parts(
        slot_manager: Arc<OnceLock<ConnectorSlotManager<String>>>,
        page_size: usize,
        kvbm_metrics: KvbmMetrics,
    ) -> Self {
        Self {
            core: KvConnectorLeaderCore::new(slot_manager, page_size, kvbm_metrics),
        }
    }

    fn new(
        worker_id: String,
        page_size: usize,
        leader_py: PyKvbmLeader,
        consolidator_vllm_endpoint: Option<String>,
        consolidator_output_endpoint: Option<String>,
    ) -> Self {
        tracing::info!("KvConnectorLeader initialized with worker_id: {}", worker_id);

        let leader = leader_py.get_inner().clone();
        let handle: Handle = get_current_tokio_handle();

        let kvbm_metrics = KvbmMetrics::new(
            &KvbmMetricsRegistry::default(),
            kvbm_metrics_endpoint_enabled(),
            parse_kvbm_metrics_port(),
        );
        let kvbm_metrics_clone = kvbm_metrics.clone();

        let slot_manager_cell = Arc::new(OnceLock::new());
        let (leader_ready_tx, leader_ready_rx) = oneshot::channel::<String>();

        {
            let slot_manager_cell = slot_manager_cell.clone();
            let consolidator_vllm_ep = consolidator_vllm_endpoint.clone();
            let consolidator_output_ep = consolidator_output_endpoint.clone();

            handle.spawn(async move {
                let ready = leader.wait_worker_sync_ready().await;
                if !ready {
                    tracing::error!(
                        "KvConnectorLeader init aborted: leader worker barrier not ready!",
                    );
                    return;
                }

                if create_distributed_registry_client().map(|client| leader.set_remote_registry(client))
                    == Some(false)
                {
                    tracing::warn!("Remote registry was already set on leader");
                }

                let mut block_manager_builder = BlockManagerBuilder::new()
                    .worker_id(0)
                    .leader(leader_py)
                    .page_size(page_size)
                    .disable_device_pool(false)
                    .kvbm_metrics(kvbm_metrics_clone.clone());

                if let (Some(vllm_ep), Some(output_ep)) =
                    (consolidator_vllm_ep, consolidator_output_ep)
                {
                    block_manager_builder = block_manager_builder.consolidator_config(
                        vllm_ep,
                        Some(output_ep),
                        EventSource::Vllm,
                    );
                }

                let block_manager = match block_manager_builder.build().await {
                    Ok(bm) => bm,
                    Err(e) => {
                        tracing::error!("Failed to build BlockManager: {}", e);
                        return;
                    }
                };

                let sm = ConnectorSlotManager::new(
                    block_manager.get_block_manager().clone(),
                    leader.clone(),
                    kvbm_metrics_clone.clone(),
                    Some(format!("worker-{}", worker_id)),
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

        Self::from_parts(slot_manager_cell, page_size, kvbm_metrics)
    }
}

impl Leader for KvConnectorLeader {
    #[inline]
    fn slot_manager(&self) -> &ConnectorSlotManager<String> {
        self.core.slot_manager()
    }

    fn get_num_new_matched_tokens(
        &self,
        request_id: String,
        request_num_tokens: usize,
        num_computed_tokens: usize,
    ) -> anyhow::Result<(Option<usize>, bool)> {
        self.core
            .get_num_new_matched_tokens(request_id, request_num_tokens, num_computed_tokens)
    }

    fn update_state_after_alloc(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> anyhow::Result<()> {
        self.core
            .update_state_after_alloc(request_id, block_ids, num_external_tokens)
    }

    fn build_connector_metadata(&mut self, scheduler_output: SchedulerOutput) -> anyhow::Result<Vec<u8>> {
        self.core.build_connector_metadata(scheduler_output.into())
    }

    fn request_finished(&mut self, request_id: String, block_ids: Vec<BlockId>) -> anyhow::Result<bool> {
        self.core.request_finished(request_id, block_ids)
    }

    fn has_slot(&self, request_id: String) -> bool {
        self.core.has_slot(&request_id)
    }

    fn create_slot(&mut self, request: KvbmRequest, tokens: Vec<u32>) -> anyhow::Result<()> {
        self.core
            .create_slot(request.request_id, request.salt_hash, tokens)
    }

    fn clear_pool(&mut self, pool: String) -> anyhow::Result<()> {
        self.core.clear_pool(pool)
    }
}

#[pyclass]
pub struct PyKvConnectorLeader {
    connector_leader: Box<dyn Leader>,
}

#[pymethods]
impl PyKvConnectorLeader {
    #[new]
    #[pyo3(signature = (worker_id, drt, page_size, leader, consolidator_vllm_endpoint=None, consolidator_output_endpoint=None))]
    pub fn new(
        worker_id: String,
        drt: Option<PyObject>,
        page_size: usize,
        leader: PyKvbmLeader,
        consolidator_vllm_endpoint: Option<String>,
        consolidator_output_endpoint: Option<String>,
    ) -> PyResult<Self> {
        let _ = &drt;

        dynamo_runtime::logging::init();

        let enable_kvbm_record = std::env::var(env_kvbm::DYN_KVBM_ENABLE_RECORD)
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let connector_leader: Box<dyn Leader> = if enable_kvbm_record {
            Box::new(recorder::KvConnectorLeaderRecorder::new(
                worker_id,
                page_size,
                leader,
                consolidator_vllm_endpoint,
                consolidator_output_endpoint,
            ))
        } else {
            Box::new(KvConnectorLeader::new(
                worker_id,
                page_size,
                leader,
                consolidator_vllm_endpoint,
                consolidator_output_endpoint,
            ))
        };
        Ok(Self { connector_leader })
    }

    fn get_num_new_matched_tokens(
        &self,
        request_id: String,
        request_num_tokens: usize,
        num_computed_tokens: usize,
    ) -> PyResult<(Option<usize>, bool)> {
        self.connector_leader
            .get_num_new_matched_tokens(request_id, request_num_tokens, num_computed_tokens)
            .map_err(to_pyerr)
    }

    fn update_state_after_alloc(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
        num_external_tokens: usize,
    ) -> PyResult<()> {
        self.connector_leader
            .update_state_after_alloc(request_id, block_ids, num_external_tokens)
            .map_err(to_pyerr)
    }

    fn build_connector_metadata(&mut self, scheduler_output: SchedulerOutput) -> PyResult<Vec<u8>> {
        self.connector_leader
            .build_connector_metadata(scheduler_output)
            .map_err(to_pyerr)
    }

    fn request_finished(&mut self, request_id: &str, block_ids: Vec<BlockId>) -> PyResult<bool> {
        self.connector_leader
            .request_finished(request_id.to_string(), block_ids)
            .map_err(to_pyerr)
    }

    fn has_slot(&self, request_id: &str) -> bool {
        self.connector_leader.has_slot(request_id.to_string())
    }

    fn create_slot(&mut self, request: KvbmRequest, tokens: Vec<u32>) -> PyResult<()> {
        self.connector_leader
            .create_slot(request, tokens)
            .map_err(to_pyerr)
    }

    fn clear_pool(&mut self, pool: String) -> PyResult<()> {
        self.connector_leader.clear_pool(pool).map_err(to_pyerr)
    }

    fn set_cpu_cache_lookup_disabled(&self, disabled: bool) -> PyResult<()> {
        if !is_dev_mode() {
            return Err(pyo3::exceptions::PyPermissionError::new_err(
                "KVBM_DEV_MODE is not enabled. Set KVBM_DEV_MODE=TRUE to allow dev-only operations.",
            ));
        }

        set_cpu_cache_lookup_disabled(disabled);
        Ok(())
    }

    fn get_cpu_cache_lookup_disabled(&self) -> bool {
        cpu_cache_lookup_disabled()
    }

    fn cpu_cache_lookup_dirty(&self) -> bool {
        cpu_cache_lookup_dirty()
    }
}
