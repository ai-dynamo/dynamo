// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use anyhow::Result;
use dynamo_llm::block_manager::block::{
    data::logical::distributed_leader_worker::DistributedLeaderWorkerResources, locality::Logical,
};
use dynamo_llm::block_manager::offload::filter::FrequencyFilter;
use dynamo_llm::block_manager::{BasicMetadata, BlockParallelismStrategy};

use pyo3::PyResult;
use std::time::Duration;
use tokio_util::sync::CancellationToken;

/// Creates a disk offload filter based on environment configuration.
/// Returns `Ok(None)` if the filter is disabled via `DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER`,
/// otherwise constructs a `FrequencyFilter` with standard parameters.
fn create_disk_offload_filter(
    cancel_token: &CancellationToken,
    runtime: &tokio::runtime::Handle,
) -> Result<Option<Arc<FrequencyFilter>>> {
    // Check if disk offload filter is disabled via environment variable
    let disable_filter = std::env::var("DYN_KVBM_DISABLE_DISK_OFFLOAD_FILTER")
        .map(|v| v == "true" || v == "1")
        .unwrap_or(false);

    if disable_filter {
        return Ok(None);
    }

    // TODO: These values seem plausible for most use cases, but we need to figure out a better way to configure them.
    let frequency_filter = FrequencyFilter::new(
        2,
        Duration::from_secs(600),
        1_000_000,
        cancel_token.child_token(),
        runtime.clone(),
    )?;

    Ok(Some(Arc::new(frequency_filter)))
}

mod controller;
mod distributed;

pub mod vllm;

/// Add bingings from this crate to the provided module
pub fn add_to_module(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<BlockManager>()?;
    m.add_class::<distributed::KvbmWorker>()?;
    m.add_class::<distributed::KvbmLeader>()?;
    m.add_class::<controller::BlockManagerClient>()?;
    m.add_class::<controller::BlockPoolStatus>()?;
    m.add_class::<controller::ResetBlocksResponse>()?;

    vllm::add_to_module(m)?;

    Ok(())
}

type VllmBlockManager = dynamo_llm::block_manager::KvBlockManager<
    Logical<DistributedLeaderWorkerResources>,
    BasicMetadata,
>;

type VllmController = Arc<
    dynamo_llm::block_manager::controller::Controller<
        Logical<DistributedLeaderWorkerResources>,
        BasicMetadata,
    >,
>;

#[pyclass]
#[derive(Clone)]
pub struct BlockManager {
    inner: VllmBlockManager,
    drt: DistributedRuntime,
    _controller: Option<VllmController>,
}

// TODO: This is in desperate need of a massive refactor. We bind and instantiate this in Python, but we never actually use it.
#[pymethods]
#[allow(unused_variables)]
impl BlockManager {
    #[new]
    #[pyo3(signature = (worker_id, leader = None, page_size = 32, num_device_blocks = None, disable_device_pool = false))]
    fn new(
        worker_id: u64,
        leader: Option<distributed::KvbmLeader>,
        page_size: usize,
        num_device_blocks: Option<usize>,
        disable_device_pool: bool,
    ) -> PyResult<Self> {
        let cancel_token = CancellationToken::new();
        let mut config = dynamo_llm::block_manager::KvBlockManagerConfig::builder().runtime(
            dynamo_llm::block_manager::KvManagerRuntimeConfig::builder()
                .worker_id(worker_id)
                .cancellation_token(cancel_token.clone())
                .build()
                .map_err(to_pyerr)?,
        );

        let model_config = dynamo_llm::block_manager::KvManagerModelConfig::builder()
            .num_layers(1)
            .outer_dim(1)
            .page_size(page_size)
            .inner_dim(1);

        config = config.model(model_config.build().map_err(to_pyerr)?);

        let (leader, drt) = if let Some(leader) = leader {
            let (leader, rt) = leader.dissolve();

            if !disable_device_pool {
                config = config.device_layout(
                    dynamo_llm::block_manager::KvManagerLayoutConfig::builder()
                        .num_blocks(leader.num_device_blocks())
                        .logical(Some(BlockParallelismStrategy::LeaderWorkerSharded))
                        .build()
                        .map_err(to_pyerr)?,
                );
            }

            if leader.num_host_blocks() > 0 {
                tracing::info!("Using {} host blocks", leader.num_host_blocks());

                let mut host_layout_config =
                    dynamo_llm::block_manager::KvManagerLayoutConfig::builder()
                        .num_blocks(leader.num_host_blocks())
                        .logical(Some(BlockParallelismStrategy::LeaderWorkerSharded));

                if leader.num_disk_blocks() > 0 {
                    if let Some(filter) =
                        create_disk_offload_filter(&cancel_token, &rt.inner().runtime().primary())
                            .map_err(to_pyerr)?
                    {
                        host_layout_config = host_layout_config.offload_filter(Some(filter));
                    }
                }

                config = config.host_layout(host_layout_config.build().map_err(to_pyerr)?);
            }

            if leader.num_disk_blocks() > 0 {
                tracing::info!("Using {} disk blocks", leader.num_disk_blocks());
                config = config.disk_layout(
                    dynamo_llm::block_manager::KvManagerLayoutConfig::builder()
                        .num_blocks(leader.num_disk_blocks())
                        .logical(Some(BlockParallelismStrategy::LeaderWorkerSharded))
                        .build()
                        .map_err(to_pyerr)?,
                );
            }
            (Some(leader), rt)
        } else {
            tracing::info!("Leader not provided. Block transfer functionality will be disabled.");

            // let num_device_blocks = num_device_blocks
            //     .expect("num_device_blocks must be provided if leader is not provided");

            // config = config.device_layout(
            //     dynamo_llm::block_manager::KvManagerLayoutConfig::builder()
            //         .num_blocks(num_device_blocks)
            //         .logical(Some(BlockParallelismStrategy::LeaderWorkerSharded))
            //         .build()
            //         .map_err(to_pyerr)?,
            // );

            unimplemented!("Leader not provided");
            // (
            //     None,
            //     Arc::new(
            //         tokio::runtime::Builder::new_multi_thread()
            //             .enable_all()
            //             .build()
            //             .map_err(to_pyerr)?,
            //     ),
            // )
        };

        let rt = drt.inner().runtime().primary();

        let config = config.build().map_err(to_pyerr)?;
        Ok(BlockManager {
            inner: rt
                .block_on(async {
                    let resources =
                        DistributedLeaderWorkerResources::new(leader, cancel_token.child_token())?;

                    dynamo_llm::block_manager::KvBlockManager::<
                        Logical<DistributedLeaderWorkerResources>,
                        BasicMetadata,
                    >::new(config, resources)
                    .await
                })
                .map_err(to_pyerr)?,
            drt,
            _controller: None,
        })
    }

    fn block_size(&self) -> usize {
        self.inner.block_size()
    }

    fn init_controller(&mut self, component: Component) -> PyResult<()> {
        if self._controller.is_some() {
            tracing::warn!("Controller already initialized. Ignoring init_controller call.");
            return Ok(());
        }

        let block_manager = self.inner.clone();
        let controller = self
            .drt
            .inner()
            .runtime()
            .primary()
            .block_on(controller::Controller::new(
                block_manager,
                component.inner.clone(),
            ))
            .map_err(to_pyerr)?;

        self._controller = Some(Arc::new(controller));

        let instance_id = component
            .inner
            .drt()
            .primary_lease()
            .map(|lease| lease.id())
            .ok_or_else(|| to_pyerr(anyhow::anyhow!("no instance id")))?;

        tracing::info!(
            "Dynamo KVBM Controller: {}.{}:{}",
            component.inner.namespace().name(),
            component.inner.name(),
            instance_id
        );

        Ok(())
    }
}

impl BlockManager {
    #[inline(always)]
    pub fn get_block_manager(&self) -> &VllmBlockManager {
        &self.inner
    }
}

#[derive(Default)]
pub struct BlockManagerBuilder {
    worker_id: u64,
    leader: Option<distributed::KvbmLeader>,
    page_size: usize,
    disable_device_pool: bool,
    kvbm_metrics: Option<dynamo_llm::block_manager::metrics_kvbm::KvbmMetrics>,
}

impl BlockManagerBuilder {
    pub fn new() -> Self {
        Self {
            page_size: 32, // default consistent with BlockManager::new
            ..Default::default()
        }
    }

    pub fn worker_id(mut self, id: u64) -> Self {
        self.worker_id = id;
        self
    }
    pub fn page_size(mut self, ps: usize) -> Self {
        self.page_size = ps;
        self
    }
    pub fn leader(mut self, l: distributed::KvbmLeader) -> Self {
        self.leader = Some(l);
        self
    }
    pub fn disable_device_pool(mut self, yes: bool) -> Self {
        self.disable_device_pool = yes;
        self
    }
    pub fn kvbm_metrics(
        mut self,
        metrics: dynamo_llm::block_manager::metrics_kvbm::KvbmMetrics,
    ) -> Self {
        self.kvbm_metrics = Some(metrics);
        self
    }

    /// Async build (call from an async context).
    pub async fn build(self) -> Result<BlockManager> {
        let worker_id = self.worker_id;
        let leader = self.leader.ok_or_else(|| {
            anyhow::anyhow!("leader is required (runtime is always taken from leader)")
        })?;

        // Get (inner leader handle, runtime) from the provided leader.
        let (leader_inner, drt) = leader.dissolve();

        let cancel_token = CancellationToken::new();

        // Runtime & model config
        let runtime_config = dynamo_llm::block_manager::KvManagerRuntimeConfig::builder()
            .worker_id(worker_id)
            .cancellation_token(cancel_token.clone())
            .build()?;

        let mut config =
            dynamo_llm::block_manager::KvBlockManagerConfig::builder().runtime(runtime_config);

        let model_config = dynamo_llm::block_manager::KvManagerModelConfig::builder()
            .num_layers(1)
            .outer_dim(1)
            .page_size(self.page_size)
            .inner_dim(1)
            .build()?;

        config = config.model(model_config);

        // Layouts derived from leader’s counts
        if !self.disable_device_pool {
            config = config.device_layout(
                dynamo_llm::block_manager::KvManagerLayoutConfig::builder()
                    .num_blocks(leader_inner.num_device_blocks())
                    .logical(Some(BlockParallelismStrategy::LeaderWorkerSharded))
                    .build()?,
            );
        }

        if leader_inner.num_host_blocks() > 0 {
            let mut host_layout_config =
                dynamo_llm::block_manager::KvManagerLayoutConfig::builder()
                    .num_blocks(leader_inner.num_host_blocks())
                    .logical(Some(BlockParallelismStrategy::LeaderWorkerSharded));

            if leader_inner.num_disk_blocks() > 0 {
                if let Some(filter) =
                    create_disk_offload_filter(&cancel_token, &drt.inner().runtime().primary())?
                {
                    host_layout_config = host_layout_config.offload_filter(Some(filter));
                }
            }

            config = config.host_layout(host_layout_config.build()?);
        }

        if leader_inner.num_disk_blocks() > 0 {
            config = config.disk_layout(
                dynamo_llm::block_manager::KvManagerLayoutConfig::builder()
                    .num_blocks(leader_inner.num_disk_blocks())
                    .logical(Some(BlockParallelismStrategy::LeaderWorkerSharded))
                    .build()?,
            );
        }

        let mut config_builder = config;
        if let Some(kvbm_metrics) = self.kvbm_metrics {
            config_builder = config_builder.kvbm_metrics(Some(kvbm_metrics));
        }
        let config = config_builder.build()?;

        let resources =
            DistributedLeaderWorkerResources::new(Some(leader_inner), cancel_token.child_token())?;

        let inner = dynamo_llm::block_manager::KvBlockManager::<
            Logical<DistributedLeaderWorkerResources>,
            BasicMetadata,
        >::new(config, resources)
        .await?;

        Ok(BlockManager {
            inner,
            drt,
            _controller: None,
        })
    }
}
