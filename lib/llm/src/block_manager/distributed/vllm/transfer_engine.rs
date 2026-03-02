// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{sync::Arc, time::Instant};

use dynamo_runtime::config::environment_names::kvbm::remote_storage as env_g4;
use dynamo_runtime::utils::task::CriticalTaskExecutionHandle;
use once_cell::sync::Lazy;
use tokio::{runtime::Handle, sync::mpsc};
use tokio::sync::Semaphore;
use tokio_util::sync::CancellationToken;

use crate::block_manager::{
    BasicMetadata, BlockMetadata, BlockPool, NixlRegisterableStorage, Storage,
    block::{
        data::logical::distributed_leader_worker::DistributedLeaderWorkerResources,
        locality::{LocalityProvider, Logical},
    },
    config::{RemoteStorageConfig, should_bypass_cpu_cache},
    connector::protocol::{LeaderTransferRequest, RequestType},
    distributed::{BlockTransferPool, BlockTransferRequest, KvbmLeader, vllm as vllm_int},
    metrics_kvbm::KvbmMetrics,
    pool::PinRegistry,
    transfer_orchestrator::{TransferPriority, priority_channel, run_priority_worker},
};
use crate::block_manager::KvBlockManager;

use super::{DrainItem, LocalOffloadRequest, LocalOnboardRequest, LocalTransferRequest, RemoteTransferRequest};

type VllmBlockManager =
    KvBlockManager<Logical<DistributedLeaderWorkerResources>, BasicMetadata>;

const DEFAULT_MAX_CONCURRENT_H2R: usize = 8;
const DEFAULT_DRAIN_QUEUE_CAP: usize = 512;
const DEFAULT_MAX_REMOTE_INFLIGHT: usize = 64;
const DEFAULT_REMOTE_HIGH_QUEUE_CAP: usize = 256;
const DEFAULT_REMOTE_LOW_QUEUE_CAP: usize = 512;
const DEFAULT_G4_TRANSFER_TIMEOUT_SECS: u64 = 30;

static MAX_CONCURRENT_H2R: Lazy<usize> = Lazy::new(|| {
    std::env::var(env_g4::DYN_KVBM_G4_MAX_CONCURRENT_H2R)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_CONCURRENT_H2R)
});

static G4_TRANSFER_TIMEOUT: Lazy<std::time::Duration> = Lazy::new(|| {
    let secs: u64 = std::env::var(env_g4::DYN_KVBM_G4_TRANSFER_TIMEOUT_SECS)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_G4_TRANSFER_TIMEOUT_SECS);
    std::time::Duration::from_secs(secs)
});

static DRAIN_QUEUE_CAP: Lazy<usize> = Lazy::new(|| {
    std::env::var("DYN_KVBM_G4_DRAIN_QUEUE_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_DRAIN_QUEUE_CAP)
});

static MAX_REMOTE_INFLIGHT: Lazy<usize> = Lazy::new(|| {
    std::env::var("DYN_KVBM_G4_MAX_REMOTE_INFLIGHT")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_MAX_REMOTE_INFLIGHT)
});

static REMOTE_HIGH_QUEUE_CAP: Lazy<usize> = Lazy::new(|| {
    std::env::var("DYN_KVBM_G4_REMOTE_HIGH_QUEUE_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_REMOTE_HIGH_QUEUE_CAP)
});

static REMOTE_LOW_QUEUE_CAP: Lazy<usize> = Lazy::new(|| {
    std::env::var("DYN_KVBM_G4_REMOTE_LOW_QUEUE_CAP")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_REMOTE_LOW_QUEUE_CAP)
});

pub struct LocalTransferEngine {
    block_manager: VllmBlockManager,
    leader: Arc<KvbmLeader>,
    xfer_rx: mpsc::UnboundedReceiver<LocalTransferRequest>,
}

impl LocalTransferEngine {
    pub fn new(
        block_manager: VllmBlockManager,
        leader: Arc<KvbmLeader>,
        xfer_rx: mpsc::UnboundedReceiver<LocalTransferRequest>,
    ) -> Self {
        Self {
            block_manager,
            leader,
            xfer_rx,
        }
    }

    pub async fn execute(
        &mut self,
        cancellation_token: CancellationToken,
        task_handle: Handle,
        task_token: CancellationToken,
        kvbm_metrics: KvbmMetrics,
    ) -> anyhow::Result<()> {
        let (onboard_tx, mut onboard_rx) = mpsc::unbounded_channel::<LocalOnboardRequest>();
        let (offload_tx, mut offload_rx) = mpsc::unbounded_channel::<LocalOffloadRequest>();
        let (remote_tx, remote_rx) = priority_channel::<RemoteTransferRequest>(
            *REMOTE_HIGH_QUEUE_CAP,
            *REMOTE_LOW_QUEUE_CAP,
        );
        let (drain_tx, mut drain_rx) = mpsc::channel::<DrainItem>(*DRAIN_QUEUE_CAP);

        let pin_registry = PinRegistry::new();
        let pin_registry_drain = pin_registry.clone();
        let pin_registry_remote = pin_registry.clone();
        let h2r_semaphore = Arc::new(Semaphore::new(*MAX_CONCURRENT_H2R));
        let h2r_semaphore_remote = h2r_semaphore.clone();

        let block_manager_offload = self.block_manager.clone();
        let block_manager_remote = self.block_manager.clone();
        let leader_offload = Arc::clone(&self.leader);
        let leader_onboard = Arc::clone(&self.leader);
        let leader_remote = Arc::clone(&self.leader);
        let drain_tx_for_offload = drain_tx.clone();
        let remote_tx_for_drain = remote_tx.clone();

        let kvbm_metrics_onboard = kvbm_metrics.clone();
        let kvbm_metrics_offload = kvbm_metrics.clone();
        let kvbm_metrics_remote = kvbm_metrics.clone();

        let onboard_task = CriticalTaskExecutionHandle::new_with_runtime(
            |cancellation_token_onboard| async move {
                while let Some(req) = onboard_rx.recv().await {
                    if cancellation_token_onboard.is_cancelled() {
                        break;
                    }
                    if let Err(e) =
                        process_onboard_request(req, &leader_onboard, kvbm_metrics_onboard.clone())
                            .await
                    {
                        tracing::error!("LocalOnboardTask error: {:?}", e);
                    }
                }
                Ok(())
            },
            task_token.clone(),
            "LocalOnboardTask",
            &task_handle,
        )
        .unwrap();

        let offload_task = CriticalTaskExecutionHandle::new_with_runtime(
            |cancellation_token_offload| async move {
                while let Some(req) = offload_rx.recv().await {
                    if cancellation_token_offload.is_cancelled() {
                        break;
                    }

                    let request_id = req.request_id.clone();
                    let operation_id = req.operation_id;

                    if let Err(e) = process_offload_request(
                        req,
                        &block_manager_offload,
                        &leader_offload,
                        kvbm_metrics_offload.clone(),
                        &drain_tx_for_offload,
                    )
                    .await
                    {
                        tracing::error!("LocalOffloadTask error: {:?}", e);
                        let fake_xfer = BlockTransferRequest {
                            from_pool: BlockTransferPool::Device,
                            to_pool: BlockTransferPool::Host,
                            blocks: vec![],
                            connector_req: Some(LeaderTransferRequest {
                                request_id: request_id.clone(),
                                uuid: operation_id,
                                requirement: None,
                                request_type: RequestType::Immediate,
                                chained: false,
                            }),
                            sequence_hashes: None,
                        };
                        if let Ok(notify_receiver) =
                            leader_offload.transfer_blocks_request(fake_xfer).await
                        {
                            let _ = notify_receiver.await;
                        }
                    }
                }
                Ok(())
            },
            task_token.clone(),
            "LocalOffloadTask",
            &task_handle,
        )
        .unwrap();

        let remote_task = CriticalTaskExecutionHandle::new_with_runtime(
            |cancellation_token_remote| async move {
                run_priority_worker(
                    cancellation_token_remote,
                    remote_rx,
                    *MAX_REMOTE_INFLIGHT,
                    move |req| {
                        let block_manager = block_manager_remote.clone();
                        let leader = Arc::clone(&leader_remote);
                        let metrics = kvbm_metrics_remote.clone();
                        let pin_reg = pin_registry_remote.clone();
                        let semaphore = h2r_semaphore_remote.clone();
                        async move {
                            if let Err(e) = process_remote_transfer_request(
                                req,
                                &block_manager,
                                &leader,
                                metrics,
                                &pin_reg,
                                &semaphore,
                            )
                            .await
                            {
                                tracing::error!("RemoteTransferTask error: {:?}", e);
                            }
                        }
                    },
                )
                .await;
                Ok(())
            },
            task_token.clone(),
            "RemoteTransferTask",
            &task_handle,
        )
        .unwrap();

        let drain_task = CriticalTaskExecutionHandle::new_with_runtime(
            |cancellation_token_drain| async move {
                loop {
                    let item = tokio::select! {
                        _ = cancellation_token_drain.cancelled() => break,
                        item = drain_rx.recv() => match item { Some(item) => item, None => break }
                    };

                    let _permit = match h2r_semaphore.clone().try_acquire_owned() {
                        Ok(p) => p,
                        Err(_) => continue,
                    };

                    let h2o_operation_id = uuid::Uuid::new_v4();
                    pin_registry_drain.insert(h2o_operation_id, item.pin_guard);

                    let h2o_req = RemoteTransferRequest::new_h2o(
                        item.request_id,
                        item.sequence_hashes,
                        item.host_block_ids,
                        h2o_operation_id,
                        item.block_size,
                        h2o_operation_id,
                    );

                    match remote_tx_for_drain.try_send(TransferPriority::Low, h2o_req) {
                        Ok(()) => {}
                        Err(tokio::sync::mpsc::error::TrySendError::Full(_))
                        | Err(tokio::sync::mpsc::error::TrySendError::Closed(_)) => {
                            pin_registry_drain.remove(&h2o_operation_id);
                            continue;
                        }
                    }
                    std::mem::forget(_permit);
                }
                Ok(())
            },
            task_token,
            "DrainTask",
            &task_handle,
        )
        .unwrap();

        loop {
            tokio::select! {
                _ = cancellation_token.cancelled() => break,
                req = self.xfer_rx.recv() => {
                    match req {
                        Some(LocalTransferRequest::Offload(offload_req)) => { let _ = offload_tx.send(offload_req); }
                        Some(LocalTransferRequest::Onboard(onboard_req)) => { let _ = onboard_tx.send(onboard_req); }
                        Some(LocalTransferRequest::Remote(remote_req)) => {
                            if remote_req.is_onboard {
                                let _ = remote_tx.send(TransferPriority::High, remote_req).await;
                            } else {
                                let _ = remote_tx.try_send(TransferPriority::Low, remote_req);
                            }
                        }
                        None => break,
                    }
                }
            }
        }

        drop(onboard_tx);
        drop(offload_tx);
        drop(remote_tx);
        onboard_task.cancel();
        offload_task.cancel();
        remote_task.cancel();
        drain_task.cancel();
        let _ = onboard_task.join().await;
        let _ = offload_task.join().await;
        let _ = remote_task.join().await;
        let _ = drain_task.join().await;
        Ok(())
    }
}

async fn process_offload_request(
    offload_req: LocalOffloadRequest,
    block_manager: &VllmBlockManager,
    leader: &Arc<KvbmLeader>,
    kvbm_metrics: KvbmMetrics,
    drain_tx: &mpsc::Sender<DrainItem>,
) -> anyhow::Result<()> {
    let request_id = offload_req.request_id.clone();
    let operation_id = offload_req.operation_id;

    let bypass_cpu_mem = should_bypass_cpu_cache();
    if bypass_cpu_mem {
        kvbm_metrics
            .offload_blocks_d2d
            .inc_by(offload_req.block_ids.len() as u64);
        process_offload_to_storage(
            offload_req,
            block_manager.disk().unwrap(),
            BlockTransferPool::Disk,
            leader,
            &request_id,
            &operation_id,
            "disk",
            None,
        )
        .await?;
    } else {
        kvbm_metrics
            .offload_blocks_d2h
            .inc_by(offload_req.block_ids.len() as u64);
        process_offload_to_storage(
            offload_req,
            block_manager.host().unwrap(),
            BlockTransferPool::Host,
            leader,
            &request_id,
            &operation_id,
            "host",
            Some(drain_tx),
        )
        .await?;
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn process_offload_to_storage<S, L, M>(
    offload_req: LocalOffloadRequest,
    storage_pool: &dyn BlockPool<S, L, M>,
    transfer_pool: BlockTransferPool,
    leader: &Arc<KvbmLeader>,
    _request_id: &str,
    _operation_id: &uuid::Uuid,
    _storage_name: &str,
    drain_tx: Option<&mpsc::Sender<DrainItem>>,
) -> anyhow::Result<()>
where
    S: Storage + NixlRegisterableStorage + 'static,
    L: LocalityProvider + 'static,
    M: BlockMetadata + 'static,
{
    let blocks = storage_pool.allocate_blocks(offload_req.block_ids.len()).await?;
    let token_blocks = offload_req.token_blocks;
    let allocated_block_ids: Vec<usize> = blocks.iter().map(|b| b.block_id()).collect();
    let block_pairs: Vec<(usize, usize)> = offload_req
        .block_ids
        .into_iter()
        .zip(allocated_block_ids.into_iter())
        .collect();

    let mut blocks_to_register = Vec::new();
    let priorities = offload_req.priorities;
    for ((mut mutable_block, token_block), priority) in blocks
        .into_iter()
        .zip(token_blocks.into_iter())
        .zip(priorities.into_iter())
    {
        mutable_block
            .apply_token_block(token_block.clone())
            .map_err(|e| anyhow::anyhow!("failed to apply token block: {:?}", e))?;
        let updated_metadata = mutable_block.metadata().with_priority(priority);
        mutable_block.update_metadata(updated_metadata);
        blocks_to_register.push(mutable_block);
    }

    let sequence_hashes = if transfer_pool == BlockTransferPool::Host && leader.remote_registry_enabled() {
        Some(offload_req.sequence_hashes.clone())
    } else {
        None
    };
    let block_xfer_req = BlockTransferRequest {
        from_pool: BlockTransferPool::Device,
        to_pool: transfer_pool,
        blocks: block_pairs,
        connector_req: Some(LeaderTransferRequest {
            request_id: offload_req.request_id.clone(),
            uuid: offload_req.operation_id,
            requirement: None,
            request_type: RequestType::Scheduled,
            chained: false,
        }),
        sequence_hashes,
    };
    let notify_receiver = leader.transfer_blocks_request(block_xfer_req).await?;
    notify_receiver
        .await
        .map_err(|_| anyhow::anyhow!("offload transfer completion failed"))?;

    let immutable_blocks = storage_pool.register_blocks(blocks_to_register).await?;
    let is_host_transfer = transfer_pool == BlockTransferPool::Host;
    if is_host_transfer && leader.remote_registry_enabled() {
        if let Some(drain_tx) = drain_tx {
            let host_block_ids = immutable_blocks.iter().map(|b| b.block_id()).collect();
            let pin_guard = crate::block_manager::pool::PinGuard::new(immutable_blocks);
            let item = DrainItem {
                request_id: offload_req.request_id.clone(),
                sequence_hashes: offload_req.sequence_hashes.clone(),
                host_block_ids,
                pin_guard,
                block_size: offload_req.block_size,
            };
            let _ = drain_tx.try_send(item);
            return Ok(());
        }
    }
    drop(immutable_blocks);
    Ok(())
}

async fn process_onboard_request(
    onboard_req: LocalOnboardRequest,
    leader: &Arc<KvbmLeader>,
    kvbm_metrics: KvbmMetrics,
) -> anyhow::Result<()> {
    if onboard_req.src_blocks.storage_pool() == BlockTransferPool::Host {
        kvbm_metrics
            .onboard_blocks_h2d
            .inc_by(onboard_req.src_blocks.len() as u64);
    } else if onboard_req.src_blocks.storage_pool() == BlockTransferPool::Disk {
        kvbm_metrics
            .onboard_blocks_d2d
            .inc_by(onboard_req.src_blocks.len() as u64);
    }

    let src_block_ids = onboard_req.src_blocks.block_ids();
    let block_pairs = src_block_ids
        .iter()
        .zip(onboard_req.dst_block_ids.iter())
        .map(|(src, dst)| (*src, *dst))
        .collect::<Vec<_>>();
    let block_xfer_req = BlockTransferRequest {
        from_pool: onboard_req.src_blocks.storage_pool(),
        to_pool: BlockTransferPool::Device,
        blocks: block_pairs,
        connector_req: Some(LeaderTransferRequest {
            request_id: onboard_req.request_id.clone(),
            uuid: onboard_req.operation_id,
            requirement: None,
            request_type: RequestType::Immediate,
            chained: false,
        }),
        sequence_hashes: None,
    };
    let notify_receiver = leader.transfer_blocks_request(block_xfer_req).await?;
    notify_receiver
        .await
        .map_err(|_| anyhow::anyhow!("onboarding transfer completion failed"))?;
    Ok(())
}

async fn process_remote_transfer_request(
    req: RemoteTransferRequest,
    block_manager: &VllmBlockManager,
    leader: &Arc<KvbmLeader>,
    kvbm_metrics: KvbmMetrics,
    pin_registry: &PinRegistry,
    h2o_semaphore: &Arc<Semaphore>,
) -> anyhow::Result<()> {
    let request_id = &req.request_id;
    let operation_id = &req.operation_id;
    let pin_id = req.pin_id;
    let is_h2o = req.is_h2o();
    let onboard_token_blocks = req.token_blocks.clone();

    let release_pin = |pin_registry: &PinRegistry,
                       pin_id: Option<uuid::Uuid>,
                       is_h2o: bool,
                       semaphore: &Arc<Semaphore>| {
        if let Some(guard) = pin_id.and_then(|id| pin_registry.remove(&id)) {
            tracing::debug!(pin_id = ?pin_id, num_blocks = guard.count(), "released pin guard");
            if is_h2o {
                semaphore.add_permits(1);
            }
        }
    };

    if req.is_onboard {
        let inflight_receivers = leader.g4_inflight().check_inflight(&req.sequence_hashes);
        if !inflight_receivers.is_empty() {
            for mut rx in inflight_receivers {
                let _ = tokio::time::timeout(*G4_TRANSFER_TIMEOUT, rx.changed()).await;
            }
            if let Some(host_pool) = block_manager.host() {
                if let Ok(host_matches) = host_pool.match_sequence_hashes(&req.sequence_hashes).await
                    && host_matches.len() == req.sequence_hashes.len()
                {
                    let block_pairs: Vec<(usize, usize)> = host_matches
                        .iter()
                        .zip(req.device_block_ids.iter())
                        .map(|(src, dst)| (src.block_id(), *dst))
                        .collect();
                    let block_xfer_req = BlockTransferRequest {
                        from_pool: BlockTransferPool::Host,
                        to_pool: BlockTransferPool::Device,
                        blocks: block_pairs,
                        connector_req: Some(LeaderTransferRequest {
                            request_id: request_id.clone(),
                            uuid: *operation_id,
                            requirement: None,
                            request_type: RequestType::Immediate,
                            chained: false,
                        }),
                        sequence_hashes: None,
                    };
                    if let Ok(notify_receiver) = leader.transfer_blocks_request(block_xfer_req).await
                        && notify_receiver.await.is_ok()
                    {
                        kvbm_metrics
                            .onboard_blocks_h2d
                            .inc_by(req.sequence_hashes.len() as u64);
                        return Ok(());
                    }
                }
            }
        }
    }

    let _inflight_guard = if req.is_onboard {
        Some(leader.g4_inflight().register(&req.sequence_hashes))
    } else {
        None
    };

    let (hashes_with_positions, filtered_host_ids) = if let Some(handle) = leader.remote_handle() {
        match vllm_int::filter_for_offload(
            &handle,
            &req.sequence_hashes,
            req.host_block_ids.as_deref(),
            leader.worker_id(),
            req.is_onboard,
        )
        .await
        {
            Some(filtered) => filtered,
            None => {
                release_pin(pin_registry, pin_id, is_h2o, h2o_semaphore);
                return Ok(());
            }
        }
    } else {
        let hashes_with_positions: Vec<(u64, u32)> = req
            .sequence_hashes
            .iter()
            .enumerate()
            .map(|(pos, &hash)| (hash, pos as u32))
            .collect();
        (hashes_with_positions, req.host_block_ids.clone())
    };

    let num_blocks = hashes_with_positions.len();
    let storage_config = leader.remote_storage_config().unwrap_or_else(|| RemoteStorageConfig::Object {
        default_bucket: std::env::var("AWS_DEFAULT_BUCKET").ok(),
        endpoint: None,
        region: None,
    });
    let backend_label = match &storage_config {
        RemoteStorageConfig::Object { .. } => "object",
        RemoteStorageConfig::Disk { transfer_flags, .. } => {
            use crate::block_manager::config::DISK_FLAG_GDS_WRITE;
            if transfer_flags & DISK_FLAG_GDS_WRITE != 0 {
                "gds_mt"
            } else {
                "posix"
            }
        }
    };

    let hashes: Vec<u64> = hashes_with_positions.iter().map(|&(h, _)| h).collect();
    let (bounce, device, onboard_host_blocks) = if req.is_h2o() {
        let bounce = filtered_host_ids
            .ok_or_else(|| anyhow::anyhow!("H2R transfer requires host_block_ids"))?;
        (bounce, vec![], None)
    } else {
        let host_pool = block_manager
            .host()
            .ok_or_else(|| anyhow::anyhow!("Host pool not available for bounce buffers"))?;
        let host_blocks = host_pool.allocate_blocks(num_blocks).await?;
        let bounce = host_blocks.iter().map(|b| b.block_id()).collect();
        let device = req.device_block_ids.iter().copied().collect();
        (bounce, device, Some(host_blocks))
    };
    let pipeline = vllm_int::create_transfer_pipeline(
        &hashes,
        &storage_config,
        req.block_size,
        leader.worker_id() as usize,
        leader.world_size(),
        req.is_onboard,
        req.is_h2o(),
        bounce,
        device,
    );

    let is_chained = !req.is_onboard;
    let wire_req = crate::block_manager::distributed::RemoteTransferRequest::new_with_connector_req(
        req.request_id.clone(),
        req.operation_id,
        &pipeline,
        LeaderTransferRequest {
            request_id: request_id.clone(),
            uuid: *operation_id,
            requirement: None,
            request_type: RequestType::Immediate,
            chained: is_chained,
        },
    );
    let notify_receiver = leader.remote_transfer_request(wire_req).await?;
    let transfer_start = Instant::now();
    let transfer_bytes = (num_blocks as u64).saturating_mul(req.block_size as u64);

    let result = match tokio::time::timeout(*G4_TRANSFER_TIMEOUT, notify_receiver).await {
        Ok(Ok(_)) => {
            crate::record_remote_metrics!(
                kvbm_metrics,
                req.is_onboard,
                num_blocks,
                transfer_bytes,
                "success",
                backend_label,
                transfer_start.elapsed().as_secs_f64()
            );

            if !req.is_onboard {
                if let Some(handle) = leader.remote_handle() {
                    vllm_int::register_tp(
                        &handle,
                        &hashes_with_positions,
                        &storage_config,
                        leader.world_size(),
                    )
                    .await;
                }
            }

            if req.is_onboard
                && let (Some(host_blocks), Some(token_blocks)) =
                    (onboard_host_blocks, onboard_token_blocks)
                && let Some(host_pool) = block_manager.host()
            {
                let mut blocks_to_register = Vec::new();
                for (mut block, token_block) in host_blocks.into_iter().zip(token_blocks) {
                    if block.apply_token_block(token_block).is_ok() {
                        blocks_to_register.push(block);
                    }
                }
                if !blocks_to_register.is_empty() {
                    let _ = host_pool.register_blocks(blocks_to_register).await;
                }
            }
            Ok(())
        }
        Ok(Err(_)) => {
            crate::record_remote_metrics!(
                kvbm_metrics,
                req.is_onboard,
                num_blocks,
                transfer_bytes,
                "error",
                backend_label,
                transfer_start.elapsed().as_secs_f64()
            );
            Err(anyhow::anyhow!("Remote transfer completion notification failed"))
        }
        Err(_) => {
            crate::record_remote_metrics!(
                kvbm_metrics,
                req.is_onboard,
                num_blocks,
                transfer_bytes,
                "timeout",
                backend_label,
                transfer_start.elapsed().as_secs_f64()
            );
            Err(anyhow::anyhow!(
                "Remote transfer timed out after {} seconds",
                G4_TRANSFER_TIMEOUT.as_secs()
            ))
        }
    };

    release_pin(pin_registry, pin_id, is_h2o, h2o_semaphore);
    result
}
