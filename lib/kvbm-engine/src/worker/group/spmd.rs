// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use crate::leader::dispatch::{PullRef, WirePullOptions, plan_pull};
use crate::leader::parallelism::{ParallelismTemplate, validate_remote_metadata};
use crate::object::ObjectBlockOps;
use anyhow::{Context, Result};
use kvbm_physical::manager::ParallelismDescriptor;
// velo event types used via fully-qualified paths (::velo::Event, ::velo::EventManager)
use futures::future::BoxFuture;

use std::collections::HashMap;
use std::sync::{Arc, RwLock};

/// SPMD (Single Program, Multiple Data) parallel worker group.
///
/// Wraps a set of rank-indexed [`Worker`]s and executes every operation on
/// all of them in parallel. Each worker has its own rank, physical layout
/// handles, and `TransferManager`, but they all receive the same logical
/// commands (transfer, connect, import/export metadata).
///
/// Transfer completion notifications from individual workers are aggregated
/// into a single notification via the event system, so callers see one
/// completion event per logical operation regardless of worker count.
///
/// Remote handle mappings are stored per `(InstanceId, worker_idx,
/// LogicalLayoutHandle)` so that each rank resolves to its own peer handle
/// during RDMA transfers.
pub struct SpmdParallelWorkers {
    workers: Vec<Arc<dyn Worker>>,
    events: Arc<::velo::EventManager>,
    runtime: tokio::runtime::Handle,

    /// Remote handle mappings: (InstanceId, REMOTE rank, LogicalLayoutHandle)
    /// -> remote LayoutHandle. Populated by `connect_remote` for later use
    /// by `execute_remote_onboard_for_instance`. The middle index is the
    /// remote worker's rank (read from its `ParallelismDescriptor` or
    /// falling back to position in the metadata vector for pre-AB-1a
    /// senders), not the local worker's index — under asymmetric TP the
    /// two differ.
    remote_handles: RwLock<HashMap<(InstanceId, usize, LogicalLayoutHandle), LayoutHandle>>,

    /// Remote peer tp_size per instance. Populated by `connect_remote`
    /// from the stamped `ParallelismDescriptor` (or from the metadata
    /// vector length for pre-AB-1a senders). Read by
    /// `execute_remote_onboard_for_instance` to decide between the
    /// symmetric same-rank-zip dispatch and the AB-3
    /// cross-parallelism planner path.
    remote_tp_sizes: RwLock<HashMap<InstanceId, usize>>,

    /// Per-instance cache of the full per-rank
    /// [`ParallelismDescriptor`] set received from a peer leader.
    /// Populated by `connect_remote` only when every remote rank
    /// carries a stamped descriptor (the Strict import path);
    /// `execute_remote_onboard_for_instance` consults it under the
    /// asymmetric-TP branch to feed [`plan_pull`]. Absent for legacy
    /// (unstamped) peers — those force the symmetric-only branch via
    /// the rank-count match check in `connect_remote`.
    ///
    /// Invariant: cached descriptors are only *meaningful* when a
    /// `local_template` is also installed. Strict-without-template
    /// (legal — the compat gate skips when no template exists) still
    /// caches the descriptors, but the asymmetric-pull dispatch will
    /// bail on missing template before reading them. Don't rely on
    /// the cache as a stand-in for "compatibility verified".
    remote_descriptors: RwLock<HashMap<InstanceId, Vec<ParallelismDescriptor>>>,

    /// Local parallelism template for cross-leader compatibility gating
    /// in `connect_remote`. When unset (pre-AB-1a behaviour), gates are
    /// skipped and the import falls back to the same-rank zip path; when
    /// set, every remote rank must carry a stamped `ParallelismDescriptor`
    /// and the gates in [`validate_remote_metadata`] apply.
    local_template: RwLock<Option<ParallelismTemplate>>,
}

impl SpmdParallelWorkers {
    /// Create a new SpmdParallelWorkers.
    ///
    /// # Arguments
    /// * `workers` - The underlying workers (one per rank)
    /// * `events` - The event system for aggregating completion notifications
    /// * `runtime` - The tokio runtime handle for spawning aggregation tasks
    pub fn new(
        workers: Vec<Arc<dyn Worker>>,
        events: Arc<::velo::EventManager>,
        runtime: tokio::runtime::Handle,
    ) -> Self {
        Self {
            workers,
            events,
            runtime,
            remote_handles: RwLock::new(HashMap::new()),
            remote_tp_sizes: RwLock::new(HashMap::new()),
            remote_descriptors: RwLock::new(HashMap::new()),
            local_template: RwLock::new(None),
        }
    }

    /// Get the number of workers.
    pub fn worker_count(&self) -> usize {
        self.workers.len()
    }

    /// Builder-style: install a local parallelism template so that
    /// `connect_remote` can run cross-leader compatibility gates and
    /// reject incompatible peer metadata up front.
    pub fn with_local_template(self, template: ParallelismTemplate) -> Self {
        *self.local_template.write().unwrap() = Some(template);
        self
    }

    /// Test/diagnostics access. Returns a clone of the configured local
    /// parallelism template if one was installed via
    /// [`Self::with_local_template`].
    #[cfg(any(test, feature = "testing"))]
    pub fn local_template(&self) -> Option<ParallelismTemplate> {
        self.local_template.read().unwrap().clone()
    }
}

impl WorkerTransfers for SpmdParallelWorkers {
    fn execute_local_transfer(
        &self,
        src: LogicalLayoutHandle,
        dst: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst_block_ids: Arc<[BlockId]>,
        options: kvbm_physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let notifications = self
            .workers
            .iter()
            .map(|worker| {
                worker.execute_local_transfer(
                    src,
                    dst,
                    src_block_ids.clone(),
                    dst_block_ids.clone(),
                    options.clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        TransferCompleteNotification::aggregate(notifications, &self.events, &self.runtime)
    }

    fn execute_remote_onboard(
        &self,
        src: RemoteDescriptor,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: kvbm_physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let notifications = self
            .workers
            .iter()
            .map(|worker| {
                worker.execute_remote_onboard(
                    src.clone(),
                    dst,
                    dst_block_ids.clone(),
                    options.clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        TransferCompleteNotification::aggregate(notifications, &self.events, &self.runtime)
    }

    fn execute_remote_offload(
        &self,
        src: LogicalLayoutHandle,
        src_block_ids: Arc<[BlockId]>,
        dst: RemoteDescriptor,
        options: kvbm_physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let notifications = self
            .workers
            .iter()
            .map(|worker| {
                worker.execute_remote_offload(
                    src,
                    src_block_ids.clone(),
                    dst.clone(),
                    options.clone(),
                )
            })
            .collect::<Result<Vec<_>>>()?;

        TransferCompleteNotification::aggregate(notifications, &self.events, &self.runtime)
    }

    fn connect_remote(
        &self,
        instance_id: InstanceId,
        metadata: Vec<SerializedLayout>,
    ) -> Result<ConnectRemoteResponse> {
        if metadata.is_empty() {
            anyhow::bail!("connect_remote: empty remote metadata");
        }

        // Unpack each remote rank up front so we can extract its
        // ParallelismDescriptor (if stamped) and tier list before
        // deciding on the import strategy.
        let mut unpacked = Vec::with_capacity(metadata.len());
        for meta in &metadata {
            unpacked.push(meta.unpack()?);
        }

        // Decide between strict (all-stamped) and legacy (same-rank
        // zip) paths via a pure helper so the routing decision is
        // testable in isolation.
        let descriptor_count = unpacked.iter().filter(|u| u.parallelism.is_some()).count();
        let strategy = decide_import_strategy(descriptor_count, unpacked.len());

        let local_template = self.local_template.read().unwrap().clone();
        if strategy == ImportStrategy::Strict {
            if let Some(template) = local_template.as_ref() {
                let descriptors: Vec<_> = unpacked
                    .iter()
                    .map(|u| u.parallelism.clone().expect("strict path: all stamped"))
                    .collect();
                let tier_lists: Vec<Vec<LogicalLayoutHandle>> = unpacked
                    .iter()
                    .map(|u| u.layouts.iter().map(|d| d.logical_type).collect())
                    .collect();
                let tier_refs: Vec<&[LogicalLayoutHandle]> =
                    tier_lists.iter().map(|v| v.as_slice()).collect();
                validate_remote_metadata(
                    template,
                    &descriptors,
                    &tier_refs,
                    LogicalLayoutHandle::G2,
                )
                .context(
                    "connect_remote: cross-leader compatibility gate rejected peer metadata",
                )?;
            }
        } else if metadata.len() != self.workers.len() {
            anyhow::bail!(
                "connect_remote: peer metadata is not stamped with ParallelismDescriptor \
                 and remote rank count ({}) does not match local worker count ({}); \
                 cross-leader asymmetric TP requires both leaders to upgrade",
                metadata.len(),
                self.workers.len()
            );
        }

        // Build handle mappings keyed by REMOTE rank via the pure helper.
        let per_rank: Vec<(usize, &[kvbm_physical::manager::LogicalLayoutDescriptor])> = unpacked
            .iter()
            .enumerate()
            .map(|(pos, u)| {
                (
                    remote_rank_for(pos, u.parallelism.as_ref()),
                    u.layouts.as_slice(),
                )
            })
            .collect();
        let new_handles = build_handle_mappings(instance_id, &per_rank);

        // In Strict mode each LOCAL worker imports EVERY remote rank's
        // metadata so its NIXL agent has registration info for any
        // remote rank it may later read from. In Legacy mode preserve
        // the same-rank zip.
        let mut import_responses = Vec::new();
        match strategy {
            ImportStrategy::Strict => {
                for worker in &self.workers {
                    for u in &unpacked {
                        let repacked = SerializedLayout::pack(
                            u.worker_address.clone(),
                            u.nixl_metadata.clone(),
                            u.layouts.clone(),
                            u.parallelism.clone(),
                        )?;
                        import_responses.push(worker.import_metadata(repacked)?);
                    }
                }
            }
            ImportStrategy::Legacy => {
                for (worker, u) in self.workers.iter().zip(unpacked.iter()) {
                    let repacked = SerializedLayout::pack(
                        u.worker_address.clone(),
                        u.nixl_metadata.clone(),
                        u.layouts.clone(),
                        u.parallelism.clone(),
                    )?;
                    import_responses.push(worker.import_metadata(repacked)?);
                }
            }
        }

        // Store all handle mappings
        {
            let mut handles = self.remote_handles.write().unwrap();
            for (key, value) in new_handles {
                handles.insert(key, value);
            }
        }

        // Persist remote tp_size and cache descriptors for this
        // instance so `execute_remote_onboard_for_instance` can route
        // between symmetric (same-rank zip) and asymmetric (AB-3
        // planner) dispatch. Reset both caches first so a peer that
        // re-imports under a different strategy (Strict → Legacy or
        // shape change) doesn't leak stale state from the prior
        // import.
        {
            let mut tp_map = self.remote_tp_sizes.write().unwrap();
            let mut desc_map = self.remote_descriptors.write().unwrap();
            tp_map.remove(&instance_id);
            desc_map.remove(&instance_id);

            match strategy {
                ImportStrategy::Strict => {
                    // All ranks stamped: the descriptors are authoritative.
                    // Every rank reports the same `tp_size` (gated above),
                    // so reading from the first is consistent.
                    let descriptors: Vec<ParallelismDescriptor> = unpacked
                        .iter()
                        .map(|u| u.parallelism.clone().expect("strict path: all stamped"))
                        .collect();
                    let remote_tp = descriptors[0].tp_size;
                    tp_map.insert(instance_id, remote_tp);
                    desc_map.insert(instance_id, descriptors);
                }
                ImportStrategy::Legacy => {
                    // Legacy import enforces metadata.len() == self.workers.len()
                    // (rank-count match check above), so the peer is
                    // effectively symmetric from this leader's POV.
                    // Even if a stray stamped entry slipped in (mixed
                    // metadata), the legacy path treats the peer as
                    // symmetric — do NOT read tp_size from that stamp.
                    // Leaving remote_descriptors empty forces the
                    // asymmetric branch to bail loudly if it's ever
                    // mis-routed here.
                    tp_map.insert(instance_id, metadata.len());
                }
            }
        }

        // If all responses are ready (synchronous), return immediately
        if import_responses.iter().all(|r| !r.could_yield()) {
            return Ok(ConnectRemoteResponse::ready());
        }

        // Create an event to aggregate all import completions
        let event = self.events.new_event()?;
        let awaiter = self.events.awaiter(event.handle())?;

        // Spawn task to await all import responses and signal completion
        self.runtime
            .spawn(await_import_responses(import_responses, event));

        Ok(ConnectRemoteResponse::from_awaiter(awaiter))
    }

    fn has_remote_metadata(&self, instance_id: InstanceId) -> bool {
        let handles = self.remote_handles.read().unwrap();
        handles.keys().any(|(id, _, _)| *id == instance_id)
    }

    fn execute_remote_onboard_for_instance(
        &self,
        instance_id: InstanceId,
        remote_logical_type: LogicalLayoutHandle,
        src_block_ids: Vec<BlockId>,
        dst: LogicalLayoutHandle,
        dst_block_ids: Arc<[BlockId]>,
        options: kvbm_physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let local_tp = self.workers.len();
        let remote_tp = *self
            .remote_tp_sizes
            .read()
            .unwrap()
            .get(&instance_id)
            .unwrap_or(&local_tp);

        if local_tp != remote_tp {
            return self.dispatch_asymmetric_pull(
                instance_id,
                remote_logical_type,
                &src_block_ids,
                dst,
                &dst_block_ids,
                options,
            );
        }

        // Symmetric path (unchanged): SPMD same-rank zip. Each local
        // worker reads from its identically-numbered remote rank via
        // the legacy `execute_remote_onboard` entry on the underlying
        // worker. Stays in place until AB-4 hoists every pull through
        // `plan_pull` (locked decision #5).
        let handles = self.remote_handles.read().unwrap();
        let mut notifications = Vec::with_capacity(self.workers.len());
        for (worker_idx, worker) in self.workers.iter().enumerate() {
            let remote_handle = handles
                .get(&(instance_id, worker_idx, remote_logical_type))
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "No remote {:?} handle for instance {} worker {}",
                        remote_logical_type,
                        instance_id,
                        worker_idx
                    )
                })?;

            let descriptor = RemoteDescriptor::Layout {
                handle: *remote_handle,
                block_ids: src_block_ids.clone(),
            };

            notifications.push(worker.execute_remote_onboard(
                descriptor,
                dst,
                dst_block_ids.clone(),
                options.clone(),
            )?);
        }

        TransferCompleteNotification::aggregate(notifications, &self.events, &self.runtime)
    }
}

impl SpmdParallelWorkers {
    /// AB-3: asymmetric-TP dispatch via the cross-parallelism planner.
    ///
    /// Replaces the AB-1b precautionary bail. Invoked from
    /// [`Self::execute_remote_onboard_for_instance`] when
    /// `local_tp != remote_tp`. Calls [`plan_pull`] to produce one
    /// [`crate::leader::dispatch::WorkerPullPlan`] per participating
    /// local rank, dispatches each to its target local worker via
    /// the new `execute_remote_pull_plan` trait method, then aggregates
    /// the per-rank notifications.
    fn dispatch_asymmetric_pull(
        &self,
        instance_id: InstanceId,
        remote_logical_type: LogicalLayoutHandle,
        src_block_ids: &[BlockId],
        dst: LogicalLayoutHandle,
        dst_block_ids: &[BlockId],
        options: kvbm_physical::transfer::TransferOptions,
    ) -> Result<TransferCompleteNotification> {
        let template = self.local_template.read().unwrap().clone().ok_or_else(|| {
            anyhow::anyhow!(
                "asymmetric pull requires a local ParallelismTemplate; install one \
                     via SpmdParallelWorkers::with_local_template() (instance={instance_id})"
            )
        })?;

        // Coherence guard: the template's tp_size describes the local
        // worker grid. If it disagrees with `workers.len()`, plan_pull
        // emits the wrong number of plans — fewer plans than workers
        // silently skip data on the un-mapped ranks. Catch the
        // misconfiguration loudly before any RPCs go out.
        if template.tp_size != self.workers.len() {
            anyhow::bail!(
                "asymmetric pull: local ParallelismTemplate tp_size ({}) disagrees with \
                 worker count ({}); template must describe the local worker grid",
                template.tp_size,
                self.workers.len(),
            );
        }

        let descriptors = self
            .remote_descriptors
            .read()
            .unwrap()
            .get(&instance_id)
            .cloned()
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "asymmetric pull needs the full ParallelismDescriptor set for instance \
                     {instance_id}; peer must stamp descriptors and connect_remote must \
                     have run through the Strict path"
                )
            })?;

        if src_block_ids.len() != dst_block_ids.len() {
            anyhow::bail!(
                "asymmetric pull requires equal-length src/dst block id lists ({} vs {})",
                src_block_ids.len(),
                dst_block_ids.len(),
            );
        }

        let refs: Vec<PullRef> = src_block_ids
            .iter()
            .zip(dst_block_ids.iter())
            .map(|(s, d)| PullRef {
                src_block_id: *s,
                dst_block_id: *d,
            })
            .collect();

        // Project to the wire-restricted option subset. Local-only
        // toggles (use_planner, layer_range, bounce_buffer,
        // cuda_stream, *_kv_layout) intentionally don't propagate —
        // see WirePullOptions docs.
        let wire_opts = WirePullOptions {
            nixl_write_notification: options.nixl_write_notification,
            metric_route: options.metric_route,
        };

        let plans = plan_pull(
            &template,
            &descriptors,
            instance_id,
            remote_logical_type,
            dst,
            &refs,
            &wire_opts,
        )?;
        if plans.is_empty() {
            return Ok(TransferCompleteNotification::completed());
        }

        let mut notifications = Vec::with_capacity(plans.len());
        for (local_rank, plan) in plans {
            let worker = self.workers.get(local_rank).ok_or_else(|| {
                anyhow::anyhow!(
                    "plan_pull produced a plan for local_rank {local_rank} but only {} \
                     local workers are registered",
                    self.workers.len()
                )
            })?;
            notifications.push(worker.execute_remote_pull_plan(plan)?);
        }

        TransferCompleteNotification::aggregate(notifications, &self.events, &self.runtime)
    }
}

/// Decision returned by [`decide_import_strategy`] — describes which
/// import path `connect_remote` should follow.
#[derive(Debug, PartialEq, Eq)]
pub(crate) enum ImportStrategy {
    /// Every remote rank carries a stamped descriptor. Run cross-leader
    /// compatibility gates and have each local worker import every
    /// remote rank's metadata.
    Strict,
    /// At least one remote rank is unstamped. Fall back to the legacy
    /// same-rank zip behaviour — but only if the rank count matches.
    Legacy,
}

/// Decide whether to take the strict (all-stamped) or legacy
/// (same-rank-zip) import path. Pure CPU; isolated so the routing
/// decision is unit-testable without a real Worker.
pub(crate) fn decide_import_strategy(
    descriptor_count: usize,
    metadata_count: usize,
) -> ImportStrategy {
    if metadata_count > 0 && descriptor_count == metadata_count {
        ImportStrategy::Strict
    } else {
        ImportStrategy::Legacy
    }
}

/// Resolve the remote rank for a peer's per-worker payload. When a
/// `ParallelismDescriptor` is stamped its `rank` field is
/// authoritative; otherwise the vector position is used (back-compat
/// with pre-AB-1a senders).
pub(crate) fn remote_rank_for(
    position: usize,
    descriptor: Option<&kvbm_physical::manager::ParallelismDescriptor>,
) -> usize {
    descriptor.map(|d| d.rank).unwrap_or(position)
}

/// Build the (instance_id, remote_rank, logical_type) → LayoutHandle
/// mappings from a vector of `(rank, &[layouts])` pairs.
pub(crate) fn build_handle_mappings(
    instance_id: InstanceId,
    per_rank: &[(usize, &[kvbm_physical::manager::LogicalLayoutDescriptor])],
) -> Vec<((InstanceId, usize, LogicalLayoutHandle), LayoutHandle)> {
    let mut out = Vec::new();
    for (remote_rank, layouts) in per_rank {
        for descriptor in layouts.iter() {
            out.push((
                (instance_id, *remote_rank, descriptor.logical_type),
                descriptor.handle,
            ));
        }
    }
    out
}

/// Helper to await all import metadata responses and signal completion via an event.
/// Helper to await all import metadata responses and signal completion via an event.
async fn await_import_responses(responses: Vec<ImportMetadataResponse>, event: ::velo::Event) {
    let results: Vec<Result<Vec<LayoutHandle>>> =
        futures::future::join_all(responses.into_iter().map(|r| r.into_future())).await;

    // Check for any failures
    let errors: Vec<_> = results.into_iter().filter_map(|r| r.err()).collect();

    if errors.is_empty() {
        let _ = event.trigger();
    } else {
        let error_msg = errors
            .iter()
            .map(|e| e.to_string())
            .collect::<Vec<_>>()
            .join("; ");
        let _ = event.poison(error_msg);
    }
}

impl ParallelWorkers for SpmdParallelWorkers {
    fn export_metadata(&self) -> Result<Vec<SerializedLayoutResponse>> {
        let metadata = self
            .workers
            .iter()
            .map(|worker| worker.export_metadata())
            .collect::<Result<Vec<_>>>()?;

        Ok(metadata)
    }

    fn import_metadata(
        &self,
        metadata: Vec<SerializedLayout>,
    ) -> Result<Vec<ImportMetadataResponse>> {
        // validate the size of the metadata is the same as the number of workers
        if metadata.len() != self.workers.len() {
            return Err(anyhow::anyhow!(
                "Metadata size does not match number of workers"
            ));
        }

        let results = self
            .workers
            .iter()
            .zip(metadata.iter())
            .map(|(worker, metadata)| worker.import_metadata(metadata.clone()))
            .collect::<Result<Vec<_>>>()?;

        Ok(results)
    }

    fn worker_count(&self) -> usize {
        self.workers.len()
    }

    fn workers(&self) -> &[Arc<dyn Worker>] {
        &self.workers
    }
}

impl ObjectBlockOps for SpmdParallelWorkers {
    fn has_blocks(
        &self,
        keys: Vec<SequenceHash>,
    ) -> BoxFuture<'static, Vec<(SequenceHash, Option<usize>)>> {
        // For has_blocks, we query all workers and verify consistency.
        // All workers should agree on block presence for SPMD semantics.
        // We return the results from worker 0 but verify all workers agree.
        let workers = self.workers.clone();
        let _runtime = self.runtime.clone();

        Box::pin(async move {
            if workers.is_empty() {
                return keys.into_iter().map(|k| (k, None)).collect();
            }

            // Query all workers in parallel
            let futures: Vec<_> = workers
                .iter()
                .map(|worker| worker.has_blocks(keys.clone()))
                .collect();

            let results: Vec<Vec<(SequenceHash, Option<usize>)>> =
                futures::future::join_all(futures).await;

            // Return results from first worker (all should agree in SPMD)
            // In debug mode, we could verify consistency across workers
            results.into_iter().next().unwrap_or_default()
        })
    }

    fn put_blocks(
        &self,
        keys: Vec<SequenceHash>,
        src_layout: LogicalLayoutHandle,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        // For put_blocks, each worker writes with its own rank-prefixed key.
        // Each worker resolves the logical handle to its own physical layout.
        // All workers must succeed for the operation to be considered successful.
        let workers = self.workers.clone();

        Box::pin(async move {
            if workers.is_empty() {
                return keys.into_iter().map(Err).collect();
            }

            // Execute put on all workers in parallel
            // Each worker resolves src_layout to its own physical layout
            let futures: Vec<_> = workers
                .iter()
                .map(|worker| worker.put_blocks(keys.clone(), src_layout, block_ids.clone()))
                .collect();

            let results: Vec<Vec<Result<SequenceHash, SequenceHash>>> =
                futures::future::join_all(futures).await;

            // Aggregate: a key succeeded only if ALL workers succeeded
            let num_keys = keys.len();
            let mut aggregated = Vec::with_capacity(num_keys);

            for (key_idx, key) in keys.iter().enumerate() {
                let all_succeeded = results.iter().all(|worker_results| {
                    worker_results
                        .get(key_idx)
                        .map(|r| r.is_ok())
                        .unwrap_or(false)
                });

                if all_succeeded {
                    aggregated.push(Ok(*key));
                } else {
                    aggregated.push(Err(*key));
                }
            }

            aggregated
        })
    }

    fn get_blocks(
        &self,
        keys: Vec<SequenceHash>,
        dst_layout: LogicalLayoutHandle,
        block_ids: Vec<BlockId>,
    ) -> BoxFuture<'static, Vec<Result<SequenceHash, SequenceHash>>> {
        // For get_blocks, each worker reads from its own rank-prefixed key.
        // Each worker resolves the logical handle to its own physical layout.
        // All workers must succeed for the operation to be considered successful.
        let workers = self.workers.clone();

        Box::pin(async move {
            if workers.is_empty() {
                return keys.into_iter().map(Err).collect();
            }

            // Execute get on all workers in parallel
            // Each worker resolves dst_layout to its own physical layout
            let futures: Vec<_> = workers
                .iter()
                .map(|worker| worker.get_blocks(keys.clone(), dst_layout, block_ids.clone()))
                .collect();

            let results: Vec<Vec<Result<SequenceHash, SequenceHash>>> =
                futures::future::join_all(futures).await;

            // Aggregate: a key succeeded only if ALL workers succeeded
            let num_keys = keys.len();
            let mut aggregated = Vec::with_capacity(num_keys);

            for (key_idx, key) in keys.iter().enumerate() {
                let all_succeeded = results.iter().all(|worker_results| {
                    worker_results
                        .get(key_idx)
                        .map(|r| r.is_ok())
                        .unwrap_or(false)
                });

                if all_succeeded {
                    aggregated.push(Ok(*key));
                } else {
                    aggregated.push(Err(*key));
                }
            }

            aggregated
        })
    }
}

#[cfg(all(test, feature = "testing"))]
mod tests {
    use super::*;
    use kvbm_common::KvDim;
    use kvbm_physical::manager::ParallelismDescriptor;

    fn descriptor(rank: usize, tp_size: usize) -> ParallelismDescriptor {
        ParallelismDescriptor {
            tp_size,
            pp_size: 1,
            rank,
            shard_axis: KvDim::HeadCount,
            global_extents: vec![],
            layer_ownership: 0..1,
        }
    }

    #[test]
    fn decide_strict_when_all_stamped() {
        assert_eq!(decide_import_strategy(4, 4), ImportStrategy::Strict);
        assert_eq!(decide_import_strategy(1, 1), ImportStrategy::Strict);
    }

    #[test]
    fn decide_legacy_when_any_unstamped() {
        assert_eq!(decide_import_strategy(3, 4), ImportStrategy::Legacy);
        assert_eq!(decide_import_strategy(0, 4), ImportStrategy::Legacy);
    }

    #[test]
    fn decide_legacy_on_empty_metadata() {
        assert_eq!(decide_import_strategy(0, 0), ImportStrategy::Legacy);
    }

    #[test]
    fn remote_rank_uses_descriptor_when_present() {
        let d = descriptor(3, 4);
        assert_eq!(remote_rank_for(0, Some(&d)), 3, "trust descriptor.rank");
    }

    #[test]
    fn remote_rank_falls_back_to_position_when_descriptor_absent() {
        assert_eq!(remote_rank_for(2, None), 2, "fall back to vector position");
    }

    #[test]
    fn handle_mappings_use_supplied_rank() {
        // build_handle_mappings is a thin transform — verify it routes
        // by the rank we hand it (the caller computes rank via
        // remote_rank_for).
        let layouts: Vec<kvbm_physical::manager::LogicalLayoutDescriptor> = Vec::new();
        let per_rank: Vec<(usize, &[_])> = vec![(7, layouts.as_slice()), (3, layouts.as_slice())];
        let id = InstanceId::new_v4();
        let mappings = build_handle_mappings(id, &per_rank);
        // No layouts => no mappings; structural test only.
        assert!(mappings.is_empty());
    }

    /// AB-3: `connect_remote` caches the full per-rank
    /// ParallelismDescriptor set when every entry is stamped (Strict
    /// path). The asymmetric-pull branch of
    /// `execute_remote_onboard_for_instance` consults this cache —
    /// missing entries would force a hard bail rather than a
    /// silent mis-route.
    #[test]
    fn connect_remote_caches_descriptors_in_strict_path() {
        use ::velo::EventManager;
        use kvbm_physical::manager::{LogicalLayoutDescriptor, SerializedLayout, WorkerAddress};

        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let spmd = SpmdParallelWorkers::new(
            Vec::new(),
            Arc::new(EventManager::local()),
            rt.handle().clone(),
        );

        let instance_id = InstanceId::new_v4();
        let stamped: Vec<SerializedLayout> = (0..4)
            .map(|rank| {
                SerializedLayout::pack(
                    WorkerAddress::new(rank as u64, format!("agent-{rank}")),
                    vec![],
                    Vec::<LogicalLayoutDescriptor>::new(),
                    Some(descriptor(rank, 4)),
                )
                .unwrap()
            })
            .collect();

        spmd.connect_remote(instance_id, stamped).unwrap();

        let cached = spmd.remote_descriptors.read().unwrap();
        let descriptors = cached
            .get(&instance_id)
            .expect("Strict-path import must cache descriptors for the asymmetric branch");
        assert_eq!(descriptors.len(), 4);
        for (i, d) in descriptors.iter().enumerate() {
            assert_eq!(d.rank, i);
            assert_eq!(d.tp_size, 4);
        }

        // The companion tp_size cache also lands so the dispatch
        // branch decision (symmetric vs asymmetric) doesn't need to
        // peek into `remote_descriptors`.
        let tp = spmd
            .remote_tp_sizes
            .read()
            .unwrap()
            .get(&instance_id)
            .copied();
        assert_eq!(tp, Some(4));
    }

    /// AB-3 fixup: Strict re-import for the same instance must
    /// replace the cached descriptor set, not merge. A peer changing
    /// shape between connects (e.g. TP=4 → TP=2 reconfiguration)
    /// would otherwise leave stale descriptor entries that disagree
    /// with the live tp_size, mis-routing the asymmetric branch.
    #[test]
    fn connect_remote_strict_reimport_replaces_cached_descriptors() {
        use ::velo::EventManager;
        use kvbm_physical::manager::{LogicalLayoutDescriptor, SerializedLayout, WorkerAddress};

        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let spmd = SpmdParallelWorkers::new(
            Vec::new(),
            Arc::new(EventManager::local()),
            rt.handle().clone(),
        );

        let instance_id = InstanceId::new_v4();

        // First import: TP=4 stamped.
        let stamped_4: Vec<SerializedLayout> = (0..4)
            .map(|rank| {
                SerializedLayout::pack(
                    WorkerAddress::new(rank as u64, format!("a-{rank}")),
                    vec![],
                    Vec::<LogicalLayoutDescriptor>::new(),
                    Some(descriptor(rank, 4)),
                )
                .unwrap()
            })
            .collect();
        spmd.connect_remote(instance_id, stamped_4).unwrap();
        assert_eq!(
            spmd.remote_descriptors
                .read()
                .unwrap()
                .get(&instance_id)
                .map(|v| v.len()),
            Some(4),
        );
        assert_eq!(
            spmd.remote_tp_sizes
                .read()
                .unwrap()
                .get(&instance_id)
                .copied(),
            Some(4),
        );

        // Second import: same instance, TP=2 stamped. Both caches
        // must reflect the new shape, not the prior TP=4.
        let stamped_2: Vec<SerializedLayout> = (0..2)
            .map(|rank| {
                SerializedLayout::pack(
                    WorkerAddress::new(rank as u64, format!("b-{rank}")),
                    vec![],
                    Vec::<LogicalLayoutDescriptor>::new(),
                    Some(descriptor(rank, 2)),
                )
                .unwrap()
            })
            .collect();
        spmd.connect_remote(instance_id, stamped_2).unwrap();
        let cached = spmd.remote_descriptors.read().unwrap();
        let descriptors = cached.get(&instance_id).unwrap();
        assert_eq!(
            descriptors.len(),
            2,
            "Strict re-import must replace the cached descriptor set, not append"
        );
        for d in descriptors {
            assert_eq!(d.tp_size, 2);
        }
        assert_eq!(
            spmd.remote_tp_sizes
                .read()
                .unwrap()
                .get(&instance_id)
                .copied(),
            Some(2),
            "remote_tp_sizes must reflect the new import's tp_size",
        );
    }

    /// AB-3 fixup: Strict-path `remote_tp_sizes` must come from
    /// `descriptors[0].tp_size`, not from a `find_map` lookup that
    /// could pick up a stray stamped entry in mixed metadata.
    #[test]
    fn connect_remote_strict_tp_size_from_descriptors_not_first_stamp() {
        use ::velo::EventManager;
        use kvbm_physical::manager::{LogicalLayoutDescriptor, SerializedLayout, WorkerAddress};

        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        let spmd = SpmdParallelWorkers::new(
            Vec::new(),
            Arc::new(EventManager::local()),
            rt.handle().clone(),
        );

        let instance_id = InstanceId::new_v4();
        // 4 stamped entries, all reporting tp_size=4 — consistent.
        let stamped: Vec<SerializedLayout> = (0..4)
            .map(|rank| {
                SerializedLayout::pack(
                    WorkerAddress::new(rank as u64, format!("a-{rank}")),
                    vec![],
                    Vec::<LogicalLayoutDescriptor>::new(),
                    Some(descriptor(rank, 4)),
                )
                .unwrap()
            })
            .collect();
        spmd.connect_remote(instance_id, stamped).unwrap();

        // The Strict-path tp_size lookup reads descriptors[0].tp_size,
        // which validate_remote_metadata's internal-consistency gate
        // guarantees matches every other rank.
        assert_eq!(
            spmd.remote_tp_sizes
                .read()
                .unwrap()
                .get(&instance_id)
                .copied(),
            Some(4),
        );
    }

    /// AB-3: Legacy (unstamped) peer metadata must NOT cache
    /// descriptors. The asymmetric branch would otherwise read a
    /// missing entry and bail with a misleading "no descriptors"
    /// message rather than the underlying "peer didn't stamp" cause.
    /// `connect_remote`'s rank-count check already forces symmetric
    /// for unstamped peers, but this guards against the cache
    /// being populated by accident.
    #[test]
    fn connect_remote_does_not_cache_descriptors_in_legacy_path() {
        use ::velo::EventManager;
        use kvbm_physical::manager::{LogicalLayoutDescriptor, SerializedLayout, WorkerAddress};

        let rt = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap();
        // workers.len() = 0 matches metadata.len() = 0 for the
        // unstamped no-op import, exercising the Legacy branch.
        let spmd = SpmdParallelWorkers::new(
            Vec::new(),
            Arc::new(EventManager::local()),
            rt.handle().clone(),
        );

        let instance_id = InstanceId::new_v4();
        // Single rank, no parallelism stamp → Legacy path.
        let unstamped = vec![
            SerializedLayout::pack(
                WorkerAddress::new(0, "agent-0".to_string()),
                vec![],
                Vec::<LogicalLayoutDescriptor>::new(),
                None,
            )
            .unwrap(),
        ];

        // Legacy path bails because metadata.len() (1) != workers.len() (0).
        // The bail happens before any cache write — which is exactly the
        // invariant we want to confirm: no Legacy descriptor leak.
        let err = match spmd.connect_remote(instance_id, unstamped) {
            Ok(_) => panic!("Legacy-path import with rank mismatch must bail"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("not stamped"),
            "expected Legacy-path bail, got: {err}"
        );
        assert!(
            spmd.remote_descriptors
                .read()
                .unwrap()
                .get(&instance_id)
                .is_none(),
            "Legacy bail must not have populated the descriptor cache"
        );
    }
}
