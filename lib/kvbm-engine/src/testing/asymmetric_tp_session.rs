// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Asymmetric-TP disagg round-trip integration test (AB-6).
//!
//! Drives the new stamped cross-parallelism path end-to-end:
//!
//! 1. TP=4 holder fills 8 G2 blocks with worker-distinct bytes and opens
//!    a disagg [`crate::disagg::session::Session`].
//! 2. TP=2 puller attaches, RDMA-pulls all 8 blocks via
//!    [`crate::disagg::session::Session::pull`] (routes through
//!    `VeloSession::pull` → `InstanceLeader::rdma_pull_with_opts` →
//!    `plan_pull` → multi-shard worker RPC →
//!    `TransferManager::execute_transfer_selection`).
//! 3. Both sides finalize session A; lifecycle Detached events confirm
//!    teardown; holder's pins release.
//! 4. Holder overwrites the original 8 blocks with a sentinel pattern.
//!    The wipe assertion checks the overwrite took, ensuring Phase 9
//!    cannot accidentally pass against still-intact source memory.
//! 5. Puller completes the returned mutables with the SAME token sequence
//!    (asserting hash parity) and registers them under its own G2
//!    manager, then opens session B.
//! 6. Puller advertises its 8 blocks in two waves of 4 via
//!    `commit` + `make_available`. Holder pulls each wave into NEW G2
//!    destination block ids (read dynamically from
//!    [`kvbm_logical::blocks::MutableBlock::block_id`]).
//! 7. Final byte-equality check: per holder worker, the round-tripped
//!    checksums must match the Phase 1 BASELINE checksums positionally.
//!    Cross-TP checksums are not comparable directly, so the test relies
//!    on lossless round-trip — the only way the bytes equal is if
//!    `plan_pull`/`build_axis_intersections` slice the HeadCount axis
//!    correctly in both directions.
//!
//! Every `await` on a stream/pull is wrapped in `tokio::time::timeout`
//! so a hang surfaces with file:line context instead of silently
//! consuming the CI slot.

use std::collections::BTreeSet;
use std::net::TcpListener;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Result, anyhow};

use kvbm_common::KvDim;
use kvbm_config::ParallelismMode;
use kvbm_logical::manager::BlockManager;
use kvbm_physical::layout::{KvBlockLayout, LayoutConfig, PhysicalLayout};
use kvbm_physical::manager::TransferManager;
use kvbm_physical::transfer::{NixlAgent, StorageKind};

use velo::Transport;
use velo::transports::tcp::TcpTransportBuilder;

use crate::leader::InstanceLeader;
use crate::leader::parallelism::ParallelismTemplate;
use crate::p2p::session::VeloSessionFactory;
use crate::worker::{DirectWorker, Worker};
use crate::{G2, G3, InstanceId};

use super::distributed::TestWorker;
use super::{managers, physical};

// -----------------------------------------------------------------------
// Shape constants — both sides share globals except HeadCount per worker
// -----------------------------------------------------------------------

const BLOCK_SIZE: usize = 4;
const NUM_LAYERS: usize = 2;
const OUTER_DIM: usize = 1;
const PAGE_SIZE: usize = 4;
const HEAD_SIZE: usize = 16;
const GLOBAL_HEAD_COUNT: usize = 16;
const DTYPE_WIDTH: usize = 2;
const MANAGER_BLOCKS: usize = 32;

const TP_HOLDER: usize = 4;
const TP_PULLER: usize = 2;

fn build_layout_config(tp: usize) -> LayoutConfig {
    let per_worker_heads = GLOBAL_HEAD_COUNT / tp;
    let inner_dim = per_worker_heads * HEAD_SIZE;
    LayoutConfig::builder()
        .num_blocks(MANAGER_BLOCKS)
        .num_layers(NUM_LAYERS)
        .outer_dim(OUTER_DIM)
        .page_size(PAGE_SIZE)
        .inner_dim(inner_dim)
        .num_heads(Some(per_worker_heads))
        .dtype_width_bytes(DTYPE_WIDTH)
        .build()
        .expect("layout config")
}

// -----------------------------------------------------------------------
// Velo + worker plumbing
// -----------------------------------------------------------------------

async fn new_velo() -> Result<Arc<velo::Velo>> {
    let listener = TcpListener::bind("127.0.0.1:0")?;
    let transport: Arc<dyn Transport> = Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)?
            .build()?,
    );
    velo::Velo::builder().add_transport(transport).build().await
}

// Default `create_fc_layout_with_config` leaves `KvBlockLayout::Unknown`,
// which the cross-TP `layout_view()` rejects. This helper sets
// `OperationalNHD` explicitly so the planner can drive HeadCount-axis
// slicing.
fn create_fc_layout_nhd(
    agent: NixlAgent,
    storage: StorageKind,
    config: LayoutConfig,
) -> Result<PhysicalLayout> {
    let builder = PhysicalLayout::builder(agent)
        .with_config(config)
        .with_block_layout(KvBlockLayout::OperationalNHD)
        .fully_contiguous();
    let layout = match storage {
        StorageKind::Pinned => builder.allocate_pinned(None).build()?,
        StorageKind::Device(d) => builder.allocate_device(d).build()?,
        StorageKind::System => builder.allocate_system().build()?,
        StorageKind::Disk(_) => {
            anyhow::bail!("disk storage unsupported for asymmetric TP test")
        }
    };
    Ok(layout)
}

fn create_test_worker_nhd(
    instance_id: InstanceId,
    agent_name: &str,
    layout_config: &LayoutConfig,
    storage: StorageKind,
) -> Result<TestWorker> {
    let worker_id = instance_id.worker_id().as_u64();
    let event_system = velo::EventManager::local();
    let test_agent = physical::TestAgentBuilder::new(agent_name)
        .require_backend("UCX")
        .build()?;
    let agent = test_agent.into_nixl_agent();
    let manager = TransferManager::builder()
        .event_system(Arc::new(event_system))
        .nixl_agent(agent.clone())
        .cuda_device_id(0)
        .build()?;
    let layout = create_fc_layout_nhd(agent, storage, layout_config.clone())?;
    let g2_handle = manager.register_layout(layout)?;
    let direct_worker = DirectWorker::builder()
        .manager(manager.clone())
        .g2_handle(g2_handle)
        .build()?;
    Ok(TestWorker {
        instance_id,
        worker_id,
        worker: Arc::new(direct_worker),
        manager: Arc::new(manager),
        g2_handle,
    })
}

// -----------------------------------------------------------------------
// Fixture
// -----------------------------------------------------------------------

pub struct AsymmetricSide {
    pub velo: Arc<velo::Velo>,
    pub leader: Arc<InstanceLeader>,
    pub factory: Arc<VeloSessionFactory>,
    pub g2_manager: Arc<BlockManager<G2>>,
    pub workers: Vec<TestWorker>,
    pub instance_id: InstanceId,
    pub tp: usize,
}

pub struct AsymmetricPair {
    pub holder: AsymmetricSide,
    pub puller: AsymmetricSide,
}

async fn build_side(
    velo: Arc<velo::Velo>,
    tp: usize,
    agent_name_prefix: &str,
    storage: StorageKind,
) -> Result<AsymmetricSide> {
    let layout_config = build_layout_config(tp);

    let mut workers = Vec::with_capacity(tp);
    for rank in 0..tp {
        let instance_id = InstanceId::new_v4();
        let agent_name = format!(
            "{agent_name_prefix}-w{rank}-{}",
            instance_id.worker_id().as_u64()
        );
        workers.push(create_test_worker_nhd(
            instance_id,
            &agent_name,
            &layout_config,
            storage,
        )?);
    }
    let worker_refs: Vec<Arc<dyn Worker>> = workers
        .iter()
        .map(|w| w.worker.clone() as Arc<dyn Worker>)
        .collect();

    let registry = managers::TestRegistryBuilder::new().build();
    let g2_manager = Arc::new(
        managers::TestManagerBuilder::<G2>::new()
            .block_count(MANAGER_BLOCKS)
            .block_size(BLOCK_SIZE)
            .registry(registry.clone())
            .build(),
    );
    let g3_manager = Arc::new(
        managers::TestManagerBuilder::<G3>::new()
            .block_count(MANAGER_BLOCKS)
            .block_size(BLOCK_SIZE)
            .registry(registry.clone())
            .build(),
    );

    let template = ParallelismTemplate::from_layout_config(
        &layout_config,
        ParallelismMode::TensorParallel,
        tp,
    )?;

    let leader = InstanceLeader::builder()
        .messenger(velo.messenger().clone())
        .registry(registry)
        .g2_manager(g2_manager.clone())
        .g3_manager(g3_manager)
        .workers(worker_refs)
        .parallelism_template(template)
        .build()?;
    leader.register_handlers()?;
    let leader = Arc::new(leader);

    // Verify stamping took. `assemble_export_metadata` is the supported
    // way to inspect ParallelismDescriptor without a builder getter.
    let exported = leader.assemble_export_metadata().await?;
    anyhow::ensure!(
        exported.len() == tp,
        "expected {tp} SerializedLayout entries, got {}",
        exported.len()
    );
    let mut seen_ranks: BTreeSet<usize> = BTreeSet::new();
    for (i, sl) in exported.iter().enumerate() {
        let unpacked = sl.unpack()?;
        let p = unpacked.parallelism.ok_or_else(|| {
            anyhow!("worker {i}: ParallelismDescriptor missing — stamping did not take")
        })?;
        anyhow::ensure!(
            p.tp_size == tp,
            "worker {i}: stamped tp_size {} != expected {tp}",
            p.tp_size
        );
        anyhow::ensure!(p.pp_size == 1, "worker {i}: pp_size != 1");
        anyhow::ensure!(
            p.shard_axis == KvDim::HeadCount,
            "worker {i}: shard_axis != HeadCount"
        );
        let hc = p
            .global_extents
            .iter()
            .find(|(k, _)| *k == KvDim::HeadCount)
            .map(|(_, v)| *v);
        anyhow::ensure!(
            hc == Some(GLOBAL_HEAD_COUNT),
            "worker {i}: global HeadCount mismatch (got {hc:?}, expected {GLOBAL_HEAD_COUNT})"
        );
        anyhow::ensure!(
            seen_ranks.insert(p.rank),
            "worker {i}: duplicate rank {}",
            p.rank
        );
        anyhow::ensure!(
            p.rank < tp,
            "worker {i}: rank {} out of range 0..{tp}",
            p.rank
        );
    }
    anyhow::ensure!(
        seen_ranks.len() == tp,
        "expected ranks 0..{tp}, got {seen_ranks:?}"
    );

    let factory = VeloSessionFactory::new(
        Arc::clone(&velo),
        Arc::clone(&leader),
        tokio::runtime::Handle::current(),
    );

    let instance_id = velo.instance_id();
    Ok(AsymmetricSide {
        velo,
        leader,
        factory,
        g2_manager,
        workers,
        instance_id,
        tp,
    })
}

/// Build a connected pair of leaders with mismatched TP (4 holder /
/// 2 puller), real velo + UCX wiring, stamped parallelism templates,
/// and per-side `VeloSessionFactory` instances.
pub async fn create_asymmetric_leader_pair_with_workers(
    storage: StorageKind,
) -> Result<AsymmetricPair> {
    let velo_holder = new_velo().await?;
    let velo_puller = new_velo().await?;
    velo_holder.register_peer(velo_puller.peer_info())?;
    velo_puller.register_peer(velo_holder.peer_info())?;
    tokio::time::sleep(Duration::from_millis(100)).await;

    let holder = build_side(velo_holder, TP_HOLDER, "holder", storage).await?;
    let puller = build_side(velo_puller, TP_PULLER, "puller", storage).await?;

    Ok(AsymmetricPair { holder, puller })
}

// -----------------------------------------------------------------------
// Tests
// -----------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::p2p::session::{
        AvailabilityDelta, CommittedBlock, LifecycleEvent, LifecycleStream, SessionFactory,
    };
    use crate::testing::token_blocks;
    use crate::{BlockId, SequenceHash};
    use futures::StreamExt;
    use kvbm_logical::blocks::ImmutableBlock;
    use kvbm_physical::transfer::{BlockChecksum, FillPattern};
    use std::collections::HashMap;

    const INITIAL_BLOCKS: usize = 8;
    const WAVE_SIZE: usize = 4;

    /// Fill each block on `worker` with a CONSTANT byte derived from
    /// `(worker_rank, block_id, position_in_list)` so a TP-rank or
    /// block-id mixup never produces a passing checksum.
    ///
    /// `FillPattern::Sequential` is uniform across same-shaped TP workers
    /// at the same `(block_id, layer_id)` and would mask such a bug.
    fn fill_blocks_worker_distinct(
        worker: &TestWorker,
        worker_rank: usize,
        block_ids: &[BlockId],
    ) -> Result<HashMap<BlockId, BlockChecksum>> {
        let mut all = HashMap::new();
        for (i, &block_id) in block_ids.iter().enumerate() {
            let byte = (worker_rank as u8)
                .wrapping_mul(53)
                .wrapping_add((block_id as u8).wrapping_mul(17))
                .wrapping_add((i as u8).wrapping_mul(7))
                .wrapping_add(1);
            let cks = worker.fill_g2_blocks(&[block_id], FillPattern::Constant(byte))?;
            all.extend(cks);
        }
        Ok(all)
    }

    fn timeout_short() -> Duration {
        Duration::from_secs(15)
    }

    fn timeout_long() -> Duration {
        Duration::from_secs(60)
    }

    async fn wait_for_detached(side_name: &str, stream: &mut LifecycleStream) -> Result<()> {
        loop {
            let evt = tokio::time::timeout(timeout_short(), stream.next())
                .await
                .map_err(|_| anyhow!("{side_name}: timeout waiting for lifecycle Detached"))?
                .ok_or_else(|| anyhow!("{side_name}: lifecycle stream ended without Detached"))?;
            match evt {
                LifecycleEvent::Detached { .. } => return Ok(()),
                LifecycleEvent::Failed { reason } => {
                    anyhow::bail!("{side_name}: session entered Failed: {reason}")
                }
                LifecycleEvent::Attached { .. } => continue,
            }
        }
    }

    /// Wait for `Attached` on a lifecycle stream. The holder side's
    /// spawned `Frame::Attach` handler runs `ensure_remote_metadata`
    /// for the puller before pushing `Attached` — so blocking on this
    /// event guarantees the import has completed and any subsequent
    /// holder→puller `ensure_remote_metadata` call (e.g. a later
    /// reverse-direction attach) hits the cache instead of racing.
    async fn wait_for_attached(side_name: &str, stream: &mut LifecycleStream) -> Result<()> {
        let evt = tokio::time::timeout(timeout_short(), stream.next())
            .await
            .map_err(|_| anyhow!("{side_name}: timeout waiting for lifecycle Attached"))?
            .ok_or_else(|| anyhow!("{side_name}: lifecycle stream ended without Attached"))?;
        match evt {
            LifecycleEvent::Attached { .. } => Ok(()),
            LifecycleEvent::Failed { reason } => {
                anyhow::bail!("{side_name}: session entered Failed before Attached: {reason}")
            }
            LifecycleEvent::Detached { .. } => {
                anyhow::bail!("{side_name}: Detached before Attached")
            }
        }
    }

    async fn next_available(
        label: &str,
        stream: &mut crate::p2p::session::AvailabilityStream,
    ) -> Result<Vec<CommittedBlock>> {
        let delta = tokio::time::timeout(timeout_short(), stream.next())
            .await
            .map_err(|_| anyhow!("{label}: availability stream timeout"))?
            .ok_or_else(|| anyhow!("{label}: availability stream ended"))?;
        match delta {
            AvailabilityDelta::Available(b) => Ok(b),
            AvailabilityDelta::Drained => {
                anyhow::bail!("{label}: Drained before Available")
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 8)]
    async fn test_asymmetric_tp_session_round_trip() -> Result<()> {
        let pair = create_asymmetric_leader_pair_with_workers(StorageKind::Pinned).await?;
        let AsymmetricPair { holder, puller } = pair;

        // -----------------------------------------------------------
        // Phase 1: holder allocates, fills (worker-distinct), captures baselines
        // -----------------------------------------------------------
        let mut h_src_mutables = holder
            .g2_manager
            .allocate_blocks(INITIAL_BLOCKS)
            .ok_or_else(|| anyhow!("holder: failed to allocate {INITIAL_BLOCKS} mutables"))?;
        let h_src_ids: Vec<BlockId> = h_src_mutables.iter().map(|m| m.block_id()).collect();

        let mut baselines: Vec<HashMap<BlockId, BlockChecksum>> = Vec::with_capacity(holder.tp);
        for (rank, worker) in holder.workers.iter().enumerate() {
            baselines.push(fill_blocks_worker_distinct(worker, rank, &h_src_ids)?);
        }
        // Cross-rank sanity: the same block_id must have DIFFERENT
        // checksums across holder workers, else a rank mixup would slip.
        for &id in &h_src_ids {
            for ra in 0..holder.tp {
                for rb in (ra + 1)..holder.tp {
                    anyhow::ensure!(
                        baselines[ra][&id] != baselines[rb][&id],
                        "baseline collision: rank {ra} == rank {rb} at block {id}"
                    );
                }
            }
        }

        let token_seq = token_blocks::create_token_sequence(INITIAL_BLOCKS, BLOCK_SIZE, 0);
        let completes: Vec<_> = h_src_mutables
            .drain(..)
            .zip(token_seq.blocks().iter())
            .map(|(m, tb)| {
                m.complete(tb)
                    .map_err(|e| anyhow!("holder: complete: {e:?}"))
            })
            .collect::<Result<Vec<_>>>()?;
        let h_immutables: Vec<ImmutableBlock<G2>> = holder.g2_manager.register_blocks(completes);
        let h_hashes: Vec<SequenceHash> = h_immutables.iter().map(|b| b.sequence_hash()).collect();
        anyhow::ensure!(
            h_hashes.len() == INITIAL_BLOCKS,
            "register_blocks dropped some blocks: got {}",
            h_hashes.len()
        );

        // -----------------------------------------------------------
        // Phase 2: open session A, attach, subscribe to availability + lifecycle
        // -----------------------------------------------------------
        let session_a_id = uuid::Uuid::new_v4();
        let h_session_a = holder.factory.open(session_a_id)?;
        // Subscribe to holder's lifecycle BEFORE the puller attaches
        // so the Attached event (pushed AFTER the spawned
        // `ensure_remote_metadata(puller_id)` completes) is captured.
        // Lifecycle streams are subscribe-once and replay prior
        // events, so subscribing here also covers the later Detached
        // wait in Phase 4.
        let mut h_lc_a = h_session_a.lifecycle();
        let h_endpoint = h_session_a
            .endpoint()
            .ok_or_else(|| anyhow!("holder session_a: missing endpoint"))?;
        let p_session_a = tokio::time::timeout(
            timeout_short(),
            puller
                .factory
                .attach(session_a_id, holder.instance_id, h_endpoint),
        )
        .await
        .map_err(|_| anyhow!("attach session_a timed out"))??;
        let mut p_lc_a = p_session_a.lifecycle();
        let mut p_avail_a = p_session_a.availability();

        // Wait for holder.Attached so the holder's spawned reverse
        // `ensure_remote_metadata(puller_id)` is guaranteed complete
        // before Phase 6 opens session B. Without this gate, the
        // session_b attach could race the still-in-flight session_a
        // import and trip the per-manager `loaded_remotes` duplicate
        // check.
        wait_for_attached("holder.session_a", &mut h_lc_a).await?;

        // -----------------------------------------------------------
        // Phase 3: holder publishes, puller pulls (synced via .await)
        // -----------------------------------------------------------
        h_session_a.commit(h_hashes.clone())?;
        h_session_a.make_available(h_immutables)?;
        h_session_a.finish_commits()?;
        h_session_a.finish_availability()?;

        let avail_blocks = next_available("session_a", &mut p_avail_a).await?;
        anyhow::ensure!(
            avail_blocks.len() == INITIAL_BLOCKS,
            "session_a: expected {INITIAL_BLOCKS} blocks, got {}",
            avail_blocks.len()
        );
        let avail_hashes: Vec<SequenceHash> = avail_blocks.iter().map(|b| b.hash).collect();
        anyhow::ensure!(
            avail_hashes == h_hashes,
            "session_a: available hashes don't match holder publication order"
        );

        let p_dst_mutables = puller
            .g2_manager
            .allocate_blocks(INITIAL_BLOCKS)
            .ok_or_else(|| anyhow!("puller: alloc {INITIAL_BLOCKS} dst"))?;
        let p_dst_ids: Vec<BlockId> = p_dst_mutables.iter().map(|m| m.block_id()).collect();

        let returned_mutables = tokio::time::timeout(
            timeout_long(),
            p_session_a.pull(h_hashes.clone(), p_dst_mutables),
        )
        .await
        .map_err(|_| anyhow!("pull session_a timed out"))??;
        anyhow::ensure!(
            returned_mutables.len() == INITIAL_BLOCKS,
            "session_a: pull returned {}",
            returned_mutables.len()
        );
        let returned_ids: Vec<BlockId> = returned_mutables.iter().map(|m| m.block_id()).collect();
        anyhow::ensure!(
            returned_ids == p_dst_ids,
            "returned mutables have unexpected block_id ordering"
        );

        // -----------------------------------------------------------
        // Phase 4: mutually finalize session A; wait for Detached; drop
        // -----------------------------------------------------------
        h_session_a.finalize(None);
        p_session_a.finalize(None);
        let (h_res, p_res) = tokio::join!(
            wait_for_detached("holder.session_a", &mut h_lc_a),
            wait_for_detached("puller.session_a", &mut p_lc_a),
        );
        h_res?;
        p_res?;
        drop(h_session_a);
        drop(p_session_a);

        // -----------------------------------------------------------
        // Phase 5: holder wipes source blocks + asserts wipe took
        // -----------------------------------------------------------
        for worker in &holder.workers {
            worker.fill_g2_blocks(&h_src_ids, FillPattern::Constant(0xEE))?;
        }
        for (rank, worker) in holder.workers.iter().enumerate() {
            let post = worker.compute_g2_checksums(&h_src_ids)?;
            for &id in &h_src_ids {
                anyhow::ensure!(
                    post[&id] != baselines[rank][&id],
                    "wipe did not change bytes on holder rank {rank} block {id}"
                );
            }
        }

        // -----------------------------------------------------------
        // Phase 6: puller completes returned mutables (hash parity),
        //          registers, opens session B; holder attaches.
        // -----------------------------------------------------------
        let p_completes: Vec<_> = returned_mutables
            .into_iter()
            .zip(token_seq.blocks().iter())
            .map(|(m, tb)| {
                m.complete(tb)
                    .map_err(|e| anyhow!("puller: complete: {e:?}"))
            })
            .collect::<Result<Vec<_>>>()?;
        for (i, c) in p_completes.iter().enumerate() {
            anyhow::ensure!(
                c.sequence_hash() == h_hashes[i],
                "puller: completed block {i} hash {:?} != holder hash {:?}",
                c.sequence_hash(),
                h_hashes[i]
            );
        }
        let p_immutables: Vec<ImmutableBlock<G2>> = puller.g2_manager.register_blocks(p_completes);
        anyhow::ensure!(
            p_immutables.len() == INITIAL_BLOCKS,
            "puller register_blocks: dropped some"
        );

        let session_b_id = uuid::Uuid::new_v4();
        let p_session_b = puller.factory.open(session_b_id)?;
        let mut p_lc_b = p_session_b.lifecycle();
        let p_endpoint_b = p_session_b
            .endpoint()
            .ok_or_else(|| anyhow!("puller session_b: missing endpoint"))?;
        let h_session_b = tokio::time::timeout(
            timeout_short(),
            holder
                .factory
                .attach(session_b_id, puller.instance_id, p_endpoint_b),
        )
        .await
        .map_err(|_| anyhow!("attach session_b timed out"))??;
        let mut h_lc_b = h_session_b.lifecycle();
        let mut h_avail_b = h_session_b.availability();
        wait_for_attached("puller.session_b", &mut p_lc_b).await?;

        // -----------------------------------------------------------
        // Phase 7: wave 1 — puller publishes 4, holder pulls 4
        // -----------------------------------------------------------
        let (wave1_imm, wave2_imm) = {
            let mut iter = p_immutables.into_iter();
            let w1: Vec<_> = (&mut iter).take(WAVE_SIZE).collect();
            let w2: Vec<_> = iter.collect();
            (w1, w2)
        };
        let wave1_hashes: Vec<SequenceHash> = h_hashes[..WAVE_SIZE].to_vec();
        let wave2_hashes: Vec<SequenceHash> = h_hashes[WAVE_SIZE..].to_vec();

        p_session_b.commit(wave1_hashes.clone())?;
        p_session_b.make_available(wave1_imm)?;

        let w1_blocks = next_available("session_b wave1", &mut h_avail_b).await?;
        anyhow::ensure!(
            w1_blocks.len() == WAVE_SIZE,
            "wave1: expected {WAVE_SIZE} blocks, got {}",
            w1_blocks.len()
        );

        let rt_w1_mutables = holder
            .g2_manager
            .allocate_blocks(WAVE_SIZE)
            .ok_or_else(|| anyhow!("holder: alloc wave1 rt"))?;
        let rt_w1_ids: Vec<BlockId> = rt_w1_mutables.iter().map(|m| m.block_id()).collect();
        // None of the wave1 dst block_ids should collide with the
        // (now-wiped) source ids — the manager pool keeps unique ids.
        for &id in &rt_w1_ids {
            anyhow::ensure!(
                !h_src_ids.contains(&id),
                "wave1 rt_id {id} collides with holder src ids"
            );
        }
        let _w1_back = tokio::time::timeout(
            timeout_long(),
            h_session_b.pull(wave1_hashes.clone(), rt_w1_mutables),
        )
        .await
        .map_err(|_| anyhow!("pull wave1 timed out"))??;

        // -----------------------------------------------------------
        // Phase 8: wave 2 — puller publishes 4, holder pulls 4
        // -----------------------------------------------------------
        p_session_b.commit(wave2_hashes.clone())?;
        p_session_b.make_available(wave2_imm)?;
        p_session_b.finish_commits()?;
        p_session_b.finish_availability()?;

        let w2_blocks = next_available("session_b wave2", &mut h_avail_b).await?;
        anyhow::ensure!(
            w2_blocks.len() == WAVE_SIZE,
            "wave2: expected {WAVE_SIZE} blocks, got {}",
            w2_blocks.len()
        );

        let rt_w2_mutables = holder
            .g2_manager
            .allocate_blocks(WAVE_SIZE)
            .ok_or_else(|| anyhow!("holder: alloc wave2 rt"))?;
        let rt_w2_ids: Vec<BlockId> = rt_w2_mutables.iter().map(|m| m.block_id()).collect();
        for &id in &rt_w2_ids {
            anyhow::ensure!(
                !h_src_ids.contains(&id),
                "wave2 rt_id {id} collides with holder src ids"
            );
            anyhow::ensure!(
                !rt_w1_ids.contains(&id),
                "wave2 rt_id {id} collides with wave1 rt ids"
            );
        }
        let _w2_back = tokio::time::timeout(
            timeout_long(),
            h_session_b.pull(wave2_hashes.clone(), rt_w2_mutables),
        )
        .await
        .map_err(|_| anyhow!("pull wave2 timed out"))??;

        // -----------------------------------------------------------
        // Phase 9: per-worker byte equality vs CAPTURED baselines
        // -----------------------------------------------------------
        let rt_ids: Vec<BlockId> = rt_w1_ids.iter().chain(rt_w2_ids.iter()).copied().collect();
        anyhow::ensure!(
            rt_ids.len() == INITIAL_BLOCKS,
            "rt_ids count {} != {INITIAL_BLOCKS}",
            rt_ids.len()
        );
        for (rank, worker) in holder.workers.iter().enumerate() {
            let post = worker.compute_g2_checksums(&rt_ids)?;
            for i in 0..INITIAL_BLOCKS {
                let baseline_id = h_src_ids[i];
                let rt_id = rt_ids[i];
                let baseline_cks = &baselines[rank][&baseline_id];
                let rt_cks = post.get(&rt_id).ok_or_else(|| {
                    anyhow!("holder rank {rank}: checksum for block {rt_id} missing")
                })?;
                anyhow::ensure!(
                    baseline_cks == rt_cks,
                    "holder rank {rank}: round-trip block {rt_id} bytes differ from baseline block {baseline_id} (baseline {baseline_cks:?} vs rt {rt_cks:?})"
                );
            }
        }

        // -----------------------------------------------------------
        // Phase 10: mutually finalize session B; wait for Detached
        // -----------------------------------------------------------
        h_session_b.finalize(None);
        p_session_b.finalize(None);
        let (hr, pr) = tokio::join!(
            wait_for_detached("holder.session_b", &mut h_lc_b),
            wait_for_detached("puller.session_b", &mut p_lc_b),
        );
        hr?;
        pr?;

        Ok(())
    }
}
