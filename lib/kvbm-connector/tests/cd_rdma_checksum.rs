// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use kvbm_connector::connector::leader::disagg::testing::InMemoryRemotePrefillQueue;
use kvbm_connector::connector::leader::disagg::{
    AlwaysRemote, ConditionalDisaggPolicy, PolicyInputs, RemotePrefillCoordinator,
    RemotePrefillStatus, SessionBlocks, VeloPrefillSessionFactory,
};
use kvbm_engine::BlockId;
use kvbm_engine::disagg::{BlockSetRequest, DisaggSession, HashSelection, UnpinRequest};
use kvbm_engine::testing::distributed::create_instance_leader_pair_with_workers;
use kvbm_engine::testing::physical;
use kvbm_physical::transfer::{FillPattern, StorageKind};
use tokio::time::timeout;
use velo::backend::tcp::TcpTransportBuilder;

const NUM_WORKERS: usize = 2;
const LAYOUT_BLOCKS: usize = 16;
const TEST_BLOCKS: usize = 4;
const BLOCK_SIZE: usize = 4;
const NUM_LAYERS: usize = 2;
const OUTER_DIM: usize = 1;
const PAGE_SIZE: usize = 4;
const INNER_DIM: usize = 64;
const DTYPE_WIDTH: usize = 2;
const MANAGER_BLOCKS: usize = 16;

fn new_velo_transport() -> Arc<velo::backend::tcp::TcpTransport> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)
            .unwrap()
            .build()
            .unwrap(),
    )
}

async fn new_velo() -> Arc<velo::Velo> {
    velo::Velo::builder()
        .add_transport(new_velo_transport())
        .build()
        .await
        .unwrap()
}

async fn with_timeout<T>(
    label: &'static str,
    fut: impl std::future::Future<Output = T>,
) -> Result<T> {
    timeout(Duration::from_secs(10), fut)
        .await
        .map_err(|_| anyhow::anyhow!("{label} timed out"))
}

async fn wait_attached(coord: &RemotePrefillCoordinator, request_id: &str) -> Result<()> {
    with_timeout("coordinator attach", async move {
        loop {
            match coord.status_for(request_id) {
                Some(RemotePrefillStatus::Attached) => return Ok(()),
                Some(RemotePrefillStatus::Failed) => {
                    anyhow::bail!("coordinator failed before attach")
                }
                _ => tokio::time::sleep(Duration::from_millis(10)).await,
            }
        }
    })
    .await?
}

fn policy_inputs() -> PolicyInputs {
    PolicyInputs {
        total_tokens: 64,
        num_computed_tokens: TEST_BLOCKS * BLOCK_SIZE,
        num_connector_tokens: TEST_BLOCKS * BLOCK_SIZE,
        transfer_params: None,
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
#[ignore = "requires RDMA-capable UCX/CUDA worker setup"]
async fn connector_remote_prefill_pulls_decode_block_sets_by_checksum() {
    let layout_config = physical::custom_config(
        LAYOUT_BLOCKS,
        NUM_LAYERS,
        OUTER_DIM,
        PAGE_SIZE,
        INNER_DIM,
        DTYPE_WIDTH,
    );
    let pair = create_instance_leader_pair_with_workers(
        MANAGER_BLOCKS,
        BLOCK_SIZE,
        NUM_WORKERS,
        &layout_config,
        StorageKind::Pinned,
    )
    .await
    .expect("leader pair with workers");

    let (src_block_ids, hashes) = pair
        .decode
        .populate_g2_blocks(TEST_BLOCKS, BLOCK_SIZE, 0)
        .expect("populate decode blocks");
    let ready_blocks = pair.decode.leader.g2_manager().match_blocks(&hashes);

    let dst_block_ids: Vec<BlockId> =
        (TEST_BLOCKS as BlockId..(TEST_BLOCKS * 2) as BlockId).collect();
    let mut decode_checksums_before = Vec::new();
    for worker in &pair.decode.workers {
        decode_checksums_before.push(
            worker
                .fill_g2_blocks(&src_block_ids, FillPattern::Constant(0xAA))
                .expect("fill decode source blocks"),
        );
    }
    for worker in &pair.prefill.workers {
        worker
            .fill_g2_blocks(&dst_block_ids, FillPattern::Constant(0xBB))
            .expect("fill prefill destination blocks");
    }

    let d_velo = new_velo().await;
    let p_velo = new_velo().await;
    d_velo.register_peer(p_velo.peer_info()).unwrap();
    p_velo.register_peer(d_velo.peer_info()).unwrap();

    let queue = InMemoryRemotePrefillQueue::new();
    let coord = RemotePrefillCoordinator::with_attach_timeout(
        Arc::new(AlwaysRemote) as Arc<dyn ConditionalDisaggPolicy>,
        VeloPrefillSessionFactory::new(Arc::clone(&d_velo)),
        queue.clone(),
        tokio::runtime::Handle::current(),
        Duration::from_secs(30),
    );

    coord
        .begin_remote_prefill(
            "req-rdma-checksum",
            &policy_inputs(),
            pair.decode.instance_id,
            SessionBlocks::new(ready_blocks, Vec::new()),
            hashes.clone(),
            vec![1, 2, 3, 4],
        )
        .expect("begin remote prefill");

    let request = queue
        .snapshot()
        .pop()
        .expect("decode enqueued remote prefill request");
    let prefill_session = with_timeout(
        "prefill attach",
        DisaggSession::attach_prefill(
            Arc::clone(&p_velo),
            request.session_id,
            request.decode_endpoint.as_ref().expect("decode endpoint"),
        ),
    )
    .await
    .expect("prefill attach timeout")
    .expect("prefill attaches to decode session");
    wait_attached(&coord, "req-rdma-checksum")
        .await
        .expect("coordinator sees attached session");

    let response = with_timeout(
        "block-set request",
        prefill_session.request_block_sets(BlockSetRequest {
            request_id: "initial-blocks".to_string(),
            hashes: HashSelection::All,
        }),
    )
    .await
    .expect("block-set request timeout")
    .expect("request block sets");

    let notification = with_timeout(
        "block-set pull start",
        pair.prefill.leader.pull_remote_block_sets(
            pair.decode.instance_id,
            &response.ready,
            &dst_block_ids,
        ),
    )
    .await
    .expect("block-set pull start timeout")
    .expect("start block-set pull");

    with_timeout(
        "block-set pull completion",
        async move { notification.await },
    )
    .await
    .expect("block-set pull completion timeout")
    .expect("block-set pull completes");

    with_timeout(
        "unpin after pull",
        prefill_session.request_unpin_from_prefill(UnpinRequest {
            request_id: "unpin-after-pull".to_string(),
            hashes: HashSelection::All,
        }),
    )
    .await
    .expect("unpin timeout")
    .expect("decode acks unpin after pull");

    for (idx, (decode_worker, prefill_worker)) in pair
        .decode
        .workers
        .iter()
        .zip(pair.prefill.workers.iter())
        .enumerate()
    {
        let decode_after = decode_worker
            .compute_g2_checksums(&src_block_ids)
            .expect("decode checksums after pull");
        let prefill_after = prefill_worker
            .compute_g2_checksums(&dst_block_ids)
            .expect("prefill checksums after pull");

        for block_idx in 0..TEST_BLOCKS {
            let src = src_block_ids[block_idx];
            let dst = dst_block_ids[block_idx];
            assert_eq!(
                decode_checksums_before[idx][&src], decode_after[&src],
                "decode source block was modified"
            );
            assert_eq!(
                decode_checksums_before[idx][&src], prefill_after[&dst],
                "prefill destination does not match decode source"
            );
        }
    }
}
