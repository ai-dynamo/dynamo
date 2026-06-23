// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::mem::ManuallyDrop;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Result;
use dynamo_llm::block_manager::KvBlockManager;
use dynamo_llm::block_manager::block::BasicMetadata;
use dynamo_llm::block_manager::block::data::logical::distributed_leader_worker::DistributedLeaderWorkerResources;
use dynamo_llm::block_manager::config::{
    BlockParallelismStrategy, KvBlockManagerConfig, KvManagerLayoutConfig, KvManagerModelConfig,
    KvManagerRuntimeConfig,
};
use dynamo_llm::block_manager::distributed::{
    KvbmLeader, KvbmLeaderConfig, KvbmLeaderNumBlocksConfig, KvbmWorker, KvbmWorkerConfig,
};
use dynamo_llm::block_manager::locality::Logical;
use dynamo_llm::block_manager::storage::{
    DeviceAllocator, Storage, StorageAllocator,
    torch::{TorchDevice, TorchTensor},
};
use tokio_util::sync::CancellationToken;

const NUM_BLOCKS: usize = 8;
const BLOCK_SIZE: usize = 4;

#[derive(Clone, Debug)]
struct MockTensor {
    ptr: u64,
    size: usize,
    shape: Vec<usize>,
}

impl MockTensor {
    fn new(shape: Vec<usize>, device_id: usize, dtype_width_bytes: usize) -> Self {
        let allocator = DeviceAllocator::new(device_id).unwrap();
        let size = shape.iter().product::<usize>() * dtype_width_bytes;
        let device_storage = ManuallyDrop::new(allocator.allocate(size).unwrap());

        Self {
            ptr: device_storage.addr(),
            size,
            shape,
        }
    }
}

impl TorchTensor for MockTensor {
    fn device(&self) -> TorchDevice {
        TorchDevice::Cuda(0)
    }

    fn data_ptr(&self) -> u64 {
        self.ptr
    }

    fn size_bytes(&self) -> usize {
        self.size
    }

    fn shape(&self) -> Vec<usize> {
        self.shape.clone()
    }

    fn stride(&self) -> Vec<usize> {
        let mut stride = vec![1];
        for i in (0..self.shape.len() - 1).rev() {
            stride.push(stride.last().unwrap() * self.shape[i]);
        }
        stride.reverse();
        stride
    }
}

#[derive(Clone, Debug)]
struct Args {
    num_workers: usize,
    page_size: usize,
    dtype_width_bytes: usize,
    device_id: usize,
    leader_pub_url: String,
    leader_ack_url: String,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            num_workers: 1,
            page_size: 32,
            dtype_width_bytes: 2,
            device_id: 0,
            leader_pub_url: "tcp://127.0.0.1:56211".to_string(),
            leader_ack_url: "tcp://127.0.0.1:56212".to_string(),
        }
    }
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = Self::default();
        let mut it = std::env::args().skip(1);

        while let Some(flag) = it.next() {
            match flag.as_str() {
                "--num-workers" => {
                    args.num_workers = it.next().unwrap().parse()?;
                }
                "--page-size" => {
                    args.page_size = it.next().unwrap().parse()?;
                }
                "--dtype-width-bytes" => {
                    args.dtype_width_bytes = it.next().unwrap().parse()?;
                }
                "--device-id" => {
                    args.device_id = it.next().unwrap().parse()?;
                }
                "--leader-pub-url" => args.leader_pub_url = it.next().unwrap(),
                "--leader-ack-url" => args.leader_ack_url = it.next().unwrap(),
                "--help" | "-h" => {
                    println!(
                        "kvbm_nixl_transfer_smoke
  --num-workers <n>               worker count (default 1)
  --page-size <n>                 worker page size (default 32)
  --dtype-width-bytes <n>         dtype width bytes (default 2)
  --device-id <id>                CUDA device id for worker tensors (default 0)
  --leader-pub-url <url>          ZMQ leader pub url
  --leader-ack-url <url>          ZMQ leader ack url"
                    );
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown flag: {other}"),
            }
        }

        Ok(args)
    }
}

async fn build_leader_and_workers(args: &Args) -> Result<(KvbmLeader, Vec<KvbmWorker>)> {
    let mut workers = Vec::with_capacity(args.num_workers);
    let cancel_token = CancellationToken::new();

    for worker_id in 0..args.num_workers {
        let tensors: Vec<Arc<dyn TorchTensor>> = vec![Arc::new(MockTensor::new(
            vec![NUM_BLOCKS, 1, 2, args.page_size, 128],
            args.device_id,
            args.dtype_width_bytes,
        ))];

        let config = KvbmWorkerConfig::builder()
            .cancel_token(cancel_token.child_token())
            .num_device_blocks(NUM_BLOCKS)
            .page_size(args.page_size)
            .dtype_width_bytes(args.dtype_width_bytes)
            .tensors(tensors)
            .device_id(worker_id)
            .leader_pub_url(args.leader_pub_url.clone())
            .leader_ack_url(args.leader_ack_url.clone())
            .build()?;

        workers.push(KvbmWorker::new(config, false).await?);
    }

    let leader_config = KvbmLeaderConfig::builder()
        .world_size(args.num_workers)
        .host_blocks_config(KvbmLeaderNumBlocksConfig {
            cache_size_in_gb: 0.0,
            num_blocks_overriden: NUM_BLOCKS,
        })
        .disk_blocks_config(KvbmLeaderNumBlocksConfig {
            cache_size_in_gb: 0.0,
            // Keep this smoke on the host/device path only. Configuring disk
            // blocks causes the worker to eagerly initialize the GDS backend.
            num_blocks_overriden: 0,
        })
        .leader_pub_url(args.leader_pub_url.clone())
        .leader_ack_url(args.leader_ack_url.clone())
        .build()?;

    let leader = KvbmLeader::new(leader_config).await?;
    anyhow::ensure!(
        leader.wait_worker_sync_ready().await,
        "leader/worker handshake did not become ready"
    );
    Ok((leader, workers))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse()?;
    let (leader, _workers) = build_leader_and_workers(&args).await?;

    let cancel_token = CancellationToken::new();
    let config = KvBlockManagerConfig::builder()
        .runtime(
            KvManagerRuntimeConfig::builder()
                .worker_id(0)
                .cancellation_token(cancel_token.clone())
                .build()?,
        )
        .model(
            KvManagerModelConfig::builder()
                .num_layers(1)
                .outer_dim(1)
                .page_size(BLOCK_SIZE)
                .inner_dim(1)
                .build()?,
        )
        .device_layout(
            KvManagerLayoutConfig::builder()
                .num_blocks(NUM_BLOCKS)
                .logical(Some(BlockParallelismStrategy::LeaderWorkerSharded))
                .build()?,
        )
        .host_layout(
            KvManagerLayoutConfig::builder()
                .num_blocks(NUM_BLOCKS)
                .logical(Some(BlockParallelismStrategy::LeaderWorkerSharded))
                .build()?,
        )
        .build()?;

    let resources =
        DistributedLeaderWorkerResources::new(Some(Arc::new(leader)), cancel_token.child_token())?;

    let block_manager =
        KvBlockManager::<Logical<DistributedLeaderWorkerResources>, BasicMetadata>::new(
            config, resources,
        )
        .await?;

    let device_pool = block_manager.device().unwrap();
    let host_pool = block_manager.host().unwrap();
    let mut device_blocks = device_pool.allocate_blocks(NUM_BLOCKS).await?;
    let mut sequence_hashes = Vec::with_capacity(NUM_BLOCKS);

    for block in &mut device_blocks {
        block.init_sequence(42)?;
        for _ in 0..BLOCK_SIZE {
            block.add_token(42)?;
        }
        block.commit()?;
        sequence_hashes.push(block.sequence_hash()?);
    }

    let immutable_device_blocks = device_pool.register_blocks(device_blocks).await?;
    tokio::time::sleep(Duration::from_millis(100)).await;

    let host_blocks = host_pool.match_sequence_hashes(&sequence_hashes).await?;

    anyhow::ensure!(
        host_blocks.len() == NUM_BLOCKS,
        "expected {} host blocks, got {}",
        NUM_BLOCKS,
        host_blocks.len()
    );

    drop(immutable_device_blocks);
    tokio::time::sleep(Duration::from_millis(100)).await;
    let _ = device_pool.allocate_blocks(NUM_BLOCKS).await?;
    tokio::time::sleep(Duration::from_millis(100)).await;

    let new_device_blocks = block_manager.onboard_blocks(host_blocks, None).await??;
    anyhow::ensure!(
        new_device_blocks.len() == NUM_BLOCKS,
        "expected {} onboarded device blocks, got {}",
        NUM_BLOCKS,
        new_device_blocks.len()
    );

    println!(
        "nixl smoke transfer complete: device->host {} blocks, host->device {} blocks",
        NUM_BLOCKS,
        new_device_blocks.len()
    );
    println!("sequence_hashes={:?}", sequence_hashes);
    Ok(())
}
