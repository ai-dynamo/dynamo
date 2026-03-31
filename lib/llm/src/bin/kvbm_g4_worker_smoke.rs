// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::mem::ManuallyDrop;
use std::sync::Arc;

use anyhow::{Context, Result};
use dynamo_llm::block_manager::Storage;
use dynamo_llm::block_manager::distributed::{
    G4BlockIndex, G4FetchRequest, G4FetchResponse, G4HealthResponse, G4OfferRequest,
    G4OfferResponse, G4PutBlock, G4PutPayloadRequest, G4QueryHit, G4QueryRequest, G4StorageWorker,
    G4TransferBlock, KvbmLeader, KvbmLeaderConfig, KvbmLeaderNumBlocksConfig, KvbmWorker,
    KvbmWorkerConfig, route_g4_put_blocks_by_owner, route_g4_sequence_hashes_by_owner,
    route_g4_transfer_blocks_by_owner, select_g4_owner,
};
use dynamo_llm::block_manager::storage::{
    DeviceAllocator, StorageAllocator,
    torch::{TorchDevice, TorchTensor},
};
use futures::future::join_all;
use reqwest::Client;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug)]
struct MockTensor {
    ptr: u64,
    size: usize,
    shape: Vec<usize>,
}

impl MockTensor {
    fn new(shape: Vec<usize>, device_id: usize, dtype_width_bytes: usize) -> Result<Self> {
        let allocator = DeviceAllocator::new(device_id)?;
        let size = shape.iter().product::<usize>() * dtype_width_bytes;
        let device_storage = ManuallyDrop::new(allocator.allocate(size)?);

        Ok(Self {
            ptr: device_storage.addr(),
            size,
            shape,
        })
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
    backend_urls: Vec<String>,
    worker_id: u64,
    device_id: usize,
    num_device_blocks: usize,
    page_size: usize,
    dtype_width_bytes: usize,
    leader_pub_url: String,
    leader_ack_url: String,
    host_blocks: usize,
    disk_blocks: usize,
    sequence_start: u64,
    count: usize,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            backend_urls: vec!["http://127.0.0.1:58080".to_string()],
            worker_id: 11,
            device_id: 0,
            num_device_blocks: 8,
            page_size: 32,
            dtype_width_bytes: 2,
            leader_pub_url: "tcp://127.0.0.1:56021".to_string(),
            leader_ack_url: "tcp://127.0.0.1:56022".to_string(),
            host_blocks: 8,
            disk_blocks: 8,
            sequence_start: 1000,
            count: 4,
        }
    }
}

impl Args {
    fn parse() -> Result<Self> {
        let mut args = Self::default();
        let mut it = std::env::args().skip(1);
        let mut saw_backend_url = false;

        while let Some(flag) = it.next() {
            match flag.as_str() {
                "--backend-url" => {
                    if !saw_backend_url {
                        args.backend_urls.clear();
                        saw_backend_url = true;
                    }
                    args.backend_urls
                        .push(it.next().context("missing value for --backend-url")?)
                }
                "--worker-id" => {
                    args.worker_id = it
                        .next()
                        .context("missing value for --worker-id")?
                        .parse()?
                }
                "--device-id" => {
                    args.device_id = it
                        .next()
                        .context("missing value for --device-id")?
                        .parse()?
                }
                "--num-device-blocks" => {
                    args.num_device_blocks = it
                        .next()
                        .context("missing value for --num-device-blocks")?
                        .parse()?
                }
                "--page-size" => {
                    args.page_size = it
                        .next()
                        .context("missing value for --page-size")?
                        .parse()?
                }
                "--dtype-width-bytes" => {
                    args.dtype_width_bytes = it
                        .next()
                        .context("missing value for --dtype-width-bytes")?
                        .parse()?
                }
                "--leader-pub-url" => {
                    args.leader_pub_url = it.next().context("missing value for --leader-pub-url")?
                }
                "--leader-ack-url" => {
                    args.leader_ack_url = it.next().context("missing value for --leader-ack-url")?
                }
                "--host-blocks" => {
                    args.host_blocks = it
                        .next()
                        .context("missing value for --host-blocks")?
                        .parse()?
                }
                "--disk-blocks" => {
                    args.disk_blocks = it
                        .next()
                        .context("missing value for --disk-blocks")?
                        .parse()?
                }
                "--sequence-start" => {
                    args.sequence_start = it
                        .next()
                        .context("missing value for --sequence-start")?
                        .parse()?
                }
                "--count" => {
                    args.count = it.next().context("missing value for --count")?.parse()?
                }
                "--help" | "-h" => {
                    println!(
                        "kvbm_g4_worker_smoke
  --backend-url <url>             remote backend base url; repeat to add owners
                                  (default http://127.0.0.1:58080)
  --worker-id <id>                local worker id (default 11)
  --device-id <id>                CUDA device id (default 0)
  --num-device-blocks <n>         local worker device blocks (default 8)
  --page-size <n>                 KVBM page size (default 32)
  --dtype-width-bytes <n>         dtype width bytes (default 2)
  --leader-pub-url <url>          local worker/leader pub url
  --leader-ack-url <url>          local worker/leader ack url
  --host-blocks <n>               leader host blocks (default 8)
  --disk-blocks <n>               leader disk blocks (default 8)
  --sequence-start <n>            first demo sequence hash (default 1000)
  --count <n>                     number of demo blocks (default 4)"
                    );
                    std::process::exit(0);
                }
                other => anyhow::bail!("unknown flag: {other}"),
            }
        }

        let mut unique_backend_urls = Vec::new();
        let mut seen_backend_urls = HashSet::new();
        for backend_url in args.backend_urls {
            if seen_backend_urls.insert(backend_url.clone()) {
                unique_backend_urls.push(backend_url);
            }
        }
        args.backend_urls = unique_backend_urls;

        Ok(args)
    }
}

async fn build_local_worker(args: &Args) -> Result<(KvbmLeader, KvbmWorker, u64)> {
    let shape = vec![args.num_device_blocks, 1, 2, args.page_size, 128];
    let tensors: Vec<Arc<dyn TorchTensor>> = vec![Arc::new(MockTensor::new(
        shape,
        args.device_id,
        args.dtype_width_bytes,
    )?)];

    let worker_config = KvbmWorkerConfig::builder()
        .cancel_token(CancellationToken::new())
        .num_device_blocks(args.num_device_blocks)
        .page_size(args.page_size)
        .dtype_width_bytes(args.dtype_width_bytes)
        .device_id(args.device_id)
        .tensors(tensors)
        .leader_pub_url(args.leader_pub_url.clone())
        .leader_ack_url(args.leader_ack_url.clone())
        .build()?;

    let mut worker = KvbmWorker::new(worker_config, false).await?;

    let leader_config = KvbmLeaderConfig::builder()
        .world_size(1)
        .leader_pub_url(args.leader_pub_url.clone())
        .leader_ack_url(args.leader_ack_url.clone())
        .host_blocks_config(KvbmLeaderNumBlocksConfig {
            cache_size_in_gb: 0.0,
            num_blocks_overriden: args.host_blocks,
        })
        .disk_blocks_config(KvbmLeaderNumBlocksConfig {
            cache_size_in_gb: 0.0,
            num_blocks_overriden: args.disk_blocks,
        })
        .build()?;

    let leader = KvbmLeader::new(leader_config).await?;
    let block_index = Arc::new(G4BlockIndex::default());
    let storage_worker = G4StorageWorker {
        worker_id: args.worker_id,
        endpoint: "local-smoke".to_string(),
    };

    let agent = worker
        .into_g4_storage_agent(storage_worker, block_index)
        .await?;
    Ok((leader, worker, agent.worker_id()))
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse()?;
    let (_leader, _worker, local_worker_id) = build_local_worker(&args).await?;
    println!("local worker ready: worker_id={local_worker_id}");

    let client = Client::new();
    let health_checks = join_all(args.backend_urls.iter().map(|backend_url| {
        let client = client.clone();
        let backend_url = backend_url.clone();
        async move {
            let health: G4HealthResponse = client
                .get(format!("{backend_url}/health"))
                .send()
                .await?
                .error_for_status()?
                .json()
                .await?;

            Ok::<_, anyhow::Error>((backend_url, health))
        }
    }))
    .await;

    let mut backend_urls_by_worker = HashMap::<u64, String>::new();
    let mut remote_workers = Vec::new();
    for result in health_checks {
        let (backend_url, health) = result?;
        if let Some(previous) = backend_urls_by_worker.insert(health.worker_id, backend_url.clone())
        {
            anyhow::bail!(
                "duplicate remote worker_id {} discovered at {} and {}",
                health.worker_id,
                previous,
                backend_url
            );
        }

        remote_workers.push(G4StorageWorker {
            worker_id: health.worker_id,
            endpoint: backend_url.clone(),
        });
        println!(
            "remote backend ready: worker_id={} listen={} base_url={}",
            health.worker_id, health.listen, backend_url
        );
    }

    let blocks: Vec<G4PutBlock> = (0..args.count)
        .map(|offset| G4PutBlock {
            sequence_hash: args.sequence_start + offset as u64,
            disk_block_idx: offset,
            size_bytes: args.page_size * 128 * args.dtype_width_bytes,
            checksum: None,
        })
        .collect();
    let transfer_blocks: Vec<G4TransferBlock> = blocks
        .iter()
        .enumerate()
        .map(|(offset, meta)| G4TransferBlock {
            meta: meta.clone(),
            payload: (0..meta.size_bytes)
                .map(|i| ((i + offset) % 251) as u8)
                .collect(),
        })
        .collect();

    let offered_by_owner = route_g4_put_blocks_by_owner(blocks.clone(), &remote_workers)?;
    let offers = join_all(offered_by_owner.into_iter().map(|(worker_id, blocks)| {
        let client = client.clone();
        let backend_url = backend_urls_by_worker
            .get(&worker_id)
            .cloned()
            .with_context(|| format!("missing backend url for worker {worker_id}"));
        async move {
            let backend_url = backend_url?;
            let offer: G4OfferResponse = client
                .post(format!("{backend_url}/offer"))
                .json(&G4OfferRequest { blocks })
                .send()
                .await?
                .error_for_status()?
                .json()
                .await?;
            Ok::<_, anyhow::Error>((worker_id, offer.accepted))
        }
    }))
    .await;

    let mut accepted_by_worker = HashMap::<u64, HashSet<u64>>::new();
    for result in offers {
        let (worker_id, accepted) = result?;
        println!("worker {worker_id} offer accepted hashes: {:?}", accepted);
        accepted_by_worker.insert(worker_id, accepted.into_iter().collect());
    }

    let accepted_blocks: Vec<G4TransferBlock> = transfer_blocks
        .iter()
        .filter(|block| {
            select_g4_owner(block.meta.sequence_hash, &remote_workers)
                .and_then(|owner| accepted_by_worker.get(&owner.worker_id))
                .is_some_and(|accepted| accepted.contains(&block.meta.sequence_hash))
        })
        .cloned()
        .collect();

    let payloads_by_owner =
        route_g4_transfer_blocks_by_owner(accepted_blocks.clone(), &remote_workers)?;
    for result in join_all(payloads_by_owner.into_iter().map(|(worker_id, blocks)| {
        let client = client.clone();
        let backend_url = backend_urls_by_worker
            .get(&worker_id)
            .cloned()
            .with_context(|| format!("missing backend url for worker {worker_id}"));
        async move {
            let backend_url = backend_url?;
            client
                .post(format!("{backend_url}/put_payload"))
                .json(&G4PutPayloadRequest { blocks })
                .send()
                .await?
                .error_for_status()?;
            Ok::<_, anyhow::Error>(worker_id)
        }
    }))
    .await
    {
        let worker_id = result?;
        println!("uploaded accepted payloads to worker {worker_id}");
    }

    let query_hashes: Vec<u64> = (0..=args.count)
        .map(|offset| args.sequence_start + offset as u64)
        .collect();
    let query_routes = route_g4_sequence_hashes_by_owner(&query_hashes, &remote_workers)?;
    let mut hits = Vec::<G4QueryHit>::new();
    for result in join_all(
        query_routes
            .into_iter()
            .map(|(worker_id, sequence_hashes)| {
                let client = client.clone();
                let backend_url = backend_urls_by_worker
                    .get(&worker_id)
                    .cloned()
                    .with_context(|| format!("missing backend url for worker {worker_id}"));
                async move {
                    let backend_url = backend_url?;
                    let hits: Vec<G4QueryHit> = client
                        .post(format!("{backend_url}/query"))
                        .json(&G4QueryRequest { sequence_hashes })
                        .send()
                        .await?
                        .error_for_status()?
                        .json()
                        .await?;
                    Ok::<_, anyhow::Error>(hits)
                }
            }),
    )
    .await
    {
        hits.extend(result?);
    }

    let fetch_hashes: Vec<u64> = blocks.iter().map(|block| block.sequence_hash).collect();
    let fetch_routes = route_g4_sequence_hashes_by_owner(&fetch_hashes, &remote_workers)?;
    let mut fetched_blocks = Vec::new();
    for result in join_all(
        fetch_routes
            .into_iter()
            .map(|(worker_id, sequence_hashes)| {
                let client = client.clone();
                let backend_url = backend_urls_by_worker
                    .get(&worker_id)
                    .cloned()
                    .with_context(|| format!("missing backend url for worker {worker_id}"));
                async move {
                    let backend_url = backend_url?;
                    let fetched: G4FetchResponse = client
                        .post(format!("{backend_url}/fetch"))
                        .json(&G4FetchRequest { sequence_hashes })
                        .send()
                        .await?
                        .error_for_status()?
                        .json()
                        .await?;
                    Ok::<_, anyhow::Error>(fetched.blocks)
                }
            }),
    )
    .await
    {
        fetched_blocks.extend(result?);
    }

    fetched_blocks.sort_by_key(|block| {
        fetch_hashes
            .iter()
            .position(|sequence_hash| *sequence_hash == block.meta.sequence_hash)
            .unwrap_or(usize::MAX)
    });
    let fetched = G4FetchResponse {
        blocks: fetched_blocks,
    };

    for expected in &transfer_blocks {
        let actual = fetched
            .blocks
            .iter()
            .find(|block| block.meta.sequence_hash == expected.meta.sequence_hash)
            .with_context(|| format!("missing fetched block {}", expected.meta.sequence_hash))?;
        anyhow::ensure!(
            actual.payload == expected.payload,
            "payload mismatch for sequence hash {}",
            expected.meta.sequence_hash
        );
    }

    let transferred_bytes: usize = fetched.blocks.iter().map(|block| block.payload.len()).sum();

    println!("queried hashes: {:?}", query_hashes);
    println!("remote hits:");
    for hit in hits {
        println!(
            "  worker_id={} sequence_hash={} disk_block_idx={} size_bytes={}",
            hit.worker_id, hit.sequence_hash, hit.disk_block_idx, hit.size_bytes
        );
    }
    println!(
        "transferred {} blocks / {} bytes via the smoke HTTP transfer backend",
        fetched.blocks.len(),
        transferred_bytes
    );
    println!(
        "note: this validates actual byte transfer in the smoke backend; real KVBM remote data-plane integration is still a separate step."
    );

    Ok(())
}
