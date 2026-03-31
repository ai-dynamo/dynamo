// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use anyhow::{Context, Result};
use dynamo_llm::block_manager::KvBlockManager;
use dynamo_llm::block_manager::Storage;
use dynamo_llm::block_manager::block::{
    BasicMetadata, BlockDataProvider, BlockDataProviderMut, MutableBlock, data::BlockDataExt,
    locality::LocalityProvider,
};
use dynamo_llm::block_manager::config::{
    KvBlockManagerConfig, KvManagerLayoutConfig, KvManagerModelConfig, KvManagerRuntimeConfig,
};
use dynamo_llm::block_manager::distributed::{
    G3pbFetchRequest, G3pbFetchResponse, G3pbHealthResponse, G3pbOfferRequest, G3pbOfferResponse,
    G3pbPeer, G3pbPutBlock, G3pbPutPayloadRequest, G3pbQueryHit, G3pbQueryRequest,
    G3pbTransferBlock, route_g3pb_put_blocks_by_owner, route_g3pb_sequence_hashes_by_owner,
    route_g3pb_transfer_blocks_by_owner, select_g3pb_owner,
};
use dynamo_llm::block_manager::locality::Local;
use dynamo_llm::block_manager::storage::{DeviceAllocator, PinnedAllocator, PinnedStorage};
use futures::future::join_all;
use reqwest::Client;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug)]
struct Args {
    backend_urls: Vec<String>,
    worker_id: u64,
    device_id: usize,
    num_device_blocks: usize,
    page_size: usize,
    dtype_width_bytes: usize,
    host_blocks: usize,
    sequence_start: u64,
    count: usize,
}

type LocalBlockManager = KvBlockManager<Local, BasicMetadata>;

#[derive(Clone, Debug)]
struct DemoBlock {
    tokens: Vec<u32>,
    sequence_hash: u64,
    payload: Vec<u8>,
    size_bytes: usize,
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
            host_blocks: 8,
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
                "--host-blocks" => {
                    args.host_blocks = it
                        .next()
                        .context("missing value for --host-blocks")?
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
                        "kvbm_g3pb_worker_smoke
  --backend-url <url>             remote backend base url; repeat to add owners
                                  (default http://127.0.0.1:58080)
  --worker-id <id>                local worker id (default 11)
  --device-id <id>                CUDA device id (default 0)
  --num-device-blocks <n>         local worker device blocks (default 8)
  --page-size <n>                 KVBM page size (default 32)
  --dtype-width-bytes <n>         dtype width bytes (default 2)
  --host-blocks <n>               leader host blocks (default 8)
  --sequence-start <n>            first demo token seed (default 1000)
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

fn demo_tokens(args: &Args, offset: usize) -> Vec<u32> {
    let start = args.sequence_start as u32 + (offset as u32 * args.page_size as u32);
    (0..args.page_size).map(|idx| start + idx as u32).collect()
}

fn block_size_bytes<S: Storage>(block: &impl BlockDataProvider<StorageType = S>) -> Result<usize> {
    let block_data = block.block_data();
    let mut size_bytes = 0usize;
    for layer_idx in 0..block_data.num_layers() {
        for outer_idx in 0..block_data.num_outer_dims() {
            size_bytes += block_data.layer_view(layer_idx, outer_idx)?.size();
        }
    }
    Ok(size_bytes)
}

fn write_block_payload<S: Storage>(
    block: &mut impl BlockDataProviderMut<StorageType = S>,
    payload: &[u8],
) -> Result<()> {
    let block_data = block.block_data_mut();
    let mut copied = 0usize;

    for layer_idx in 0..block_data.num_layers() {
        for outer_idx in 0..block_data.num_outer_dims() {
            let mut layer_view = block_data.layer_view_mut(layer_idx, outer_idx)?;
            let layer_size = layer_view.size();
            let end = copied + layer_size;
            anyhow::ensure!(
                end <= payload.len(),
                "payload shorter than destination block: copied={copied} layer_size={layer_size} payload_len={}",
                payload.len()
            );

            unsafe {
                std::ptr::copy_nonoverlapping(
                    payload[copied..end].as_ptr(),
                    layer_view.as_mut_ptr(),
                    layer_size,
                );
            }
            copied = end;
        }
    }

    anyhow::ensure!(
        copied == payload.len(),
        "payload longer than destination block: copied={copied} payload_len={}",
        payload.len()
    );

    Ok(())
}

fn read_block_payload(
    block: &impl BlockDataProvider<StorageType = PinnedStorage>,
) -> Result<Vec<u8>> {
    let block_data = block.block_data();
    let mut payload = Vec::with_capacity(block_size_bytes(block)?);

    for layer_idx in 0..block_data.num_layers() {
        for outer_idx in 0..block_data.num_outer_dims() {
            let layer_view = block_data.layer_view(layer_idx, outer_idx)?;
            unsafe {
                payload.extend_from_slice(std::slice::from_raw_parts(
                    layer_view.as_ptr(),
                    layer_view.size(),
                ));
            }
        }
    }

    Ok(payload)
}

fn complete_block<
    S: Storage,
    L: LocalityProvider,
    M: dynamo_llm::block_manager::block::BlockMetadata,
>(
    block: &mut MutableBlock<S, L, M>,
    tokens: &[u32],
) -> Result<()> {
    block.init_sequence(42)?;
    for token in tokens {
        block.add_token(*token)?;
    }
    block.commit()?;
    Ok(())
}

async fn prepare_demo_blocks(
    args: &Args,
    block_manager: &LocalBlockManager,
) -> Result<Vec<DemoBlock>> {
    let device_pool = block_manager
        .device()
        .context("local block manager has no device pool")?;
    let mut blocks = device_pool.allocate_blocks(args.count + 1).await?;
    let mut demo_blocks = Vec::with_capacity(blocks.len());

    for (offset, block) in blocks.iter_mut().enumerate() {
        let tokens = demo_tokens(args, offset);
        complete_block(block, &tokens)?;
        let sequence_hash = block.sequence_hash()?;
        let size_bytes = block_size_bytes(block)?;
        let payload = (0..size_bytes)
            .map(|idx| ((idx + offset) % 251) as u8)
            .collect();

        demo_blocks.push(DemoBlock {
            tokens,
            sequence_hash,
            payload,
            size_bytes,
        });
    }

    drop(blocks);

    Ok(demo_blocks)
}

async fn build_local_runtime(args: &Args) -> Result<LocalBlockManager> {
    let cancel_token = CancellationToken::new();
    let mut block_manager_config = KvBlockManagerConfig::builder()
        .runtime(
            KvManagerRuntimeConfig::builder()
                .worker_id(args.worker_id)
                .cancellation_token(cancel_token.clone())
                .disable_nixl()
                .build()?,
        )
        .model(
            KvManagerModelConfig::builder()
                .num_layers(1)
                .outer_dim(1)
                .page_size(args.page_size)
                .inner_dim(128)
                .dtype_width_bytes(args.dtype_width_bytes)
                .build()?,
        )
        .device_layout(
            KvManagerLayoutConfig::builder()
                .num_blocks(args.num_device_blocks)
                .allocator(DeviceAllocator::new(args.device_id)?)
                .build()?,
        );

    if args.host_blocks > 0 {
        block_manager_config = block_manager_config.host_layout(
            KvManagerLayoutConfig::builder()
                .num_blocks(args.host_blocks)
                .allocator(PinnedAllocator::new(args.device_id)?)
                .build()?,
        );
    }

    let block_manager_config = block_manager_config.build()?;
    LocalBlockManager::new(block_manager_config).await
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse()?;
    let block_manager = build_local_runtime(&args).await?;
    println!(
        "local worker ready: worker_id={} device_blocks={} host_blocks={}",
        block_manager.worker_id(),
        args.num_device_blocks,
        args.host_blocks
    );

    let demo_blocks = prepare_demo_blocks(&args, &block_manager).await?;
    let uploaded_demo_blocks = &demo_blocks[..args.count];
    let missing_demo_block = &demo_blocks[args.count];

    let client = Client::new();
    let health_checks = join_all(args.backend_urls.iter().map(|backend_url| {
        let client = client.clone();
        let backend_url = backend_url.clone();
        async move {
            let health: G3pbHealthResponse = client
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

        remote_workers.push(G3pbPeer {
            worker_id: health.worker_id,
            endpoint: backend_url.clone(),
        });
        println!(
            "remote backend ready: worker_id={} listen={} base_url={}",
            health.worker_id, health.listen, backend_url
        );
    }

    let blocks: Vec<G3pbPutBlock> = uploaded_demo_blocks
        .iter()
        .map(|block| G3pbPutBlock {
            sequence_hash: block.sequence_hash,
            size_bytes: block.size_bytes,
            checksum: None,
        })
        .collect();
    let transfer_blocks: Vec<G3pbTransferBlock> = uploaded_demo_blocks
        .iter()
        .zip(blocks.iter())
        .map(|(demo, meta)| G3pbTransferBlock {
            meta: meta.clone(),
            payload: demo.payload.clone(),
        })
        .collect();

    let offered_by_owner = route_g3pb_put_blocks_by_owner(blocks.clone(), &remote_workers)?;
    let offers = join_all(offered_by_owner.into_iter().map(|(worker_id, blocks)| {
        let client = client.clone();
        let backend_url = backend_urls_by_worker
            .get(&worker_id)
            .cloned()
            .with_context(|| format!("missing backend url for worker {worker_id}"));
        async move {
            let backend_url = backend_url?;
            let offer: G3pbOfferResponse = client
                .post(format!("{backend_url}/offer"))
                .json(&G3pbOfferRequest { blocks })
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

    let accepted_blocks: Vec<G3pbTransferBlock> = transfer_blocks
        .iter()
        .filter(|block| {
            select_g3pb_owner(block.meta.sequence_hash, &remote_workers)
                .and_then(|owner| accepted_by_worker.get(&owner.worker_id))
                .is_some_and(|accepted| accepted.contains(&block.meta.sequence_hash))
        })
        .cloned()
        .collect();

    let payloads_by_owner =
        route_g3pb_transfer_blocks_by_owner(accepted_blocks.clone(), &remote_workers)?;
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
                .json(&G3pbPutPayloadRequest { blocks })
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

    let duplicate_offer_blocks: Vec<G3pbPutBlock> = accepted_blocks
        .iter()
        .map(|block| block.meta.clone())
        .collect();
    if !duplicate_offer_blocks.is_empty() {
        let duplicate_offer_routes =
            route_g3pb_put_blocks_by_owner(duplicate_offer_blocks, &remote_workers)?;
        for result in join_all(
            duplicate_offer_routes
                .into_iter()
                .map(|(worker_id, blocks)| {
                    let client = client.clone();
                    let backend_url = backend_urls_by_worker
                        .get(&worker_id)
                        .cloned()
                        .with_context(|| format!("missing backend url for worker {worker_id}"));
                    async move {
                        let backend_url = backend_url?;
                        let offer: G3pbOfferResponse = client
                            .post(format!("{backend_url}/offer"))
                            .json(&G3pbOfferRequest { blocks })
                            .send()
                            .await?
                            .error_for_status()?
                            .json()
                            .await?;
                        Ok::<_, anyhow::Error>((worker_id, offer.accepted))
                    }
                }),
        )
        .await
        {
            let (worker_id, accepted) = result?;
            anyhow::ensure!(
                accepted.is_empty(),
                "duplicate offer unexpectedly accepted hashes on worker {worker_id}: {:?}",
                accepted
            );
            println!("duplicate offer on worker {worker_id} correctly accepted nothing");
        }
    }

    let query_hashes: Vec<u64> = demo_blocks
        .iter()
        .map(|block| block.sequence_hash)
        .collect();
    let query_routes = route_g3pb_sequence_hashes_by_owner(&query_hashes, &remote_workers)?;
    let mut hits = Vec::<G3pbQueryHit>::new();
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
                    let hits: Vec<G3pbQueryHit> = client
                        .post(format!("{backend_url}/query"))
                        .json(&G3pbQueryRequest { sequence_hashes })
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

    let hit_by_sequence_hash: HashMap<u64, G3pbQueryHit> = hits
        .iter()
        .cloned()
        .map(|hit| (hit.sequence_hash, hit))
        .collect();
    let cache_miss_hashes: Vec<u64> = query_hashes
        .iter()
        .copied()
        .filter(|sequence_hash| !hit_by_sequence_hash.contains_key(sequence_hash))
        .collect();
    let fetch_hashes: Vec<u64> = query_hashes
        .iter()
        .copied()
        .filter(|sequence_hash| hit_by_sequence_hash.contains_key(sequence_hash))
        .collect();

    anyhow::ensure!(
        cache_miss_hashes == vec![missing_demo_block.sequence_hash],
        "unexpected cache-miss set: {:?}",
        cache_miss_hashes
    );

    let fetch_routes = route_g3pb_sequence_hashes_by_owner(&fetch_hashes, &remote_workers)?;
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
                    let fetched: G3pbFetchResponse = client
                        .post(format!("{backend_url}/fetch"))
                        .json(&G3pbFetchRequest { sequence_hashes })
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
    let fetched = G3pbFetchResponse {
        blocks: fetched_blocks,
    };

    for expected in uploaded_demo_blocks
        .iter()
        .filter(|block| fetch_hashes.contains(&block.sequence_hash))
    {
        let actual = fetched
            .blocks
            .iter()
            .find(|block| block.meta.sequence_hash == expected.sequence_hash)
            .with_context(|| format!("missing fetched block {}", expected.sequence_hash))?;
        anyhow::ensure!(
            actual.payload == expected.payload,
            "payload mismatch for sequence hash {}",
            expected.sequence_hash
        );
    }

    let host_pool = block_manager
        .host()
        .context("local block manager has no host pool for fetched G3pb blocks")?;
    let device_pool = block_manager
        .device()
        .context("local block manager has no device pool for onboarded G3pb blocks")?;
    let fetched_demo_blocks: Vec<_> = uploaded_demo_blocks
        .iter()
        .filter(|block| fetch_hashes.contains(&block.sequence_hash))
        .collect();
    let mut mutable_host_blocks = host_pool.allocate_blocks(fetched_demo_blocks.len()).await?;
    for (mutable_host_block, expected) in mutable_host_blocks.iter_mut().zip(&fetched_demo_blocks) {
        complete_block(mutable_host_block, &expected.tokens)?;
        write_block_payload(mutable_host_block, &expected.payload)?;
    }

    let immutable_host_blocks = host_pool.register_blocks(mutable_host_blocks).await?;
    anyhow::ensure!(
        immutable_host_blocks.len() == fetched_demo_blocks.len(),
        "expected {} registered host blocks, got {}",
        fetched_demo_blocks.len(),
        immutable_host_blocks.len()
    );

    for expected in &fetched_demo_blocks {
        let host_block = immutable_host_blocks
            .iter()
            .find(|block| block.sequence_hash() == expected.sequence_hash)
            .with_context(|| {
                format!(
                    "missing registered host block for fetched sequence hash {}",
                    expected.sequence_hash
                )
            })?;
        anyhow::ensure!(
            read_block_payload(host_block)? == expected.payload,
            "host registration payload mismatch for sequence hash {}",
            expected.sequence_hash
        );
    }

    let onboarded_blocks = block_manager
        .onboard_blocks(immutable_host_blocks.clone(), None)
        .await??;
    anyhow::ensure!(
        onboarded_blocks.len() == fetched_demo_blocks.len(),
        "expected {} onboarded device blocks, got {}",
        fetched_demo_blocks.len(),
        onboarded_blocks.len()
    );

    let onboarded_sequence_hashes: Vec<_> = onboarded_blocks
        .iter()
        .map(|block| block.sequence_hash())
        .collect();
    let matched_device_blocks = device_pool
        .match_sequence_hashes(fetch_hashes.as_slice())
        .await?;
    anyhow::ensure!(
        matched_device_blocks.len() == fetched_demo_blocks.len(),
        "expected {} device-pool matches after onboard, got {}",
        fetched_demo_blocks.len(),
        matched_device_blocks.len()
    );

    let transferred_bytes: usize = fetched.blocks.iter().map(|block| block.payload.len()).sum();

    println!("queried hashes: {:?}", query_hashes);
    println!("cache misses carried as fallback: {:?}", cache_miss_hashes);
    println!("remote hits:");
    for hit in hits {
        println!(
            "  worker_id={} sequence_hash={} size_bytes={}",
            hit.worker_id, hit.sequence_hash, hit.size_bytes
        );
    }
    println!(
        "onboarded {} fetched blocks into local device pool with sequence hashes {:?}",
        onboarded_blocks.len(),
        onboarded_sequence_hashes
    );
    println!(
        "transferred {} blocks / {} bytes via the smoke HTTP transfer backend",
        fetched.blocks.len(),
        transferred_bytes
    );
    println!(
        "note: this validates HTTP query/fetch plus local host registration and device onboard; real remote KVBM data-plane integration is still a separate step."
    );

    Ok(())
}
