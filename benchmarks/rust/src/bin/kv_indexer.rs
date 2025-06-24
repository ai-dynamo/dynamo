use clap::Parser;
use futures::future::join_all;
use std::{collections::HashSet, fs::read_to_string, time::Instant};
use tokio_util::sync::CancellationToken;

use dynamo_llm::kv_router::{
    indexer::{KvIndexer, KvIndexerInterface, KvIndexerSharded, RouterEvent},
    protocols::{
        ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData, KvCacheRemoveData,
        KvCacheStoreData, KvCacheStoredBlockData, LocalBlockHash,
    },
};

fn parse_percent(s: &str) -> Result<f64, String> {
    let value: f64 = s.parse().map_err(|_| format!("Invalid percent: {}", s))?;
    if !(0.0..=1.0).contains(&value) {
        return Err(format!("Percent must be between 0.0 and 1.0: {}", s));
    }
    Ok(value)
}

#[derive(Parser, Debug)]
#[clap(version, about, long_about = None)]
struct Args {
    /// Path to mooncake trace.
    mooncake_trace_path: String,

    /// Amount of shards to use for the indexer. This directly corresponds to the degree of parallelism.
    /// More parallelism means higher event processing rates at the expense of slower request rates.
    #[arg(short, long, default_value_t = 1)]
    num_shards: usize,

    /// A mock for remove block events.
    /// When a stored event is created, we also create a corresponding remove event that removes a fraction of the blocks that were stored.
    // TODO: This is a bit of a hack to simulate block removal. We should instead more closely model an LRU behavior.
    #[arg(short, long, default_value_t = 0.25, value_parser = parse_percent)]
    removal_fraction: f64,
}

fn load_trace(args: &Args) -> (Vec<KvCacheEvent>, Vec<Vec<u64>>) {
    println!("Loading trace from {}", args.mooncake_trace_path);

    let lines: Vec<String> = read_to_string(&args.mooncake_trace_path)
        .unwrap()
        .lines()
        .map(String::from)
        .collect();

    println!("Reading {} events", lines.len());

    let mut events = Vec::new();
    let mut sequences = Vec::new();

    let mut seen_blocks = HashSet::new();

    let mut event_id = 0;

    for line in lines {
        let json_value: serde_json::Value = serde_json::from_str(&line).unwrap();

        let hash_ids = json_value["hash_ids"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| x.as_u64().unwrap())
            .collect::<Vec<_>>();

        sequences.push(hash_ids.clone());

        let block_hashes;
        let parent_hash;

        if let Some(start) = hash_ids.iter().position(|x| !seen_blocks.contains(x)) {
            block_hashes = hash_ids[start..].to_vec();
            if start == 0 {
                parent_hash = None;
            } else {
                parent_hash = Some(ExternalSequenceBlockHash(hash_ids[start - 1]));
            }
        } else {
            continue;
        };

        let event_data = KvCacheEventData::Stored(KvCacheStoreData {
            parent_hash,
            blocks: block_hashes
                .iter()
                .map(|x| KvCacheStoredBlockData {
                    block_hash: ExternalSequenceBlockHash(*x),
                    tokens_hash: LocalBlockHash(*x),
                })
                .collect(),
        });

        events.push(KvCacheEvent {
            event_id,
            data: event_data,
        });

        event_id += 1;

        let num_to_keep = (block_hashes.len() as f64 * (1.0 - args.removal_fraction)) as usize;

        for hash in block_hashes[..num_to_keep].iter() {
            assert!(!seen_blocks.contains(hash));
            seen_blocks.insert(*hash);
        }

        let remove_block_hashes: Vec<ExternalSequenceBlockHash> = block_hashes[num_to_keep..]
            .iter()
            .rev()
            .map(|x| ExternalSequenceBlockHash(*x))
            .collect();

        if !remove_block_hashes.is_empty() {
            let remove_event_data = KvCacheEventData::Removed(KvCacheRemoveData {
                block_hashes: remove_block_hashes,
            });

            events.push(KvCacheEvent {
                event_id,
                data: remove_event_data,
            });

            event_id += 1;
        }
    }

    (events, sequences)
}

fn build_indexer(args: &Args) -> Box<dyn KvIndexerInterface + Sync + Send> {
    let token = CancellationToken::new();

    let kv_indexer: Box<dyn KvIndexerInterface + Sync + Send> = if args.num_shards == 1 {
        Box::new(KvIndexer::new_with_frequency(token, None, 32))
    } else {
        Box::new(KvIndexerSharded::new_with_frequency(
            token,
            args.num_shards,
            None,
            32,
        ))
    };

    kv_indexer
}

async fn bench_events(
    indexer: &mut dyn KvIndexerInterface,
    events: &[KvCacheEvent],
    num_shards: usize,
) {
    println!("Running events.");
    let start = Instant::now();

    // create worker_ids for each shared
    let worker_ids: Vec<i64> = (0..num_shards as i64).collect();

    for event in events {
        for id in &worker_ids {
            indexer
                .apply_event(RouterEvent::new(*id, event.clone()))
                .await;
        }
    }

    let total_blocks: usize = events
        .iter()
        .map(|elem| match &elem.data {
            KvCacheEventData::Stored(i) => i.blocks.len(),
            KvCacheEventData::Removed(i) => i.block_hashes.len(),
            KvCacheEventData::Cleared => {
                panic!("Invalid event type! Benchmark only supports stored and removed events.")
            }
        })
        .sum();

    println!("Events - Elapsed time: {:.2?}", start.elapsed());
    println!(
        "Events - Event rate: {} events/s",
        to_rate_millis(events.len() * num_shards, start)
    );
    println!(
        "Events - Block rate: {} blocks/s",
        to_rate_millis(total_blocks * num_shards, start)
    );
}

async fn bench_requests(indexer: &mut dyn KvIndexerInterface, sequences: &[Vec<u64>]) {
    let total_request_blocks: usize = sequences.iter().map(|item| item.len()).sum();

    let num_requests = sequences.len();

    let mut inflight_requests = Vec::with_capacity(num_requests);
    let start = Instant::now();

    for request in sequences {
        inflight_requests
            .push(indexer.find_matches(request.iter().map(|x| LocalBlockHash(*x)).collect()));
    }

    join_all(inflight_requests).await;

    println!("Requests - Elapsed time: {:.2?}", start.elapsed());
    println!(
        "Requests - Request rate: {} requests/s",
        to_rate_millis(num_requests, start)
    );
    println!(
        "Requests - Block rate: {} blocks/s",
        to_rate_millis(total_request_blocks, start)
    );
}

fn to_rate_millis(x: usize, start: Instant) -> f32 {
    x as f32 / start.elapsed().as_millis() as f32 * 1000.0
}

#[tokio::main]
async fn main() {
    let args = Args::parse();

    let (events, sequences) = load_trace(&args);

    let mut indexer = build_indexer(&args);

    bench_events(indexer.as_mut(), &events, args.num_shards).await;

    bench_requests(indexer.as_mut(), &sequences).await;
}
