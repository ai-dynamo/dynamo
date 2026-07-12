// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;

use dynamo_backend_common::{DynamoError, KvEventPublisher, KvEventSource};
use dynamo_kv_router::protocols::{
    BlockExtraInfo, BlockMmObjectInfo, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData,
    KvCacheRemoveData, KvCacheStoreData,
};
use dynamo_kv_router::zmq_wire::create_stored_blocks;
use parking_lot::Mutex;
use tokio::sync::watch;
use tokio::task::JoinHandle;
use tokio_util::sync::CancellationToken;
use tonic::transport::Channel;

use crate::client::{self, Client};
use crate::proto as pb;

pub async fn discover_sources(
    channel: Channel,
    mut client: Client,
    expected_ranks: HashSet<u32>,
    cancel: CancellationToken,
    tasks: Arc<Mutex<Vec<JoinHandle<()>>>>,
    fatal: watch::Sender<Option<String>>,
) -> Result<Vec<KvEventSource>, DynamoError> {
    let response = client
        .get_kv_event_sources(pb::GetKvEventSourcesRequest {
            data_parallel_ranks: Vec::new(),
        })
        .await
        .map_err(|status| client::status_to_dynamo("GetKvEventSources", status))?
        .into_inner();
    let mut result = Vec::new();
    let mut seen_ranks = HashSet::new();
    for source in response.sources {
        let rank = source.data_parallel_rank.ok_or_else(|| {
            client::invalid_arg("OpenEngine KV event source omitted data_parallel_rank")
        })?;
        if !seen_ranks.insert(rank) {
            return Err(client::invalid_arg(format!(
                "OpenEngine advertised duplicate KV event source for rank {rank}"
            )));
        }
        if !expected_ranks.contains(&rank) {
            return Err(client::invalid_arg(format!(
                "OpenEngine advertised KV event source rank {rank} outside its data-parallel range"
            )));
        }
        match source.transport.as_str() {
            "zmq" => {
                let Some(endpoint) = source.endpoint_addr else {
                    return Err(client::invalid_arg("ZMQ KV source omitted endpoint_addr"));
                };
                let protocol = if endpoint.protocol.is_empty() {
                    "tcp"
                } else {
                    endpoint.protocol.as_str()
                };
                result.push(KvEventSource::Zmq {
                    endpoint: format!("{protocol}://{}:{}", endpoint.host, endpoint.port),
                    topic: source.topic,
                    dp_rank: rank,
                });
            }
            "grpc" => {
                let channel = channel.clone();
                let cancel = cancel.clone();
                let tasks = tasks.clone();
                let fatal = fatal.clone();
                result.push(KvEventSource::Push {
                    dp_rank: rank,
                    on_ready: Box::new(move |publisher| {
                        let task =
                            tokio::spawn(subscribe_loop(channel, rank, publisher, cancel, fatal));
                        tasks.lock().push(task);
                        Ok(())
                    }),
                });
            }
            unsupported => {
                tracing::warn!(unsupported, rank, "ignoring unsupported KV event transport")
            }
        }
    }
    Ok(result)
}

async fn subscribe_loop(
    channel: Channel,
    rank: u32,
    publisher: Arc<KvEventPublisher>,
    cancel: CancellationToken,
    fatal: watch::Sender<Option<String>>,
) {
    let mut client = pb::open_engine_client::OpenEngineClient::new(channel);
    let response = client.subscribe_kv_events(subscription_request(rank)).await;
    let mut stream = match response {
        Ok(response) => response.into_inner(),
        Err(error) => {
            tracing::error!(%error, rank, "OpenEngine KV subscription failed");
            let _ = fatal.send(Some(format!(
                "OpenEngine KV subscription for rank {rank} failed: {error}"
            )));
            return;
        }
    };
    let warnings = Arc::new(AtomicU32::new(0));
    let mut last_sequence = None;
    loop {
        let message = tokio::select! {
            _ = cancel.cancelled() => break,
            message = stream.message() => message,
        };
        match message {
            Ok(Some(response)) => match response.event {
                Some(pb::subscribe_kv_events_response::Event::Batch(batch)) => {
                    if batch.data_parallel_rank != rank {
                        let message = format!(
                            "OpenEngine KV stream for rank {rank} returned batch rank {}",
                            batch.data_parallel_rank
                        );
                        tracing::error!(%message);
                        let _ = fatal.send(Some(message));
                        return;
                    }
                    if !accept_sequence(&mut last_sequence, batch.sequence_number) {
                        tracing::error!(
                            rank,
                            sequence_number = batch.sequence_number,
                            ?last_sequence,
                            "OpenEngine KV stream is non-monotonic"
                        );
                        let _ = fatal.send(Some(format!(
                            "OpenEngine KV stream for rank {rank} is non-monotonic"
                        )));
                        return;
                    }
                    if batch.events.is_empty() {
                        continue;
                    }
                    if batch.events.len() != 1 {
                        let message = format!(
                            "OpenEngine KV batch {} for rank {rank} contained {} events; revision-2 TRT contract requires at most one",
                            batch.sequence_number,
                            batch.events.len()
                        );
                        let _ = fatal.send(Some(message));
                        return;
                    }
                    let event = batch.events.into_iter().next().expect("checked one event");
                    let event = match convert_event(event, rank, batch.sequence_number, &warnings) {
                        Ok(event) => event,
                        Err(message) => {
                            let message = format!(
                                "invalid OpenEngine KV batch {} for rank {rank}: {message}",
                                batch.sequence_number
                            );
                            let _ = fatal.send(Some(message));
                            return;
                        }
                    };
                    if let Some(event) = event
                        && let Err(error) = publisher.publish(event)
                    {
                        tracing::debug!(%error, rank, "Dynamo KV publisher closed");
                        return;
                    }
                }
                Some(pb::subscribe_kv_events_response::Event::Error(error)) => {
                    tracing::error!(?error.code, %error.message, rank, "OpenEngine KV stream error");
                    let _ = fatal.send(Some(format!(
                        "OpenEngine KV stream error for rank {rank}: {}",
                        error.message
                    )));
                    return;
                }
                None => {}
            },
            Ok(None) => {
                tracing::warn!(rank, "OpenEngine KV subscription ended");
                if !cancel.is_cancelled() {
                    let _ = fatal.send(Some(format!(
                        "OpenEngine KV subscription for rank {rank} ended"
                    )));
                }
                return;
            }
            Err(error) => {
                tracing::warn!(%error, rank, "OpenEngine KV subscription transport failed");
                if !cancel.is_cancelled() {
                    let _ = fatal.send(Some(format!(
                        "OpenEngine KV subscription for rank {rank} failed: {error}"
                    )));
                }
                return;
            }
        }
    }
}

fn subscription_request(rank: u32) -> pb::SubscribeKvEventsRequest {
    pb::SubscribeKvEventsRequest {
        data_parallel_ranks: vec![rank],
        // Revision 2 does not advertise snapshot/replay capability. Request
        // only the live stream until discovery can negotiate it.
        include_snapshot: false,
        start_sequence_number: 0,
    }
}

fn convert_event(
    value: pb::KvEvent,
    rank: u32,
    sequence_number: u64,
    warnings: &Arc<AtomicU32>,
) -> Result<Option<KvCacheEvent>, String> {
    let Some(wire_event) = value.event else {
        return Err("KvEvent omitted its event payload".to_string());
    };
    let data = match wire_event {
        pb::kv_event::Event::BlockStored(stored) => {
            if !is_gpu_medium(stored.medium) {
                return Ok(None);
            }
            if stored.block_size == 0 {
                return Err("BlockStored block_size must be non-zero".to_string());
            }
            if stored.block_hashes.is_empty() && stored.token_ids.is_empty() {
                return Ok(None);
            }
            if !stored.kv_cache_spec_kind.is_empty() || stored.kv_cache_spec_sliding_window != 0 {
                return Err("unsupported KV cache group/spec metadata".to_string());
            }
            if stored.lora_id != 0 && stored.lora_name.is_empty() {
                return Err("BlockStored with a non-zero LoRA ID omitted its name".to_string());
            }
            let expected_tokens = usize::try_from(stored.block_size)
                .ok()
                .and_then(|size| size.checked_mul(stored.block_hashes.len()))
                .ok_or_else(|| "BlockStored token cardinality overflow".to_string())?;
            if stored.token_ids.len() != expected_tokens {
                return Err(format!(
                    "BlockStored has {} tokens for {} block(s) of size {}",
                    stored.token_ids.len(),
                    stored.block_hashes.len(),
                    stored.block_size
                ));
            }
            let hashes = stored
                .block_hashes
                .iter()
                .map(block_hash)
                .collect::<Result<Vec<_>, _>>()?;
            let block_mm_infos = trt_mm_infos(&stored.extra_keys, hashes.len())?;
            let num_block_tokens = vec![u64::from(stored.block_size); hashes.len()];
            let blocks = create_stored_blocks(
                stored.block_size,
                &stored.token_ids,
                &num_block_tokens,
                &hashes,
                (!stored.lora_name.is_empty()).then_some(stored.lora_name.as_str()),
                None,
                warnings,
                block_mm_infos.as_deref(),
                None,
                None,
            );
            if blocks.len() != hashes.len() {
                return Err(
                    "BlockStored could not be normalized without dropping blocks".to_string(),
                );
            }
            KvCacheEventData::Stored(KvCacheStoreData {
                parent_hash: stored
                    .parent_block_hash
                    .as_ref()
                    .map(block_hash)
                    .transpose()?
                    .map(ExternalSequenceBlockHash),
                start_position: None,
                blocks,
            })
        }
        pb::kv_event::Event::BlockRemoved(removed) => {
            if !is_gpu_medium(removed.medium) {
                return Ok(None);
            }
            let block_hashes = removed
                .block_hashes
                .iter()
                .map(block_hash)
                .collect::<Result<Vec<_>, _>>()?
                .into_iter()
                .map(ExternalSequenceBlockHash)
                .collect::<Vec<_>>();
            if block_hashes.is_empty() {
                return Ok(None);
            }
            KvCacheEventData::Removed(KvCacheRemoveData { block_hashes })
        }
        pb::kv_event::Event::AllBlocksCleared(_) => KvCacheEventData::Cleared,
    };
    Ok(Some(KvCacheEvent {
        // Keep the producer sequence intact so queue drops remain visible to
        // Dynamo's indexer rather than being hidden by a local counter.
        event_id: sequence_number,
        data,
        dp_rank: rank,
    }))
}

fn accept_sequence(last: &mut Option<u64>, current: u64) -> bool {
    if last.is_some_and(|last| current <= last) {
        return false;
    }
    *last = Some(current);
    true
}

fn is_gpu_medium(value: i32) -> bool {
    matches!(
        pb::StorageMedium::try_from(value),
        Ok(pb::StorageMedium::Gpu | pb::StorageMedium::Unspecified) | Err(_)
    )
}

fn block_hash(value: &pb::KvBlockHash) -> Result<u64, String> {
    if value.encoding != "decimal_int64" {
        return Err(format!(
            "KV block hash encoding `{}` is not canonical decimal_int64",
            value.encoding
        ));
    }
    let text = std::str::from_utf8(&value.value)
        .map_err(|_| "KV block hash is not ASCII decimal".to_string())?;
    let parsed = text
        .parse::<i64>()
        .map_err(|_| "KV block hash is not a signed int64 decimal".to_string())?;
    if parsed.to_string() != text {
        return Err("KV block hash is not canonically encoded".to_string());
    }
    Ok(parsed as u64)
}

fn trt_mm_infos(
    tuples: &[pb::OpaqueKeyTuple],
    block_count: usize,
) -> Result<Option<Vec<Option<BlockExtraInfo>>>, String> {
    if tuples.is_empty() {
        return Ok(None);
    }
    let mut infos = vec![None; block_count];
    for tuple in tuples {
        let [tag, block_index, mm_hash, start_offset] = tuple.values.as_slice() else {
            return Err("TRT multimodal tuple must contain exactly four values".to_string());
        };
        if tag != "trt_mm_v1" {
            return Err(format!("unknown KV extra-key tag `{tag}`"));
        }
        let block_index = parse_canonical_decimal::<usize>(block_index, "MM block index")?;
        if block_index >= block_count {
            return Err("MM block index exceeds BlockStored cardinality".to_string());
        }
        let mm_hash = parse_canonical_decimal::<u64>(mm_hash, "MM hash")?;
        let _start_offset = parse_canonical_decimal::<usize>(start_offset, "MM start offset")?;
        infos[block_index]
            .get_or_insert_with(|| BlockExtraInfo {
                mm_objects: Vec::new(),
            })
            .mm_objects
            .push(BlockMmObjectInfo {
                mm_hash,
                // TRT supplies a logical-media-item offset, while this field
                // represents block-relative ranges. Do not mislabel it;
                // Dynamo's KV hash depends only on mm_hash.
                offsets: Vec::new(),
            });
    }
    Ok(Some(infos))
}

fn parse_canonical_decimal<T>(value: &str, name: &str) -> Result<T, String>
where
    T: std::str::FromStr + ToString,
{
    let parsed = value
        .parse::<T>()
        .map_err(|_| format!("{name} is not decimal"))?;
    if parsed.to_string() != value {
        return Err(format!("{name} is not canonically encoded"));
    }
    Ok(parsed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_only_canonical_decimal_int64_hashes() {
        assert_eq!(
            block_hash(&pb::KvBlockHash {
                value: b"-1".to_vec(),
                encoding: "decimal_int64".into()
            }),
            Ok(u64::MAX)
        );
        assert!(
            block_hash(&pb::KvBlockHash {
                value: b"01".to_vec(),
                encoding: "decimal_int64".into()
            })
            .is_err()
        );
        assert!(
            block_hash(&pb::KvBlockHash {
                value: 42u64.to_le_bytes().to_vec(),
                encoding: "bytes".into()
            })
            .is_err()
        );
    }

    #[test]
    fn preserves_gaps_and_rejects_non_monotonic_sequences() {
        let warnings = Arc::new(AtomicU32::new(0));
        let cleared = || pb::KvEvent {
            event: Some(pb::kv_event::Event::AllBlocksCleared(
                pb::AllBlocksCleared {},
            )),
            ..Default::default()
        };
        assert_eq!(
            convert_event(cleared(), 2, 7, &warnings)
                .unwrap()
                .unwrap()
                .event_id,
            7
        );
        assert_eq!(
            convert_event(cleared(), 2, 11, &warnings)
                .unwrap()
                .unwrap()
                .event_id,
            11
        );
        let mut last = None;
        assert!(accept_sequence(&mut last, 7));
        assert!(accept_sequence(&mut last, 11));
        assert!(!accept_sequence(&mut last, 11));
        assert!(!accept_sequence(&mut last, 10));
    }

    #[test]
    fn live_subscription_does_not_request_unsupported_snapshot_or_replay() {
        let request = subscription_request(3);
        assert_eq!(request.data_parallel_ranks, vec![3]);
        assert!(!request.include_snapshot);
        assert_eq!(request.start_sequence_number, 0);
    }

    #[test]
    fn accepts_filtered_nonzero_group_and_named_id_zero_lora_with_mm_metadata() {
        let warnings = Arc::new(AtomicU32::new(0));
        let event = pb::KvEvent {
            event: Some(pb::kv_event::Event::BlockStored(pb::BlockStored {
                block_hashes: vec![pb::KvBlockHash {
                    value: b"9".to_vec(),
                    encoding: "decimal_int64".into(),
                }],
                token_ids: vec![1, 2],
                block_size: 2,
                lora_id: 0,
                lora_name: "vision-lora".into(),
                medium: pb::StorageMedium::Gpu as i32,
                extra_keys: vec![pb::OpaqueKeyTuple {
                    values: vec!["trt_mm_v1".into(), "0".into(), "42".into(), "99".into()],
                }],
                group_idx: 1,
                ..Default::default()
            })),
            ..Default::default()
        };
        let converted = convert_event(event, 3, 7, &warnings).unwrap().unwrap();
        let KvCacheEventData::Stored(stored) = converted.data else {
            panic!("expected stored event");
        };
        assert_eq!(stored.blocks.len(), 1);
        assert_eq!(
            stored.blocks[0].mm_extra_info.as_ref().unwrap().mm_objects[0].mm_hash,
            42
        );
        assert_eq!(
            stored.blocks[0].mm_extra_info.as_ref().unwrap().mm_objects[0].offsets,
            Vec::<(usize, usize)>::new()
        );
        assert_eq!(converted.dp_rank, 3);
    }
}
