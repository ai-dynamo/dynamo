// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::AtomicU32;
use std::time::Duration;

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

use crate::client::{self, Control};
use crate::proto as pb;

pub(crate) struct SourceDiscovery {
    pub expected_ranks: HashSet<u32>,
    pub routing_image_token_id: Option<u32>,
    pub deadline: Duration,
    pub cancel: CancellationToken,
    pub tasks: Arc<Mutex<Vec<JoinHandle<()>>>>,
    pub fatal: watch::Sender<Option<String>>,
}

pub(crate) async fn discover_sources(
    channel: Channel,
    mut client: Control,
    discovery: SourceDiscovery,
) -> Result<Vec<KvEventSource>, DynamoError> {
    let SourceDiscovery {
        expected_ranks,
        routing_image_token_id,
        deadline,
        cancel,
        tasks,
        fatal,
    } = discovery;
    let response = tokio::time::timeout(
        deadline,
        client.get_kv_event_sources(pb::GetKvEventSourcesRequest {
            data_parallel_ranks: Vec::new(),
        }),
    )
    .await
    .map_err(|_| client::engine_shutdown("OpenEngine GetKvEventSources timed out"))?
    .map_err(|status| client::status_to_dynamo("GetKvEventSources", status))?
    .into_inner();
    let mut result = Vec::new();
    let mut seen_ranks = HashSet::new();
    let mut usable_ranks = HashSet::new();
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
                if source.encoding != "msgpack" {
                    return Err(client::invalid_arg(format!(
                        "ZMQ KV source for rank {rank} uses unsupported encoding `{}`",
                        source.encoding
                    )));
                }
                let Some(endpoint) = source.endpoint_addr else {
                    return Err(client::invalid_arg("ZMQ KV source omitted endpoint_addr"));
                };
                if endpoint.host.is_empty()
                    || matches!(endpoint.host.as_str(), "*" | "0.0.0.0" | "::" | "[::]")
                    || endpoint.port == 0
                    || endpoint.port > u32::from(u16::MAX)
                {
                    return Err(client::invalid_arg(format!(
                        "ZMQ KV source for rank {rank} omitted a connectable endpoint"
                    )));
                }
                let protocol = if endpoint.protocol.is_empty() {
                    "tcp"
                } else {
                    endpoint.protocol.as_str()
                };
                result.push(KvEventSource::Zmq {
                    endpoint: format_zmq_endpoint(protocol, &endpoint.host, endpoint.port),
                    topic: source.topic,
                    dp_rank: rank,
                    image_token_id: routing_image_token_id,
                });
                usable_ranks.insert(rank);
            }
            "grpc" => {
                if source.encoding != "protobuf" {
                    return Err(client::invalid_arg(format!(
                        "gRPC KV source for rank {rank} uses unsupported encoding `{}`",
                        source.encoding
                    )));
                }
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
                usable_ranks.insert(rank);
            }
            unsupported => {
                tracing::warn!(unsupported, rank, "ignoring unsupported KV event transport")
            }
        }
    }
    // An empty source list means KV events are disabled. Once the server
    // advertises any source, every advertised DP rank must have a transport
    // this sidecar can actually consume.
    if !seen_ranks.is_empty() && usable_ranks != expected_ranks {
        let mut missing = expected_ranks
            .difference(&usable_ranks)
            .copied()
            .collect::<Vec<_>>();
        missing.sort_unstable();
        return Err(client::invalid_arg(format!(
            "OpenEngine KV event discovery omitted data-parallel ranks {missing:?}"
        )));
    }
    Ok(result)
}

fn format_zmq_endpoint(protocol: &str, host: &str, port: u32) -> String {
    if host.contains(':') && !(host.starts_with('[') && host.ends_with(']')) {
        format!("{protocol}://[{host}]:{port}")
    } else {
        format!("{protocol}://{host}:{port}")
    }
}

async fn subscribe_loop(
    channel: Channel,
    rank: u32,
    publisher: Arc<KvEventPublisher>,
    cancel: CancellationToken,
    fatal: watch::Sender<Option<String>>,
) {
    let mut client = pb::control_client::ControlClient::new(channel);
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
                    let events = match convert_batch_events(batch.events, rank, &warnings) {
                        Ok(events) => events,
                        Err(message) => {
                            let message = format!(
                                "invalid OpenEngine KV batch {} for rank {rank}: {message}",
                                batch.sequence_number
                            );
                            let _ = fatal.send(Some(message));
                            return;
                        }
                    };
                    for mut event in events {
                        // Batch continuity is verified above. Dynamo requires a
                        // distinct ID for every normalized event, so expand each
                        // lossless producer batch into local consecutive IDs.
                        event.event_id = publisher.next_event_id();
                        if let Err(error) = publisher.publish(event) {
                            tracing::debug!(%error, rank, "Dynamo KV publisher closed");
                            return;
                        }
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

fn convert_batch_events(
    events: Vec<pb::KvEvent>,
    rank: u32,
    warnings: &Arc<AtomicU32>,
) -> Result<Vec<KvCacheEvent>, String> {
    events
        .into_iter()
        .filter_map(|event| convert_event(event, rank, 0, warnings).transpose())
        .collect()
}

fn subscription_request(rank: u32) -> pb::SubscribeKvEventsRequest {
    pb::SubscribeKvEventsRequest {
        data_parallel_ranks: vec![rank],
        // OpenEngine does not advertise snapshot/replay capability. Request
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
        // The caller assigns a distinct local ID after validating the producer
        // batch sequence. A batch may contain more than one normalized event.
        event_id: sequence_number,
        data,
        dp_rank: rank,
    }))
}

fn accept_sequence(last: &mut Option<u64>, current: u64) -> bool {
    let accepted = match *last {
        None => true,
        Some(previous) => previous.checked_add(1) == Some(current),
    };
    if !accepted {
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
    fn preserves_batch_ids_and_rejects_sequence_gaps() {
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
        assert!(accept_sequence(&mut last, 8));
        assert!(!accept_sequence(&mut last, 10));
        assert!(!accept_sequence(&mut last, 8));
    }

    #[test]
    fn live_subscription_does_not_request_unsupported_snapshot_or_replay() {
        let request = subscription_request(3);
        assert_eq!(request.data_parallel_ranks, vec![3]);
        assert!(!request.include_snapshot);
        assert_eq!(request.start_sequence_number, 0);
    }

    #[test]
    fn repeated_events_in_one_batch_are_all_retained() {
        let warnings = Arc::new(AtomicU32::new(0));
        let cleared = || pb::KvEvent {
            event: Some(pb::kv_event::Event::AllBlocksCleared(
                pb::AllBlocksCleared {},
            )),
            ..Default::default()
        };
        let events = convert_batch_events(vec![cleared(), cleared()], 3, &warnings).unwrap();
        assert_eq!(events.len(), 2);
        assert!(events.iter().all(|event| event.dp_rank == 3));
    }

    #[test]
    fn brackets_ipv6_zmq_hosts() {
        assert_eq!(
            format_zmq_endpoint("tcp", "2001:db8::1", 5557),
            "tcp://[2001:db8::1]:5557"
        );
        assert_eq!(
            format_zmq_endpoint("tcp", "127.0.0.1", 5557),
            "tcp://127.0.0.1:5557"
        );
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
