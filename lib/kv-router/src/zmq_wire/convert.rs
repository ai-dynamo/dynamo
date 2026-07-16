// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::protocols::{
    BlockExtraInfo, BlockHashOptions, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData,
    KvCacheRemoveData, KvCacheStoreData, KvCacheStoredBlockData, Placement, PlacementEvent,
    StorageTier, WorkerWithDpRank, compute_block_hash_for_seq,
};

use super::types::{BlockHashValue, RawKvEvent};

/// Convert a raw event coming from the ZMQ channel into a placement-aware worker event.
pub fn convert_event(
    raw: RawKvEvent,
    event_id: u64,
    kv_block_size: u32,
    worker: WorkerWithDpRank,
    warning_count: &Arc<AtomicU32>,
    image_token_id: Option<u32>,
    video_token_id: Option<u32>,
) -> Option<PlacementEvent> {
    let storage_tier = match &raw {
        RawKvEvent::BlockStored { medium, .. } | RawKvEvent::BlockRemoved { medium, .. } => {
            StorageTier::from_kv_medium_or_default(medium.as_deref())
        }
        RawKvEvent::AllBlocksCleared => StorageTier::Device,
        RawKvEvent::Ignored => return None,
    };
    let dp_rank = worker.dp_rank;
    let event = match raw {
        RawKvEvent::BlockStored {
            block_hashes,
            parent_block_hash,
            token_ids,
            block_size,
            lora_name,
            cache_namespace,
            block_mm_infos,
            medium: _,
            is_eagle,
            group_idx: _,
            kv_cache_spec_kind: _,
            kv_cache_spec_sliding_window: _,
        } => {
            // Reject self-referencing blocks: all block hashes (including parent) must be unique.
            {
                let mut seen = HashSet::with_capacity(block_hashes.len() + 1);
                if let Some(parent) = parent_block_hash {
                    seen.insert(parent.into_u64());
                }
                let has_duplicate = block_hashes.iter().any(|h| !seen.insert(h.into_u64()));
                if has_duplicate {
                    tracing::warn!(
                        event_id,
                        "Self-referencing block detected: duplicate hash in store event; dropping"
                    );
                    // Return an empty Removed instead of Cleared to avoid nuking
                    // the worker's entire index state. An empty Removed is a no-op
                    // in the radix tree (zero iterations, returns Ok(())).
                    return Some(PlacementEvent::new(
                        Placement::local_worker(worker.worker_id, worker.dp_rank, storage_tier),
                        KvCacheEvent {
                            event_id,
                            data: KvCacheEventData::Removed(KvCacheRemoveData {
                                block_hashes: vec![],
                            }),
                            dp_rank,
                        },
                    ));
                }
            }

            let num_block_tokens = vec![block_size as u64; block_hashes.len()];
            let block_hashes_u64: Vec<u64> = block_hashes
                .into_iter()
                .map(BlockHashValue::into_u64)
                .collect();
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: parent_block_hash
                        .map(BlockHashValue::into_u64)
                        .map(ExternalSequenceBlockHash::from),
                    start_position: None,
                    blocks: create_stored_blocks(
                        kv_block_size,
                        &token_ids,
                        &num_block_tokens,
                        &block_hashes_u64,
                        lora_name.as_deref(),
                        cache_namespace.as_deref(),
                        warning_count,
                        block_mm_infos.as_deref(),
                        is_eagle,
                        image_token_id,
                        video_token_id,
                    ),
                }),
                dp_rank,
            }
        }
        RawKvEvent::BlockRemoved { block_hashes, .. } => {
            let hashes = block_hashes
                .into_iter()
                .map(BlockHashValue::into_u64)
                .map(ExternalSequenceBlockHash::from)
                .collect();
            KvCacheEvent {
                event_id,
                data: KvCacheEventData::Removed(KvCacheRemoveData {
                    block_hashes: hashes,
                }),
                dp_rank,
            }
        }
        RawKvEvent::AllBlocksCleared => KvCacheEvent {
            event_id,
            data: KvCacheEventData::Cleared,
            dp_rank,
        },
        RawKvEvent::Ignored => unreachable!("ignored events return before conversion"),
    };

    Some(PlacementEvent::new(
        Placement::local_worker(worker.worker_id, worker.dp_rank, storage_tier),
        event,
    ))
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MmTokenKind {
    Image,
    Video,
}

fn mm_token_kind(
    token_id: u32,
    image_token_id: Option<u32>,
    video_token_id: Option<u32>,
) -> Option<MmTokenKind> {
    if image_token_id == Some(token_id) {
        Some(MmTokenKind::Image)
    } else if video_token_id == Some(token_id) {
        Some(MmTokenKind::Video)
    } else {
        None
    }
}

fn find_unique_object_mapping(
    group_kinds: &[MmTokenKind],
    mm_objects: &[u64],
    inferred_object_kinds: &HashMap<u64, MmTokenKind>,
) -> Option<Vec<u64>> {
    fn visit(
        group_kinds: &[MmTokenKind],
        mm_objects: &[u64],
        inferred_object_kinds: &HashMap<u64, MmTokenKind>,
        group_index: usize,
        object_start: usize,
        current: &mut Vec<u64>,
        solution: &mut Option<Vec<u64>>,
        ambiguous: &mut bool,
    ) {
        if *ambiguous {
            return;
        }
        if group_index == group_kinds.len() {
            if solution.is_some() {
                *ambiguous = true;
            } else {
                *solution = Some(current.clone());
            }
            return;
        }

        let remaining_groups = group_kinds.len() - group_index;
        if mm_objects.len().saturating_sub(object_start) < remaining_groups {
            return;
        }
        let last_candidate = mm_objects.len() - remaining_groups;
        for object_index in object_start..=last_candidate {
            let mm_hash = mm_objects[object_index];
            if inferred_object_kinds.get(&mm_hash) != Some(&group_kinds[group_index]) {
                continue;
            }
            current.push(mm_hash);
            visit(
                group_kinds,
                mm_objects,
                inferred_object_kinds,
                group_index + 1,
                object_index + 1,
                current,
                solution,
                ambiguous,
            );
            current.pop();
            if *ambiguous {
                return;
            }
        }
    }

    let mut solution = None;
    let mut ambiguous = false;
    visit(
        group_kinds,
        mm_objects,
        inferred_object_kinds,
        0,
        0,
        &mut Vec::with_capacity(group_kinds.len()),
        &mut solution,
        &mut ambiguous,
    );
    (!ambiguous).then_some(solution).flatten()
}

fn infer_mm_object_kinds(
    token_ids: &[u32],
    num_block_tokens: &[u64],
    block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
    is_eagle: Option<bool>,
    image_token_id: Option<u32>,
    video_token_id: Option<u32>,
) -> HashMap<u64, MmTokenKind> {
    let Some(block_mm_infos) = block_mm_infos else {
        return HashMap::new();
    };
    if image_token_id.is_some() && image_token_id == video_token_id {
        return HashMap::new();
    }

    let append = is_eagle.unwrap_or(false) as usize;
    let mut inferred: HashMap<u64, Option<MmTokenKind>> = HashMap::new();
    let mut token_offset = 0usize;
    for (block_index, num_block_tokens) in num_block_tokens.iter().enumerate() {
        let end = token_offset + append + *num_block_tokens as usize;
        if end > token_ids.len() {
            break;
        }
        let Some(info) = block_mm_infos.get(block_index).and_then(Option::as_ref) else {
            token_offset += *num_block_tokens as usize;
            continue;
        };
        if info.mm_objects.len() != 1 {
            token_offset += *num_block_tokens as usize;
            continue;
        }

        let mut observed_kind = None;
        let mut mixed_kinds = false;
        for &token_id in &token_ids[token_offset..end] {
            let Some(kind) = mm_token_kind(token_id, image_token_id, video_token_id) else {
                continue;
            };
            if observed_kind.is_some_and(|observed| observed != kind) {
                mixed_kinds = true;
                break;
            }
            observed_kind = Some(kind);
        }
        if !mixed_kinds && let Some(kind) = observed_kind {
            let mm_hash = info.mm_objects[0].mm_hash;
            inferred
                .entry(mm_hash)
                .and_modify(|existing| {
                    if *existing != Some(kind) {
                        *existing = None;
                    }
                })
                .or_insert(Some(kind));
        }
        token_offset += *num_block_tokens as usize;
    }

    inferred
        .into_iter()
        .filter_map(|(mm_hash, kind)| kind.map(|kind| (mm_hash, kind)))
        .collect()
}

/// Rewrite model placeholder runs to Dynamo's canonical `pad_value(mm_hash)`.
/// Images contribute one run per object. A Qwen video can contribute several
/// timestamp-separated runs, so consecutive video runs belong to one object.
/// vLLM may attach the next MM object to a boundary block before its placeholder
/// appears. In that case, use kinds inferred from unambiguous blocks in the same
/// event only when they identify one ordered mapping. Otherwise, do not guess:
/// the caller keeps the event's native MM hash path for that block.
fn substitute_pad_values(
    token_ids: &[u32],
    image_token_id: Option<u32>,
    video_token_id: Option<u32>,
    mm_objects: &[u64],
    inferred_object_kinds: Option<&HashMap<u64, MmTokenKind>>,
) -> Option<Vec<u32>> {
    if image_token_id.is_some() && image_token_id == video_token_id {
        return None;
    }

    let mut runs = Vec::new();
    let mut token_index = 0usize;
    while token_index < token_ids.len() {
        let Some(kind) = mm_token_kind(token_ids[token_index], image_token_id, video_token_id)
        else {
            token_index += 1;
            continue;
        };
        let start = token_index;
        token_index += 1;
        while token_index < token_ids.len()
            && mm_token_kind(token_ids[token_index], image_token_id, video_token_id) == Some(kind)
        {
            token_index += 1;
        }
        runs.push((start, token_index, kind));
    }

    // Blocks containing only timestamp/structural tokens need no substitution.
    if runs.is_empty() {
        return Some(token_ids.to_vec());
    }

    let mut run_groups = Vec::with_capacity(runs.len());
    let mut group_kinds = Vec::with_capacity(runs.len());
    let mut group_count = 0usize;
    let mut previous_kind = None;
    for &(_, _, kind) in &runs {
        let starts_new_object =
            kind == MmTokenKind::Image || previous_kind != Some(MmTokenKind::Video);
        if starts_new_object {
            group_count += 1;
            group_kinds.push(kind);
        }
        run_groups.push(group_count - 1);
        previous_kind = Some(kind);
    }

    let group_hashes = if group_count == mm_objects.len() {
        mm_objects.to_vec()
    } else {
        let mapping = find_unique_object_mapping(&group_kinds, mm_objects, inferred_object_kinds?);
        let Some(mapping) = mapping else {
            tracing::debug!(
                inferred_objects = group_count,
                event_objects = mm_objects.len(),
                "multimodal placeholder runs cannot be mapped to event objects exactly; preserving native event hashing"
            );
            return None;
        };
        mapping
    };

    let mut out = token_ids.to_vec();
    for ((start, end, _), group_index) in runs.into_iter().zip(run_groups) {
        let pad = crate::protocols::pad_value_for_mm_hash(group_hashes[group_index]);
        out[start..end].fill(pad);
    }
    Some(out)
}

#[derive(Default)]
pub struct StoredBlockOptions<'a> {
    pub lora_name: Option<&'a str>,
    pub cache_namespace: Option<&'a str>,
    pub mm_extra_info: Option<BlockExtraInfo>,
    pub is_eagle: Option<bool>,
    pub image_token_id: Option<u32>,
    pub video_token_id: Option<u32>,
}

pub fn create_stored_block_from_parts(
    kv_block_size: u32,
    block_hash: u64,
    token_ids: &[u32],
    options: StoredBlockOptions<'_>,
) -> KvCacheStoredBlockData {
    create_stored_block_from_parts_with_kinds(kv_block_size, block_hash, token_ids, options, None)
}

fn create_stored_block_from_parts_with_kinds(
    kv_block_size: u32,
    block_hash: u64,
    token_ids: &[u32],
    options: StoredBlockOptions<'_>,
    inferred_object_kinds: Option<&HashMap<u64, MmTokenKind>>,
) -> KvCacheStoredBlockData {
    let StoredBlockOptions {
        lora_name,
        cache_namespace,
        mm_extra_info,
        is_eagle,
        image_token_id,
        video_token_id,
    } = options;

    // When the model exposes routing placeholder tokens and this block carries
    // MM objects (vLLM events), normalize to the canonical pad_value scheme and
    // hash without block_mm_infos, matching the frontend. SGLang events carry
    // neither model placeholders nor mm_extra_info, so they remain unchanged.
    let normalized_tokens = match mm_extra_info.as_ref() {
        Some(info)
            if (image_token_id.is_some() || video_token_id.is_some())
                && !info.mm_objects.is_empty() =>
        {
            let mm_hashes: Vec<u64> = info.mm_objects.iter().map(|o| o.mm_hash).collect();
            substitute_pad_values(
                token_ids,
                image_token_id,
                video_token_id,
                &mm_hashes,
                inferred_object_kinds,
            )
        }
        _ => None,
    };
    let fallback_mm_infos = if normalized_tokens.is_none() {
        mm_extra_info.as_ref().map(|info| vec![Some(info.clone())])
    } else {
        None
    };
    let tokens_hash = compute_block_hash_for_seq(
        normalized_tokens.as_deref().unwrap_or(token_ids),
        kv_block_size,
        BlockHashOptions {
            block_mm_infos: fallback_mm_infos.as_deref(),
            lora_name,
            cache_namespace,
            is_eagle,
        },
    )[0];

    tracing::trace!(
        "Creating stored block: external_block_hash={}, tokens_hash={}, token_ids={:?}, kv_block_size={}, mm_extra_info={:?}",
        block_hash,
        tokens_hash.0,
        token_ids,
        kv_block_size,
        mm_extra_info
    );
    KvCacheStoredBlockData {
        block_hash: ExternalSequenceBlockHash::from(block_hash),
        tokens_hash,
        mm_extra_info,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn create_stored_blocks(
    kv_block_size: u32,
    token_ids: &[u32],
    num_block_tokens: &[u64],
    block_hashes: &[u64],
    lora_name: Option<&str>,
    cache_namespace: Option<&str>,
    warning_count: &Arc<AtomicU32>,
    block_mm_infos: Option<&[Option<BlockExtraInfo>]>,
    is_eagle: Option<bool>,
    image_token_id: Option<u32>,
    video_token_id: Option<u32>,
) -> Vec<KvCacheStoredBlockData> {
    let mut blocks: Vec<KvCacheStoredBlockData> = Vec::new();

    let mut token_offset: usize = 0;
    let append = is_eagle.unwrap_or(false) as usize;
    let inferred_object_kinds = infer_mm_object_kinds(
        token_ids,
        num_block_tokens,
        block_mm_infos,
        is_eagle,
        image_token_id,
        video_token_id,
    );

    for (block_idx, (num_tokens_it, block_hash_it)) in
        num_block_tokens.iter().zip(block_hashes.iter()).enumerate()
    {
        if *num_tokens_it != kv_block_size as u64 {
            if warning_count.fetch_add(1, Ordering::Relaxed) < 3 {
                tracing::warn!(
                    "Block not published. Block size must be {} tokens to be published. Block size is: {}",
                    kv_block_size,
                    *num_tokens_it
                );
            }
            break;
        }

        let end = token_offset + append + *num_tokens_it as usize;
        if end > token_ids.len() {
            if warning_count.fetch_add(1, Ordering::Relaxed) < 3 {
                tracing::warn!(
                    "Block not published. token_ids too short: need {}, got {}",
                    end,
                    token_ids.len()
                );
            }
            break;
        }

        let tokens = &token_ids[token_offset..end];
        let mm_extra_info = block_mm_infos
            .and_then(|infos| infos.get(block_idx))
            .and_then(|opt| opt.clone());

        blocks.push(create_stored_block_from_parts_with_kinds(
            kv_block_size,
            *block_hash_it,
            tokens,
            StoredBlockOptions {
                lora_name,
                cache_namespace,
                mm_extra_info,
                is_eagle,
                image_token_id,
                video_token_id,
            },
            Some(&inferred_object_kinds),
        ));
        token_offset += *num_tokens_it as usize;
    }

    blocks
}

#[cfg(test)]
mod normalize_tests {
    use super::*;
    use crate::protocols::{BlockMmObjectInfo, pad_value_for_mm_hash};

    /// A normalized vLLM block (image_token_id run + mm_hash) must hash
    /// identically to the frontend's pad_value scheme. The parity the
    /// consolidation rests on.
    #[test]
    fn vllm_event_normalizes_to_frontend_pad_value_hash() {
        let block_size = 4u32;
        let image_token_id = 151655u32;
        let mm_hash = 9_533_257_059_414_191_570u64;
        // vLLM-style block: two real tokens then an image run.
        let vllm_tokens = vec![10u32, 20, image_token_id, image_token_id];
        let mm_info = BlockExtraInfo {
            mm_objects: vec![BlockMmObjectInfo {
                mm_hash,
                offsets: vec![],
            }],
        };

        let stored = create_stored_block_from_parts(
            block_size,
            0xabcd,
            &vllm_tokens,
            StoredBlockOptions {
                mm_extra_info: Some(mm_info),
                image_token_id: Some(image_token_id),
                ..Default::default()
            },
        );

        // Frontend side: same tokens but image positions already pad_value,
        // hashed WITHOUT block_mm_infos.
        let pad = pad_value_for_mm_hash(mm_hash);
        let frontend_tokens = vec![10u32, 20, pad, pad];
        let expected =
            compute_block_hash_for_seq(&frontend_tokens, block_size, BlockHashOptions::default())
                [0];

        assert_eq!(
            stored.tokens_hash, expected,
            "normalized vLLM event hash must match frontend pad_value hash"
        );
    }

    #[test]
    fn vllm_video_event_normalizes_to_frontend_pad_value_hash() {
        let block_size = 4u32;
        let video_token_id = 151656u32;
        let mm_hash = 9_533_257_059_414_191_570u64;
        let vllm_tokens = vec![10u32, 20, video_token_id, video_token_id];
        let mm_info = BlockExtraInfo {
            mm_objects: vec![BlockMmObjectInfo {
                mm_hash,
                offsets: vec![],
            }],
        };

        let stored = create_stored_block_from_parts(
            block_size,
            0xabcd,
            &vllm_tokens,
            StoredBlockOptions {
                mm_extra_info: Some(mm_info),
                video_token_id: Some(video_token_id),
                ..Default::default()
            },
        );
        let pad = pad_value_for_mm_hash(mm_hash);
        let expected = compute_block_hash_for_seq(
            &[10u32, 20, pad, pad],
            block_size,
            BlockHashOptions::default(),
        )[0];

        assert_eq!(stored.tokens_hash, expected);
    }

    #[test]
    fn timestamped_video_runs_share_one_hash_before_an_image() {
        let image_token_id = 151655u32;
        let video_token_id = 151656u32;
        let video_hash = 41u64;
        let image_hash = 42u64;
        let tokens = [
            video_token_id,
            video_token_id,
            7,
            video_token_id,
            video_token_id,
            8,
            image_token_id,
            image_token_id,
        ];

        let normalized = substitute_pad_values(
            &tokens,
            Some(image_token_id),
            Some(video_token_id),
            &[video_hash, image_hash],
            None,
        )
        .unwrap();

        let video_pad = pad_value_for_mm_hash(video_hash);
        let image_pad = pad_value_for_mm_hash(image_hash);
        assert_eq!(
            normalized,
            [
                video_pad, video_pad, 7, video_pad, video_pad, 8, image_pad, image_pad,
            ]
        );
    }

    #[test]
    fn consecutive_video_objects_fail_closed_without_offsets() {
        let video_token_id = 151656u32;
        let tokens = [video_token_id, 7, video_token_id];

        assert!(
            substitute_pad_values(&tokens, None, Some(video_token_id), &[41, 42], None).is_none()
        );
    }

    #[test]
    fn image_run_object_mismatch_fails_closed() {
        let image_token_id = 151655u32;

        assert!(
            substitute_pad_values(
                &[image_token_id, 7, image_token_id],
                Some(image_token_id),
                None,
                &[41],
                None,
            )
            .is_none()
        );
        assert!(
            substitute_pad_values(
                &[image_token_id, image_token_id],
                Some(image_token_id),
                None,
                &[41, 42],
                None,
            )
            .is_none()
        );
    }

    #[test]
    fn mixed_boundary_uses_kind_inferred_from_neighboring_blocks() {
        let block_size = 4u32;
        let image_token_id = 151655u32;
        let video_token_id = 151656u32;
        let image_hash = 41u64;
        let video_hash = 42u64;
        let tokens = [
            image_token_id,
            image_token_id,
            image_token_id,
            image_token_id,
            image_token_id,
            image_token_id,
            7,
            8,
            9,
            10,
            video_token_id,
            video_token_id,
        ];
        let block_mm_infos = [
            Some(BlockExtraInfo {
                mm_objects: vec![BlockMmObjectInfo {
                    mm_hash: image_hash,
                    offsets: vec![],
                }],
            }),
            Some(BlockExtraInfo {
                mm_objects: vec![
                    BlockMmObjectInfo {
                        mm_hash: image_hash,
                        offsets: vec![],
                    },
                    BlockMmObjectInfo {
                        mm_hash: video_hash,
                        offsets: vec![],
                    },
                ],
            }),
            Some(BlockExtraInfo {
                mm_objects: vec![BlockMmObjectInfo {
                    mm_hash: video_hash,
                    offsets: vec![],
                }],
            }),
        ];

        let stored = create_stored_blocks(
            block_size,
            &tokens,
            &[4, 4, 4],
            &[101, 102, 103],
            None,
            None,
            &Arc::new(AtomicU32::new(0)),
            Some(&block_mm_infos),
            None,
            Some(image_token_id),
            Some(video_token_id),
        );

        let image_pad = pad_value_for_mm_hash(image_hash);
        let video_pad = pad_value_for_mm_hash(video_hash);
        let expected_tokens = [
            image_pad, image_pad, image_pad, image_pad, image_pad, image_pad, 7, 8, 9, 10,
            video_pad, video_pad,
        ];
        let expected_hashes =
            compute_block_hash_for_seq(&expected_tokens, block_size, BlockHashOptions::default());

        assert_eq!(stored.len(), 3);
        assert_eq!(
            stored
                .iter()
                .map(|block| block.tokens_hash)
                .collect::<Vec<_>>(),
            expected_hashes
        );
    }

    /// sglang-style events carry no image_token_id tokens nor mm_extra_info, so
    /// passing image_token_id is a no-op: the hash is over the raw tokens.
    #[test]
    fn sglang_event_unaffected_by_image_token_id() {
        let block_size = 4u32;
        let pad = pad_value_for_mm_hash(42);
        let tokens = vec![1u32, 2, pad, pad];

        let with_img = create_stored_block_from_parts(
            block_size,
            0x1,
            &tokens,
            StoredBlockOptions {
                image_token_id: Some(151655),
                ..Default::default()
            },
        );
        let without =
            create_stored_block_from_parts(block_size, 0x1, &tokens, StoredBlockOptions::default());
        assert_eq!(with_img.tokens_hash, without.tokens_hash);
    }
}
