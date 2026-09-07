// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::protocols::{
    BlockExtraInfo, BlockHashOptions, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData,
    KvCacheRemoveData, KvCacheStoreData, KvCacheStoredBlockData, Placement, PlacementEvent,
    StorageTier, WorkerWithDpRank, compute_block_hash_for_seq,
};

use super::types::{BlockHashValue, Locality, RawKvEvent};

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
    // Read the wire tier/locality facts up front, before any indexing work.
    let (medium, locality) = match &raw {
        RawKvEvent::BlockStored {
            medium, locality, ..
        }
        | RawKvEvent::BlockRemoved {
            medium, locality, ..
        } => (medium.as_deref(), *locality),
        RawKvEvent::AllBlocksCleared { .. } => (None, None),
        RawKvEvent::Ignored => return None,
    };

    // No consumer exists for a shared/global index yet (dynamo #10457), so
    // REMOTE and any unrecognized locality fail closed. Absent or LOCAL keeps
    // the event worker-local, matching legacy CPU-offload events that never
    // carried a locality field. The normalizer's preprocess step classifies
    // these as filtered first; this guard is a defensive backstop for direct
    // convert_event callers that bypass preprocess.
    if matches!(locality, Some(Locality::Remote | Locality::Unknown)) {
        tracing::trace!(event_id, "Dropping non-local KV event (locality != LOCAL)");
        return None;
    }

    // Fail closed on unrecognized media instead of silently indexing them on
    // the device (G1) primary tree. vLLM 0.26.0 ships `FS` / `OBJ` (pre-#48123
    // wire); those and any future medium strings are dropped, not misfiled. The
    // normalizer's preprocess step classifies these as filtered first (so no
    // event id is burned); this guard is a defensive backstop for direct
    // convert_event callers that bypass preprocess, mirroring the locality gate.
    let storage_tier = match medium {
        None => StorageTier::Device,
        Some(medium) => match StorageTier::from_kv_medium(medium) {
            Some(tier) => tier,
            None => {
                if warning_count.fetch_add(1, Ordering::Relaxed) < 3 {
                    tracing::warn!(event_id, medium, "Dropping KV event with unknown medium");
                }
                return None;
            }
        },
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
            locality: _,
            ownership: _,
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
        RawKvEvent::AllBlocksCleared { .. } => KvCacheEvent {
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

/// Rewrite model placeholder runs to Dynamo's canonical `pad_value(mm_hash)`.
/// Images contribute one run per object. A Qwen video can contribute several
/// timestamp-separated runs, so consecutive video runs belong to one object.
/// Only an exact ordered run/object mapping is normalized. Boundary or mixed
/// blocks that cannot be mapped exactly preserve vLLM's native MM hash path.
pub fn normalize_mm_placeholder_runs(
    token_ids: &[u32],
    image_token_id: Option<u32>,
    video_token_id: Option<u32>,
    mm_objects: &[u64],
) -> Option<(Vec<u32>, usize)> {
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

    // Dropping MM metadata from a runless boundary block would collapse
    // different media objects onto the same token-only hash.
    if runs.is_empty() {
        return None;
    }

    let mut run_groups = Vec::with_capacity(runs.len());
    let mut group_count = 0usize;
    let mut previous_kind = None;
    for &(_, _, kind) in &runs {
        let starts_new_object =
            kind == MmTokenKind::Image || previous_kind != Some(MmTokenKind::Video);
        if starts_new_object {
            group_count += 1;
        }
        run_groups.push(group_count - 1);
        previous_kind = Some(kind);
    }

    if group_count != mm_objects.len() {
        tracing::debug!(
            inferred_objects = group_count,
            event_objects = mm_objects.len(),
            "multimodal placeholder runs cannot be mapped to event objects exactly; preserving native event hashing"
        );
        return None;
    }

    let mut out = token_ids.to_vec();
    for ((start, end, _), group_index) in runs.into_iter().zip(run_groups) {
        let pad = crate::protocols::pad_value_for_mm_hash(mm_objects[group_index]);
        out[start..end].fill(pad);
    }
    Some((out, group_count))
}

/// Rewrite each `image_token_id` run in `token_ids` to `pad_value(mm_hash)`,
/// assigning one MM hash per run in order and clamping excess runs to the last
/// hash. Returns the normalized tokens and the number of discovered runs.
///
/// This is the existing image-only request/event normalization contract used
/// by `/generate`. Video events use the stricter helper above.
pub fn normalize_mm_token_runs(
    token_ids: &[u32],
    image_token_id: u32,
    mm_hashes: &[u64],
) -> Option<(Vec<u32>, usize)> {
    let last_mm_hash = *mm_hashes.last()?;
    let mut out = Vec::with_capacity(token_ids.len());
    let mut object_index = 0usize;
    let mut in_run = false;
    let mut runs = 0usize;
    let mut run_pad = 0u32;
    for &token_id in token_ids {
        if token_id == image_token_id {
            if !in_run {
                in_run = true;
                runs += 1;
                let mm_hash = mm_hashes.get(object_index).copied().unwrap_or(last_mm_hash);
                run_pad = crate::protocols::pad_value_for_mm_hash(mm_hash);
            }
            out.push(run_pad);
        } else {
            if in_run {
                in_run = false;
                object_index += 1;
            }
            out.push(token_id);
        }
    }
    Some((out, runs))
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
    let requires_exact_mm_mapping = options.video_token_id.is_some();
    create_stored_block_from_parts_with_video_context(
        kv_block_size,
        block_hash,
        token_ids,
        options,
        requires_exact_mm_mapping,
    )
}

fn create_stored_block_from_parts_with_video_context(
    kv_block_size: u32,
    block_hash: u64,
    token_ids: &[u32],
    options: StoredBlockOptions<'_>,
    requires_exact_mm_mapping: bool,
) -> KvCacheStoredBlockData {
    let StoredBlockOptions {
        lora_name,
        cache_namespace,
        mm_extra_info,
        is_eagle,
        image_token_id,
        video_token_id,
    } = options;

    // Preserve the existing image-only run-order contract for models without
    // video support. Video-capable models always use exact modality-aware
    // mapping: an incremental vLLM batch can omit a later video placeholder
    // even when a boundary block already carries its MM identity. Both
    // canonical paths hash without block_mm_infos; ambiguous blocks preserve
    // vLLM's native MM hash instead. SGLang events carry neither placeholder
    // ids nor mm_extra_info and are unchanged.
    let normalized_tokens = match mm_extra_info.as_ref() {
        Some(info) if requires_exact_mm_mapping && !info.mm_objects.is_empty() => {
            let mm_hashes: Vec<u64> = info.mm_objects.iter().map(|o| o.mm_hash).collect();
            normalize_mm_placeholder_runs(token_ids, image_token_id, video_token_id, &mm_hashes)
                .map(|(tokens, _)| tokens)
        }
        Some(info)
            if image_token_id.is_some_and(|image_token_id| token_ids.contains(&image_token_id))
                && !info.mm_objects.is_empty() =>
        {
            let mm_hashes: Vec<u64> = info.mm_objects.iter().map(|o| o.mm_hash).collect();
            normalize_mm_token_runs(
                token_ids,
                image_token_id.expect("image token checked above"),
                &mm_hashes,
            )
            .map(|(tokens, runs)| {
                if runs != mm_hashes.len() {
                    tracing::debug!(
                        runs,
                        mm_objects = mm_hashes.len(),
                        "image_token_id run count != mm_object count; pad_value assignment is best-effort by run order"
                    );
                }
                tokens
            })
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
    let requires_exact_mm_mapping = video_token_id.is_some();

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

        blocks.push(create_stored_block_from_parts_with_video_context(
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
            requires_exact_mm_mapping,
        ));
        token_offset += *num_tokens_it as usize;
    }

    blocks
}

#[cfg(test)]
mod normalize_tests {
    use super::*;
    use crate::protocols::{BlockMmObjectInfo, pad_value_for_mm_hash};

    #[test]
    fn image_only_normalization_preserves_worker_order_and_clamping() {
        let image_token_id = 99;
        let (normalized, runs) =
            normalize_mm_token_runs(&[10, 99, 42, 99, 20, 99], image_token_id, &[7, 8])
                .expect("non-empty hashes normalize");

        assert_eq!(runs, 3);
        assert_eq!(
            normalized,
            vec![
                10,
                pad_value_for_mm_hash(7),
                42,
                pad_value_for_mm_hash(8),
                20,
                pad_value_for_mm_hash(8),
            ]
        );
        assert!(normalize_mm_token_runs(&[99], image_token_id, &[]).is_none());
    }

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
    fn two_separated_images_preserve_frontend_hash_parity() {
        let block_size = 6u32;
        let image_token_id = 151655u32;
        let image_hashes = [41u64, 42u64];
        let tokens = [
            image_token_id,
            image_token_id,
            7,
            image_token_id,
            image_token_id,
            8,
        ];
        let mm_info = BlockExtraInfo {
            mm_objects: image_hashes
                .iter()
                .map(|mm_hash| BlockMmObjectInfo {
                    mm_hash: *mm_hash,
                    offsets: vec![],
                })
                .collect(),
        };

        let stored = create_stored_block_from_parts(
            block_size,
            0xabcd,
            &tokens,
            StoredBlockOptions {
                mm_extra_info: Some(mm_info),
                image_token_id: Some(image_token_id),
                ..Default::default()
            },
        );
        let expected_tokens = [
            pad_value_for_mm_hash(image_hashes[0]),
            pad_value_for_mm_hash(image_hashes[0]),
            7,
            pad_value_for_mm_hash(image_hashes[1]),
            pad_value_for_mm_hash(image_hashes[1]),
            8,
        ];
        let expected =
            compute_block_hash_for_seq(&expected_tokens, block_size, BlockHashOptions::default())
                [0];

        assert_eq!(stored.tokens_hash, expected);
    }

    #[test]
    fn video_capable_model_preserves_unambiguous_image_normalization() {
        let block_size = 6u32;
        let image_token_id = 99u32;
        let video_token_id = 100u32;
        let tokens = [10, image_token_id, 20, image_token_id, 30, 40];
        let mm_info = BlockExtraInfo {
            mm_objects: [41u64, 42u64]
                .into_iter()
                .map(|mm_hash| BlockMmObjectInfo {
                    mm_hash,
                    offsets: vec![],
                })
                .collect(),
        };
        let expected_tokens = normalize_mm_token_runs(&tokens, image_token_id, &[41, 42])
            .expect("image-only event normalizes")
            .0;
        let expected =
            compute_block_hash_for_seq(&expected_tokens, block_size, BlockHashOptions::default())
                [0];

        let stored = create_stored_block_from_parts(
            block_size,
            0xabcd,
            &tokens,
            StoredBlockOptions {
                mm_extra_info: Some(mm_info),
                image_token_id: Some(image_token_id),
                video_token_id: Some(video_token_id),
                ..Default::default()
            },
        );

        assert_eq!(stored.tokens_hash, expected);
    }

    #[test]
    fn video_capable_ambiguous_image_mapping_preserves_native_hash() {
        let block_size = 6u32;
        let image_token_id = 99u32;
        let video_token_id = 100u32;
        let tokens = [10, image_token_id, 20, image_token_id, 30, image_token_id];
        let mm_info = BlockExtraInfo {
            mm_objects: [41u64, 42u64]
                .into_iter()
                .map(|mm_hash| BlockMmObjectInfo {
                    mm_hash,
                    offsets: vec![],
                })
                .collect(),
        };

        let stored = create_stored_block_from_parts(
            block_size,
            0xabcd,
            &tokens,
            StoredBlockOptions {
                mm_extra_info: Some(mm_info.clone()),
                image_token_id: Some(image_token_id),
                video_token_id: Some(video_token_id),
                ..Default::default()
            },
        );
        let expected = create_stored_block_from_parts(
            block_size,
            0xabcd,
            &tokens,
            StoredBlockOptions {
                mm_extra_info: Some(mm_info),
                ..Default::default()
            },
        );

        assert_eq!(stored.tokens_hash, expected.tokens_hash);
    }

    #[test]
    fn incremental_batch_preserves_later_video_identity_on_boundary_block() {
        let block_size = 4u32;
        let image_token_id = 151655u32;
        let video_token_id = 151656u32;
        let image_hash = 41u64;
        // This incremental batch ends before the later video placeholder.
        let tokens = [
            image_token_id,
            image_token_id,
            image_token_id,
            image_token_id,
            image_token_id,
            image_token_id,
            7,
            8,
        ];
        let make_blocks = |video_hash| {
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
            ];
            create_stored_blocks(
                block_size,
                &tokens,
                &[4, 4],
                &[101, 102],
                None,
                None,
                &Arc::new(AtomicU32::new(0)),
                Some(&block_mm_infos),
                None,
                Some(image_token_id),
                Some(video_token_id),
            )
        };

        let first_video = make_blocks(42);
        let second_video = make_blocks(43);

        assert_eq!(first_video[0].tokens_hash, second_video[0].tokens_hash);
        assert_ne!(first_video[1].tokens_hash, second_video[1].tokens_hash);
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

        let normalized = normalize_mm_placeholder_runs(
            &tokens,
            Some(image_token_id),
            Some(video_token_id),
            &[video_hash, image_hash],
        )
        .unwrap()
        .0;

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
            normalize_mm_placeholder_runs(&tokens, None, Some(video_token_id), &[41, 42]).is_none()
        );
    }

    #[test]
    fn exact_video_mapping_rejects_image_object_count_mismatch() {
        let image_token_id = 151655u32;
        let video_token_id = 151656u32;

        assert!(
            normalize_mm_placeholder_runs(
                &[image_token_id, 7, image_token_id],
                Some(image_token_id),
                Some(video_token_id),
                &[41],
            )
            .is_none()
        );
        assert!(
            normalize_mm_placeholder_runs(
                &[image_token_id, image_token_id],
                Some(image_token_id),
                Some(video_token_id),
                &[41, 42],
            )
            .is_none()
        );
    }

    #[test]
    fn large_mismatched_mapping_fails_closed_without_search() {
        let image_token_id = 151655u32;
        let video_token_id = 151656u32;
        let mut tokens = Vec::new();
        for separator in 0..14 {
            tokens.extend([image_token_id, separator]);
        }
        tokens.push(video_token_id);
        let mm_objects: Vec<u64> = (0..28).collect();

        assert!(
            normalize_mm_placeholder_runs(
                &tokens,
                Some(image_token_id),
                Some(video_token_id),
                &mm_objects,
            )
            .is_none()
        );
    }

    #[test]
    fn mixed_boundary_preserves_native_hash_when_mapping_is_not_exact() {
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
        let expected_image = compute_block_hash_for_seq(
            &[image_pad, image_pad, image_pad, image_pad],
            block_size,
            BlockHashOptions::default(),
        )[0];
        let expected_boundary = create_stored_block_from_parts(
            block_size,
            102,
            &tokens[4..8],
            StoredBlockOptions {
                mm_extra_info: block_mm_infos[1].clone(),
                ..Default::default()
            },
        );
        let expected_video = compute_block_hash_for_seq(
            &[9, 10, video_pad, video_pad],
            block_size,
            BlockHashOptions::default(),
        )[0];

        assert_eq!(stored.len(), 3);
        assert_eq!(stored[0].tokens_hash, expected_image);
        assert_eq!(stored[1].tokens_hash, expected_boundary.tokens_hash);
        assert_eq!(stored[2].tokens_hash, expected_video);
    }

    #[test]
    fn runless_video_boundary_preserves_native_mm_identity() {
        let block_size = 4u32;
        let video_token_id = 151656u32;
        let tokens = [1, 2, 3, 4, video_token_id, video_token_id, 5, 6];
        let make_blocks = |mm_hash| {
            let info = Some(BlockExtraInfo {
                mm_objects: vec![BlockMmObjectInfo {
                    mm_hash,
                    offsets: vec![],
                }],
            });
            create_stored_blocks(
                block_size,
                &tokens,
                &[4, 4],
                &[101, 102],
                None,
                None,
                &Arc::new(AtomicU32::new(0)),
                Some(&[info.clone(), info]),
                None,
                Some(151655),
                Some(video_token_id),
            )
        };

        let first_video = make_blocks(41);
        let second_video = make_blocks(42);
        let native_first = create_stored_block_from_parts(
            block_size,
            101,
            &tokens[..4],
            StoredBlockOptions {
                mm_extra_info: Some(BlockExtraInfo {
                    mm_objects: vec![BlockMmObjectInfo {
                        mm_hash: 41,
                        offsets: vec![],
                    }],
                }),
                ..Default::default()
            },
        );

        assert_eq!(first_video[0].tokens_hash, native_first.tokens_hash);
        assert_ne!(first_video[0].tokens_hash, second_video[0].tokens_hash);
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
