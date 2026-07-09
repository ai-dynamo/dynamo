// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashSet;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use crate::protocols::{
    BlockExtraInfo, BlockHashOptions, ExternalSequenceBlockHash, KvCacheEvent, KvCacheEventData,
    KvCacheRemoveData, KvCacheStoreBlockInput, KvCacheStoreData, KvCacheStoreInput,
    KvCacheStoredBlockData, Placement, PlacementEvent, PlacementEventInput, StorageTier,
    WorkerWithDpRank, compute_block_hash_for_seq,
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
) -> Option<PlacementEvent> {
    convert_event_impl(
        raw,
        event_id,
        kv_block_size,
        worker,
        warning_count,
        image_token_id,
        false,
    )
    .map(|input| input.event)
}

/// Convert a raw event while retaining worker-local token context for optional
/// publisher-side block coalescing.
pub fn convert_event_with_input(
    raw: RawKvEvent,
    event_id: u64,
    kv_block_size: u32,
    worker: WorkerWithDpRank,
    warning_count: &Arc<AtomicU32>,
    image_token_id: Option<u32>,
) -> Option<PlacementEventInput> {
    convert_event_impl(
        raw,
        event_id,
        kv_block_size,
        worker,
        warning_count,
        image_token_id,
        true,
    )
}

#[allow(clippy::too_many_arguments)]
fn convert_event_impl(
    raw: RawKvEvent,
    event_id: u64,
    kv_block_size: u32,
    worker: WorkerWithDpRank,
    warning_count: &Arc<AtomicU32>,
    image_token_id: Option<u32>,
    capture_store_input: bool,
) -> Option<PlacementEventInput> {
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
                    return Some(PlacementEventInput::without_store_input(
                        PlacementEvent::new(
                            Placement::local_worker(worker.worker_id, worker.dp_rank, storage_tier),
                            KvCacheEvent {
                                event_id,
                                data: KvCacheEventData::Removed(KvCacheRemoveData {
                                    block_hashes: vec![],
                                }),
                                dp_rank,
                            },
                        ),
                    ));
                }
            }

            let num_block_tokens = vec![block_size as u64; block_hashes.len()];
            let block_hashes_u64: Vec<u64> = block_hashes
                .into_iter()
                .map(BlockHashValue::into_u64)
                .collect();
            let (blocks, store_input) = if capture_store_input {
                let (blocks, input) = create_stored_blocks_with_input(
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
                );
                (blocks, Some(input))
            } else {
                (
                    create_stored_blocks(
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
                    ),
                    None,
                )
            };
            let event = KvCacheEvent {
                event_id,
                data: KvCacheEventData::Stored(KvCacheStoreData {
                    parent_hash: parent_block_hash
                        .map(BlockHashValue::into_u64)
                        .map(ExternalSequenceBlockHash::from),
                    start_position: None,
                    blocks,
                }),
                dp_rank,
            };
            return Some(PlacementEventInput::new(
                PlacementEvent::new(
                    Placement::local_worker(worker.worker_id, worker.dp_rank, storage_tier),
                    event,
                ),
                store_input,
            ));
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

    Some(PlacementEventInput::without_store_input(
        PlacementEvent::new(
            Placement::local_worker(worker.worker_id, worker.dp_rank, storage_tier),
            event,
        ),
    ))
}

/// Rewrite each `image_token_id` run in `token_ids` to `pad_value(mm_hash)`,
/// one mm_hash per run in order, so the recomputed `tokens_hash` matches the
/// frontend's pad_value expansion. Exact when images are separated by a
/// non-image token (true for Qwen2/2.5/3-VL); a run-vs-mm_object count mismatch
/// (adjacent images, no separator) is logged below rather than silent.
fn substitute_pad_values(token_ids: &[u32], image_token_id: u32, mm_objects: &[u64]) -> Vec<u32> {
    let mut out = Vec::with_capacity(token_ids.len());
    // `obj_idx` advances once per completed run, so run N fills with
    // mm_objects[N], clamped to the last object if runs outnumber mm_objects.
    let mut obj_idx = 0usize;
    let mut in_run = false;
    let mut runs = 0usize;
    // pad_value for the current run, computed once on entry and reused for the
    // rest of the run (one mm_hash per run, so it's constant within a run).
    let mut run_pad = 0u32;
    for &t in token_ids {
        if t == image_token_id {
            if !in_run {
                in_run = true;
                runs += 1;
                // Safety: the sole caller (`create_stored_block_from_parts`)
                // only reaches here with a non-empty `mm_objects`, so `last()`
                // is `Some`.
                let mm_hash = mm_objects
                    .get(obj_idx)
                    .copied()
                    .unwrap_or_else(|| *mm_objects.last().unwrap());
                run_pad = crate::protocols::pad_value_for_mm_hash(mm_hash);
            }
            out.push(run_pad);
        } else {
            if in_run {
                in_run = false;
                obj_idx += 1;
            }
            out.push(t);
        }
    }
    if runs != mm_objects.len() {
        tracing::debug!(
            runs,
            mm_objects = mm_objects.len(),
            "image_token_id run count != mm_object count; pad_value assignment is best-effort by run order"
        );
    }
    out
}

#[derive(Default)]
pub struct StoredBlockOptions<'a> {
    pub lora_name: Option<&'a str>,
    pub cache_namespace: Option<&'a str>,
    pub mm_extra_info: Option<BlockExtraInfo>,
    pub is_eagle: Option<bool>,
    pub image_token_id: Option<u32>,
}

pub fn create_stored_block_from_parts(
    kv_block_size: u32,
    block_hash: u64,
    token_ids: &[u32],
    options: StoredBlockOptions<'_>,
) -> KvCacheStoredBlockData {
    let StoredBlockOptions {
        lora_name,
        cache_namespace,
        mm_extra_info,
        is_eagle,
        image_token_id,
    } = options;
    let tokens_hash = match (image_token_id, mm_extra_info.as_ref()) {
        (Some(img_tok), Some(info)) if !info.mm_objects.is_empty() => {
            let mm_hashes: Vec<u64> = info.mm_objects.iter().map(|o| o.mm_hash).collect();
            let substituted = substitute_pad_values(token_ids, img_tok, &mm_hashes);
            compute_block_hash_for_seq(
                &substituted,
                kv_block_size,
                BlockHashOptions {
                    block_mm_infos: None,
                    lora_name,
                    cache_namespace,
                    is_eagle,
                },
            )[0]
        }
        _ => {
            let block_mm_infos = mm_extra_info.as_ref().map(|info| vec![Some(info.clone())]);
            compute_block_hash_for_seq(
                token_ids,
                kv_block_size,
                BlockHashOptions {
                    block_mm_infos: block_mm_infos.as_deref(),
                    lora_name,
                    cache_namespace,
                    is_eagle,
                },
            )[0]
        }
    };
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

pub fn create_stored_block_from_parts_with_input(
    kv_block_size: u32,
    block_hash: u64,
    token_ids: &[u32],
    options: StoredBlockOptions<'_>,
) -> (KvCacheStoredBlockData, KvCacheStoreBlockInput) {
    let StoredBlockOptions {
        lora_name,
        cache_namespace,
        mm_extra_info,
        is_eagle,
        image_token_id,
    } = options;

    // When the model has a routing image token and this block carries mm
    // objects (vLLM events), normalize to the canonical pad_value scheme:
    // substitute pad_value over the image_token_id runs and hash WITHOUT
    // block_mm_infos, matching the frontend. sglang events carry no
    // image_token_id tokens nor mm_extra_info, so they take the else branch
    // unchanged.
    let (canonical_tokens, hash_mm_extra_info) = match (image_token_id, mm_extra_info.as_ref()) {
        (Some(img_tok), Some(info)) if !info.mm_objects.is_empty() => {
            let mm_hashes: Vec<u64> = info.mm_objects.iter().map(|o| o.mm_hash).collect();
            let substituted = substitute_pad_values(token_ids, img_tok, &mm_hashes);
            (substituted, None)
        }
        _ => (token_ids.to_vec(), mm_extra_info.clone()),
    };
    let block_mm_infos = hash_mm_extra_info
        .as_ref()
        .map(|info| vec![Some(info.clone())]);
    let tokens_hash = compute_block_hash_for_seq(
        &canonical_tokens,
        kv_block_size,
        BlockHashOptions {
            block_mm_infos: block_mm_infos.as_deref(),
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
    (
        KvCacheStoredBlockData {
            block_hash: ExternalSequenceBlockHash::from(block_hash),
            tokens_hash,
            mm_extra_info,
        },
        KvCacheStoreBlockInput {
            token_ids: canonical_tokens,
            lora_name: lora_name.map(ToOwned::to_owned),
            cache_namespace: cache_namespace.map(ToOwned::to_owned),
            is_eagle: is_eagle.unwrap_or(false),
            hash_mm_extra_info,
        },
    )
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
) -> Vec<KvCacheStoredBlockData> {
    let mut blocks = Vec::new();
    let mut token_offset = 0usize;
    let append = usize::from(is_eagle.unwrap_or(false));

    for (block_idx, (num_tokens, block_hash)) in
        num_block_tokens.iter().zip(block_hashes).enumerate()
    {
        if *num_tokens != u64::from(kv_block_size) {
            if warning_count.fetch_add(1, Ordering::Relaxed) < 3 {
                tracing::warn!(
                    "Block not published. Block size must be {} tokens to be published. Block size is: {}",
                    kv_block_size,
                    *num_tokens
                );
            }
            break;
        }
        let end = token_offset + append + *num_tokens as usize;
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
        let mm_extra_info = block_mm_infos
            .and_then(|infos| infos.get(block_idx))
            .and_then(Clone::clone);
        blocks.push(create_stored_block_from_parts(
            kv_block_size,
            *block_hash,
            &token_ids[token_offset..end],
            StoredBlockOptions {
                lora_name,
                cache_namespace,
                mm_extra_info,
                is_eagle,
                image_token_id,
            },
        ));
        token_offset += *num_tokens as usize;
    }
    blocks
}

#[allow(clippy::too_many_arguments)]
pub fn create_stored_blocks_with_input(
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
) -> (Vec<KvCacheStoredBlockData>, KvCacheStoreInput) {
    let mut blocks: Vec<KvCacheStoredBlockData> = Vec::new();
    let mut inputs: Vec<KvCacheStoreBlockInput> = Vec::new();

    let mut token_offset: usize = 0;
    let append = is_eagle.unwrap_or(false) as usize;

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

        let (block, input) = create_stored_block_from_parts_with_input(
            kv_block_size,
            *block_hash_it,
            tokens,
            StoredBlockOptions {
                lora_name,
                cache_namespace,
                mm_extra_info,
                is_eagle,
                image_token_id,
            },
        );
        blocks.push(block);
        inputs.push(input);
        token_offset += *num_tokens_it as usize;
    }

    (blocks, KvCacheStoreInput { blocks: inputs })
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

    #[test]
    fn sidecar_path_preserves_published_blocks() {
        let warning_count = Arc::new(AtomicU32::new(0));
        let standard = create_stored_blocks(
            2,
            &[10, 20, 30, 40],
            &[2, 2],
            &[100, 200],
            Some("adapter"),
            Some("namespace"),
            &warning_count,
            None,
            None,
            None,
        );
        let (with_sidecar, input) = create_stored_blocks_with_input(
            2,
            &[10, 20, 30, 40],
            &[2, 2],
            &[100, 200],
            Some("adapter"),
            Some("namespace"),
            &warning_count,
            None,
            None,
            None,
        );
        assert_eq!(with_sidecar, standard);
        assert_eq!(input.blocks[0].token_ids, vec![10, 20]);
        assert_eq!(input.blocks[1].token_ids, vec![30, 40]);
    }
}
