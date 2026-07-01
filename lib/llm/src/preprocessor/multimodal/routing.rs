// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use anyhow::Result;

use super::MultimodalTokenCounter;
use crate::{
    preprocessor::MmImageEntry,
    protocols::{TokenIdType, common::preprocessor::MmRoutingInfo},
};

pub(in crate::preprocessor) struct MmRoutingModel {
    pub(in crate::preprocessor) counter: MultimodalTokenCounter,
    pub(in crate::preprocessor) placeholder_token_id: TokenIdType,
    pub(in crate::preprocessor) prepend_bos_token_id: Option<TokenIdType>,
}

/// Bound routing-only prompt materialization even for malformed model configs.
pub(in crate::preprocessor) const MAX_MM_ROUTING_TOKENS: usize = 1_048_576;

fn build_mm_routing_info(
    token_ids: &[TokenIdType],
    find_token_id: TokenIdType,
    mm_image_entries: &[MmImageEntry],
    image_token_counts: &[usize],
    routing_prepend_bos: Option<TokenIdType>,
    block_size: usize,
    max_expanded_prompt_len: usize,
) -> Result<MmRoutingInfo> {
    anyhow::ensure!(
        mm_image_entries.len() == image_token_counts.len(),
        "MM image/count cardinality mismatch"
    );
    anyhow::ensure!(block_size > 0, "MM routing block size must be non-zero");
    let placeholder_count = token_ids
        .iter()
        .filter(|&&token_id| token_id == find_token_id)
        .count();
    anyhow::ensure!(
        placeholder_count == mm_image_entries.len(),
        "MM placeholder/image cardinality mismatch: found {placeholder_count} placeholders for {} images",
        mm_image_entries.len()
    );

    let replacement_tokens = image_token_counts.iter().try_fold(0usize, |total, count| {
        total
            .checked_add(*count)
            .ok_or_else(|| anyhow::anyhow!("MM routing image-token count overflowed usize"))
    })?;
    let bos_extra = usize::from(routing_prepend_bos.is_some());
    let retained_text_tokens = token_ids
        .len()
        .checked_sub(mm_image_entries.len())
        .ok_or_else(|| anyhow::anyhow!("MM placeholders exceed prompt token count"))?;
    let expanded_prompt_len = retained_text_tokens
        .checked_add(replacement_tokens)
        .and_then(|total| total.checked_add(bos_extra))
        .ok_or_else(|| anyhow::anyhow!("MM routing expanded prompt length overflowed usize"))?;
    anyhow::ensure!(
        expanded_prompt_len <= max_expanded_prompt_len,
        "MM routing expanded prompt length {expanded_prompt_len} exceeds limit {max_expanded_prompt_len}"
    );
    let total_tokens = expanded_prompt_len
        .checked_add(block_size - 1)
        .and_then(|rounded| rounded.checked_div(block_size))
        .and_then(|blocks| blocks.checked_mul(block_size))
        .ok_or_else(|| anyhow::anyhow!("MM routing block padding overflowed usize"))?;
    anyhow::ensure!(
        total_tokens <= MAX_MM_ROUTING_TOKENS,
        "MM routing padded token length {total_tokens} exceeds hard limit {MAX_MM_ROUTING_TOKENS}"
    );

    let mut expanded = Vec::new();
    expanded
        .try_reserve_exact(total_tokens)
        .map_err(|error| anyhow::anyhow!("failed to reserve MM routing token buffer: {error}"))?;
    if let Some(bos) = routing_prepend_bos {
        expanded.push(bos);
    }

    let mut image_index = 0usize;
    for &token_id in token_ids {
        if token_id == find_token_id && image_index < mm_image_entries.len() {
            let fill_token = dynamo_kv_router::protocols::pad_value_for_mm_hash(
                mm_image_entries[image_index].mm_hash,
            );
            expanded.extend(std::iter::repeat_n(
                fill_token,
                image_token_counts[image_index],
            ));
            image_index += 1;
        } else {
            expanded.push(token_id);
        }
    }

    debug_assert_eq!(expanded.len(), expanded_prompt_len);
    expanded.resize(total_tokens, 0);

    Ok(MmRoutingInfo {
        routing_token_ids: expanded,
        block_mm_infos: Vec::new(),
        expanded_prompt_len,
    })
}

/// Routing tokens and worker image identities must be installed atomically.
pub(in crate::preprocessor) struct MmRoutingPayload {
    pub(in crate::preprocessor) routing_info: MmRoutingInfo,
    pub(in crate::preprocessor) mm_hashes: Vec<serde_json::Value>,
}

pub(in crate::preprocessor) fn build_mm_routing_payload(
    token_ids: &[TokenIdType],
    find_token_id: TokenIdType,
    mm_image_entries: &[MmImageEntry],
    image_token_counts: &[usize],
    routing_prepend_bos: Option<TokenIdType>,
    block_size: usize,
    max_expanded_prompt_len: usize,
) -> Result<MmRoutingPayload> {
    let routing_info = build_mm_routing_info(
        token_ids,
        find_token_id,
        mm_image_entries,
        image_token_counts,
        routing_prepend_bos,
        block_size,
        max_expanded_prompt_len,
    )?;
    let mm_hashes = mm_image_entries
        .iter()
        .map(|entry| serde_json::Value::String(format!("{:016x}", entry.mm_hash)))
        .collect();
    Ok(MmRoutingPayload {
        routing_info,
        mm_hashes,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_mm_routing_info_expands_multiple_images_and_pads_blocks() {
        let entries = [
            MmImageEntry {
                mm_hash: 0x1234,
                width: 640,
                height: 480,
            },
            MmImageEntry {
                mm_hash: 0x5678,
                width: 320,
                height: 240,
            },
        ];
        let counts = [2, 3];
        let first_fill = dynamo_kv_router::protocols::pad_value_for_mm_hash(entries[0].mm_hash);
        let second_fill = dynamo_kv_router::protocols::pad_value_for_mm_hash(entries[1].mm_hash);

        let info = build_mm_routing_info(
            &[10, 99, 11, 99, 12],
            99,
            &entries,
            &counts,
            Some(1),
            4,
            1024,
        )
        .unwrap();

        assert_eq!(info.expanded_prompt_len, 9);
        assert_eq!(info.routing_token_ids.len(), 12);
        assert_eq!(
            &info.routing_token_ids[..9],
            &[
                1,
                10,
                first_fill,
                first_fill,
                11,
                second_fill,
                second_fill,
                second_fill,
                12,
            ]
        );
        assert_eq!(&info.routing_token_ids[9..], &[0, 0, 0]);
        assert!(info.block_mm_infos.is_empty());
    }

    #[test]
    fn build_mm_routing_info_uses_image_hash_as_identity() {
        let count = [2];
        let make = |mm_hash| {
            build_mm_routing_info(
                &[7],
                7,
                &[MmImageEntry {
                    mm_hash,
                    width: 1,
                    height: 1,
                }],
                &count,
                None,
                2,
                1024,
            )
            .unwrap()
            .routing_token_ids
        };

        assert_eq!(make(42), make(42));
        assert_ne!(make(42), make(43));
    }

    #[test]
    fn build_mm_routing_payload_rejects_oversized_expansion_atomically() {
        let entries = [MmImageEntry {
            mm_hash: 42,
            width: 1,
            height: 1,
        }];
        let error = build_mm_routing_payload(&[7], 7, &entries, &[1025], None, 16, 1024)
            .err()
            .expect("oversized expansion must fail atomically");

        assert!(error.to_string().contains("expanded prompt length 1025"));
    }

    #[test]
    fn build_mm_routing_payload_formats_hashes_in_image_order() {
        let entries = [
            MmImageEntry {
                mm_hash: 0x1234,
                width: 640,
                height: 480,
            },
            MmImageEntry {
                mm_hash: 0x5678,
                width: 320,
                height: 240,
            },
        ];
        let payload =
            build_mm_routing_payload(&[99, 10, 99], 99, &entries, &[2, 1], None, 4, 1024).unwrap();
        let first_fill = dynamo_kv_router::protocols::pad_value_for_mm_hash(entries[0].mm_hash);
        let second_fill = dynamo_kv_router::protocols::pad_value_for_mm_hash(entries[1].mm_hash);

        assert_eq!(
            &payload.routing_info.routing_token_ids[..4],
            &[first_fill, first_fill, 10, second_fill]
        );
        assert_eq!(
            payload.mm_hashes,
            vec![
                serde_json::Value::String("0000000000001234".to_string()),
                serde_json::Value::String("0000000000005678".to_string()),
            ]
        );
    }

    #[test]
    fn build_mm_routing_info_rejects_count_overflow() {
        let entries = [
            MmImageEntry {
                mm_hash: 42,
                width: 1,
                height: 1,
            },
            MmImageEntry {
                mm_hash: 43,
                width: 1,
                height: 1,
            },
        ];
        let error = build_mm_routing_info(
            &[7, 7],
            7,
            &entries,
            &[usize::MAX, 1],
            None,
            16,
            MAX_MM_ROUTING_TOKENS,
        )
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("image-token count overflowed usize")
        );
    }

    #[test]
    fn build_mm_routing_info_rejects_oversized_block_padding() {
        let entries = [MmImageEntry {
            mm_hash: 42,
            width: 1,
            height: 1,
        }];
        let error = build_mm_routing_info(
            &[7],
            7,
            &entries,
            &[1],
            None,
            MAX_MM_ROUTING_TOKENS + 1,
            MAX_MM_ROUTING_TOKENS,
        )
        .unwrap_err();

        assert!(error.to_string().contains("padded token length"));
    }
}
