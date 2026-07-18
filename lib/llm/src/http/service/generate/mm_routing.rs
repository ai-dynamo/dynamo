// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::GenerateRequest;
use crate::protocols::common::preprocessor::MmRoutingInfo;

/// Build the routing-only token sequence used by vLLM KV events for multimodal
/// prompts. The caller-provided `features` object remains opaque to execution;
/// this projection reads only the hashes and placeholder ranges required to
/// make request-side KV hashes match worker-side event hashes.
pub(super) fn generate_mm_routing_info(
    request: &GenerateRequest,
    kv_cache_block_size: u32,
) -> Result<Option<MmRoutingInfo>, &'static str> {
    let Some(features) = request.passthrough.get("features") else {
        return Ok(None);
    };
    if features.is_null() {
        return Ok(None);
    }

    let features = features
        .as_object()
        .ok_or("features must be a JSON object")?;
    let mm_hashes = features
        .get("mm_hashes")
        .and_then(serde_json::Value::as_object)
        .ok_or("features.mm_hashes must be a JSON object")?;
    let mm_placeholders = features
        .get("mm_placeholders")
        .and_then(serde_json::Value::as_object)
        .ok_or("features.mm_placeholders must be a JSON object")?;

    if mm_hashes
        .keys()
        .chain(mm_placeholders.keys())
        .any(|modality| modality != "image")
    {
        return Err("exact /generate MM routing currently supports image placeholders only");
    }
    if kv_cache_block_size == 0 {
        return Err("KV cache block size must be non-zero");
    }

    let (hashes, placeholders) = match (mm_hashes.get("image"), mm_placeholders.get("image")) {
        (None, None) => return Ok(None),
        (Some(hashes), Some(placeholders)) => (
            hashes
                .as_array()
                .ok_or("features.mm_hashes.image must be an array")?,
            placeholders
                .as_array()
                .ok_or("features.mm_placeholders.image must be an array")?,
        ),
        _ => return Err("image hashes and placeholders must both be present"),
    };
    if hashes.len() != placeholders.len() {
        return Err("image hashes and placeholders must have equal lengths");
    }

    let mut ranges = Vec::with_capacity(hashes.len());
    for (hash, placeholder) in hashes.iter().zip(placeholders) {
        let hash = hash
            .as_str()
            .and_then(dynamo_kv_router::zmq_wire::hash_mm_identifier)
            .ok_or("multimodal hashes must be non-empty strings")?;
        let placeholder = placeholder
            .as_object()
            .ok_or("multimodal placeholders must be JSON objects")?;
        let offset = placeholder
            .get("offset")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .ok_or("multimodal placeholder offsets must be non-negative integers")?;
        let length = placeholder
            .get("length")
            .and_then(serde_json::Value::as_u64)
            .and_then(|value| usize::try_from(value).ok())
            .filter(|value| *value > 0)
            .ok_or("multimodal placeholder lengths must be positive integers")?;
        let end = offset
            .checked_add(length)
            .filter(|end| *end <= request.token_ids.len())
            .ok_or("multimodal placeholder range exceeds token_ids")?;
        let is_embed = match placeholder.get("is_embed") {
            None | Some(serde_json::Value::Null) => {
                // vLLM 0.24 render responses omit sparse masks. A uniform
                // placeholder span is safely dense; a mixed span is ambiguous
                // and must retain token-only routing rather than over-substitute.
                if request.token_ids[offset..end]
                    .windows(2)
                    .any(|pair| pair[0] != pair[1])
                {
                    return Err("mixed multimodal placeholder spans require is_embed");
                }
                None
            }
            Some(value) => {
                let mask = value
                    .as_array()
                    .ok_or("multimodal placeholder is_embed must be an array")?;
                if mask.len() != length {
                    return Err(
                        "multimodal placeholder is_embed length must match placeholder length",
                    );
                }
                let mut parsed = Vec::with_capacity(mask.len());
                for entry in mask {
                    parsed.push(
                        entry
                            .as_bool()
                            .ok_or("multimodal placeholder is_embed entries must be booleans")?,
                    );
                }
                Some(parsed)
            }
        };
        ranges.push((offset, end, hash, is_embed));
    }

    if ranges.is_empty() {
        return Ok(None);
    }

    ranges.sort_unstable_by_key(|(offset, _, _, _)| *offset);
    for pair in ranges.windows(2) {
        let (_, previous_end, previous_hash, _) = &pair[0];
        let (next_offset, _, next_hash, _) = &pair[1];
        if previous_end > next_offset {
            return Err("multimodal placeholder ranges must not overlap");
        }
        if previous_end == next_offset && previous_hash != next_hash {
            return Err("adjacent multimodal placeholders must share an identifier");
        }
    }

    // vLLM's current event normalizer associates MM objects with contiguous
    // image-token runs by order, clamping excess runs to the last object in a
    // block. A sparse mask can split one object into multiple runs, so verify
    // that this run-order mapping still produces the request-side identity.
    // If it does not, retain correctness by using ordinary token routing.
    let block_size = kv_cache_block_size as usize;
    let mut first_relevant_range = 0;
    for block_start in (0..request.token_ids.len()).step_by(block_size) {
        let block_end = (block_start + block_size).min(request.token_ids.len());
        let mut worker_objects = Vec::new();
        let mut expected_by_position = vec![None; block_end - block_start];

        while first_relevant_range < ranges.len() && ranges[first_relevant_range].1 <= block_start {
            first_relevant_range += 1;
        }
        for (offset, end, hash, is_embed) in ranges[first_relevant_range..]
            .iter()
            .take_while(|(offset, _, _, _)| *offset < block_end)
        {
            let intersection_start = (*offset).max(block_start);
            let intersection_end = (*end).min(block_end);
            debug_assert!(intersection_start < intersection_end);
            worker_objects.push(*hash);
            for global_position in intersection_start..intersection_end {
                let should_embed = is_embed
                    .as_ref()
                    .is_none_or(|mask| mask[global_position - *offset]);
                if should_embed {
                    expected_by_position[global_position - block_start] = Some(*hash);
                }
            }
        }

        let mut expected_runs = Vec::new();
        let mut current_run = None;
        for expected_hash in expected_by_position {
            match (current_run, expected_hash) {
                (None, Some(hash)) => {
                    current_run = Some(hash);
                    expected_runs.push(hash);
                }
                (Some(current), Some(hash)) if current != hash => {
                    return Err("adjacent multimodal embed positions must share an identifier");
                }
                (Some(_), None) => current_run = None,
                _ => {}
            }
        }

        for (run_index, expected_hash) in expected_runs.into_iter().enumerate() {
            let worker_hash = worker_objects
                .get(run_index)
                .or_else(|| worker_objects.last())
                .copied();
            if worker_hash != Some(expected_hash) {
                return Err(
                    "sparse multimodal layout cannot be normalized exactly by worker events",
                );
            }
        }
    }

    let mut routing_token_ids = request.token_ids.clone();
    for (offset, end, hash, is_embed) in ranges {
        let pad = dynamo_kv_router::protocols::pad_value_for_mm_hash(hash);
        if let Some(mask) = is_embed {
            for (token, should_embed) in routing_token_ids[offset..end].iter_mut().zip(mask) {
                if should_embed {
                    *token = pad;
                }
            }
        } else {
            routing_token_ids[offset..end].fill(pad);
        }
    }

    let padded_len = routing_token_ids
        .len()
        .div_ceil(block_size)
        .checked_mul(block_size)
        .ok_or("multimodal routing token length overflow")?;
    routing_token_ids.resize(padded_len, 0);

    Ok(Some(MmRoutingInfo {
        routing_token_ids,
        // vLLM events are normalized to the same pad-value token scheme, so
        // MM identity is already present in the alternate routing tokens.
        block_mm_infos: Vec::new(),
        expanded_prompt_len: request.token_ids.len(),
    }))
}
