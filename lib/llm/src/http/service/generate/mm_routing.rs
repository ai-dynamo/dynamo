// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::GenerateRequest;
use crate::protocols::common::preprocessor::MmRoutingInfo;
pub(crate) use crate::protocols::multimodal::preprocessed_mm_cache_identifier;
use crate::protocols::multimodal::{
    MAX_PREPROCESSED_MM_BYTES, MAX_PREPROCESSED_MM_FEATURES, MAX_PREPROCESSED_MM_MODALITY_BYTES,
    MAX_PREPROCESSED_MM_ROUTING_HASH_BYTES,
};
use base64::Engine as _;

/// Validate the execution contract for renderer-produced multimodal features.
/// Routing may remain conservative, but execution metadata must never reach a
/// worker in a shape the sidecar would drop or reinterpret.
pub(super) fn validate_generate_mm_features(request: &GenerateRequest) -> Result<(), String> {
    let Some(features) = request.passthrough.get("features") else {
        return Ok(());
    };
    if features.is_null() {
        return Ok(());
    }
    let features = features
        .as_object()
        .ok_or_else(|| "features must be a JSON object".to_string())?;
    if let Some(field) = features.keys().find(|field| {
        !matches!(
            field.as_str(),
            "mm_hashes" | "mm_placeholders" | "kwargs_data"
        )
    }) {
        return Err(format!("unsupported features field `{field}`"));
    }
    let hashes = features
        .get("mm_hashes")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| "features.mm_hashes must be a JSON object".to_string())?;
    let placeholders = features
        .get("mm_placeholders")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| "features.mm_placeholders must be a JSON object".to_string())?;
    let kwargs = features
        .get("kwargs_data")
        .and_then(serde_json::Value::as_object)
        .ok_or_else(|| {
            "features.kwargs_data is required; unverified cache-hit nulls are unsupported"
                .to_string()
        })?;
    let same_modalities = |other: &serde_json::Map<String, serde_json::Value>| {
        hashes.len() == other.len() && hashes.keys().all(|modality| other.contains_key(modality))
    };
    if !same_modalities(placeholders) || !same_modalities(kwargs) {
        return Err(
            "features hashes, placeholders, and kwargs_data must have identical modality keys"
                .to_string(),
        );
    }

    let mut feature_count = 0usize;
    let mut decoded_bytes = 0usize;
    let mut ranges = Vec::new();
    for (modality, hashes) in hashes {
        if modality.is_empty() || modality.len() > MAX_PREPROCESSED_MM_MODALITY_BYTES {
            return Err(format!(
                "feature modality names must contain between 1 and {MAX_PREPROCESSED_MM_MODALITY_BYTES} bytes"
            ));
        }
        let hashes = hashes
            .as_array()
            .ok_or_else(|| format!("features.mm_hashes.{modality} must be an array"))?;
        let placeholders = placeholders[modality]
            .as_array()
            .ok_or_else(|| format!("features.mm_placeholders.{modality} must be an array"))?;
        let kwargs = kwargs[modality]
            .as_array()
            .ok_or_else(|| format!("features.kwargs_data.{modality} must be an array"))?;
        if hashes.len() != placeholders.len() || hashes.len() != kwargs.len() {
            return Err(format!(
                "feature lists for modality {modality:?} must have equal lengths"
            ));
        }
        feature_count = feature_count
            .checked_add(hashes.len())
            .ok_or_else(|| "too many multimodal features".to_string())?;
        for ((hash, placeholder), encoded) in hashes.iter().zip(placeholders).zip(kwargs) {
            let hash = hash
                .as_str()
                .filter(|hash| {
                    !hash.is_empty() && hash.len() <= MAX_PREPROCESSED_MM_ROUTING_HASH_BYTES
                })
                .ok_or_else(|| {
                    "multimodal hashes must contain between 1 and 512 bytes".to_string()
                })?;
            let _ = hash;
            let placeholder = placeholder
                .as_object()
                .ok_or_else(|| "multimodal placeholders must be JSON objects".to_string())?;
            if let Some(field) = placeholder
                .keys()
                .find(|field| !matches!(field.as_str(), "offset" | "length" | "is_embed"))
            {
                return Err(format!(
                    "unsupported multimodal placeholder field `{field}`"
                ));
            }
            let offset = placeholder
                .get("offset")
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .ok_or_else(|| {
                    "multimodal placeholder offsets must be non-negative integers".to_string()
                })?;
            let length = placeholder
                .get("length")
                .and_then(serde_json::Value::as_u64)
                .and_then(|value| usize::try_from(value).ok())
                .filter(|value| *value > 0)
                .ok_or_else(|| {
                    "multimodal placeholder lengths must be positive integers".to_string()
                })?;
            let end = offset
                .checked_add(length)
                .filter(|end| *end <= request.token_ids.len())
                .ok_or_else(|| "multimodal placeholder range exceeds token_ids".to_string())?;
            if let Some(mask) = placeholder.get("is_embed")
                && !mask.is_null()
            {
                let mask = mask.as_array().ok_or_else(|| {
                    "multimodal placeholder is_embed must be an array".to_string()
                })?;
                if mask.len() != length || mask.iter().any(|value| !value.is_boolean()) {
                    return Err(
                        "multimodal placeholder is_embed must contain one boolean per position"
                            .to_string(),
                    );
                }
            }
            let encoded = encoded.as_str().ok_or_else(|| {
                "each multimodal feature must carry base64 kwargs_data; cache-hit nulls are unsupported"
                    .to_string()
            })?;
            let bytes = base64::engine::general_purpose::STANDARD
                .decode(encoded)
                .map_err(|error| format!("invalid multimodal kwargs_data base64: {error}"))?;
            decoded_bytes = decoded_bytes
                .checked_add(bytes.len())
                .ok_or_else(|| "multimodal feature payload is too large".to_string())?;
            ranges.push((offset, end));
        }
    }
    if feature_count == 0 || feature_count > MAX_PREPROCESSED_MM_FEATURES {
        return Err(format!(
            "features must contain between 1 and {MAX_PREPROCESSED_MM_FEATURES} multimodal items"
        ));
    }
    if decoded_bytes > MAX_PREPROCESSED_MM_BYTES {
        return Err(format!(
            "multimodal feature payload exceeds {} MiB",
            MAX_PREPROCESSED_MM_BYTES / (1024 * 1024)
        ));
    }
    ranges.sort_unstable_by_key(|range| range.0);
    if ranges.windows(2).any(|pair| pair[0].1 > pair[1].0) {
        return Err("multimodal feature ranges must not overlap".to_string());
    }
    Ok(())
}

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
    let kwargs_data = features
        .get("kwargs_data")
        .and_then(serde_json::Value::as_object)
        .ok_or("features.kwargs_data must be a JSON object")?;

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

    let (hashes, placeholders, kwargs) = match (
        mm_hashes.get("image"),
        mm_placeholders.get("image"),
        kwargs_data.get("image"),
    ) {
        (None, None, None) => return Ok(None),
        (Some(hashes), Some(placeholders), Some(kwargs)) => (
            hashes
                .as_array()
                .ok_or("features.mm_hashes.image must be an array")?,
            placeholders
                .as_array()
                .ok_or("features.mm_placeholders.image must be an array")?,
            kwargs
                .as_array()
                .ok_or("features.kwargs_data.image must be an array")?,
        ),
        _ => return Err("image hashes, placeholders, and kwargs_data must all be present"),
    };
    if hashes.len() != placeholders.len() || hashes.len() != kwargs.len() {
        return Err("image hashes, placeholders, and kwargs_data must have equal lengths");
    }

    let mut ranges = Vec::with_capacity(hashes.len());
    for ((hash, placeholder), kwargs) in hashes.iter().zip(placeholders).zip(kwargs) {
        hash.as_str()
            .ok_or("multimodal hashes must be non-empty strings")?;
        let kwargs = kwargs
            .as_str()
            .ok_or("multimodal kwargs_data entries must be base64 strings")?;
        let kwargs = base64::engine::general_purpose::STANDARD
            .decode(kwargs)
            .map_err(|_| "multimodal kwargs_data must be valid base64")?;
        let identifier = preprocessed_mm_cache_identifier("image", &kwargs);
        let hash = dynamo_kv_router::zmq_wire::hash_mm_identifier(&identifier)
            .ok_or("canonical multimodal identifier must be non-empty")?;
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
