// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-free tokenizer/preprocessor construction for router-only mode.
//!
//! In full Dynamo mode the EPP fetches a [`ModelDeploymentCard`] from the
//! distributed discovery plane and builds an [`OpenAIPreprocessor`] from it
//! (see [`crate::epp::Router::from_discovery`]). Router-only mode has no
//! control plane and no Dynamo runtime, so the preprocessor is built entirely
//! offline from a model id:
//!
//! 1. resolve the model files into the local HF cache (download if absent), and
//! 2. load a [`ModelDeploymentCard`] from that directory, then
//! 3. build the [`OpenAIPreprocessor`] (tokenizer + chat template).
//!
//! Only the tokenizer/config files are fetched — model weights are skipped.
//! The preprocessor is used solely to tokenize prompts for routing; the worker
//! re-tokenizes the forwarded request, so no `token_ids` are injected.

use std::sync::Arc;

use anyhow::{Context, Result};
use dynamo_llm::hub::from_hf;
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;

/// Build an [`OpenAIPreprocessor`] offline from a model id (HF repo or local
/// path), with no Dynamo runtime or discovery plane.
///
/// `block_size` MUST equal the engine's `--block-size`; it is recorded on the
/// card so any block-size-derived preprocessing stays consistent with the
/// engine that produces the KV-cache events the selector ingests.
pub async fn build_offline_preprocessor(
    model_name: &str,
    block_size: u32,
) -> Result<Arc<OpenAIPreprocessor>> {
    // Tokenizer/config only — never download weights for the routing path.
    let model_path = from_hf(model_name, true)
        .await
        .with_context(|| format!("resolving tokenizer/config for model {model_name:?}"))?;

    let mut card = ModelDeploymentCard::load_from_disk(&model_path, None)
        .with_context(|| format!("loading ModelDeploymentCard from {}", model_path.display()))?;
    card.set_name(model_name);
    card.kv_cache_block_size = block_size;

    OpenAIPreprocessor::new(card)
        .with_context(|| format!("building offline preprocessor for model {model_name:?}"))
}
