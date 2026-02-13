// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shared utilities for benchmark binaries.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::time::Duration;

// ---------------------------------------------------------------------------
// Latency statistics
// ---------------------------------------------------------------------------

pub struct LatencyStats {
    pub min: Duration,
    pub max: Duration,
    pub p50: Duration,
    pub p95: Duration,
    pub p99: Duration,
}

impl LatencyStats {
    pub fn from_durations(durations: &[Duration]) -> Option<Self> {
        if durations.is_empty() {
            return None;
        }

        let mut sorted = durations.to_vec();
        sorted.sort();
        let n = sorted.len();

        Some(Self {
            min: sorted[0],
            max: sorted[n - 1],
            p50: sorted[n / 2],
            p95: sorted[n * 95 / 100],
            p99: sorted[n * 99 / 100],
        })
    }
}

// ---------------------------------------------------------------------------
// OpenAI-style chat types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

// ---------------------------------------------------------------------------
// Model auto-detection
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelInfo>,
}

#[derive(Debug, Deserialize)]
struct ModelInfo {
    id: String,
}

pub async fn fetch_model_name(frontend_url: &str) -> Result<String> {
    let client = reqwest::Client::new();
    let url = format!("{}/v1/models", frontend_url);

    println!("  Auto-detecting model from {}...", url);

    let response = client
        .get(&url)
        .send()
        .await
        .context("Failed to connect to frontend /v1/models endpoint")?;

    if !response.status().is_success() {
        anyhow::bail!("Models endpoint returned status: {}", response.status());
    }

    let models: ModelsResponse = response
        .json()
        .await
        .context("Failed to parse models response")?;

    match models.data.len() {
        0 => anyhow::bail!("No models found at endpoint. Is a backend running?"),
        1 => {
            let model_id = models.data[0].id.clone();
            println!("  Auto-detected model: {}", model_id);
            Ok(model_id)
        }
        n => {
            println!("  Multiple models available ({}):", n);
            for m in &models.data {
                println!("    - {}", m.id);
            }
            anyhow::bail!("Multiple models available. Please specify --model explicitly.")
        }
    }
}
