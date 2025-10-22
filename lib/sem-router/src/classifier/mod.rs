// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Text classifiers for semantic routing

use std::sync::Arc;

use async_trait::async_trait;
use anyhow::Result;

use super::config::{ClassifierConfig, SemRouterConfig};

pub mod fasttext;

/// One score per label (supports N classes).
#[derive(Debug, Clone)]
pub struct LabelScore {
    pub label: String,
    pub score: f32, // 0..1
}

#[derive(Debug, Clone)]
pub struct Classification {
    /// Sorted in descending order by score.
    pub labels: Vec<LabelScore>,
}

#[async_trait]
pub trait Classifier: Send + Sync {
    async fn predict(&self, text: &str) -> Result<Classification>;
    fn name(&self) -> &'static str;
}

pub fn build_classifier(cfg: &SemRouterConfig) -> Result<Arc<dyn Classifier>> {
    match &cfg.classifier {
        ClassifierConfig::Fasttext { model_path } => {
            let clf = fasttext::FasttextClassifier::new(model_path)?;
            Ok(Arc::new(clf) as Arc<dyn Classifier>)
        }
    }
}

