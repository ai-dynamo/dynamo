// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Text classifiers for semantic routing

use std::{collections::HashMap, sync::Arc};

use async_trait::async_trait;
use anyhow::Result;

use crate::semrouter::config::{ClassifierConfig, SemRouterConfig};

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

/// Simple, dependency-free baseline that maps keywords -> labels.
/// Use it to validate routing plumbing; swap in ONNX/Candle later.
struct KeywordClassifier {
    classes: HashMap<String, Vec<String>>,
}

#[async_trait]
impl Classifier for KeywordClassifier {
    async fn predict(&self, text: &str) -> Result<Classification> {
        let text_lc = text.to_ascii_lowercase();
        let mut out: Vec<LabelScore> = self
            .classes
            .iter()
            .map(|(label, kws)| {
                let mut hits = 0usize;
                for kw in kws {
                    if !kw.is_empty() && text_lc.contains(&kw.to_ascii_lowercase()) {
                        hits += 1;
                    }
                }
                let score = if kws.is_empty() { 0.0 } else { hits as f32 / kws.len() as f32 };
                LabelScore { label: label.clone(), score }
            })
            .collect();

        out.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(Classification { labels: out })
    }

    fn name(&self) -> &'static str { "keyword" }
}

pub fn build_classifier(cfg: &SemRouterConfig) -> Result<Arc<dyn Classifier>> {
    match &cfg.classifier {
        ClassifierConfig::Keyword { classes } => {
            Ok(Arc::new(KeywordClassifier { classes: classes.clone() }) as Arc<dyn Classifier>)
        }
        ClassifierConfig::Fasttext { model_path } => {
            let clf = fasttext::FasttextClassifier::new(model_path)?;
            Ok(Arc::new(clf) as Arc<dyn Classifier>)
        }
        // Future extensions:
        // ClassifierConfig::Onnx { .. } => unimplemented!(),
        // ClassifierConfig::Candle { .. } => unimplemented!(),
        // ClassifierConfig::Http { .. } => unimplemented!(),
    }
}
