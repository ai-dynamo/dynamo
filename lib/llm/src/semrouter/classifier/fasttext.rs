use super::Classifier;
use anyhow::{anyhow, Result};
use std::collections::HashMap;

#[cfg(feature = "fasttext-classifier")]
use fasttext::FastText;

/// fastText-based classifier for binary classification
#[cfg(feature = "fasttext-classifier")]
pub struct FasttextClassifier {
    ft: FastText,
}

#[cfg(feature = "fasttext-classifier")]
impl FasttextClassifier {
    pub fn new(model_path: &str) -> Result<Self> {
        let mut ft = FastText::new();
        ft.load_model(model_path)
            .map_err(|e| anyhow!("Failed to load fastText model from {}: {}", model_path, e))?;

        // Verify we have the expected labels
        let (labels, _) = ft
            .get_labels()
            .map_err(|e| anyhow!("Failed to get labels: {}", e))?;
        let has_reasoning = labels.iter().any(|s| s == "__label__reasoning");
        let has_non = labels.iter().any(|s| s == "__label__non-reasoning");

        if !has_reasoning || !has_non {
            return Err(anyhow!(
                "Model must have __label__reasoning and __label__non-reasoning labels. Found: {:?}",
                labels
            ));
        }

        tracing::info!(
            "Initialized fastText classifier from {}: {} labels",
            model_path,
            labels.len()
        );

        Ok(Self { ft })
    }
}

#[cfg(feature = "fasttext-classifier")]
impl Classifier for FasttextClassifier {
    fn classify(&self, text: &str) -> Result<HashMap<String, f32>> {
        let start = std::time::Instant::now();

        // Preview text for logging (truncate long inputs)
        let text_preview = if text.len() > 60 {
            format!("{}...", &text[..60])
        } else {
            text.to_string()
        };

        // Get top 2 predictions with all probabilities (threshold = -1.0)
        let preds = self
            .ft
            .predict(text, 2, -1.0)
            .map_err(|e| anyhow!("fastText prediction failed: {}", e))?;

        let mut p_reason = 0.0f32;
        let mut p_non = 0.0f32;

        for p in preds {
            if p.label == "__label__reasoning" {
                p_reason = p.prob as f32;
            } else if p.label == "__label__non-reasoning" {
                p_non = p.prob as f32;
            }
        }

        // Normalize to ensure probabilities sum to 1.0
        let sum = (p_reason + p_non).max(1e-6);
        p_reason /= sum;
        p_non /= sum;

        let latency = start.elapsed();
        let predicted_class = if p_reason > p_non { "reasoning" } else { "non-reasoning" };
        let confidence = p_reason.max(p_non);

        tracing::info!(
            latency_us = latency.as_micros(),
            text = %text_preview,
            predicted_class = %predicted_class,
            confidence = %format!("{:.3}", confidence),
            reasoning_prob = %format!("{:.3}", p_reason),
            non_reasoning_prob = %format!("{:.3}", p_non),
            "fastText classification"
        );

        let mut result = HashMap::with_capacity(2);
        result.insert("non-reasoning".to_string(), p_non);
        result.insert("reasoning".to_string(), p_reason);

        Ok(result)
    }
}

// Stub implementation when feature is not enabled
#[cfg(not(feature = "fasttext-classifier"))]
pub struct FasttextClassifier;

#[cfg(not(feature = "fasttext-classifier"))]
impl FasttextClassifier {
    pub fn new(_model_path: &str) -> Result<Self> {
        Err(anyhow!(
            "FasttextClassifier requires 'fasttext-classifier' feature"
        ))
    }
}

#[cfg(not(feature = "fasttext-classifier"))]
impl Classifier for FasttextClassifier {
    fn classify(&self, _text: &str) -> Result<HashMap<String, f32>> {
        Err(anyhow!(
            "FasttextClassifier requires 'fasttext-classifier' feature"
        ))
    }
}

