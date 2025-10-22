use std::collections::HashMap;

pub mod mock;

#[cfg(feature = "candle-classifier")]
pub mod candle;

#[cfg(feature = "fasttext-classifier")]
pub mod fasttext;

/// Unified classifier trait for both binary and multi-class classification
/// Returns label probabilities as a HashMap
///
/// For binary classification (e.g., reasoning vs non-reasoning):
///   Returns {"non-reasoning": 0.3, "reasoning": 0.7}
///
/// For multi-class classification:
///   Returns {"math": 0.1, "code": 0.2, "reasoning": 0.6, "general": 0.1}
pub trait Classifier: Send + Sync {
    fn classify(&self, text: &str) -> anyhow::Result<HashMap<String, f32>>;
}

