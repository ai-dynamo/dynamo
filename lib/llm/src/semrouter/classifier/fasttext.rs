use super::Classifier;
use anyhow::{anyhow, Result};
use std::collections::HashMap;

#[cfg(feature = "fasttext-classifier")]
use fasttext::FastText;

#[cfg(feature = "fasttext-classifier")]
use sha2::{Digest, Sha256};

/// fastText-based classifier for binary classification (pure Rust path)
#[cfg(feature = "fasttext-classifier")]
pub struct FasttextClassifier {
    ft: FastText,
}

#[cfg(feature = "fasttext-classifier")]
impl FasttextClassifier {
    pub fn new(model_path: &str) -> Result<Self> {
        // Log file size + sha256 to be 100% sure we’re loading the same bytes
        let meta = std::fs::metadata(model_path)?;
        let size_bytes = meta.len();
        let bytes = std::fs::read(model_path)?;
        let sha256 = {
            let mut hasher = Sha256::new();
            hasher.update(&bytes);
            format!("{:x}", hasher.finalize())
        };

        let mut ft = FastText::new();
        ft.load_model(model_path)
            .map_err(|e| anyhow!("Failed to load fastText model from {}: {}", model_path, e))?;

        // Verify expected labels exist
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

        // Introspect args so we know the loss/ngrams actually used by the model
        let args = ft.get_args();
        let loss = format!("{:?}", args.loss());
        let word_ngrams = args.word_ngrams();
        let dim = args.dim();
        let minn = args.minn();
        let maxn = args.maxn();

        tracing::info!(
            "fastText model: path={} size_bytes={} sha256={}",
            model_path,
            size_bytes,
            sha256
        );
        tracing::info!(
            "Initialized fastText classifier: {} labels, loss={}, wordNgrams={}, dim={}, minn={}, maxn={}",
            labels.len(),
            loss,
            word_ngrams,
            dim,
            minn,
            maxn
        );

        Ok(Self { ft })
    }
}

#[cfg(feature = "fasttext-classifier")]
impl Classifier for FasttextClassifier {
    fn classify(&self, text: &str) -> Result<HashMap<String, f32>> {
        let start = std::time::Instant::now();

        // Preview for logs
        let text_preview = if text.len() > 60 {
            format!("{}...", &text[..60])
        } else {
            text.to_string()
        };
        let bytes = text.as_bytes().len();
        tracing::info!("fastText classify: bytes={} preview={:?}", bytes, text_preview);

        // === IMPORTANT: newline for parity with CLI/Python ===
        // (Some bindings/paths expect line-terminated input; this helps parity.)
        let mut query = String::with_capacity(text.len() + 1);
        query.push_str(text);
        if !query.ends_with('\n') {
            query.push('\n');
        }

        // Ask for ALL labels; threshold 0.0 means don't filter
        // NOTE: API is `predict<T: AsRef<str>>(text, k, threshold)`
        let preds = self
            .ft
            .predict(&query, -1, 0.0)
            .map_err(|e| anyhow!("fastText prediction failed: {}", e))?;

        tracing::info!("fastText returned {} preds", preds.len());
        for (i, p) in preds.iter().enumerate() {
            tracing::info!("pred[{}]: label='{}' prob={:.6}", i, p.label, p.prob);
        }

        // Extract our two labels (if one is missing, we’ll complement)
        let mut p_reason: Option<f32> = None;
        let mut p_non: Option<f32> = None;
        for p in &preds {
            match p.label.as_str() {
                "__label__reasoning" => p_reason = Some(p.prob as f32),
                "__label__non-reasoning" => p_non = Some(p.prob as f32),
                _ => {}
            }
        }

        // If only one label arrives, use complement. If both, trust them.
        let (mut p_reason, mut p_non) = match (p_reason, p_non) {
            (Some(r), Some(n)) => (r, n),
            (Some(r), None) => (r, (1.0 - r).max(0.0).min(1.0)),
            (None, Some(n)) => ((1.0 - n).max(0.0).min(1.0), n),
            (None, None) => {
                tracing::warn!("No known labels returned; defaulting to 0.5/0.5");
                (0.5, 0.5)
            }
        };

        // Normalize gently (fastText can emit 1.00000x from rounding)
        // See: known rounding >1.0 behavior.
        let sum = (p_reason + p_non).clamp(1e-9, 1e9);
        p_reason = (p_reason / sum).clamp(0.0, 1.0);
        p_non = (p_non / sum).clamp(0.0, 1.0);

        let latency = start.elapsed();
        let predicted_class = if p_reason > p_non { "reasoning" } else { "non-reasoning" };
        let confidence = p_reason.max(p_non);

        tracing::info!(
            latency_us = latency.as_micros(),
            predicted_class = %predicted_class,
            confidence = %format!("{:.3}", confidence),
            reasoning_prob = %format!("{:.3}", p_reason),
            non_reasoning_prob = %format!("{:.3}", p_non),
            "fastText classification complete"
        );

        let mut result = HashMap::with_capacity(2);
        result.insert("reasoning".to_string(), p_reason);
        result.insert("non-reasoning".to_string(), p_non);
        Ok(result)
    }
}

// Stub when the feature is not enabled
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
