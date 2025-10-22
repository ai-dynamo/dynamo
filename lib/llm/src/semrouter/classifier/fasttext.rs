use super::{Classifier, Classification};
use anyhow::{anyhow, Result};

#[cfg(feature = "clf-fasttext")]
use fasttext::FastText;

#[cfg(feature = "clf-fasttext")]
use sha2::{Digest, Sha256};

/// fastText-based classifier for multiclass classification (pure Rust path)
#[cfg(feature = "clf-fasttext")]
pub struct FasttextClassifier {
    ft: FastText,
}

#[cfg(feature = "clf-fasttext")]
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

#[cfg(feature = "clf-fasttext")]
impl FasttextClassifier {
    fn classify_sync(&self, text: &str) -> Result<Classification> {
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

        // Convert fastText labels (with __label__ prefix) to our format
        let mut labels: Vec<LabelScore> = preds
            .into_iter()
            .filter_map(|p| {
                // Strip __label__ prefix if present
                let label = p.label.strip_prefix("__label__").unwrap_or(&p.label).to_string();
                let score = p.prob.clamp(0.0, 1.0) as f32;
                Some(LabelScore { label, score })
            })
            .collect();

        // Sort by score descending
        labels.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));

        let latency = start.elapsed();
        if !labels.is_empty() {
            tracing::info!(
                latency_us = latency.as_micros(),
                top_label = %labels[0].label,
                top_score = %format!("{:.3}", labels[0].score),
                "fastText classification complete"
            );
        } else {
            tracing::warn!("fastText returned no predictions");
        }

        Ok(Classification { labels })
    }
}

#[cfg(feature = "clf-fasttext")]
#[async_trait::async_trait]
impl Classifier for FasttextClassifier {
    async fn predict(&self, text: &str) -> Result<Classification> {
        // FastText is CPU-bound sync operation, but wrap in async for trait compatibility
        // Could use spawn_blocking for true non-blocking behavior if needed
        self.classify_sync(text)
    }

    fn name(&self) -> &'static str {
        "fasttext"
    }
}

// Stub when the feature is not enabled
#[cfg(not(feature = "clf-fasttext"))]
pub struct FasttextClassifier;

#[cfg(not(feature = "clf-fasttext"))]
impl FasttextClassifier {
    pub fn new(_model_path: &str) -> Result<Self> {
        Err(anyhow!(
            "FasttextClassifier requires 'clf-fasttext' feature"
        ))
    }
}

#[cfg(not(feature = "clf-fasttext"))]
#[async_trait::async_trait]
impl Classifier for FasttextClassifier {
    async fn predict(&self, _text: &str) -> Result<Classification> {
        Err(anyhow!(
            "FasttextClassifier requires 'clf-fasttext' feature"
        ))
    }

    fn name(&self) -> &'static str {
        "fasttext"
    }
}
