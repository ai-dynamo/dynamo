use crate::semrouter::{
    classifier::Classifier,
    metrics::{CLASSIFIER_LATENCY, ROUTE_DECISIONS},
    policy::CategoryPolicy,
    types::{RequestMeta, RoutePlan, RoutingMode, Target},
    PolicyConfig,
};
use anyhow::{anyhow, Result};
use std::sync::Arc;
use tracing::debug;

#[cfg(feature = "candle-classifier")]
use crate::semrouter::CandleClassifier;

/// Semantic router using a unified classifier
/// Supports both binary and multi-class classification transparently
pub struct SemRouter {
    policy: CategoryPolicy,
    classifier: Arc<dyn Classifier>,
}

impl SemRouter {
    /// Create a new semantic router with the given policy and classifier
    pub fn new(policy: CategoryPolicy, classifier: Arc<dyn Classifier>) -> Self {
        Self { policy, classifier }
    }

    /// Create a SemRouter from a config file using Candle classifier
    ///
    /// Reads classifier configuration from environment variables:
    /// - SEMROUTER_MODEL_ID: HuggingFace model ID (e.g., "CodeIsAbstract/ReasoningTextClassifier")
    /// - SEMROUTER_MAX_LENGTH: Optional max sequence length (default: 256)
    /// - SEMROUTER_DEVICE: Optional device ("cpu" or "cuda:0", default: "cpu")
    #[cfg(feature = "candle-classifier")]
    pub fn from_config(config_path: impl AsRef<std::path::Path>) -> Result<Self> {
        let policy_config = PolicyConfig::load(config_path)?;
        let policy = CategoryPolicy::new(policy_config);

        // Load Candle classifier from environment variables
        let model_id = std::env::var("SEMROUTER_MODEL_ID")
            .unwrap_or_else(|_| "CodeIsAbstract/ReasoningTextClassifier".to_string());
        let max_length = std::env::var("SEMROUTER_MAX_LENGTH")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(256);

        let device_str = std::env::var("SEMROUTER_DEVICE").unwrap_or_else(|_| "cpu".to_string());
        let device = if device_str.starts_with("cuda") {
            #[cfg(feature = "cuda")]
            {
                let device_id = device_str
                    .strip_prefix("cuda:")
                    .and_then(|s| s.parse::<usize>().ok())
                    .unwrap_or(0);
                candle_core::Device::new_cuda(device_id)?
            }
            #[cfg(not(feature = "cuda"))]
            {
                tracing::warn!("CUDA requested but not available, using CPU");
                candle_core::Device::Cpu
            }
        } else {
            candle_core::Device::Cpu
        };

        tracing::info!("Loading Candle classifier: model={}, device={:?}", model_id, device);
        let classifier = Arc::new(CandleClassifier::from_pretrained(&model_id, max_length, device)?);

        tracing::info!("Semantic router initialized with Candle classifier");
        Ok(Self::new(policy, classifier))
    }

    /// Create a SemRouter from a config file (uses fastText classifier)
    #[cfg(all(feature = "fasttext-classifier", not(feature = "candle-classifier")))]
    pub fn from_config(config_path: impl AsRef<std::path::Path>) -> Result<Self> {
        let policy_config = PolicyConfig::load(config_path)?;
        let policy = CategoryPolicy::new(policy_config);

        let model_path = std::env::var("SEMROUTER_MODEL_PATH")
            .map_err(|_| anyhow!("SEMROUTER_MODEL_PATH not set"))?;

        tracing::info!("Loading fastText classifier from {}", model_path);
        let classifier = Arc::new(crate::semrouter::FasttextClassifier::new(&model_path)?);

        tracing::info!("Semantic router initialized with fastText classifier");
        Ok(Self::new(policy, classifier))
    }

    /// Create a SemRouter from a config file (uses MockClassifier when no ML classifier is available)
    #[cfg(not(any(feature = "candle-classifier", feature = "fasttext-classifier")))]
    pub fn from_config(config_path: impl AsRef<std::path::Path>) -> Result<Self> {
        use crate::semrouter::MockClassifier;

        let policy_config = PolicyConfig::load(config_path)?;
        let policy = CategoryPolicy::new(policy_config.clone());

        // Use MockClassifier in binary mode if reasoning_model is set, multi-class otherwise
        let binary_mode = policy_config.reasoning_model.is_some();
        let classifier = Arc::new(MockClassifier::new(binary_mode));

        tracing::info!("Semantic router initialized with MockClassifier (binary_mode={})", binary_mode);
        Ok(Self::new(policy, classifier))
    }

    /// Create a SemRouter from a config file with a custom classifier
    ///
    /// This allows using any classifier implementation
    pub fn from_config_with_classifier(
        config_path: impl AsRef<std::path::Path>,
        classifier: Arc<dyn Classifier>,
    ) -> Result<Self> {
        let policy_config = PolicyConfig::load(config_path)?;
        let policy = CategoryPolicy::new(policy_config);
        Ok(Self::new(policy, classifier))
    }

    /// Apply semantic routing to a request
    pub fn apply(
        &self,
        req_json: &mut serde_json::Value,
        meta: &RequestMeta<'_>,
    ) -> Option<RouteDecision> {
        let mode = meta.routing_mode;
        let model = meta.model_field.unwrap_or("");
        let is_alias = model.is_empty() || model == "router";

        if matches!(mode, RoutingMode::Shadow)
            || (matches!(mode, RoutingMode::Auto) && !is_alias)
        {
            let decision = self.decide_and_record(meta, "shadow");
            return Some(decision);
        }

        if matches!(mode, RoutingMode::Off) {
            return None;
        }

        if matches!(mode, RoutingMode::Auto) && !is_alias {
            return None;
        }

        match self.decide_plan(meta, "enforce") {
            Some((plan, decision)) => {
                match plan.target {
                    Target::OnPrem { model } => {
                        if let Some(m) = req_json.get_mut("model") {
                            *m = serde_json::Value::String(model.clone());
                        }
                        debug!(
                            "Semantic router enforced model: {} (rationale: {}, winner: {})",
                            model, plan.rationale, plan.winner_label
                        );
                    }
                }
                Some(decision)
            }
            None => None,
        }
    }

    fn decide_and_record(&self, meta: &RequestMeta<'_>, route_type: &str) -> RouteDecision {
        if let Some((winner, rationale, target)) = self.decide(meta) {
            ROUTE_DECISIONS
                .with_label_values(&[route_type, &target, rationale, &winner, meta.transport])
                .inc();
            RouteDecision {
                winner_label: winner,
                rationale,
                target_model: target,
            }
        } else {
            RouteDecision {
                winner_label: "none".into(),
                rationale: "error",
                target_model: "none".into(),
            }
        }
    }

    fn decide_plan(
        &self,
        meta: &RequestMeta<'_>,
        route_type: &str,
    ) -> Option<(RoutePlan, RouteDecision)> {
        if let Some((winner, rationale, target, plan)) = self.decide_full(meta) {
            ROUTE_DECISIONS
                .with_label_values(&[route_type, &target, rationale, &winner, meta.transport])
                .inc();
            let decision = RouteDecision {
                winner_label: winner,
                rationale,
                target_model: target,
            };
            Some((plan, decision))
        } else {
            None
        }
    }

    fn decide(&self, meta: &RequestMeta<'_>) -> Option<(String, &'static str, String)> {
        self.decide_full(meta).map(|(w, r, t, _)| (w, r, t))
    }

    fn decide_full(
        &self,
        meta: &RequestMeta<'_>,
    ) -> Option<(String, &'static str, String, RoutePlan)> {
        let text = meta.request_text?;
        if text.is_empty() {
            return None;
        }

        let t0 = std::time::Instant::now();

        // Run classifier (works for both binary and multi-class)
        let probs = self.classifier.classify(text).ok()?;

        CLASSIFIER_LATENCY
            .with_label_values(&[meta.transport])
            .observe(t0.elapsed().as_secs_f64() * 1000.0);

        // Policy decides based on probabilities
        let plan = self.policy.decide(&probs);

        let target_str = match &plan.target {
            Target::OnPrem { model } => model.clone(),
        };
        Some((
            plan.winner_label.clone(),
            plan.rationale,
            target_str,
            plan,
        ))
    }
}

#[derive(Clone, Debug)]
pub struct RouteDecision {
    pub winner_label: String,
    pub rationale: &'static str,
    pub target_model: String,
}

