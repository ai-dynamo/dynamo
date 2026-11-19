// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shortest-predicted-reply-first admission on the backend engine.
//!
//! Predicts each request's output length from its session's earlier replies and its prompt
//! length, and forwards the negated prediction as the request's backend priority. An engine whose
//! waiting queue orders by priority then admits the shortest predicted reply first; an engine
//! that does not order by priority is unaffected. The prediction replaces any priority the client
//! sent.
//!
//! Ported from TensorRT-LLM's `tensorrt_llm/serve/output_length_predictor.py`.

mod predictor;
mod trees;

use std::collections::HashMap;
use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Arc;
use std::time::{Duration, Instant};

use async_trait::async_trait;
use dynamo_kv_router::plugins::prelude::*;
use serde::Deserialize;

use self::predictor::{Features, MAX_TOKENS, Predictor};

const CLASSIFIER_TYPE: &str = "output_length_prediction";
// Terminal events clear a request's row, so this only bounds memory if they stop arriving. A
// request past the cap is still prioritized; its reply is not learned.
const MAX_PENDING: usize = 200_000;
const STATS_PERIOD: Duration = Duration::from_secs(60);

// The plugin takes no parameters; this rejects any that a policy file supplies.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Parameters {}

fn provider(
    parameters: &RequestClassifierParameters,
) -> Result<RequestClassifierFactory, RequestClassifierProviderError> {
    let Parameters {} = parameters.deserialize()?;
    Ok(Arc::new(|_context| Box::new(OutputLengthClassifier::new())))
}

pub(super) fn register(
    registry: &mut RouterPluginRegistry,
) -> Result<(), RequestClassifierRegistryError> {
    registry.register_request_classifier(CLASSIFIER_TYPE, Arc::new(provider))
}

struct Pending {
    session: Option<u64>,
    isl: usize,
    features: Features,
}

pub struct OutputLengthClassifier {
    predictor: Predictor,
    pending: HashMap<String, Pending>,
    stats_logged: Instant,
}

impl Default for OutputLengthClassifier {
    fn default() -> Self {
        Self::new()
    }
}

// Session IDs are client-chosen and unbounded, so histories are keyed by a digest. A collision
// only blends two sessions' histories.
fn session_key(session_id: &str) -> u64 {
    let mut hasher = DefaultHasher::new();
    session_id.hash(&mut hasher);
    hasher.finish()
}

fn backend_priority(predicted_tokens: f64) -> i32 {
    -(predicted_tokens.clamp(0.0, MAX_TOKENS).round() as i32)
}

impl OutputLengthClassifier {
    pub fn new() -> Self {
        tracing::info!(
            target: "output_length_prediction",
            "output-length prediction enabled: backend priority follows the predicted reply length"
        );
        Self {
            predictor: Predictor::new(),
            pending: HashMap::new(),
            stats_logged: Instant::now(),
        }
    }

    fn predict(&mut self, request: &mut ClassifyRequest, now: Instant) {
        let session = request
            .session_context()
            .map(|session| session_key(session.session_id()));
        let isl = request.isl_tokens();
        let prediction = self.predictor.predict(session, isl, now);
        tracing::trace!(
            target: "output_length_prediction",
            request_id = request.request_id(),
            predicted_tokens = prediction.tokens,
            source = ?prediction.source,
            "predicted output length"
        );
        request.set_backend_priority(backend_priority(prediction.tokens));
        if let Some(request_id) = request.request_id()
            && self.pending.len() < MAX_PENDING
        {
            self.pending.insert(
                request_id.to_owned(),
                Pending {
                    session,
                    isl,
                    features: prediction.features,
                },
            );
        }
    }

    fn observe(&mut self, event: ClassifyEvent, now: Instant) {
        match event {
            ClassifyEvent::Completed {
                request_id,
                context_tokens,
                ..
            } => {
                let Some(pending) = self.pending.remove(&request_id) else {
                    return;
                };
                // The context total is the prompt plus what was generated; without it the reply
                // length is unknown.
                if let Some(context_tokens) = context_tokens {
                    let out = context_tokens.saturating_sub(pending.isl);
                    self.predictor
                        .learn(pending.session, pending.isl, out, &pending.features, now);
                }
            }
            ClassifyEvent::Aborted { request_id, .. } => {
                self.pending.remove(&request_id);
            }
            _ => {}
        }
        self.log_stats(now);
    }

    fn log_stats(&mut self, now: Instant) {
        if now.duration_since(self.stats_logged) < STATS_PERIOD {
            return;
        }
        self.stats_logged = now;
        let stats = self.predictor.stats();
        tracing::info!(
            target: "output_length_prediction",
            sessions = stats.sessions,
            replies_learned = stats.learned,
            replies_skipped = stats.skipped,
            ridge_fits = stats.ridge_fits,
            tree_fits = stats.tree_fits,
            predicted_by_prior = stats.from_prior,
            predicted_by_median = stats.from_median,
            predicted_by_session = stats.from_session,
            predicted_by_ridge = stats.from_ridge,
            predicted_by_trees = stats.from_trees,
            pending = self.pending.len(),
            "output-length prediction"
        );
    }
}

#[async_trait]
impl RequestClassifier for OutputLengthClassifier {
    fn classify(
        &mut self,
        mut request: ClassifyRequest,
        _workers: &dyn WorkerView,
    ) -> Result<ClassifyRequest, Box<ClassifierError>> {
        self.predict(&mut request, Instant::now());
        Ok(request)
    }

    async fn on_event(&mut self, event: ClassifyEvent) {
        self.observe(event, Instant::now());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WORKER: WorkerWithDpRank = WorkerWithDpRank {
        worker_id: 1,
        dp_rank: 0,
    };

    fn request(request_id: &str, session: &str, isl: usize) -> ClassifyRequest {
        ClassifyRequest::with_timing(isl, 0, tokio::time::Instant::now())
            .with_request_id(request_id)
            .with_session_context(SessionContext::new(session.to_owned(), None, None, None))
    }

    fn priority(
        classifier: &mut OutputLengthClassifier,
        request_id: &str,
        session: &str,
        isl: usize,
        now: Instant,
    ) -> i32 {
        let mut request = request(request_id, session, isl);
        classifier.predict(&mut request, now);
        request.backend_priority().unwrap()
    }

    fn complete(
        classifier: &mut OutputLengthClassifier,
        request_id: &str,
        context_tokens: Option<usize>,
        now: Instant,
    ) {
        classifier.observe(
            ClassifyEvent::Completed {
                request_id: request_id.to_owned(),
                worker: WORKER,
                context_tokens,
            },
            now,
        );
    }

    #[test]
    fn a_session_with_shorter_replies_gets_the_higher_priority() {
        let now = Instant::now();
        let mut classifier = OutputLengthClassifier::new();
        assert_eq!(priority(&mut classifier, "a1", "short", 1_000, now), -400);
        assert_eq!(priority(&mut classifier, "b1", "long", 1_000, now), -400);
        complete(&mut classifier, "a1", Some(1_050), now);
        complete(&mut classifier, "b1", Some(4_000), now);

        let short = priority(&mut classifier, "a2", "short", 1_100, now);
        let long = priority(&mut classifier, "b2", "long", 4_100, now);
        assert_eq!((short, long), (-50, -3_000));
        assert!(short > long);
    }

    #[test]
    fn a_reply_without_a_length_or_an_aborted_request_is_not_learned() {
        let now = Instant::now();
        let mut classifier = OutputLengthClassifier::new();
        priority(&mut classifier, "a1", "s", 1_000, now);
        priority(&mut classifier, "a2", "s", 1_000, now);
        complete(&mut classifier, "a1", None, now);
        classifier.observe(
            ClassifyEvent::Aborted {
                request_id: "a2".to_owned(),
                worker: Some(WORKER),
                error: None,
            },
            now,
        );

        assert!(classifier.pending.is_empty());
        assert_eq!(classifier.predictor.stats().learned, 0);
        assert_eq!(priority(&mut classifier, "a3", "s", 1_000, now), -400);
    }

    #[test]
    fn a_request_without_a_session_is_predicted_from_all_replies() {
        let now = Instant::now();
        let mut classifier = OutputLengthClassifier::new();
        priority(&mut classifier, "a1", "s", 1_000, now);
        complete(&mut classifier, "a1", Some(1_250), now);

        let mut anonymous = ClassifyRequest::with_timing(2_000, 0, tokio::time::Instant::now());
        classifier.predict(&mut anonymous, now);
        assert_eq!(anonymous.backend_priority(), Some(-250));
        assert!(classifier.pending.is_empty());
    }

    #[test]
    fn the_policy_file_selects_it_and_rejects_parameters() {
        let resolve = |yaml: &str| {
            let mut registry = crate::default_registry();
            crate::register(&mut registry).unwrap();
            let policy = tempfile::NamedTempFile::new().unwrap();
            std::fs::write(policy.path(), yaml).unwrap();
            let config = dynamo_kv_router::config::KvRouterConfig {
                router_policy_config: Some(policy.path().display().to_string()),
                ..Default::default()
            };
            registry.resolve_plugins(&config)
        };

        let plugins = resolve("request_classifier:\n  type: output_length_prediction\n").unwrap();
        let factory = plugins.request_classifier().unwrap();
        let _classifier = factory(RequestClassifierContext::new(16, Vec::new));

        assert!(
            resolve(
                "request_classifier:\n  type: output_length_prediction\n  parameters:\n    predictor: ridge\n"
            )
            .is_err()
        );
    }
}
