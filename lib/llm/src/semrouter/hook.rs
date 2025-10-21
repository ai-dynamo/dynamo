use crate::semrouter::{
    classifier::MultiClassifier,
    metrics::{CLASSIFIER_LATENCY, ROUTE_DECISIONS},
    policy::CategoryPolicy,
    types::{RequestMeta, RoutePlan, RoutingMode, Target},
};
use std::sync::Arc;
use tracing::debug;

pub struct SemRouter<C: MultiClassifier + ?Sized> {
    policy: CategoryPolicy,
    clf: Arc<C>,
}

impl<C: MultiClassifier + ?Sized> SemRouter<C> {
    pub fn new(policy: CategoryPolicy, clf: Arc<C>) -> Self {
        Self { policy, clf }
    }

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
        let dist = self.clf.probs(text).ok()?;
        CLASSIFIER_LATENCY
            .with_label_values(&[meta.transport])
            .observe(t0.elapsed().as_secs_f64() * 1000.0);

        let plan = self.policy.decide(&dist);
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

