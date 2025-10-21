use crate::semrouter::config::{PolicyConfig, Rule};
use crate::semrouter::types::{RoutePlan, Target};
use std::collections::HashMap;

pub struct CategoryPolicy {
    cfg: PolicyConfig,
}

impl CategoryPolicy {
    pub fn new(cfg: PolicyConfig) -> Self {
        Self { cfg }
    }

    pub fn decide(&self, dist: &HashMap<String, f32>) -> RoutePlan {
        for Rule {
            when_any,
            route_onprem_model,
            rationale,
        } in &self.cfg.rules
        {
            if when_any
                .iter()
                .any(|c| dist.get(&c.label).copied().unwrap_or(0.0) >= c.min_conf)
            {
                let winner = self.top_label(dist);
                return RoutePlan {
                    target: Target::OnPrem {
                        model: route_onprem_model.clone(),
                    },
                    rationale: Box::leak(rationale.clone().into_boxed_str()),
                    winner_label: winner,
                };
            }
        }

        let mut best: Option<(&str, f32, i32)> = None;
        for (label, &p) in dist {
            if p >= self.cfg.threshold_min_conf {
                let w = *self.cfg.weights.get(label).unwrap_or(&0);
                match best {
                    None => best = Some((label.as_str(), p, w)),
                    Some((_, bp, bw)) => {
                        if w > bw || (w == bw && p > bp) {
                            best = Some((label.as_str(), p, w));
                        }
                    }
                }
            }
        }
        if let Some((label, _p, _w)) = best
            && let Some(rule) = self
                .cfg
                .rules
                .iter()
                .find(|r| r.when_any.iter().any(|c| c.label == label))
        {
            return RoutePlan {
                target: Target::OnPrem {
                    model: rule.route_onprem_model.clone(),
                },
                rationale: Box::leak(rule.rationale.clone().into_boxed_str()),
                winner_label: label.to_string(),
            };
        }

        RoutePlan {
            target: Target::OnPrem {
                model: self.cfg.abstain_onprem_model.clone(),
            },
            rationale: "abstain",
            winner_label: self.top_label(dist),
        }
    }

    fn top_label(&self, dist: &HashMap<String, f32>) -> String {
        dist.iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(l, _)| l.clone())
            .unwrap_or_else(|| "none".into())
    }
}

