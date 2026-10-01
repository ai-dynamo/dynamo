// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Temporary, opt-in POC adapter for Frontend-owned scheduler gauges.
//!
//! Relay does not yet publish these scheduler snapshots. This adapter is valid
//! only for one Frontend replica/model per pool, static prefill accounting and
//! no scheduler replica sync. Gauge timestamps are collection times, not event
//! emission times. Explicit membership prevents an absent rank looking idle.
//! Replace this adapter with complete Relay scheduler snapshots when available.

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::sync::Arc;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use anyhow::{Result, bail};
use dynamo_kv_router::global_view::PoolId;
use parking_lot::RwLock;
use serde::Deserialize;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[serde(deny_unknown_fields)]
pub struct SchedulerMember {
    pub worker_id: String,
    pub dp_rank: u32,
}

#[derive(Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetricsConfig {
    pub url: String,
    pub members: Vec<SchedulerMember>,
    pub block_size: u32,
}

impl MetricsConfig {
    pub fn validate(&self) -> Result<()> {
        let url = reqwest::Url::parse(&self.url)?;
        if !matches!(url.scheme(), "http" | "https")
            || url.host().is_none()
            || !url.username().is_empty()
            || url.password().is_some()
            || self.block_size == 0
            || self.members.is_empty()
            || self
                .members
                .iter()
                .any(|m| m.worker_id.parse::<u64>().is_err())
            || self.members.iter().collect::<BTreeSet<_>>().len() != self.members.len()
        {
            bail!("invalid experimental scheduler metrics configuration");
        }
        Ok(())
    }
}

pub struct MetricsSource {
    pub pool_id: PoolId,
    pub model: String,
    pub config: MetricsConfig,
}

#[derive(Clone, Copy, Debug)]
pub struct SchedulerSnapshot {
    /// Sum across every configured worker/rank. These are scheduled active
    /// blocks, NOT occupied cache blocks reported by KV usage.
    pub prefill_tokens: u64,
    pub decode_blocks: u64,
    pub block_size: u32,
    pub collected_at_unix_ms: u64,
}

pub trait SchedulerLoadRepository: Send + Sync {
    fn snapshot(&self, pool: &PoolId, model: &str, now_ms: u64) -> Option<SchedulerSnapshot>;
}

pub struct SchedulerMetrics {
    sources: Vec<MetricsSource>,
    samples: RwLock<HashMap<(PoolId, String), SchedulerSnapshot>>,
    client: reqwest::Client,
    interval: Duration,
    max_age_ms: u64,
}

impl SchedulerMetrics {
    pub fn new(
        sources: Vec<MetricsSource>,
        interval_ms: u64,
        max_age_ms: u64,
    ) -> Result<Arc<Self>> {
        if sources.is_empty() || interval_ms == 0 || max_age_ms <= interval_ms {
            bail!("scheduler metrics need sources and max age greater than scrape interval");
        }
        for source in &sources {
            source.config.validate()?;
        }
        Ok(Arc::new(Self {
            sources,
            samples: RwLock::new(HashMap::new()),
            client: reqwest::Client::builder()
                .timeout(Duration::from_millis(max_age_ms))
                .redirect(reqwest::redirect::Policy::none())
                .build()?,
            interval: Duration::from_millis(interval_ms),
            max_age_ms,
        }))
    }

    pub async fn run(&self, cancel: CancellationToken) {
        let mut tick = tokio::time::interval(self.interval);
        tick.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
        loop {
            tokio::select! {
                _ = cancel.cancelled() => return,
                _ = tick.tick() => {}
            }
            let refresh = futures::future::join_all(self.sources.iter().map(|source| async move {
                let collected_at_unix_ms = now_ms();
                let sample = async {
                    let text = self.client.get(&source.config.url).send().await?
                        .error_for_status()?.text().await?;
                    let (prefill_tokens, decode_blocks) = parse_snapshot(&text, &source.config.members)?;
                    Ok::<_, anyhow::Error>(SchedulerSnapshot {
                        prefill_tokens, decode_blocks,
                        block_size: source.config.block_size,
                        collected_at_unix_ms,
                    })
                }.await;
                let key = (source.pool_id.clone(), source.model.clone());
                match sample {
                    Ok(sample) => { self.samples.write().insert(key, sample); }
                    Err(cause) => {
                        // Drop coverage failures immediately, rather than continuing to
                        // price a pool with obsolete membership until its age expires.
                        self.samples.write().remove(&key);
                        tracing::warn!(pool_id = %source.pool_id, %cause, "scheduler metrics unavailable");
                    }
                }
            }));
            tokio::select! {
                _ = cancel.cancelled() => return,
                _ = refresh => {}
            }
        }
    }
}

impl SchedulerLoadRepository for SchedulerMetrics {
    fn snapshot(&self, pool: &PoolId, model: &str, now_ms: u64) -> Option<SchedulerSnapshot> {
        let sample = *self.samples.read().get(&(pool.clone(), model.to_owned()))?;
        (now_ms.checked_sub(sample.collected_at_unix_ms)? <= self.max_age_ms).then_some(sample)
    }
}

pub(super) fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|v| u64::try_from(v.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or(0)
}

/// Parse only the exact scheduler gauges. Every configured rank must have both
/// values, exactly once. Unexpected ranks also invalidate the snapshot.
fn parse_snapshot(text: &str, members: &[SchedulerMember]) -> Result<(u64, u64)> {
    let expected: BTreeSet<_> = members.iter().cloned().collect();
    let mut values = [BTreeMap::new(), BTreeMap::new()];
    for line in text.lines().filter(|line| !line.starts_with('#')) {
        let kind = if line.starts_with("dynamo_frontend_worker_active_prefill_tokens{") {
            0
        } else if line.starts_with("dynamo_frontend_worker_active_decode_blocks{") {
            1
        } else {
            continue;
        };
        let (labels, value) = line
            .split_once('}')
            .ok_or_else(|| anyhow::anyhow!("missing metric labels"))?;
        let labels = labels.split_once('{').unwrap().1;
        let labels: HashMap<_, _> = labels
            .split(',')
            .filter_map(|label| label.split_once('='))
            .map(|(k, v)| (k.trim(), v.trim_matches('"')))
            .collect();
        let member = SchedulerMember {
            worker_id: labels
                .get("worker_id")
                .ok_or_else(|| anyhow::anyhow!("missing worker_id"))?
                .to_string(),
            dp_rank: labels
                .get("dp_rank")
                .ok_or_else(|| anyhow::anyhow!("missing dp_rank"))?
                .parse()?,
        };
        // Dynamo uses the decode worker label for aggregated schedulers too.
        if labels.get("worker_type") != Some(&"decode") || !expected.contains(&member) {
            bail!("unexpected scheduler membership or role");
        }
        let value: f64 = value
            .split_whitespace()
            .next()
            .ok_or_else(|| anyhow::anyhow!("missing gauge"))?
            .parse()?;
        if !value.is_finite() || value < 0.0 || value.fract() != 0.0 || value >= u64::MAX as f64 {
            bail!("invalid scheduler gauge");
        }
        if values[kind].insert(member, value as u64).is_some() {
            bail!("duplicate scheduler gauge");
        }
    }
    if values
        .iter()
        .any(|v| v.keys().cloned().collect::<BTreeSet<_>>() != expected)
    {
        bail!("incomplete scheduler rank coverage");
    }
    let sum = |v: &BTreeMap<SchedulerMember, u64>| {
        v.values()
            .try_fold(0u64, |s, v| s.checked_add(*v))
            .ok_or_else(|| anyhow::anyhow!("scheduler gauge sum overflow"))
    };
    Ok((sum(&values[0])?, sum(&values[1])?))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn gauges(worker: &str, prefill: &str, decode: &str) -> String {
        format!(
            "dynamo_frontend_worker_active_prefill_tokens{{worker_id=\"{worker}\",dp_rank=\"0\",worker_type=\"decode\"}} {prefill}\ndynamo_frontend_worker_active_decode_blocks{{worker_id=\"{worker}\",dp_rank=\"0\",worker_type=\"decode\"}} {decode}\n"
        )
    }
    #[tokio::test]
    async fn stale_and_future_collections_are_unavailable() {
        use dynamo_kv_router::global_view::{PoolIdDeriver, PoolKey, V1PoolIdDeriver};
        let pool = V1PoolIdDeriver.derive(&PoolKey::new("site", "ns", "dgd").unwrap());
        let metrics = SchedulerMetrics::new(
            vec![MetricsSource {
                pool_id: pool.clone(),
                model: "model".into(),
                config: MetricsConfig {
                    url: "http://127.0.0.1:1/metrics".into(),
                    members: vec![SchedulerMember {
                        worker_id: "1".into(),
                        dp_rank: 0,
                    }],
                    block_size: 64,
                },
            }],
            500,
            2000,
        )
        .unwrap();
        metrics.samples.write().insert(
            (pool.clone(), "model".into()),
            SchedulerSnapshot {
                prefill_tokens: 0,
                decode_blocks: 0,
                block_size: 64,
                collected_at_unix_ms: 1000,
            },
        );
        assert!(metrics.snapshot(&pool, "model", 2999).is_some());
        assert!(metrics.snapshot(&pool, "model", 3001).is_none());
        assert!(metrics.snapshot(&pool, "model", 999).is_none());
        assert!(metrics.snapshot(&pool, "other", 1000).is_none());
    }

    #[test]
    fn require_complete_exact_membership_and_valid_gauges() {
        let members = vec![
            SchedulerMember {
                worker_id: "1".into(),
                dp_rank: 0,
            },
            SchedulerMember {
                worker_id: "2".into(),
                dp_rank: 0,
            },
        ];
        let complete = gauges("1", "64", "2") + &gauges("2", "128", "3");
        assert_eq!(parse_snapshot(&complete, &members).unwrap(), (192, 5));
        assert!(parse_snapshot(&gauges("1", "0", "0"), &members).is_err());
        assert!(parse_snapshot(&(complete.clone() + &gauges("1", "0", "0")), &members).is_err());
        assert!(parse_snapshot(&(complete + &gauges("3", "0", "0")), &members).is_err());
        assert!(
            parse_snapshot(
                &(gauges("1", "NaN", "0") + &gauges("2", "0", "0")),
                &members
            )
            .is_err()
        );
    }
}
