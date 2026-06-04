// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Hub → decode push of the CD circuit-breaker tier (P2).
//!
//! The breaker state machine lives on the [`PrefillRouterManager`](super::manager::PrefillRouterManager)
//! and is stepped by its dedicated tick task. When the tier CHANGES, and when a
//! NEW decode worker registers, the hub must push the current
//! [`TierSignal`](crate::handlers::TierSignal) to the **decode** participants so
//! they can cache it and read it synchronously inside GNMT.
//!
//! ## Fan-out target = the decode set ONLY
//!
//! The decode set is owned by the [`ConditionalDisaggManager`](crate::features::disagg::ConditionalDisaggManager)
//! (`inner.decode`), NOT the prefill-router fleet and NOT the full peer
//! registry. A PREFILL instance must NEVER be pushed to (it has no GNMT and the
//! tier is meaningless there). We reach the decode set through a late-bound
//! [`DecodeSetProvider`] so the two feature managers don't form an Arc cycle:
//! the binary constructs the broadcaster (owned by the prefill router, which
//! owns the breaker) and late-binds the CD manager as the provider.
//!
//! ## Push transport
//!
//! Each push is a velo `typed_unary` to the decode instance over
//! [`TIER_SIGNAL_HANDLER`](crate::handlers::TIER_SIGNAL_HANDLER), spawned
//! detached and bounded by a short timeout — mirroring `fan_out_probes` in
//! `server.rs`. A slow/dead decode cannot wedge the tick task. Delivery is
//! best-effort: the decode default is `Calm` (== prior behavior) and the
//! absolute-epoch cache means a later push always reconciles a missed one.

use std::sync::{Arc, OnceLock, Weak};
use std::time::Duration;

use kvbm_observability::CdMetrics;
use kvbm_protocols::disagg::BreakerTier;
use velo_ext::InstanceId;

use super::breaker::CircuitBreaker;
use crate::handlers::{TIER_SIGNAL_HANDLER, TierSignal, TierSignalAck};

/// Wall-clock guard on a single tier-signal push. Short — the push is
/// fire-and-forget control traffic; a hung decode must not back up the tick
/// task or the registration path.
const TIER_PUSH_TIMEOUT: Duration = Duration::from_secs(5);

/// Snake-case tier name for metrics labels / logging.
pub(crate) fn tier_name(t: BreakerTier) -> &'static str {
    match t {
        BreakerTier::Calm => "calm",
        BreakerTier::Warm => "warm",
        BreakerTier::Hot => "hot",
    }
}

/// Integer encoding of a tier for the `kvbm_cd_breaker_tier` gauge.
pub(crate) fn tier_gauge(t: BreakerTier) -> i64 {
    match t {
        BreakerTier::Calm => 0,
        BreakerTier::Warm => 1,
        BreakerTier::Hot => 2,
    }
}

/// Supplies the current set of DECODE instance ids to push a tier signal to.
///
/// Implemented by [`ConditionalDisaggManager`](crate::features::disagg::ConditionalDisaggManager),
/// which owns the decode/prefill role split. Returning ONLY decode ids is the
/// MUST-FIX that keeps prefill instances off the push path.
pub trait DecodeSetProvider: Send + Sync {
    /// Snapshot of the currently-registered decode instance ids.
    fn decode_instances(&self) -> Vec<InstanceId>;
}

/// Fans the current breaker tier out to the decode set over velo.
///
/// Owned (behind an `Arc`) by the [`PrefillRouterManager`](super::manager::PrefillRouterManager);
/// the CD manager holds a clone of the same `Arc` so its `on_register` can push
/// the current tier to a freshly-registered decode. Both the velo handle and
/// the decode-set provider are late-bound (`OnceLock`) because the hub builder
/// constructs the managers and the transport in an order that would otherwise
/// require an Arc cycle.
pub struct TierBroadcaster {
    breaker: Arc<CircuitBreaker>,
    velo: OnceLock<Arc<velo::Velo>>,
    decode_provider: OnceLock<Weak<dyn DecodeSetProvider>>,
    /// Optional hub-process CD metrics. When present, the tick task records the
    /// gauge + transition counter here. `None` until a hub /metrics surface is
    /// wired (the gauge is otherwise inert — the breaker still pushes).
    metrics: Option<CdMetrics>,
}

impl std::fmt::Debug for TierBroadcaster {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TierBroadcaster")
            .field("tier", &self.breaker.tier())
            .field("epoch", &self.breaker.epoch())
            .field("velo_attached", &self.velo.get().is_some())
            .field("provider_bound", &self.decode_provider.get().is_some())
            .field("metrics", &self.metrics.is_some())
            .finish()
    }
}

impl TierBroadcaster {
    /// Build a broadcaster over the given breaker. The velo handle and the
    /// decode-set provider are bound later via [`Self::set_velo`] /
    /// [`Self::set_decode_provider`].
    pub fn new(breaker: Arc<CircuitBreaker>, metrics: Option<CdMetrics>) -> Arc<Self> {
        Arc::new(Self {
            breaker,
            velo: OnceLock::new(),
            decode_provider: OnceLock::new(),
            metrics,
        })
    }

    /// Bind the hub's velo handle (idempotent; first writer wins).
    pub fn set_velo(&self, velo: Arc<velo::Velo>) {
        let _ = self.velo.set(velo);
    }

    /// Bind the decode-set provider (idempotent; first writer wins). Stored as
    /// a `Weak` so the broadcaster does not keep the CD manager alive.
    pub fn set_decode_provider(&self, provider: Weak<dyn DecodeSetProvider>) {
        let _ = self.decode_provider.set(provider);
    }

    /// The breaker this broadcaster fans out.
    pub fn breaker(&self) -> &Arc<CircuitBreaker> {
        &self.breaker
    }

    /// Record a tier transition into the optional hub-process metrics and push
    /// the new tier to every decode. Called by the tick task on each
    /// `evaluate()` that reports a change.
    pub fn on_transition(&self, from: BreakerTier, to: BreakerTier) {
        if let Some(m) = &self.metrics {
            m.set_breaker_tier(tier_gauge(to));
            m.record_breaker_transition(tier_name(from), tier_name(to));
        }
        let epoch = self.breaker.epoch();
        let targets = self.decode_targets();
        if targets.is_empty() {
            return;
        }
        tracing::info!(
            ?from,
            ?to,
            epoch,
            decode_count = targets.len(),
            "CD breaker: pushing tier transition to decode set"
        );
        for id in targets {
            self.spawn_push(id, to, epoch);
        }
    }

    /// Push the CURRENT tier to a single (freshly-registered) decode instance.
    /// Used by the CD manager's `on_register`. A no-op (logged at debug) if the
    /// velo handle is unbound (discovery-only hub).
    pub fn push_current_to(&self, id: InstanceId) {
        let tier = self.breaker.tier();
        let epoch = self.breaker.epoch();
        tracing::debug!(
            ?tier,
            epoch,
            decode = %id,
            "CD breaker: seeding tier on decode registration"
        );
        self.spawn_push(id, tier, epoch);
    }

    /// Current decode instance ids (empty if the provider is unbound or gone).
    fn decode_targets(&self) -> Vec<InstanceId> {
        match self.decode_provider.get().and_then(Weak::upgrade) {
            Some(p) => p.decode_instances(),
            None => Vec::new(),
        }
    }

    /// Spawn a detached, timeout-bounded velo push of `(tier, epoch)` to `id`.
    /// Best-effort: errors are logged and dropped. No-op without a velo handle.
    fn spawn_push(&self, id: InstanceId, tier: BreakerTier, epoch: u64) {
        let Some(velo) = self.velo.get().cloned() else {
            return;
        };
        tokio::spawn(async move {
            let signal = TierSignal { tier, epoch };
            let push = async {
                let unary = velo.typed_unary::<TierSignalAck>(TIER_SIGNAL_HANDLER)?;
                let ack = unary.payload(&signal)?.instance(id).send().await?;
                Ok::<TierSignalAck, anyhow::Error>(ack)
            };
            match tokio::time::timeout(TIER_PUSH_TIMEOUT, push).await {
                Ok(Ok(ack)) => {
                    tracing::debug!(decode = %id, epoch = ack.epoch, ok = ack.ok, "tier push acked");
                }
                Ok(Err(e)) => {
                    tracing::debug!(decode = %id, error = %format!("{e:#}"), "tier push failed");
                }
                Err(_) => {
                    tracing::debug!(decode = %id, "tier push timed out");
                }
            }
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    use super::super::breaker::BreakerConfig;

    struct StaticProvider(Vec<InstanceId>);
    impl DecodeSetProvider for StaticProvider {
        fn decode_instances(&self) -> Vec<InstanceId> {
            self.0.clone()
        }
    }

    #[test]
    fn tier_name_and_gauge_round_trip() {
        for (t, name, g) in [
            (BreakerTier::Calm, "calm", 0),
            (BreakerTier::Warm, "warm", 1),
            (BreakerTier::Hot, "hot", 2),
        ] {
            assert_eq!(tier_name(t), name);
            assert_eq!(tier_gauge(t), g as i64);
        }
    }

    #[test]
    fn decode_targets_empty_without_provider() {
        let breaker = Arc::new(CircuitBreaker::new(BreakerConfig::default(), 1));
        let b = TierBroadcaster::new(breaker, None);
        assert!(b.decode_targets().is_empty(), "no provider ⇒ no targets");
    }

    #[test]
    fn decode_targets_reads_provider() {
        let breaker = Arc::new(CircuitBreaker::new(BreakerConfig::default(), 1));
        let b = TierBroadcaster::new(breaker, None);
        let ids = vec![InstanceId::new_v4(), InstanceId::new_v4()];
        let provider: Arc<dyn DecodeSetProvider> = Arc::new(StaticProvider(ids.clone()));
        b.set_decode_provider(Arc::downgrade(&provider));
        let got = b.decode_targets();
        assert_eq!(got.len(), 2);
        assert_eq!(got, ids);
    }

    #[test]
    fn on_transition_records_metrics() {
        let breaker = Arc::new(CircuitBreaker::new(BreakerConfig::default(), 1));
        let metrics = CdMetrics::new();
        let b = TierBroadcaster::new(breaker, Some(metrics.clone()));
        // No velo, no provider: on_transition still records the metrics and
        // is a no-op on the (empty) push set.
        b.on_transition(BreakerTier::Calm, BreakerTier::Hot);
        assert_eq!(metrics.breaker_tier.get(), tier_gauge(BreakerTier::Hot));
        assert_eq!(
            metrics
                .breaker_transitions_total
                .with_label_values(&["calm", "hot"])
                .get(),
            1
        );
    }

    #[tokio::test]
    async fn spawn_push_without_velo_is_noop() {
        // No velo bound ⇒ spawn_push returns immediately, no panic.
        let breaker = Arc::new(CircuitBreaker::new(BreakerConfig::default(), 1));
        let b = TierBroadcaster::new(breaker, None);
        b.push_current_to(InstanceId::new_v4());
        // Nothing to await; just assert we got here.
    }

    // A dropped provider (Weak fails to upgrade) yields no targets.
    #[test]
    fn dropped_provider_yields_no_targets() {
        let breaker = Arc::new(CircuitBreaker::new(BreakerConfig::default(), 1));
        let b = TierBroadcaster::new(breaker, None);
        let provider: Arc<dyn DecodeSetProvider> =
            Arc::new(StaticProvider(vec![InstanceId::new_v4()]));
        b.set_decode_provider(Arc::downgrade(&provider));
        drop(provider);
        assert!(b.decode_targets().is_empty(), "dropped provider ⇒ no targets");
    }
}
