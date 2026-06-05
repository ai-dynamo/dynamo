// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the P2 CD-breaker hub→decode tier PUSH.
//!
//! These exercise [`TierBroadcaster`] against real `velo::Velo` peers — no
//! GPU, no vLLM, no Python. A "hub" velo fans a [`TierSignal`] out to a "decode"
//! velo that installs the [`TIER_SIGNAL_HANDLER`] (caching into a shared
//! `AtomicU8`). The tests assert:
//!   1. a tier TRANSITION is delivered to the decode's handler,
//!   2. push-on-register seeds a fresh decode with the CURRENT tier,
//!   3. a PREFILL instance (absent from the decode-set provider) is NEVER
//!      pushed to (the MUST-FIX: fan-out targets `inner.decode` only).

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, AtomicU64, Ordering};
use std::time::Duration;

use kvbm_hub::handlers::{TIER_SIGNAL_HANDLER, TierSignal, TierSignalAck};
use kvbm_hub::{BreakerConfig, CircuitBreaker, DecodeSetProvider, TierBroadcaster};
use kvbm_protocols::disagg::BreakerTier;
use velo::Handler;
use velo::transports::tcp::TcpTransportBuilder;
use velo_ext::InstanceId;

fn new_velo_transport() -> Arc<velo::transports::tcp::TcpTransport> {
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    Arc::new(
        TcpTransportBuilder::new()
            .from_listener(listener)
            .unwrap()
            .build()
            .unwrap(),
    )
}

async fn new_velo() -> Arc<velo::Velo> {
    velo::Velo::builder()
        .add_transport(new_velo_transport())
        .build()
        .await
        .unwrap()
}

/// A decode-side tier cache + the velo handler that writes it. Mirrors the
/// connector's `DecodeTierCache` (kept local here so the hub test crate doesn't
/// depend on the connector). Epoch-gated, absolute state.
#[derive(Default)]
struct TestTierCache {
    tier: AtomicU8,
    epoch: AtomicU64,
}

impl TestTierCache {
    fn tier(&self) -> u8 {
        self.tier.load(Ordering::Acquire)
    }
    fn epoch(&self) -> u64 {
        self.epoch.load(Ordering::Acquire)
    }
    fn apply(&self, tier: u8, epoch: u64) -> bool {
        let cached = self.epoch.load(Ordering::Acquire);
        if epoch < cached {
            return false;
        }
        self.tier.store(tier, Ordering::Release);
        self.epoch.store(epoch, Ordering::Release);
        true
    }
}

fn tier_u8(t: BreakerTier) -> u8 {
    match t {
        BreakerTier::Calm => 0,
        BreakerTier::Warm => 1,
        BreakerTier::Hot => 2,
    }
}

/// Install the TIER_SIGNAL handler on `velo`, caching into `cache`.
fn install_tier_handler(velo: &velo::Velo, cache: Arc<TestTierCache>) {
    velo.register_handler(
        Handler::typed_unary_async::<TierSignal, TierSignalAck, _, _>(
            TIER_SIGNAL_HANDLER,
            move |ctx| {
                let cache = Arc::clone(&cache);
                async move {
                    let ok = cache.apply(tier_u8(ctx.input.tier), ctx.input.epoch);
                    Ok(TierSignalAck {
                        epoch: ctx.input.epoch,
                        ok,
                    })
                }
            },
        )
        .build(),
    )
    .unwrap();
}

struct StaticDecodeSet(Vec<InstanceId>);
impl DecodeSetProvider for StaticDecodeSet {
    fn decode_instances(&self) -> Vec<InstanceId> {
        self.0.clone()
    }
}

/// Build a hub velo + a decode velo as mutual peers, with the decode's tier
/// handler installed. Returns (hub_velo, decode_velo, decode_cache).
async fn hub_and_decode() -> (Arc<velo::Velo>, Arc<velo::Velo>, Arc<TestTierCache>) {
    let hub = new_velo().await;
    let decode = new_velo().await;
    hub.messenger().register_peer(decode.peer_info()).unwrap();
    decode.messenger().register_peer(hub.peer_info()).unwrap();
    let cache = Arc::new(TestTierCache::default());
    install_tier_handler(&decode, Arc::clone(&cache));
    // Let peer registration settle.
    tokio::time::sleep(Duration::from_millis(200)).await;
    (hub, decode, cache)
}

/// Poll until `cond` or a deadline.
async fn wait_until(mut cond: impl FnMut() -> bool, budget: Duration) -> bool {
    let deadline = std::time::Instant::now() + budget;
    while std::time::Instant::now() < deadline {
        if cond() {
            return true;
        }
        tokio::time::sleep(Duration::from_millis(25)).await;
    }
    cond()
}

/// A tier TRANSITION (on_transition) is delivered to the decode's handler.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn transition_delivers_tier_to_decode() {
    let (hub, decode, cache) = hub_and_decode().await;

    let breaker = Arc::new(CircuitBreaker::new(BreakerConfig::default(), 100));
    let broadcaster = TierBroadcaster::new(Arc::clone(&breaker), None);
    broadcaster.set_velo(Arc::clone(&hub));
    let provider: Arc<dyn DecodeSetProvider> =
        Arc::new(StaticDecodeSet(vec![decode.instance_id()]));
    broadcaster.set_decode_provider(Arc::downgrade(&provider));

    // Drive the breaker to HOT (the tick task would do this; we drive evaluate
    // directly), then push the transition.
    let new_tier = breaker.evaluate(0.0, 0).expect("breaker trips to HOT");
    assert_eq!(new_tier, BreakerTier::Hot);
    broadcaster.on_transition(BreakerTier::Calm, new_tier);

    assert!(
        wait_until(
            || cache.tier() == tier_u8(BreakerTier::Hot),
            Duration::from_secs(3)
        )
        .await,
        "decode handler should have received the HOT tier push"
    );
    assert_eq!(cache.epoch(), breaker.epoch(), "epoch propagated");

    let _ = decode; // keep alive
}

/// push-on-register seeds a fresh decode with the CURRENT tier (even though no
/// transition happens after it registers).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn push_on_register_seeds_fresh_decode() {
    let (hub, decode, cache) = hub_and_decode().await;

    // Breaker already HOT before the decode "registers".
    let breaker = Arc::new(CircuitBreaker::new(BreakerConfig::default(), 100));
    assert_eq!(breaker.evaluate(0.0, 0), Some(BreakerTier::Hot));
    let broadcaster = TierBroadcaster::new(Arc::clone(&breaker), None);
    broadcaster.set_velo(Arc::clone(&hub));
    // Provider not even needed for push_current_to (it targets one id).
    broadcaster.push_current_to(decode.instance_id());

    assert!(
        wait_until(
            || cache.tier() == tier_u8(BreakerTier::Hot),
            Duration::from_secs(3)
        )
        .await,
        "freshly-registered decode should be seeded with the current HOT tier"
    );
}

/// A PREFILL instance is NEVER pushed to: the decode-set provider returns only
/// decode ids, so on_transition reaches the decode but not the prefill velo.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn prefill_instance_is_not_pushed() {
    let hub = new_velo().await;
    let decode = new_velo().await;
    let prefill = new_velo().await;
    for p in [&decode, &prefill] {
        hub.messenger().register_peer(p.peer_info()).unwrap();
        p.messenger().register_peer(hub.peer_info()).unwrap();
    }
    let decode_cache = Arc::new(TestTierCache::default());
    let prefill_cache = Arc::new(TestTierCache::default());
    install_tier_handler(&decode, Arc::clone(&decode_cache));
    install_tier_handler(&prefill, Arc::clone(&prefill_cache));
    tokio::time::sleep(Duration::from_millis(200)).await;

    let breaker = Arc::new(CircuitBreaker::new(BreakerConfig::default(), 100));
    let broadcaster = TierBroadcaster::new(Arc::clone(&breaker), None);
    broadcaster.set_velo(Arc::clone(&hub));
    // The provider lists ONLY the decode — prefill is deliberately excluded
    // (the CD manager's DecodeSetProvider impl returns inner.decode only).
    let provider: Arc<dyn DecodeSetProvider> =
        Arc::new(StaticDecodeSet(vec![decode.instance_id()]));
    broadcaster.set_decode_provider(Arc::downgrade(&provider));

    assert_eq!(breaker.evaluate(0.0, 0), Some(BreakerTier::Hot));
    broadcaster.on_transition(BreakerTier::Calm, BreakerTier::Hot);

    // Decode receives it.
    assert!(
        wait_until(
            || decode_cache.tier() == tier_u8(BreakerTier::Hot),
            Duration::from_secs(3)
        )
        .await,
        "decode must receive the tier push"
    );
    // Give any (erroneous) prefill push ample time to land, then assert it did
    // NOT: the prefill cache stays at the default Calm @ epoch 0.
    tokio::time::sleep(Duration::from_millis(500)).await;
    assert_eq!(
        prefill_cache.tier(),
        tier_u8(BreakerTier::Calm),
        "PREFILL instance must NOT be pushed to"
    );
    assert_eq!(prefill_cache.epoch(), 0, "prefill cache untouched");

    let _ = prefill;
}

/// An older-epoch push must not regress a newer cached tier (absolute state /
/// latest-by-recency). Exercised directly on the cache semantics that the
/// production `DecodeTierCache` shares.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn lower_epoch_push_is_ignored() {
    let (hub, decode, cache) = hub_and_decode().await;

    // First push HOT @ epoch 50.
    let signal_hot = TierSignal {
        tier: BreakerTier::Hot,
        epoch: 50,
    };
    let _: TierSignalAck = hub
        .typed_unary(TIER_SIGNAL_HANDLER)
        .unwrap()
        .payload(&signal_hot)
        .unwrap()
        .instance(decode.instance_id())
        .send()
        .await
        .unwrap();
    assert!(
        wait_until(
            || cache.epoch() == 50 && cache.tier() == tier_u8(BreakerTier::Hot),
            Duration::from_secs(3)
        )
        .await
    );

    // Now a STALE push CALM @ epoch 10 must be ignored.
    let signal_stale = TierSignal {
        tier: BreakerTier::Calm,
        epoch: 10,
    };
    let ack: TierSignalAck = hub
        .typed_unary(TIER_SIGNAL_HANDLER)
        .unwrap()
        .payload(&signal_stale)
        .unwrap()
        .instance(decode.instance_id())
        .send()
        .await
        .unwrap();
    assert!(!ack.ok, "stale push must be rejected");
    assert_eq!(cache.tier(), tier_u8(BreakerTier::Hot), "tier not regressed");
    assert_eq!(cache.epoch(), 50, "epoch not regressed");
}
