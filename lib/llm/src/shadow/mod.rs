// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Shadow taps mirror the live request stream, and optionally the responses,
//! of a frontend to event-plane topics, so that a shadow deployment can be
//! driven with production traffic.
//!
//! The tap is a pipeline operator linked just above migration. There it sees
//! every request as a `PreprocessedRequest`, whichever public API or chat
//! processor produced it, and sees each client request once: migration
//! replays a request below the tap when a worker fails.
//!
//! With `DYN_SHADOW_TAP_CONFIG` unset the operator is not linked, so a
//! frontend without taps pays nothing.

mod config;
mod envelope;
mod filter;
mod publisher;
mod tap;

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::{Arc, LazyLock, Mutex};

use anyhow::{Context, Result};
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::config::environment_names::llm::shadow::DYN_SHADOW_TAP_CONFIG;
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::transports::event_plane::EventPublisher;
use tokio_util::sync::CancellationToken;

pub use config::{Capture, ResponseOptions, ShadowConfig, TapSpec};
pub use envelope::{
    ENVELOPE_SCHEMA_VERSION, ShadowChoice, ShadowEnvelope, ShadowOrigin, ShadowOutcome,
    ShadowResponse,
};
pub(crate) use tap::ShadowTap;
use tap::{TapCounters, TapQueue};

pub(crate) type ShadowTaps = Arc<[Arc<TapQueue>]>;

/// Taps belong to the `DistributedRuntime` that created them: their
/// publishers use its transport and stop on its shutdown token. One process
/// can hold several runtimes, at once or one after another, so a process-wide
/// tap set would hand a second runtime queues that nothing drains.
#[derive(Default)]
struct Registry {
    runtimes: Mutex<HashMap<u64, (ShadowTaps, CancellationToken)>>,
    /// Held across `get_or_init`. Several inputs can share one runtime and
    /// start at the same time; a second start would register the same metrics
    /// again and fail, or open a second set of publishers.
    starting: tokio::sync::Mutex<()>,
}

impl Registry {
    fn live(&self, runtime_id: u64) -> Option<ShadowTaps> {
        let runtimes = self.runtimes.lock().expect("shadow tap registry poisoned");
        runtimes
            .get(&runtime_id)
            .filter(|(_, shutdown)| !shutdown.is_cancelled())
            .map(|(taps, _)| Arc::clone(taps))
    }

    async fn get_or_init<F, Fut>(
        &self,
        runtime_id: u64,
        shutdown: CancellationToken,
        start: F,
    ) -> Result<ShadowTaps>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<ShadowTaps>>,
    {
        let _starting = self.starting.lock().await;
        if let Some(taps) = self.live(runtime_id) {
            return Ok(taps);
        }
        let taps = start().await?;
        self.insert(runtime_id, Arc::clone(&taps), shutdown);
        Ok(taps)
    }

    fn insert(&self, runtime_id: u64, taps: ShadowTaps, shutdown: CancellationToken) {
        let mut runtimes = self.runtimes.lock().expect("shadow tap registry poisoned");
        runtimes.retain(|_, (_, shutdown)| !shutdown.is_cancelled());
        runtimes.insert(runtime_id, (taps, shutdown));
    }
}

static REGISTRY: LazyLock<Registry> = LazyLock::new(Registry::default);

/// Read the tap config and start one publisher task per tap. A frontend whose
/// config is wrong must not start: a typo in a filter name would otherwise
/// mirror more data than the operator intended.
pub(crate) async fn init_from_env(drt: &DistributedRuntime) -> Result<()> {
    let Some(path) = std::env::var_os(DYN_SHADOW_TAP_CONFIG).filter(|path| !path.is_empty()) else {
        return Ok(());
    };
    let config = ShadowConfig::from_path(&PathBuf::from(path))?;
    if config.taps.is_empty() {
        return Ok(());
    }
    REGISTRY
        .get_or_init(drt.connection_id(), drt.child_token(), || {
            start_taps(drt, config)
        })
        .await?;
    Ok(())
}

/// Every step that can fail for a transient reason runs before anything that
/// outlives a failure. A start that fails must leave no metric collectors and
/// no publisher tasks, or a second start on the same runtime could not succeed.
async fn start_taps(drt: &DistributedRuntime, config: ShadowConfig) -> Result<ShadowTaps> {
    let namespace = drt.namespace(config.namespace.clone())?;

    let mut publishers = Vec::with_capacity(config.taps.len());
    for spec in config.taps {
        // `capacity` bounds what the frontend holds for a shadow that has
        // stopped reading. The ZMQ socket queues behind the tap queue, and its
        // default mark of 100,000 messages would let long prompts reach
        // gigabytes, so it gets the same bound.
        let send_hwm = i32::try_from(spec.capacity).unwrap_or(i32::MAX);
        let publisher = EventPublisher::for_namespace_with_zmq_send_hwm(
            &namespace,
            spec.topic.clone(),
            send_hwm,
        )
        .await
        .with_context(|| format!("shadow tap `{}`: creating publisher", spec.name))?;
        publishers.push((spec, publisher));
    }

    let metrics = namespace.metrics();
    let counter = |name: &str, help: &str| metrics.create_intcountervec(name, help, &["tap"], &[]);
    let queued = counter(
        "shadow_tap_queued_total",
        "Records a shadow tap queued for publishing",
    )?;
    let dropped = counter(
        "shadow_tap_dropped_total",
        "Records a shadow tap dropped because its queue was full",
    )?;
    let publish_errors = counter(
        "shadow_tap_publish_errors_total",
        "Records a shadow tap failed to publish",
    )?;

    let mut taps = Vec::with_capacity(publishers.len());
    for (spec, publisher) in publishers {
        let name = spec.name.to_string();
        tracing::info!(
            tap = name,
            namespace = config.namespace,
            topic = spec.topic,
            capture = ?spec.capture,
            filters = ?spec.filters.names,
            capacity = spec.capacity,
            "shadow tap enabled"
        );
        let counters = TapCounters {
            queued: queued.with_label_values(&[&name]),
            dropped: dropped.with_label_values(&[&name]),
        };
        let (queue, receiver) = TapQueue::new(spec, counters);
        drt.runtime().secondary().spawn(publisher::run(
            name.clone(),
            receiver,
            publisher,
            publish_errors.with_label_values(&[&name]),
            drt.child_token(),
        ));
        taps.push(queue);
    }

    Ok(taps.into())
}

/// The taps of a runtime, or `None` when it has none and nothing is linked.
pub(crate) fn taps_for(drt: &DistributedRuntime) -> Option<ShadowTaps> {
    REGISTRY.live(drt.connection_id())
}

pub(crate) fn tap_for(taps: &ShadowTaps, origin: ShadowOrigin) -> Arc<ShadowTap> {
    ShadowTap::new(taps.to_vec(), origin)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    fn taps() -> ShadowTaps {
        let config =
            ShadowConfig::from_yaml("schema_version: 1\ntaps:\n  - {name: t, capture: request}\n")
                .unwrap();
        config
            .taps
            .into_iter()
            .map(|spec| TapQueue::new(spec, tap::tests::counters()).0)
            .collect()
    }

    #[test]
    fn taps_belong_to_one_runtime_and_end_with_it() {
        let registry = Registry::default();
        let first = CancellationToken::new();
        registry.insert(1, taps(), first.clone());

        assert!(registry.live(1).is_some());
        assert!(
            registry.live(2).is_none(),
            "another runtime has no taps yet"
        );

        first.cancel();
        assert!(
            registry.live(1).is_none(),
            "a stopped runtime has no live taps"
        );

        let second = taps();
        registry.insert(2, Arc::clone(&second), CancellationToken::new());
        assert!(Arc::ptr_eq(&registry.live(2).unwrap(), &second));
        assert_eq!(
            registry.runtimes.lock().unwrap().len(),
            1,
            "stopped runtimes are pruned"
        );
    }

    #[tokio::test]
    async fn concurrent_starts_for_one_runtime_run_once() {
        let registry = Registry::default();
        let starts = AtomicUsize::new(0);
        let start = || async {
            starts.fetch_add(1, Ordering::SeqCst);
            tokio::task::yield_now().await;
            Ok(taps())
        };
        let shutdown = CancellationToken::new();

        let (first, second) = tokio::join!(
            registry.get_or_init(7, shutdown.clone(), start),
            registry.get_or_init(7, shutdown.clone(), start),
        );
        assert_eq!(starts.load(Ordering::SeqCst), 1);
        assert!(Arc::ptr_eq(&first.unwrap(), &second.unwrap()));
    }

    #[tokio::test]
    async fn a_failed_start_registers_nothing() {
        let registry = Registry::default();
        let failed = registry
            .get_or_init(7, CancellationToken::new(), || async {
                anyhow::bail!("no transport")
            })
            .await;
        assert!(failed.is_err());
        assert!(registry.live(7).is_none());
    }
}
