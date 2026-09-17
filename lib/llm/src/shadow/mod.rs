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

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result};
use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::config::environment_names::llm::shadow::DYN_SHADOW_TAP_CONFIG;
use dynamo_runtime::metrics::MetricsHierarchy;
use dynamo_runtime::transports::event_plane::EventPublisher;

pub use config::{Capture, ResponseOptions, ShadowConfig, TapSpec};
pub use envelope::{
    ENVELOPE_SCHEMA_VERSION, ShadowEnvelope, ShadowOrigin, ShadowOutcome, ShadowResponse,
};
pub(crate) use tap::ShadowTap;
use tap::{TapCounters, TapQueue};

static TAPS: OnceLock<Vec<Arc<TapQueue>>> = OnceLock::new();

/// Read the tap config and start one publisher task per tap. A frontend whose
/// config is wrong must not start: a typo in a filter name would otherwise
/// mirror more data than the operator intended.
pub(crate) async fn init_from_env(drt: &DistributedRuntime) -> Result<()> {
    let Some(path) = std::env::var_os(DYN_SHADOW_TAP_CONFIG).filter(|path| !path.is_empty()) else {
        return Ok(());
    };
    // Several inputs can share one process; the taps belong to the process.
    if TAPS.get().is_some() {
        return Ok(());
    }
    let config = ShadowConfig::from_path(&PathBuf::from(path))?;
    if config.taps.is_empty() {
        return Ok(());
    }

    let namespace = drt.namespace(config.namespace.clone())?;
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

    let mut taps = Vec::with_capacity(config.taps.len());
    for spec in config.taps {
        let name = spec.name.to_string();
        let publisher = EventPublisher::for_namespace(&namespace, spec.topic.clone())
            .await
            .with_context(|| format!("shadow tap `{name}`: creating publisher"))?;
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

    TAPS.set(taps)
        .map_err(|_| anyhow::anyhow!("shadow taps are already initialized"))
}

/// The tap operator for a pipeline being built, or `None` when no taps are
/// configured and nothing is linked.
pub(crate) fn tap_for(origin: ShadowOrigin) -> Option<Arc<ShadowTap>> {
    TAPS.get()
        .filter(|taps| !taps.is_empty())
        .map(|taps| ShadowTap::new(taps.clone(), origin))
}
