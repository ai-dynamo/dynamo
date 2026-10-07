// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! EXPERIMENT ONLY: static direct-ZMQ KV sources for the e2e indexer-contention campaign.
//!
//! [`STATIC_KV_SOURCES_ENV`] lists KV event publishers that the direct-ZMQ ingress subscribes to
//! without discovery and whose events it accepts without the serving-membership check. Entries
//! are separated by commas or whitespace:
//!
//! - `<worker_id>@<zmq endpoint>`: one source;
//! - `<first_worker_id>+<count>@tcp://<host>:<first_port>`: `count` sources, worker
//!   `first_worker_id + k` publishing on port `first_port + k`.
//!
//! A value starting with `@` names a file of entries (`#` starts a comment). A static source's
//! publisher ID equals its worker ID and its DP rank is 0. Static sources are live-only (no
//! recovery target) and exist only in the ingress's private membership view, so they never
//! become serving workers and are never routable. A static worker or publisher ID that collides
//! with a discovered source is skipped and logged.

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use anyhow::{Context, Result, bail, ensure};
use dynamo_kv_router::protocols::WorkerWithDpRank;
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use crate::discovery::{
    KvEventSource, KvSourceMembershipView, KvSourceMembershipWatch, KvSourceStatus,
};

pub(crate) const STATIC_KV_SOURCES_ENV: &str = "DYN_EXPERIMENT_STATIC_KV_SOURCES";

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct StaticKvSource {
    pub(crate) worker_id: u64,
    pub(crate) endpoint: String,
}

impl StaticKvSource {
    pub(crate) fn publisher_id(&self) -> u64 {
        self.worker_id
    }
}

/// Parse [`STATIC_KV_SOURCES_ENV`]; unset or empty yields no sources.
pub(crate) fn static_kv_sources_from_env() -> Result<Arc<[StaticKvSource]>> {
    let Some(value) = std::env::var_os(STATIC_KV_SOURCES_ENV) else {
        return Ok(Arc::from([]));
    };
    let value = value
        .into_string()
        .map_err(|_| anyhow::anyhow!("{STATIC_KV_SOURCES_ENV} must be valid UTF-8"))?;
    let spec = match value.trim().strip_prefix('@') {
        Some(path) => std::fs::read_to_string(path)
            .with_context(|| format!("reading {STATIC_KV_SOURCES_ENV} file {path}"))?,
        None => value,
    };
    let sources =
        parse_static_sources(&spec).with_context(|| format!("parsing {STATIC_KV_SOURCES_ENV}"))?;
    if !sources.is_empty() {
        tracing::warn!(
            count = sources.len(),
            env = STATIC_KV_SOURCES_ENV,
            "EXPERIMENT: accepting static direct-ZMQ KV sources without serving membership"
        );
    }
    Ok(sources.into())
}

fn parse_static_sources(spec: &str) -> Result<Vec<StaticKvSource>> {
    let mut sources = Vec::new();
    let mut workers = HashSet::new();
    for line in spec.lines() {
        let line = line.split('#').next().unwrap_or_default();
        for entry in line
            .split(|c: char| c == ',' || c.is_whitespace())
            .filter(|entry| !entry.is_empty())
        {
            for source in parse_entry(entry)? {
                ensure!(
                    workers.insert(source.worker_id),
                    "worker {} is listed twice",
                    source.worker_id
                );
                sources.push(source);
            }
        }
    }
    Ok(sources)
}

fn parse_entry(entry: &str) -> Result<Vec<StaticKvSource>> {
    let (ids, endpoint) = entry
        .split_once('@')
        .with_context(|| format!("entry {entry:?} is not <worker_id>@<endpoint>"))?;
    ensure!(
        !endpoint.is_empty(),
        "entry {entry:?} has an empty endpoint"
    );
    let Some((first, count)) = ids.split_once('+') else {
        let worker_id = parse_u64(ids, entry)?;
        return Ok(vec![StaticKvSource {
            worker_id,
            endpoint: endpoint.to_string(),
        }]);
    };
    let first = parse_u64(first, entry)?;
    let count = parse_u64(count, entry)?;
    let (prefix, port) = endpoint
        .rsplit_once(':')
        .filter(|(prefix, _)| prefix.starts_with("tcp://"))
        .with_context(|| format!("range entry {entry:?} needs a tcp://<host>:<port> endpoint"))?;
    let first_port = port
        .parse::<u16>()
        .with_context(|| format!("range entry {entry:?} has an invalid port"))?;
    ensure!(count > 0, "range entry {entry:?} has a zero count");
    if u64::from(first_port) + count - 1 > u64::from(u16::MAX) {
        bail!("range entry {entry:?} runs past port 65535");
    }
    (0..count)
        .map(|offset| {
            let worker_id = first
                .checked_add(offset)
                .with_context(|| format!("range entry {entry:?} overflows the worker ID"))?;
            Ok(StaticKvSource {
                worker_id,
                endpoint: format!("{prefix}:{}", u64::from(first_port) + offset),
            })
        })
        .collect()
}

fn parse_u64(value: &str, entry: &str) -> Result<u64> {
    let value = value.trim();
    let parsed = match value.strip_prefix("0x") {
        Some(hex) => u64::from_str_radix(hex, 16),
        None => value.parse::<u64>(),
    };
    parsed.with_context(|| format!("entry {entry:?} has an invalid number {value:?}"))
}

/// Derive a membership channel that adds the static sources as active live-only sources.
pub(crate) fn with_static_sources(
    membership: KvSourceMembershipWatch,
    sources: Arc<[StaticKvSource]>,
    cancel: CancellationToken,
) -> KvSourceMembershipWatch {
    let mut source = membership.clone();
    let mut reported = HashSet::new();
    let initial = augment_view(source.borrow().clone(), &sources, &mut reported);
    let (tx, rx) = watch::channel(initial);
    tokio::spawn(async move {
        loop {
            tokio::select! {
                biased;
                _ = cancel.cancelled() => break,
                changed = source.changed() => if changed.is_err() { break; },
            }
            let view = augment_view(source.borrow_and_update().clone(), &sources, &mut reported);
            tx.send_replace(view);
        }
    });
    membership.with_receiver(rx)
}

fn augment_view(
    mut view: KvSourceMembershipView,
    sources: &[StaticKvSource],
    reported: &mut HashSet<u64>,
) -> KvSourceMembershipView {
    let Some(kv_state_endpoint) = view.resolved_kv_state_endpoint().cloned() else {
        return view;
    };
    let discovered_workers: HashSet<u64> =
        view.sources.keys().map(|worker| worker.worker_id).collect();
    let discovered_publishers: HashSet<u64> = view
        .sources
        .values()
        .filter_map(|status| status.active_source().map(|source| source.publisher_id))
        .collect();
    let mut added = HashMap::with_capacity(sources.len());
    for source in sources {
        if discovered_workers.contains(&source.worker_id)
            || discovered_publishers.contains(&source.publisher_id())
        {
            if reported.insert(source.worker_id) {
                tracing::error!(
                    worker_id = source.worker_id,
                    "EXPERIMENT: static KV source collides with a discovered source; skipping it"
                );
            }
            continue;
        }
        let worker = WorkerWithDpRank::new(source.worker_id, 0);
        added.insert(
            worker,
            KvSourceStatus::ActiveLiveOnly(KvEventSource {
                kv_state_endpoint: kv_state_endpoint.clone(),
                worker,
                publisher_id: source.publisher_id(),
                recovery_target: None,
            }),
        );
    }
    for (worker, status) in added {
        view.kv_event_publishing_enabled
            .insert(worker.worker_id, Some(true));
        view.recovery_expected.insert(worker, false);
        view.sources.insert(worker, status);
    }
    view
}

#[cfg(test)]
mod tests {
    use dynamo_runtime::protocols::EndpointId;

    use super::*;
    use crate::discovery::KvStateEndpointResolution;

    fn endpoint() -> EndpointId {
        EndpointId {
            namespace: "ns".to_string(),
            component: "backend".to_string(),
            name: "generate".to_string(),
        }
    }

    fn view_with(worker_id: u64, publisher_id: u64) -> KvSourceMembershipView {
        let worker = WorkerWithDpRank::new(worker_id, 0);
        KvSourceMembershipView {
            serving_endpoint: endpoint(),
            endpoint_resolution: KvStateEndpointResolution::Resolved(endpoint()),
            sources: HashMap::from([(
                worker,
                KvSourceStatus::ActiveLiveOnly(KvEventSource {
                    kv_state_endpoint: endpoint(),
                    worker,
                    publisher_id,
                    recovery_target: None,
                }),
            )]),
            kv_event_publishing_enabled: HashMap::from([(worker_id, Some(true))]),
            kv_event_source_mode: HashMap::new(),
            recovery_expected: HashMap::from([(worker, false)]),
        }
    }

    #[test]
    fn parses_singles_ranges_and_comments() {
        let sources = parse_static_sources(
            "7@tcp://10.0.0.1:5000, 0x10+3@tcp://host-a:6000\n# comment\n100@tcp://h:1 # tail",
        )
        .unwrap();
        let ids: Vec<_> = sources.iter().map(|source| source.worker_id).collect();
        assert_eq!(ids, vec![7, 16, 17, 18, 100]);
        assert_eq!(sources[1].endpoint, "tcp://host-a:6000");
        assert_eq!(sources[3].endpoint, "tcp://host-a:6002");
        assert_eq!(sources[3].publisher_id(), 18);
    }

    #[test]
    fn rejects_duplicates_and_bad_ranges() {
        assert!(parse_static_sources("1@tcp://a:1 1@tcp://b:2").is_err());
        assert!(parse_static_sources("1+2@tcp://a:65535").is_err());
        assert!(parse_static_sources("1+0@tcp://a:1").is_err());
        assert!(parse_static_sources("1+2@ipc://sock").is_err());
        assert!(parse_static_sources("tcp://a:1").is_err());
    }

    #[test]
    fn augment_adds_live_only_sources_and_skips_collisions() {
        let sources = parse_static_sources("5+3@tcp://h:7000").unwrap();
        // Worker 5 is discovered; publisher 7 belongs to a discovered source.
        let mut view = view_with(5, 99);
        let other = WorkerWithDpRank::new(42, 0);
        view.sources.insert(
            other,
            KvSourceStatus::ActiveLiveOnly(KvEventSource {
                kv_state_endpoint: endpoint(),
                worker: other,
                publisher_id: 7,
                recovery_target: None,
            }),
        );
        let mut reported = HashSet::new();
        let augmented = augment_view(view, &sources, &mut reported);

        assert_eq!(reported, HashSet::from([5, 7]));
        let phantom = WorkerWithDpRank::new(6, 0);
        let Some(KvSourceStatus::ActiveLiveOnly(source)) = augmented.sources.get(&phantom) else {
            panic!("static worker 6 should be an active live-only source");
        };
        assert_eq!(source.publisher_id, 6);
        assert_eq!(source.kv_state_endpoint, endpoint());
        assert_eq!(augmented.kv_event_publishing_enabled(6), Some(true));
        assert_eq!(augmented.recovery_expected(&phantom), Some(false));
        // The discovered source for worker 5 is untouched.
        let Some(KvSourceStatus::ActiveLiveOnly(source)) =
            augmented.sources.get(&WorkerWithDpRank::new(5, 0))
        else {
            panic!("discovered worker 5 should remain");
        };
        assert_eq!(source.publisher_id, 99);
        assert_eq!(augmented.sources.len(), 3);
    }

    #[test]
    fn ambiguous_endpoint_adds_nothing() {
        let sources = parse_static_sources("5@tcp://h:7000").unwrap();
        let mut view = view_with(1, 1);
        view.endpoint_resolution = KvStateEndpointResolution::Ambiguous {
            endpoints: vec![endpoint()],
        };
        let augmented = augment_view(view.clone(), &sources, &mut HashSet::new());
        assert_eq!(augmented, view);
    }
}
