// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Global Discovery HTTP service for the multi-cluster Global Router POC.
//!
//! Runs in the hub cluster, watches the EndpointSlices that Karmada imports for
//! multi-cluster services, and serves the KV DC Relay pools it finds as JSON on
//! `GET /pools`. `GET /healthz` turns ready once the first EndpointSlice list
//! has synced.
//!
//! Naming conventions this POC relies on:
//! - Each pool's Relay is exported as a Service named `<service-prefix><dgd>`
//!   (default prefix `relay-`) in the DGD's namespace.
//! - Karmada imports it into the hub as `derived-<service>` and names each
//!   imported slice `imported-<cluster>-<source slice>`, where the source slice
//!   name starts with `<service>-`. Imported slices carry no source-cluster
//!   label, so the cluster (the pool's site ID) is read from the slice name.

use std::collections::{BTreeMap, HashMap};
use std::net::{IpAddr, SocketAddr};
use std::sync::{Arc, PoisonError, RwLock};

use anyhow::{Context, Result};
use axum::extract::State;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use clap::Parser;
use futures::StreamExt;
use k8s_openapi::api::discovery::v1::EndpointSlice;
use kube::runtime::{WatchStreamExt, watcher};
use kube::{Api, Client, ResourceExt};
use serde::Serialize;
use tokio::net::TcpListener;
use tokio_util::sync::CancellationToken;

/// Karmada's EndpointSlice controller marks every slice it imports with this label.
const KARMADA_SLICE_SELECTOR: &str =
    "endpointslice.kubernetes.io/managed-by=endpointslice-controller.karmada.io";
const SERVICE_NAME_LABEL: &str = "kubernetes.io/service-name";
const DERIVED_PREFIX: &str = "derived-";
const IMPORTED_PREFIX: &str = "imported-";

#[derive(Parser)]
#[command(about = "Serve Relay pools discovered from Karmada-imported EndpointSlices")]
struct Args {
    /// Address to serve HTTP on.
    #[arg(long, default_value = "0.0.0.0:8080")]
    listen: SocketAddr,
    /// Watch one namespace instead of all namespaces.
    #[arg(long)]
    namespace: Option<String>,
    /// Exported Relay Services are named `<prefix><dgd>`.
    #[arg(long, default_value = "relay-")]
    service_prefix: String,
    /// Name of the Relay port that carries the WAN gRPC stream.
    #[arg(long, default_value = "wan-grpc")]
    relay_port_name: String,
    /// Name of the Relay port that carries the stats gRPC stream.
    #[arg(long, default_value = "stats-grpc")]
    stats_port_name: String,
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt::init();
    let args = Args::parse();
    let client = Client::try_default()
        .await
        .context("create Kubernetes client")?;
    let api: Api<EndpointSlice> = match &args.namespace {
        Some(namespace) => Api::namespaced(client, namespace),
        None => Api::all(client),
    };
    let naming = Naming {
        service_prefix: args.service_prefix,
        relay_port_name: args.relay_port_name,
        stats_port_name: args.stats_port_name,
    };
    let index = SharedIndex::default();
    let cancel = CancellationToken::new();
    let watch = tokio::spawn(watch_slices(api, naming, index.clone(), cancel.clone()));

    let app = Router::new()
        .route("/pools", get(pools))
        .route("/healthz", get(healthz))
        .with_state(index);
    let listener = TcpListener::bind(args.listen)
        .await
        .with_context(|| format!("bind Global Discovery listener at {}", args.listen))?;
    tracing::info!(address = %args.listen, "Global Discovery listening");

    let signal_cancel = cancel.clone();
    tokio::spawn(async move {
        if let Err(error) = shutdown_signal().await {
            tracing::warn!(%error, "shutdown signal handler failed");
        }
        signal_cancel.cancel();
    });
    axum::serve(listener, app)
        .with_graceful_shutdown(cancel.clone().cancelled_owned())
        .await
        .context("serve Global Discovery HTTP")?;
    cancel.cancel();
    watch.await.context("join EndpointSlice watcher")
}

type SharedIndex = Arc<RwLock<Index>>;

/// Single consumer of the watch stream, so index updates are applied in order.
async fn watch_slices(
    api: Api<EndpointSlice>,
    naming: Naming,
    index: SharedIndex,
    cancel: CancellationToken,
) {
    let config = watcher::Config::default().labels(KARMADA_SLICE_SELECTOR);
    let mut stream = std::pin::pin!(watcher(api, config).default_backoff());
    loop {
        let event = tokio::select! {
            biased;
            () = cancel.cancelled() => return,
            event = stream.next() => event,
        };
        match event {
            Some(Ok(event)) => index
                .write()
                .unwrap_or_else(PoisonError::into_inner)
                .apply(&naming, event),
            // The stream keeps the last good view and retries with backoff.
            Some(Err(error)) => tracing::warn!(%error, "EndpointSlice watch error"),
            None => {
                tracing::error!("EndpointSlice watch stream ended");
                index.write().unwrap_or_else(PoisonError::into_inner).synced = false;
                return;
            }
        }
    }
}

async fn pools(State(index): State<SharedIndex>) -> Response {
    let index = index.read().unwrap_or_else(PoisonError::into_inner);
    if !index.synced {
        return (StatusCode::SERVICE_UNAVAILABLE, "EndpointSlices not synced").into_response();
    }
    Json(PoolsResponse {
        pools: index.pools(),
    })
    .into_response()
}

async fn healthz(State(index): State<SharedIndex>) -> StatusCode {
    if index.read().unwrap_or_else(PoisonError::into_inner).synced {
        StatusCode::OK
    } else {
        StatusCode::SERVICE_UNAVAILABLE
    }
}

async fn shutdown_signal() -> Result<()> {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{SignalKind, signal};

        let mut terminate = signal(SignalKind::terminate()).context("register SIGTERM handler")?;
        tokio::select! {
            result = tokio::signal::ctrl_c() => result.context("wait for Ctrl-C")?,
            _ = terminate.recv() => {}
        }
    }
    #[cfg(not(unix))]
    tokio::signal::ctrl_c().await.context("wait for Ctrl-C")?;
    Ok(())
}

/// How exported Relay Services and their ports are named.
struct Naming {
    service_prefix: String,
    relay_port_name: String,
    stats_port_name: String,
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct PoolRef {
    site_id: String,
    namespace: String,
    dgd_name: String,
}

/// What one imported EndpointSlice says about its pool.
#[derive(Debug, PartialEq)]
struct SliceEntry {
    pool: PoolRef,
    relay_port: Option<u16>,
    stats_port: Option<u16>,
    endpoints: Vec<(IpAddr, bool)>,
}

impl Naming {
    /// Returns `None` for slices that are not an imported Relay Service.
    fn slice_entry(&self, slice: &EndpointSlice) -> Option<SliceEntry> {
        let namespace = slice.metadata.namespace.as_deref()?;
        let name = slice.metadata.name.as_deref()?;
        let service = slice
            .labels()
            .get(SERVICE_NAME_LABEL)?
            .strip_prefix(DERIVED_PREFIX)?;
        let dgd_name = service
            .strip_prefix(self.service_prefix.as_str())
            .filter(|dgd| !dgd.is_empty())?;
        let site_id = source_cluster(name, service)?;
        let port = |wanted: &str| {
            slice
                .ports
                .iter()
                .flatten()
                .find(|port| port.name.as_deref() == Some(wanted))
                .and_then(|port| port.port)
                .and_then(|port| u16::try_from(port).ok())
        };
        let endpoints = slice
            .endpoints
            .iter()
            .flat_map(|endpoint| {
                // EndpointSlice API: a missing `ready` condition means ready.
                let ready = endpoint
                    .conditions
                    .as_ref()
                    .and_then(|conditions| conditions.ready)
                    .unwrap_or(true);
                endpoint
                    .addresses
                    .iter()
                    .filter_map(move |address| address.parse().ok().map(|ip| (ip, ready)))
            })
            .collect();
        Some(SliceEntry {
            pool: PoolRef {
                site_id: site_id.to_owned(),
                namespace: namespace.to_owned(),
                dgd_name: dgd_name.to_owned(),
            },
            relay_port: port(&self.relay_port_name),
            stats_port: port(&self.stats_port_name),
            endpoints,
        })
    }
}

/// Reads `<cluster>` from `imported-<cluster>-<service>-<suffix>`.
fn source_cluster<'a>(slice_name: &'a str, service: &str) -> Option<&'a str> {
    let rest = slice_name.strip_prefix(IMPORTED_PREFIX)?;
    let end = rest.rfind(&format!("-{service}-"))?;
    (end > 0).then(|| &rest[..end])
}

/// Imported Relay slices keyed by `namespace/name`, kept current one event at a time.
#[derive(Default)]
struct Index {
    synced: bool,
    slices: HashMap<String, SliceEntry>,
    /// Slices listed since the last `Init`; replaces `slices` at `InitDone`.
    relist: Option<HashMap<String, SliceEntry>>,
}

impl Index {
    fn apply(&mut self, naming: &Naming, event: watcher::Event<EndpointSlice>) {
        match event {
            watcher::Event::Init => self.relist = Some(HashMap::new()),
            watcher::Event::InitApply(slice) => {
                if let Some(relist) = self.relist.as_mut() {
                    upsert(relist, naming, &slice);
                }
            }
            watcher::Event::InitDone => {
                if let Some(relist) = self.relist.take() {
                    self.slices = relist;
                }
                self.synced = true;
            }
            watcher::Event::Apply(slice) => upsert(&mut self.slices, naming, &slice),
            watcher::Event::Delete(slice) => {
                self.slices.remove(&slice_key(&slice));
            }
        }
    }

    fn pools(&self) -> Vec<PoolView> {
        let mut grouped: BTreeMap<&PoolRef, Vec<&SliceEntry>> = BTreeMap::new();
        for entry in self.slices.values() {
            grouped.entry(&entry.pool).or_default().push(entry);
        }
        grouped
            .into_iter()
            .map(|(pool, entries)| {
                let mut endpoints: Vec<(IpAddr, bool)> = entries
                    .iter()
                    .flat_map(|entry| entry.endpoints.iter().copied())
                    .collect();
                endpoints.sort_unstable();
                endpoints.dedup();
                let first_ready = endpoints
                    .iter()
                    .find(|(_, ready)| *ready)
                    .map(|(ip, _)| *ip);
                let url = |port: Option<u16>| {
                    Some(format!("http://{}", SocketAddr::new(first_ready?, port?)))
                };
                let relay_grpc_url = url(entries.iter().find_map(|entry| entry.relay_port));
                let stats_grpc_url = url(entries.iter().find_map(|entry| entry.stats_port));
                PoolView {
                    site_id: pool.site_id.clone(),
                    namespace: pool.namespace.clone(),
                    dgd_name: pool.dgd_name.clone(),
                    ready: relay_grpc_url.is_some() && stats_grpc_url.is_some(),
                    relay_grpc_url,
                    stats_grpc_url,
                    endpoints: endpoints
                        .into_iter()
                        .map(|(address, ready)| EndpointView { address, ready })
                        .collect(),
                }
            })
            .collect()
    }
}

fn upsert(slices: &mut HashMap<String, SliceEntry>, naming: &Naming, slice: &EndpointSlice) {
    let key = slice_key(slice);
    match naming.slice_entry(slice) {
        Some(entry) => {
            slices.insert(key, entry);
        }
        None => {
            slices.remove(&key);
        }
    }
}

fn slice_key(slice: &EndpointSlice) -> String {
    format!(
        "{}/{}",
        slice.metadata.namespace.as_deref().unwrap_or_default(),
        slice.metadata.name.as_deref().unwrap_or_default()
    )
}

#[derive(Serialize)]
struct PoolsResponse {
    pools: Vec<PoolView>,
}

#[derive(Debug, Serialize)]
struct PoolView {
    site_id: String,
    namespace: String,
    dgd_name: String,
    /// True when a ready endpoint and both Relay ports are known.
    ready: bool,
    relay_grpc_url: Option<String>,
    stats_grpc_url: Option<String>,
    endpoints: Vec<EndpointView>,
}

#[derive(Debug, Serialize)]
struct EndpointView {
    address: IpAddr,
    ready: bool,
}

#[cfg(test)]
mod tests {
    use k8s_openapi::api::discovery::v1::{Endpoint, EndpointConditions, EndpointPort};
    use k8s_openapi::apimachinery::pkg::apis::meta::v1::ObjectMeta;

    use super::*;

    fn naming() -> Naming {
        Naming {
            service_prefix: "relay-".to_owned(),
            relay_port_name: "wan-grpc".to_owned(),
            stats_port_name: "stats-grpc".to_owned(),
        }
    }

    fn slice(name: &str, service: &str, endpoints: &[(&str, Option<bool>)]) -> EndpointSlice {
        EndpointSlice {
            address_type: "IPv4".to_owned(),
            metadata: ObjectMeta {
                name: Some(name.to_owned()),
                namespace: Some("dynamo".to_owned()),
                labels: Some(BTreeMap::from([(
                    SERVICE_NAME_LABEL.to_owned(),
                    format!("{DERIVED_PREFIX}{service}"),
                )])),
                ..Default::default()
            },
            ports: Some(vec![
                EndpointPort {
                    name: Some("wan-grpc".to_owned()),
                    port: Some(50051),
                    ..Default::default()
                },
                EndpointPort {
                    name: Some("stats-grpc".to_owned()),
                    port: Some(50052),
                    ..Default::default()
                },
            ]),
            endpoints: endpoints
                .iter()
                .map(|(address, ready)| Endpoint {
                    addresses: vec![(*address).to_owned()],
                    conditions: Some(EndpointConditions {
                        ready: *ready,
                        ..Default::default()
                    }),
                    ..Default::default()
                })
                .collect(),
        }
    }

    fn west_relay(endpoints: &[(&str, Option<bool>)]) -> EndpointSlice {
        slice(
            "imported-aws-west-relay-mocker-west-1-28g2h",
            "relay-mocker-west-1",
            endpoints,
        )
    }

    fn synced(events: Vec<watcher::Event<EndpointSlice>>) -> Index {
        let mut index = Index::default();
        index.apply(&naming(), watcher::Event::Init);
        for event in events {
            index.apply(&naming(), event);
        }
        index.apply(&naming(), watcher::Event::InitDone);
        index
    }

    #[test]
    fn reads_site_and_dgd_from_imported_slice() {
        let entry = naming()
            .slice_entry(&west_relay(&[("10.43.66.175", Some(true))]))
            .expect("relay slice");
        assert_eq!(
            entry.pool,
            PoolRef {
                site_id: "aws-west".to_owned(),
                namespace: "dynamo".to_owned(),
                dgd_name: "mocker-west-1".to_owned(),
            }
        );
        assert_eq!(
            (entry.relay_port, entry.stats_port),
            (Some(50051), Some(50052))
        );
    }

    #[test]
    fn ignores_services_without_relay_prefix() {
        let serve = slice(
            "imported-aws-west-serve-pw28n",
            "serve",
            &[("10.43.71.89", None)],
        );
        assert_eq!(naming().slice_entry(&serve), None);
    }

    #[test]
    fn ignores_slice_name_without_service() {
        let renamed = slice("imported-aws-west-other-x", "relay-mocker-west-1", &[]);
        assert_eq!(naming().slice_entry(&renamed), None);
    }

    #[test]
    fn serves_ready_pool_urls() {
        let index = synced(vec![watcher::Event::InitApply(west_relay(&[(
            "10.43.66.175",
            None,
        )]))]);
        let pools = index.pools();
        assert_eq!(pools.len(), 1);
        assert!(pools[0].ready);
        assert_eq!(
            pools[0].relay_grpc_url.as_deref(),
            Some("http://10.43.66.175:50051")
        );
        assert_eq!(
            pools[0].stats_grpc_url.as_deref(),
            Some("http://10.43.66.175:50052")
        );
    }

    #[test]
    fn pool_without_ready_endpoint_is_not_ready() {
        let index = synced(vec![watcher::Event::InitApply(west_relay(&[(
            "10.43.66.175",
            Some(false),
        )]))]);
        let pools = index.pools();
        assert!(!pools[0].ready);
        assert_eq!(pools[0].relay_grpc_url, None);
        assert_eq!(pools[0].endpoints.len(), 1);
    }

    #[test]
    fn brackets_ipv6_endpoints() {
        let index = synced(vec![watcher::Event::InitApply(west_relay(&[(
            "fd00::2",
            Some(true),
        )]))]);
        assert_eq!(
            index.pools()[0].relay_grpc_url.as_deref(),
            Some("http://[fd00::2]:50051")
        );
    }

    #[test]
    fn not_synced_until_init_done() {
        let mut index = Index::default();
        index.apply(&naming(), watcher::Event::Init);
        index.apply(
            &naming(),
            watcher::Event::InitApply(west_relay(&[("10.43.66.175", None)])),
        );
        assert!(!index.synced);
        assert!(index.pools().is_empty());
        index.apply(&naming(), watcher::Event::InitDone);
        assert!(index.synced);
        assert_eq!(index.pools().len(), 1);
    }

    #[test]
    fn delete_removes_pool() {
        let mut index = synced(vec![watcher::Event::InitApply(west_relay(&[(
            "10.43.66.175",
            None,
        )]))]);
        index.apply(&naming(), watcher::Event::Delete(west_relay(&[])));
        assert!(index.pools().is_empty());
    }

    #[test]
    fn relist_replaces_previous_slices() {
        let mut index = synced(vec![watcher::Event::InitApply(west_relay(&[(
            "10.43.66.175",
            None,
        )]))]);
        let ohio = slice(
            "imported-aws-ohio-relay-mocker-ohio-1-abcde",
            "relay-mocker-ohio-1",
            &[("10.42.1.2", None)],
        );
        index.apply(&naming(), watcher::Event::Init);
        index.apply(&naming(), watcher::Event::InitApply(ohio));
        // The old view is still served until the relist completes.
        assert_eq!(index.pools()[0].site_id, "aws-west");
        index.apply(&naming(), watcher::Event::InitDone);
        let sites: Vec<_> = index.pools().into_iter().map(|pool| pool.site_id).collect();
        assert_eq!(sites, ["aws-ohio"]);
    }
}
