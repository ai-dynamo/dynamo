// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Each pool cluster's region, for pools whose imported slices name none: MCS
//! names a pool's cluster, and only some implementations name its region.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock, PoisonError};
use std::time::Duration;

use dynamo_global_common::PoolLocation;
use futures::StreamExt;
use kube::api::{ApiResource, DynamicObject, GroupVersionKind};
use kube::runtime::reflector::{self, Store, store::Writer};
use kube::runtime::watcher;
use kube::{Api, Client};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;

use super::cache::{WATCH_TIMEOUT_SECS, retried};
use crate::pool_discovery::PoolListing;

/// How long a list waits for the first read of Karmada's clusters, so that a
/// replica that just started does not list its pools under a stand-in region.
/// Once one wait runs out, lists stop waiting.
const FIRST_READ_WAIT: Duration = Duration::from_secs(5);

/// Where each pool cluster is, by the name MCS gives the cluster, for pools
/// whose slices name no region: with Karmada, the `spec.region` of its member
/// `Cluster`, then its configured region, then the cluster's own name, as a
/// stand-in that is logged once.
pub struct ClusterRegions {
    configured: HashMap<String, PoolLocation>,
    karmada: Option<KarmadaRegions>,
    /// Clusters already logged as having no region.
    unplaced: Mutex<HashSet<String>>,
}

impl ClusterRegions {
    /// `karmada`, a client of the Karmada control plane, gives each member
    /// `Cluster`'s `spec.region`; the watch starts on first use and needs
    /// `get`, `list` and `watch` on `clusters.cluster.karmada.io`.
    /// `configured` maps a cluster's MCS name to its location otherwise.
    pub fn new(configured: HashMap<String, PoolLocation>, karmada: Option<Client>) -> Self {
        Self {
            configured,
            karmada: karmada.map(KarmadaRegions::new),
            unplaced: Mutex::default(),
        }
    }

    /// Waits, up to [`FIRST_READ_WAIT`], until Karmada's clusters have been
    /// read once or cannot be.
    pub(super) async fn first_read(&self) {
        if let Some(karmada) = &self.karmada {
            karmada.first_read().await;
        }
    }

    /// Gives each listing whose slices named no region (an empty one) its
    /// cluster's location.
    pub(super) fn locate(&self, listings: &mut [PoolListing]) {
        let karmada = self
            .karmada
            .as_ref()
            .map(KarmadaRegions::locations)
            .unwrap_or_default();
        for listing in listings
            .iter_mut()
            .filter(|listing| listing.location.region.is_empty())
        {
            let cluster = listing.key.cluster_id();
            listing.location = karmada
                .get(cluster)
                .or_else(|| self.configured.get(cluster))
                .cloned()
                .unwrap_or_else(|| self.stand_in(cluster));
        }
    }

    fn stand_in(&self, cluster: &str) -> PoolLocation {
        let mut unplaced = self.unplaced.lock().unwrap_or_else(PoisonError::into_inner);
        if unplaced.insert(cluster.to_owned()) {
            tracing::warn!(
                cluster,
                "no region is known for this pool cluster, so its pools are listed with the \
                 cluster's name as their region; set the region in the member Cluster's \
                 spec.region with Karmada, or with --cluster-region"
            );
        }
        PoolLocation {
            region: cluster.to_owned(),
            availability_zone: None,
            datacenter: None,
        }
    }
}

impl fmt::Debug for ClusterRegions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ClusterRegions")
            .field("configured", &self.configured)
            .field("karmada", &self.karmada.is_some())
            .finish_non_exhaustive()
    }
}

/// Karmada's member `Cluster`s, from one reflector.
struct KarmadaRegions {
    clusters: Api<DynamicObject>,
    store: Store<DynamicObject>,
    /// Taken when the watch starts.
    writer: Mutex<Option<Writer<DynamicObject>>>,
    /// Whether the clusters have been read once, or cannot be.
    settled: watch::Sender<bool>,
    /// Lists no longer wait for the first read.
    gave_up_waiting: AtomicBool,
    watching: OnceLock<()>,
    stop: CancellationToken,
}

impl KarmadaRegions {
    fn new(client: Client) -> Self {
        let gvk = GroupVersionKind::gvk("cluster.karmada.io", "v1alpha1", "Cluster");
        let resource = ApiResource::from_gvk(&gvk);
        let writer = Writer::new(resource.clone());
        Self {
            clusters: Api::all_with(client, &resource),
            store: writer.as_reader(),
            writer: Mutex::new(Some(writer)),
            settled: watch::Sender::new(false),
            gave_up_waiting: AtomicBool::new(false),
            watching: OnceLock::new(),
            stop: CancellationToken::new(),
        }
    }

    /// Each member's location, by member name, from its usable `spec.region`.
    fn locations(&self) -> HashMap<String, PoolLocation> {
        self.ensure_watching();
        self.store
            .state()
            .iter()
            .filter_map(|cluster| {
                let region = cluster.data.get("spec")?.get("region")?.as_str()?;
                Some((
                    cluster.metadata.name.clone()?,
                    PoolLocation::in_region(region)?,
                ))
            })
            .collect()
    }

    async fn first_read(&self) {
        self.ensure_watching();
        if self.gave_up_waiting.load(Ordering::Relaxed) {
            return;
        }
        let mut settled = self.settled.subscribe();
        let read = tokio::time::timeout(FIRST_READ_WAIT, settled.wait_for(|settled| *settled));
        if read.await.is_err() {
            self.gave_up_waiting.store(true, Ordering::Relaxed);
        }
    }

    /// Every caller runs in a Tokio runtime, so the spawn is safe.
    fn ensure_watching(&self) {
        self.watching.get_or_init(|| {
            let writer = self
                .writer
                .lock()
                .unwrap_or_else(PoisonError::into_inner)
                .take();
            if let Some(writer) = writer {
                tokio::spawn(watch_clusters(
                    self.clusters.clone(),
                    writer,
                    self.settled.clone(),
                    self.stop.clone(),
                ));
            }
        });
    }
}

impl Drop for KarmadaRegions {
    fn drop(&mut self) {
        self.stop.cancel();
    }
}

/// Keeps `writer`'s store current until `stop`, or until the API server
/// turns out not to serve `Cluster`s, as a member cluster does not.
async fn watch_clusters(
    clusters: Api<DynamicObject>,
    writer: Writer<DynamicObject>,
    settled: watch::Sender<bool>,
    stop: CancellationToken,
) {
    // As the EndpointSlice watch: lists from the API server's cache, and a
    // server that stops answering fails within the client's read timeout.
    let config = watcher::Config {
        page_size: None,
        ..watcher::Config::default()
            .any_semantic()
            .timeout(WATCH_TIMEOUT_SECS)
    };
    // The reflector sees `Init`, which starts each relist; `retried` then
    // drops it, so a failing list backs off.
    let mut events = std::pin::pin!(retried(reflector::reflector(
        writer,
        watcher(clusters, config)
    )));
    let mut last_error = None;
    loop {
        let event = tokio::select! {
            biased;
            () = stop.cancelled() => return,
            event = events.next() => event,
        };
        match event {
            Some(Ok(event)) => {
                if last_error.take().is_some() {
                    tracing::info!("reading Karmada's Cluster objects for pool regions again");
                }
                if matches!(event, watcher::Event::InitDone) {
                    settled.send_replace(true);
                }
            }
            Some(Err(error)) if not_served(&error) => {
                tracing::info!(
                    "this API server serves no Karmada Cluster objects, so it is not the Karmada \
                     control plane; pool regions come from --cluster-region"
                );
                settled.send_replace(true);
                return;
            }
            Some(Err(error)) => {
                let reason = error.to_string();
                if last_error.as_ref() != Some(&reason) {
                    tracing::warn!(
                        error = reason,
                        "cannot read Karmada's Cluster objects for pool regions; regions come \
                         from --cluster-region or the cluster's name until they can be read"
                    );
                    last_error = Some(reason);
                }
                settled.send_replace(true);
            }
            None => return,
        }
    }
}

/// The API server answered the list with 404: it has no such resource.
fn not_served(error: &watcher::Error) -> bool {
    matches!(
        error,
        watcher::Error::InitialListFailed(kube::Error::Api(response)) if response.code == 404
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn a_slices_region_then_a_configured_one_then_the_name_places_a_pool() {
        let east = PoolLocation::in_region("us-east-2").unwrap();
        let regions = ClusterRegions::new(HashMap::from([("east".to_owned(), east)]), None);
        regions.first_read().await;
        let listing = |cluster: &str, region: &str| PoolListing {
            key: dynamo_global_common::PoolKey::new(cluster, "dynamo", "mocker").unwrap(),
            location: PoolLocation {
                region: region.to_owned(),
                availability_zone: None,
                datacenter: None,
            },
            frontends: Vec::new(),
            relay_identities: Vec::new(),
        };
        let mut listings = [
            listing("east", "eu-west-3"),
            listing("east", ""),
            listing("west", ""),
        ];
        regions.locate(&mut listings);
        assert_eq!(listings[0].location.region, "eu-west-3");
        assert_eq!(listings[1].location.region, "us-east-2");
        assert_eq!(listings[2].location.region, "west");
        assert_eq!(listings[2].location.availability_zone, None);
    }
}
