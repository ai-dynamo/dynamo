// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Registers the Relay's routing pool with Global Discovery under the model in
//! its serving topology.

use std::collections::BTreeSet;
use std::sync::Arc;

use dynamo_global_common::{PoolKey, PoolLocation, PoolRegistration};
use dynamo_global_discovery_client::{GlobalDiscoveryClient, PoolHeartbeat, RegistrationStatus};
use tokio::sync::watch;
use tokio_util::sync::CancellationToken;
use tokio_util::task::AbortOnDropHandle;
use url::Url;

use super::topology::{TopologyReadinessState, TopologySnapshot};

/// Keeps the routing pool registered under one model of its serving topology,
/// as [`select_model`] picks it, until cancelled or dropped. With no model to
/// pick it registers nothing, and the pool drops out of Global Discovery once
/// its last lease passes.
pub(super) struct RoutingPoolRegistration {
    heartbeat: PoolHeartbeat,
    withheld: watch::Receiver<Option<String>>,
    _follow: AbortOnDropHandle<()>,
}

impl RoutingPoolRegistration {
    pub(super) fn spawn(
        client: GlobalDiscoveryClient,
        key: PoolKey,
        location: PoolLocation,
        frontends: Vec<Url>,
        topology: watch::Receiver<Arc<TopologySnapshot>>,
        cancel: CancellationToken,
    ) -> Self {
        let (registration, registration_rx) = watch::channel(None);
        let (withheld, withheld_rx) = watch::channel(None);
        let template = PoolRegistration {
            key,
            location,
            model: String::new(),
            capabilities: Vec::new(),
            frontends,
            capacity: Vec::new(),
        };
        let follow = tokio::spawn(follow_topology(
            template,
            topology,
            registration,
            withheld,
            cancel.clone(),
        ));
        Self {
            heartbeat: PoolHeartbeat::spawn(client, registration_rx, cancel),
            withheld: withheld_rx,
            _follow: AbortOnDropHandle::new(follow),
        }
    }

    /// Why the pool is not registered, or `None` while it is.
    pub(super) fn unregistered_reason(&self) -> Option<String> {
        if let Some(reason) = self.withheld.borrow().clone() {
            return Some(reason);
        }
        match self.heartbeat.status() {
            RegistrationStatus::Registered(_) => None,
            RegistrationStatus::Unregistered => Some("no registration has finished yet".to_owned()),
            RegistrationStatus::Failing(reason) => Some(reason),
        }
    }
}

/// Sets `registration` to `template` under the model [`select_model`] picks,
/// or to `None` with the reason in `withheld`. Withdraws the registration when
/// it stops.
async fn follow_topology(
    template: PoolRegistration,
    mut topology: watch::Receiver<Arc<TopologySnapshot>>,
    registration: watch::Sender<Option<PoolRegistration>>,
    withheld: watch::Sender<Option<String>>,
    cancel: CancellationToken,
) {
    // The models last warned about, so each set is logged once.
    let mut warned = BTreeSet::new();
    loop {
        let current = registration
            .borrow()
            .as_ref()
            .map(|registration| registration.model.clone());
        let (candidates, selected) = select_model(&topology.borrow_and_update(), current);
        if candidates.len() > 1 && candidates != warned {
            tracing::warn!(pool = %template.key, models = ?candidates, registered = ?selected.as_ref().ok(), "the serving topology has more than one model, and Global Routers route the routing pool only by its registered model");
            warned = candidates;
        }
        let (next, reason) = match selected {
            Ok(model) => (
                Some(PoolRegistration {
                    model,
                    ..template.clone()
                }),
                None,
            ),
            Err(reason) => (None, Some(reason)),
        };
        withheld.send_replace(reason);
        registration.send_if_modified(|current| {
            if *current == next {
                return false;
            }
            *current = next;
            true
        });
        tokio::select! {
            _ = cancel.cancelled() => break,
            _ = registration.closed() => break,
            changed = topology.changed() => if changed.is_err() {
                break;
            },
        }
    }
    // The heartbeat keeps sending the last value, so leave none.
    withheld.send_replace(Some("the Relay stopped registering the pool".to_owned()));
    registration.send_replace(None);
}

/// Returns the topology's models and the one to register. `current` stays
/// while it is in the topology in any state: membership is the same for every
/// Relay replica, but readiness is each replica's own view, so following it
/// could split replicas between models. Otherwise it is the one Ready model,
/// or the one model while none is Ready. Adapters are not models here.
fn select_model(
    topology: &TopologySnapshot,
    current: Option<String>,
) -> (BTreeSet<String>, Result<String, String>) {
    let models = |ready_only: bool| {
        topology
            .entries
            .iter()
            .filter(|entry| !ready_only || entry.state == TopologyReadinessState::Ready)
            .map(|entry| entry.model.as_str().to_owned())
            .collect::<BTreeSet<_>>()
    };
    let all = models(false);
    if let Some(current) = current.filter(|model| all.contains(model)) {
        return (all, Ok(current));
    }
    let ready = models(true);
    let candidates = if ready.is_empty() { &all } else { &ready };
    let mut candidate = candidates.iter();
    let selected = match (candidate.next(), candidate.next()) {
        (Some(model), None) => Ok(model.clone()),
        (None, _) => Err("the serving topology has no model yet".to_owned()),
        (Some(_), Some(_)) => Err(format!(
            "the serving topology has more than one model ({}), and a registration names one",
            candidates.iter().cloned().collect::<Vec<_>>().join(", ")
        )),
    };
    (all, selected)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::super::identity::CanonicalModelId;
    use super::super::topology::{AdapterReadiness, TopologyEntry};
    use super::*;

    fn entry(model: &str, state: TopologyReadinessState) -> TopologyEntry {
        TopologyEntry {
            namespace: "dynamo".to_owned(),
            model: CanonicalModelId::new(model).unwrap(),
            state,
            present_roles: Vec::new(),
            missing_roles: Vec::new(),
            members: Vec::new(),
            duplicate_role_endpoints: Vec::new(),
            legacy_fallback_active: false,
            adapters: Vec::new(),
        }
    }

    fn snapshot(entries: Vec<TopologyEntry>) -> Arc<TopologySnapshot> {
        Arc::new(TopologySnapshot {
            revision: 0,
            entries,
        })
    }

    fn template() -> PoolRegistration {
        PoolRegistration {
            key: PoolKey::new("east", "dynamo", "blue").unwrap(),
            location: PoolLocation {
                region: "us-east-1".to_owned(),
                availability_zone: None,
                datacenter: None,
            },
            model: String::new(),
            capabilities: Vec::new(),
            frontends: vec![Url::parse("http://blue-frontend:8000/").unwrap()],
            capacity: Vec::new(),
        }
    }

    fn selected(topology: Vec<TopologyEntry>, current: Option<&str>) -> Result<String, String> {
        select_model(&snapshot(topology), current.map(str::to_owned)).1
    }

    #[test]
    fn selects_the_ready_model() {
        use TopologyReadinessState::{Ready, Unavailable, Unknown};
        assert_eq!(
            selected(Vec::new(), None),
            Err("the serving topology has no model yet".to_owned())
        );
        // While nothing is Ready, the one model in any state.
        assert_eq!(
            selected(vec![entry("qwen", Unknown)], None),
            Ok("qwen".to_owned())
        );
        // A model that is not Ready yet does not make the choice ambiguous,
        // and an adapter is not a model.
        let mut qwen = entry("qwen", Ready);
        qwen.adapters.push(AdapterReadiness {
            model: CanonicalModelId::new("qwen-lora").unwrap(),
            state: Ready,
            missing_roles: Vec::new(),
        });
        assert_eq!(
            selected(vec![qwen, entry("llama", Unknown)], None),
            Ok("qwen".to_owned())
        );
        assert_eq!(
            selected(vec![entry("qwen", Ready), entry("llama", Ready)], None),
            Err(
                "the serving topology has more than one model (llama, qwen), and a registration names one"
                    .to_owned()
            )
        );
        // The registered model stays while it is in the topology in any
        // state: while a second model rolls out, or while this replica sees
        // it unavailable.
        assert_eq!(
            selected(
                vec![entry("qwen", Ready), entry("llama", Ready)],
                Some("qwen")
            ),
            Ok("qwen".to_owned())
        );
        assert_eq!(
            selected(
                vec![entry("qwen", Unavailable), entry("llama", Ready)],
                Some("qwen")
            ),
            Ok("qwen".to_owned())
        );
        // Once it leaves, the choice starts over.
        assert_eq!(
            selected(vec![entry("llama", Ready)], Some("qwen")),
            Ok("llama".to_owned())
        );
        assert!(
            selected(
                vec![entry("llama", Unknown), entry("mistral", Unknown)],
                Some("qwen")
            )
            .is_err()
        );
    }

    async fn changed<T: Clone>(receiver: &mut watch::Receiver<T>) -> T {
        tokio::time::timeout(Duration::from_secs(5), receiver.changed())
            .await
            .expect("no change")
            .unwrap();
        receiver.borrow_and_update().clone()
    }

    #[tokio::test]
    async fn follows_the_topology_and_withdraws_when_it_stops() {
        use TopologyReadinessState::Ready;
        let (topology, topology_rx) = watch::channel(snapshot(Vec::new()));
        let (registration, mut registration_rx) = watch::channel(None);
        let (withheld, mut withheld_rx) = watch::channel(None);
        let follow = tokio::spawn(follow_topology(
            template(),
            topology_rx,
            registration,
            withheld,
            CancellationToken::new(),
        ));
        topology.send_replace(snapshot(vec![entry("qwen", Ready)]));
        assert_eq!(
            changed(&mut registration_rx).await,
            Some(PoolRegistration {
                model: "qwen".to_owned(),
                ..template()
            })
        );
        assert_eq!(*withheld_rx.borrow_and_update(), None);

        // A second model does not move the registration off the first.
        topology.send_replace(snapshot(vec![entry("qwen", Ready), entry("llama", Ready)]));
        assert_eq!(changed(&mut withheld_rx).await, None);
        assert!(!registration_rx.has_changed().unwrap());
        assert_eq!(
            registration_rx
                .borrow()
                .as_ref()
                .map(|registration| registration.model.as_str()),
            Some("qwen")
        );

        drop(topology);
        assert_eq!(changed(&mut registration_rx).await, None);
        assert_eq!(
            withheld_rx.borrow().as_deref(),
            Some("the Relay stopped registering the pool")
        );
        follow.await.unwrap();
    }

    #[tokio::test]
    async fn reports_why_the_pool_is_not_registered() {
        // Nothing serves this URL, so every registration fails.
        let unused = std::net::TcpListener::bind("127.0.0.1:0")
            .unwrap()
            .local_addr()
            .unwrap();
        let client = GlobalDiscoveryClient::new(
            format!("http://{unused}/").parse().unwrap(),
            Duration::from_secs(1),
        )
        .unwrap();
        let (topology, topology_rx) = watch::channel(snapshot(Vec::new()));
        let template = template();
        let registration = RoutingPoolRegistration::spawn(
            client,
            template.key,
            template.location,
            template.frontends,
            topology_rx,
            CancellationToken::new(),
        );
        let reason = |expected: String| {
            let registration = &registration;
            async move {
                tokio::time::timeout(Duration::from_secs(5), async {
                    loop {
                        if let Some(reason) = registration.unregistered_reason()
                            && reason.starts_with(&expected)
                        {
                            return reason;
                        }
                        tokio::time::sleep(Duration::from_millis(10)).await;
                    }
                })
                .await
                .expect("the expected reason never showed")
            }
        };
        reason("the serving topology has no model yet".to_owned()).await;
        topology.send_replace(snapshot(vec![entry("qwen", TopologyReadinessState::Ready)]));
        reason(format!("request to http://{unused}/pools failed")).await;
    }
}
