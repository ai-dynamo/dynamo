// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{
    DiscoveryInstance, DistributedRuntime, Metrics, ModelManager, ModelWatcher, NamespaceFilter,
    RouterConfig,
};
use crate::{discovery::Model, model_card::ModelDeploymentCard};
use dynamo_runtime::discovery::DiscoveryEvent;
use futures::StreamExt;
use std::{sync::Arc, time::Duration};

pub(super) fn discovered_card(
    namespace: &str,
    instance_id: u64,
    card: &ModelDeploymentCard,
) -> DiscoveryEvent {
    DiscoveryEvent::Added(DiscoveryInstance::Model {
        namespace: namespace.to_string(),
        component: "workers".to_string(),
        endpoint: "generate".to_string(),
        instance_id,
        card_json: serde_json::to_value(card).unwrap(),
        model_suffix: None,
    })
}

pub(super) type TestDiscoverySender =
    tokio::sync::mpsc::UnboundedSender<(DiscoveryEvent, tokio::sync::oneshot::Sender<()>)>;

pub(super) async fn apply_discovery_event(events: &TestDiscoverySender, event: DiscoveryEvent) {
    let (applied, acknowledged) = tokio::sync::oneshot::channel();
    events.send((event, applied)).unwrap();
    tokio::time::timeout(Duration::from_secs(5), acknowledged)
        .await
        .expect("controller stopped consuming discovery events")
        .unwrap();
}

pub(super) fn watch_test_cards(
    drt: DistributedRuntime,
    manager: Arc<ModelManager>,
    router_config: RouterConfig,
) -> (TestDiscoverySender, tokio::task::JoinHandle<()>) {
    watch_test_cards_with_decisions(drt, manager, router_config, false)
}

pub(super) fn watch_test_cards_with_decisions(
    drt: DistributedRuntime,
    manager: Arc<ModelManager>,
    router_config: RouterConfig,
    decisions: bool,
) -> (TestDiscoverySender, tokio::task::JoinHandle<()>) {
    let mut watcher = ModelWatcher::new(
        drt,
        manager,
        router_config,
        0,
        None,
        None,
        None,
        Arc::new(Metrics::new()),
    );
    watcher.set_systemone_enabled(decisions);
    let watcher = Arc::new(watcher);
    let (event_tx, event_rx) = tokio::sync::mpsc::unbounded_channel();
    let stream = futures::stream::unfold(
        (event_rx, None::<tokio::sync::oneshot::Sender<()>>),
        |(mut receiver, applied)| async {
            // The controller polls again only after applying the previous event.
            if let Some(applied) = applied {
                let _ = applied.send(());
            }
            receiver
                .recv()
                .await
                .map(|(event, applied)| (Ok(event), (receiver, Some(applied))))
        },
    )
    .boxed();
    let task = tokio::spawn(watcher.watch(stream, NamespaceFilter::Global));
    (event_tx, task)
}

pub(super) async fn wait_for_model(
    manager: &ModelManager,
    name: &str,
    ready: impl Fn(&Model) -> bool,
) -> Arc<Model> {
    tokio::time::timeout(Duration::from_secs(5), async {
        loop {
            if let Some(model) = manager.get_committed_model(name)
                && ready(&model)
            {
                return model;
            }
            tokio::task::yield_now().await;
        }
    })
    .await
    .expect("model did not reach the expected serving state")
}
