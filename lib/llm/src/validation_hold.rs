// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Holds model-card discovery writes until worker validation releases.
//!
//! Under etcd discovery the frontend routes to any endpoint that has a model
//! card, whatever the pod's readiness. So a worker with a publish gate
//! (`DYN_PUBLISH_GATE`) must not write its card while its checks run.
//! [`register_model_card`](crate::local_model::register_model_card) returns as
//! usual and the card waits here. When every publish check passes, the held
//! cards are written in registration order. The ready checks
//! (`DYN_READY_GATE`) are then called against the published card, and the
//! worker reports ready once they pass too. A worker with only a ready gate
//! writes its card at once.
//!
//! A held card can be unregistered but not edited: a taint update is refused
//! until release. If a held card can't be written, the worker shuts down so
//! its gates run again on restart; its discovery lease removes any card that
//! was already written.

use std::collections::HashMap;
use std::sync::LazyLock;

use dynamo_runtime::DistributedRuntime;
use dynamo_runtime::component::Endpoint;
use dynamo_runtime::discovery::DiscoverySpec;
use dynamo_runtime::traits::DistributedRuntimeProvider;

/// A card's endpoint path and model suffix.
type CardKey = (String, Option<String>);

#[derive(Default)]
struct Held {
    cards: Vec<(CardKey, DiscoverySpec)>,
    published: bool,
}

/// Held cards per runtime, by connection id.
static HELD: LazyLock<tokio::sync::Mutex<HashMap<u64, Held>>> = LazyLock::new(Default::default);

fn card_key(endpoint: &Endpoint, model_suffix: Option<&str>) -> CardKey {
    let id = endpoint.id();
    (
        format!("{}/{}/{}", id.namespace, id.component, id.name),
        model_suffix.map(str::to_string),
    )
}

/// Hold `spec` if this worker's validation requires it. Returns the spec back
/// when it should be written now: no gates, no publish gate, or already
/// published.
pub(crate) async fn hold_card(
    endpoint: &Endpoint,
    model_suffix: Option<&str>,
    spec: DiscoverySpec,
) -> Option<DiscoverySpec> {
    let drt = endpoint.drt();
    let Some(validation) = drt.worker_validation() else {
        return Some(spec);
    };
    if !validation.holds_card() {
        return Some(spec);
    }
    let mut held = HELD.lock().await;
    let id = drt.connection_id();
    if !held.contains_key(&id) {
        let hook_drt = drt.clone();
        if !validation.set_publish_hook(Box::new(move || Box::pin(publish(hook_drt)))) {
            return Some(spec);
        }
    }
    let entry = held.entry(id).or_default();
    if entry.published {
        return Some(spec);
    }
    let key = card_key(endpoint, model_suffix);
    tracing::info!(card = ?key, "Holding model card until worker validation passes");
    entry.cards.push((key, spec));
    None
}

/// Write this runtime's held cards, in registration order.
async fn publish(drt: DistributedRuntime) -> anyhow::Result<()> {
    let mut held = HELD.lock().await;
    let entry = held.entry(drt.connection_id()).or_default();
    for (key, spec) in std::mem::take(&mut entry.cards) {
        if let Err(error) = drt.discovery().register(spec).await {
            drt.shutdown();
            return Err(error.context(format!(
                "publishing held model card {key:?}; shutting down so the gates run again"
            )));
        }
    }
    entry.published = true;
    Ok(())
}

/// Refuse a taint update while this worker's base-model card is held.
pub(crate) async fn reject_if_held(endpoint: &Endpoint) -> anyhow::Result<()> {
    let key = card_key(endpoint, None);
    let held = HELD.lock().await;
    if held
        .get(&endpoint.drt().connection_id())
        .is_some_and(|entry| entry.cards.iter().any(|(held, _)| *held == key))
    {
        anyhow::bail!(
            "the model card for {key:?} is held until worker validation passes; \
             update its taints after release"
        );
    }
    Ok(())
}

/// Drop a held card. Returns `true` if one was held, in which case it was
/// never written and there is nothing to unregister from discovery.
pub(crate) async fn forget_held(endpoint: &Endpoint, model_suffix: Option<&str>) -> bool {
    let key = card_key(endpoint, model_suffix);
    let mut held = HELD.lock().await;
    let Some(entry) = held.get_mut(&endpoint.drt().connection_id()) else {
        return false;
    };
    let before = entry.cards.len();
    entry.cards.retain(|(held, _)| *held != key);
    entry.cards.len() != before
}

#[cfg(all(test, feature = "integration"))]
mod tests {
    use crate::local_model::{LocalModel, register_model_card, update_model_taints};
    use crate::model_card::ModelDeploymentCard;
    use dynamo_runtime::discovery::DiscoveryQuery;
    use dynamo_runtime::validation::testing::CheckServer;
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
    use std::collections::HashSet;
    use std::time::Duration;

    /// A worker with these publish and ready checks and one registered model
    /// card.
    async fn gated(
        publish: &[&str],
        ready: &[&str],
    ) -> (DistributedRuntime, dynamo_runtime::component::Endpoint) {
        let (publish, ready) = (publish.join(","), ready.join(","));
        temp_env::async_with_vars(
            [
                ("DYN_SYSTEM_PORT", Some("0")),
                ("DYN_PUBLISH_GATE", Some(publish.as_str())),
                ("DYN_READY_GATE", Some(ready.as_str())),
            ],
            async {
                let drt = DistributedRuntime::new(
                    Runtime::from_current().unwrap(),
                    DistributedConfig::process_local(),
                )
                .await
                .unwrap();
                let endpoint = drt
                    .namespace("ns")
                    .unwrap()
                    .component("c")
                    .unwrap()
                    .endpoint("generate");
                let mut card = ModelDeploymentCard::with_name_only("m");
                card.runtime_config.max_num_seqs = Some(8);
                register_model_card(&endpoint, &card).await.unwrap();
                (drt, endpoint)
            },
        )
        .await
    }

    async fn cards(drt: &DistributedRuntime) -> usize {
        let all = drt
            .discovery()
            .list(DiscoveryQuery::AllModels)
            .await
            .unwrap();
        all.iter()
            .filter(|card| card.instance_id() == drt.connection_id())
            .count()
    }

    async fn release(drt: &DistributedRuntime) {
        let validation = drt.worker_validation().unwrap();
        validation.start(Default::default(), drt.child_token());
        tokio::time::timeout(Duration::from_secs(10), validation.wait_released())
            .await
            .unwrap();
    }

    /// Wait until `url`'s check at `gate` reaches `state`.
    async fn wait_for_state(drt: &DistributedRuntime, gate: &str, url: &str, state: &str) {
        let validation = drt.worker_validation().unwrap();
        for _ in 0..200 {
            if validation.status_json()[gate][url]["state"] == state {
                return;
            }
            tokio::time::sleep(Duration::from_millis(10)).await;
        }
        panic!(
            "{gate} check {url} never {state}: {}",
            validation.status_json()
        );
    }

    #[tokio::test]
    async fn card_is_written_only_on_release_and_cannot_be_tainted_while_held() {
        let checks = CheckServer::start().await;
        let (drt, endpoint) = gated(&[&checks.url("pass")], &[]).await;
        assert_eq!(cards(&drt).await, 0, "held card was written");
        let taints = HashSet::from(["drain".to_string()]);
        let error = update_model_taints(&endpoint, taints.clone())
            .await
            .unwrap_err();
        assert!(format!("{error:#}").contains("held"), "{error:#}");

        release(&drt).await;
        assert_eq!(cards(&drt).await, 1);
        update_model_taints(&endpoint, taints).await.unwrap();
    }

    #[tokio::test]
    async fn only_a_ready_gate_writes_the_card_at_once() {
        let checks = CheckServer::start().await;
        let (drt, _) = gated(&[], &[&checks.url("pass")]).await;
        assert_eq!(cards(&drt).await, 1);
        assert!(!drt.worker_validation().unwrap().is_released());
    }

    #[tokio::test]
    async fn ready_checks_run_against_the_published_card() {
        let checks = CheckServer::start().await;
        let warm = checks.url("slow");
        let (drt, endpoint) = gated(&[&checks.url("pass")], &[warm.as_str()]).await;
        assert_eq!(
            cards(&drt).await,
            0,
            "card written before the publish checks passed"
        );
        let error = update_model_taints(&endpoint, HashSet::new())
            .await
            .unwrap_err();
        assert!(format!("{error:#}").contains("held"), "{error:#}");

        let validation = drt.worker_validation().unwrap();
        validation.start(Default::default(), drt.child_token());
        wait_for_state(&drt, "ready", &warm, "running").await;
        assert_eq!(
            cards(&drt).await,
            1,
            "card not written before the ready check"
        );
        assert!(!validation.is_released());
        update_model_taints(&endpoint, HashSet::from(["drain".to_string()]))
            .await
            .unwrap();

        tokio::time::timeout(Duration::from_secs(10), validation.wait_released())
            .await
            .unwrap();
    }

    #[tokio::test]
    async fn a_failed_publish_check_never_writes_the_card() {
        let checks = CheckServer::start().await;
        let (soak, warm) = (checks.url("fail"), checks.url("pass"));
        let (drt, _) = gated(&[soak.as_str()], &[warm.as_str()]).await;
        let validation = drt.worker_validation().unwrap();
        validation.start(Default::default(), drt.child_token());
        wait_for_state(&drt, "publish", &soak, "failed").await;
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert_eq!(cards(&drt).await, 0);
        assert!(!validation.is_released());
        assert_eq!(validation.status_json()["ready"][&warm]["state"], "pending");
    }

    #[tokio::test]
    async fn a_card_unregistered_while_held_is_never_written() {
        let checks = CheckServer::start().await;
        let (drt, endpoint) = gated(&[&checks.url("pass")], &[]).await;
        LocalModel::detach_from_endpoint(&endpoint, None)
            .await
            .unwrap();
        release(&drt).await;
        assert_eq!(cards(&drt).await, 0);
    }

    #[tokio::test]
    async fn a_card_that_cannot_be_published_shuts_the_worker_down() {
        let checks = CheckServer::start().await;
        let (drt, _) = gated(&[&checks.url("pass")], &[]).await;
        // Another model now owns the held card's discovery key, so writing the
        // held card on release is refused.
        let other = ModelDeploymentCard::with_name_only("other");
        let spec = dynamo_runtime::discovery::DiscoverySpec::from_model_with_suffix(
            "ns".into(),
            "c".into(),
            "generate".into(),
            &other.for_mdc_wire(),
            None,
        )
        .unwrap();
        drt.discovery().register(spec).await.unwrap();

        let validation = drt.worker_validation().unwrap();
        validation.start(Default::default(), drt.child_token());
        let shutdown = drt.primary_token();
        tokio::time::timeout(Duration::from_secs(10), shutdown.cancelled())
            .await
            .expect("worker did not shut down");
        assert!(!validation.is_released());
    }
}
