// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Router-only ("on-ramp") mode construction.
//!
//! Builds the pieces the embedded router needs to front raw `vllm serve` pods
//! with no Dynamo control plane, and assembles them into [`RouterOnlyParts`] that
//! [`crate::epp::Router::from_router_only`] folds into a `Router`. Keeping the
//! I/O-heavy construction here keeps the `epp.rs` integration small.
//!
//! Pipeline:
//! 1. **Inert runtime** — `DistributedRuntime` on the `mem` discovery backend
//!    with the ZMQ event plane (set via `DYN_DISCOVERY_BACKEND=mem` /
//!    `DYN_EVENT_PLANE=zmq` in the deployment). No etcd/NATS is contacted.
//! 2. **Offline preprocessor** — download just the tokenizer/config for
//!    `DYN_MODEL_NAME` from HF and build an `OpenAIPreprocessor` (no GPU).
//! 3. **Decode `KvRouter`** — built against an in-process `mem` component; its
//!    event-plane subscriber consumes what the decode [`KvRepublisher`] emits.
//! 4. **Pod reflector** — [`RouterOnlyPodReflector`] over `DYN_EPP_POD_SELECTOR`.
//! 5. **KV republisher(s) + reconciler** — per Ready pod, bridge its native
//!    vLLM ZMQ KV events onto the matching router's event plane.
//!
//! **Aggregated** (`DYN_EPP_ROLE_LABEL` unset): the `PrefillRouter` is created
//! inactive, so prefill routing always falls back to decode-only.
//!
//! **Disaggregated** (`DYN_EPP_ROLE_LABEL` set): pods are partitioned by role;
//! decode pods feed the decode router and prefill pods feed a second prefill
//! `mem` component, and the `PrefillRouter` is activated against it so prefill
//! workers are KV-routed. The EPP then emits `x-prefiller-host-port`.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use anyhow::{Context, Result};
use tokio_util::sync::CancellationToken;

use dynamo_kv_router::config::kv_router_config_from_dynamo_env;
use dynamo_llm::discovery::{ModelManager, WORKER_TYPE_DECODE};
use dynamo_llm::kv_router::{KvRouter, PrefillRouter};
use dynamo_llm::local_model::LocalModel;
use dynamo_llm::model_card::ModelDeploymentCard;
use dynamo_llm::preprocessor::OpenAIPreprocessor;
use dynamo_runtime::pipeline::RouterMode;
use dynamo_runtime::{DistributedRuntime, Runtime};

use crate::kv_republisher::KvRepublisher;
use crate::router_only::RouterOnlyConfig;
use crate::router_only_reflector::{ROLE_DECODE, ROLE_PREFILL, RouterOnlyPodReflector};

/// In-process namespace the embedded routers and KV republishers share.
/// Router-only mode has no Dynamo discovery, so these names only need to match
/// between each router's event-plane subscriber and its republisher's publisher.
const ROUTER_ONLY_NAMESPACE: &str = "dynamo-epp-router-only";
/// Component backing the decode router + decode KV events.
const DECODE_COMPONENT: &str = "vllm";
/// Component backing the prefill router + prefill KV events (disagg only).
const PREFILL_COMPONENT: &str = "vllm-prefill";

/// How often the listener reconciler diffs the reflector's Ready set against
/// active per-pod ZMQ listeners.
const RECONCILE_INTERVAL: Duration = Duration::from_secs(2);

/// The constructed router-only-mode pieces, assembled into a `Router` by
/// [`crate::epp::Router::from_router_only`].
pub struct RouterOnlyParts {
    pub runtime: Runtime,
    pub preprocessor: Arc<OpenAIPreprocessor>,
    pub decode_router: Arc<KvRouter>,
    pub prefill_router: Arc<PrefillRouter>,
    pub reflector: Arc<RouterOnlyPodReflector>,
    /// Decode and (in disagg) prefill republishers, held so their per-pod ZMQ
    /// listeners + event-plane publishers stay alive for the router's lifetime.
    pub republishers: Vec<Arc<KvRepublisher>>,
}

/// Build the router-only-mode router pieces from configuration.
pub async fn build_router_only(cfg: &RouterOnlyConfig) -> Result<RouterOnlyParts> {
    let runtime = Runtime::from_settings()?;
    let drt = DistributedRuntime::from_settings(runtime.clone()).await?;

    // Router-only mode must run NATS-free. If the runtime resolved to the NATS
    // event plane, the deployment is misconfigured (expected DYN_EVENT_PLANE=zmq
    // + DYN_DISCOVERY_BACKEND=mem); warn loudly rather than silently dialing NATS.
    if drt.default_event_transport_kind() != dynamo_runtime::discovery::EventTransportKind::Zmq {
        tracing::warn!(
            transport = ?drt.default_event_transport_kind(),
            "Router-only mode expects the ZMQ event plane; set DYN_EVENT_PLANE=zmq and \
             DYN_DISCOVERY_BACKEND=mem to keep the EPP runtime-free (no NATS/etcd)."
        );
    }

    let preprocessor = build_offline_preprocessor(cfg).await?;
    let model_manager = Arc::new(ModelManager::new());

    let mut kv_router_config = kv_router_config_from_dynamo_env();
    // Raw vLLM workers never register a ModelRuntimeConfig in discovery; they are
    // admitted into the scheduler via register_workers(allowed_worker_ids) on the
    // hot path, so we must not block startup waiting for discovery registration.
    kv_router_config.skip_initial_worker_wait = true;

    // Decode router on an in-process `mem` component shared with the decode
    // republisher's publisher, so republished decode events flow into its index.
    let decode_component = drt
        .namespace(ROUTER_ONLY_NAMESPACE)?
        .component(DECODE_COMPONENT)?;
    let decode_endpoint = decode_component.endpoint("generate");
    let decode_router = model_manager
        .kv_chooser_for(
            &decode_endpoint,
            cfg.block_size,
            Some(kv_router_config.clone()),
            None,
            WORKER_TYPE_DECODE,
            Some(cfg.model_name.clone()),
            false, // enable_eagle: raw vLLM does not expose an eagle config here
        )
        .await?;

    let decode_republisher = Arc::new(
        KvRepublisher::new(
            &decode_component,
            cfg.block_size,
            cfg.kv_event_port,
            cfg.kv_event_topic.clone(),
        )
        .await?,
    );

    let mut prefill_config = kv_router_config;
    prefill_config.router_track_active_blocks = false;

    // PrefillRouter: activated against a dedicated prefill component in
    // disaggregated mode, left inactive (decode-only fallback) in aggregated mode.
    let (prefill_router, prefill_republisher) = if cfg.is_disaggregated() {
        let prefill_component = drt
            .namespace(ROUTER_ONLY_NAMESPACE)?
            .component(PREFILL_COMPONENT)?;
        let prefill_endpoint = prefill_component.endpoint("generate");

        let (prefill_tx, prefill_rx) = tokio::sync::oneshot::channel();
        let prefill_router = PrefillRouter::new(
            prefill_rx,
            model_manager.clone(),
            RouterMode::KV,
            cfg.block_size,
            Some(prefill_config),
            None,
            cfg.enforce_disagg,
            cfg.model_name.clone(),
            ROUTER_ONLY_NAMESPACE.to_string(),
            false,
        );
        // Activate immediately: router-only prefill workers are admitted via
        // register_workers on the hot path, not via discovery, so there is no
        // discovery event to wait for.
        if prefill_tx.send(prefill_endpoint).is_err() {
            anyhow::bail!("failed to activate router-only prefill router");
        }

        let prefill_republisher = Arc::new(
            KvRepublisher::new(
                &prefill_component,
                cfg.block_size,
                cfg.kv_event_port,
                cfg.kv_event_topic.clone(),
            )
            .await?,
        );
        tracing::info!("Router-only disaggregated mode: prefill router activated");
        (prefill_router, Some(prefill_republisher))
    } else {
        // Aggregated: build the PrefillRouter but never send an activation
        // endpoint, so route_prefill always errors and pick() falls back to
        // decode-only.
        let (_prefill_tx, prefill_rx) = tokio::sync::oneshot::channel();
        let prefill_router = PrefillRouter::new(
            prefill_rx,
            model_manager.clone(),
            RouterMode::KV,
            cfg.block_size,
            Some(prefill_config),
            None,
            false, // enforce_disagg forced off in aggregated router-only mode
            cfg.model_name.clone(),
            ROUTER_ONLY_NAMESPACE.to_string(),
            false,
        );
        (prefill_router, None)
    };

    let k8s_namespace = std::env::var("POD_NAMESPACE").map_err(|_| {
        anyhow::anyhow!(
            "POD_NAMESPACE environment variable is not set. \
             Inject it via the downward API (fieldRef metadata.namespace) on the EPP pod."
        )
    })?;
    let reflector = Arc::new(
        RouterOnlyPodReflector::spawn(
            &k8s_namespace,
            &cfg.pod_selector,
            cfg.target_port,
            cfg.role_label.clone(),
        )
        .await?,
    );

    let mut republishers = vec![decode_republisher.clone()];
    if let Some(prefill_republisher) = prefill_republisher.clone() {
        republishers.push(prefill_republisher);
    }

    if cfg.kv_events {
        spawn_listener_reconciler(reflector.clone(), decode_republisher, prefill_republisher);
    } else {
        tracing::info!(
            "DYN_EPP_KV_EVENTS=false: skipping per-pod ZMQ KV ingestion (load-aware routing only)"
        );
    }

    Ok(RouterOnlyParts {
        runtime,
        preprocessor,
        decode_router,
        prefill_router,
        reflector,
        republishers,
    })
}

/// Download just the tokenizer/config for `cfg.model_name` from Hugging Face and
/// build an `OpenAIPreprocessor`. No GPU and no Dynamo model card are required;
/// the worker re-tokenizes, so this is only used for routing-side tokenization.
async fn build_offline_preprocessor(cfg: &RouterOnlyConfig) -> Result<Arc<OpenAIPreprocessor>> {
    let model_path = LocalModel::fetch(&cfg.model_name, /* ignore_weights = */ true)
        .await
        .with_context(|| format!("downloading tokenizer/config for '{}'", cfg.model_name))?;
    let mut card = ModelDeploymentCard::load_from_disk(&model_path, None)
        .with_context(|| format!("loading model card for '{}'", cfg.model_name))?;
    card.set_name(&cfg.model_name);
    card.kv_cache_block_size = cfg.block_size;
    let preprocessor = OpenAIPreprocessor::new(card)
        .with_context(|| format!("building preprocessor for '{}'", cfg.model_name))?;
    tracing::info!(
        model = %cfg.model_name,
        block_size = cfg.block_size,
        "Built offline preprocessor for router-only mode"
    );
    Ok(preprocessor)
}

/// Background task: keep one per-pod ZMQ KV-event listener running for every
/// Ready vLLM pod, routed to the republisher for its role. Starts listeners for
/// newly-Ready pods and cancels them when pods leave the reflector's Ready set.
///
/// In aggregated mode (`prefill_republisher = None`) every Ready pod feeds the
/// decode republisher. In disaggregated mode, decode-role pods feed the decode
/// republisher and prefill-role pods feed the prefill republisher.
fn spawn_listener_reconciler(
    reflector: Arc<RouterOnlyPodReflector>,
    decode_republisher: Arc<KvRepublisher>,
    prefill_republisher: Option<Arc<KvRepublisher>>,
) {
    tokio::spawn(async move {
        let mut active: HashMap<u64, CancellationToken> = HashMap::new();
        loop {
            // Build the desired set: worker_id -> (republisher, "ip:port").
            let mut desired: HashMap<u64, (Arc<KvRepublisher>, String)> = HashMap::new();

            let decode_ready = if reflector.is_role_aware() {
                reflector.ready_workers_for_role(ROLE_DECODE)
            } else {
                reflector.ready_workers()
            };
            for (worker_id, endpoint) in decode_ready {
                desired.insert(worker_id, (decode_republisher.clone(), endpoint));
            }

            if let Some(prefill_republisher) = &prefill_republisher {
                for (worker_id, endpoint) in reflector.ready_workers_for_role(ROLE_PREFILL) {
                    desired.insert(worker_id, (prefill_republisher.clone(), endpoint));
                }
            }

            // Start listeners for newly-desired pods.
            for (worker_id, (republisher, endpoint)) in &desired {
                if active.contains_key(worker_id) {
                    continue;
                }
                // endpoint is "ip:port"; the republisher appends the KV-event
                // port, so it only needs the pod IP.
                let Some((ip, _port)) = endpoint.rsplit_once(':') else {
                    tracing::warn!(worker_id, endpoint = %endpoint, "Malformed endpoint; skipping");
                    continue;
                };
                let token = republisher.register_worker(*worker_id, ip);
                active.insert(*worker_id, token);
            }

            // Cancel listeners for pods that are no longer desired.
            active.retain(|worker_id, token| {
                if desired.contains_key(worker_id) {
                    true
                } else {
                    tracing::info!(worker_id, "Pod gone; cancelling its KV-event listener");
                    token.cancel();
                    false
                }
            });

            tokio::time::sleep(RECONCILE_INTERVAL).await;
        }
    });
}
