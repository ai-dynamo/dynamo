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
//!    `DYN_MODEL_NAME` from HF and build an `OpenAIPreprocessor` (no GPU, no
//!    Dynamo model card).
//! 3. **Decode `KvRouter`** — built against an in-process `mem` component; its
//!    event-plane subscriber consumes what the [`KvRepublisher`] emits.
//! 4. **Pod reflector** — [`RouterOnlyPodReflector`] over `DYN_EPP_POD_SELECTOR`.
//! 5. **KV republisher + reconciler** — per Ready pod, bridge its native vLLM
//!    ZMQ KV events onto the router's event plane.
//!
//! NOTE: this builds the **aggregated** path. The `PrefillRouter` is created
//! inactive (never receives an activation endpoint), so prefill routing always
//! falls back to decode-only. Disaggregated prefill discovery + routing is a
//! follow-up.

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
use crate::router_only_reflector::RouterOnlyPodReflector;

/// In-process namespace/component the embedded router and KV republisher share.
/// Router-only mode has no Dynamo discovery, so these names only need to match
/// between the router's event-plane subscriber and the republisher's publisher.
const ROUTER_ONLY_NAMESPACE: &str = "dynamo-epp-router-only";
const ROUTER_ONLY_COMPONENT: &str = "vllm";

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
    pub republisher: Arc<KvRepublisher>,
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

    // One in-process `mem` component shared by the decode router's event-plane
    // subscriber and the KV republisher's publisher, so republished events flow
    // into the router's index.
    let component = drt
        .namespace(ROUTER_ONLY_NAMESPACE)?
        .component(ROUTER_ONLY_COMPONENT)?;
    let endpoint = component.endpoint("generate");

    let mut kv_router_config = kv_router_config_from_dynamo_env();
    // Raw vLLM workers never register a ModelRuntimeConfig in discovery; they are
    // admitted into the scheduler via register_workers(allowed_worker_ids) on the
    // hot path, so we must not block startup waiting for discovery registration.
    kv_router_config.skip_initial_worker_wait = true;

    let model_manager = Arc::new(ModelManager::new());
    let decode_router = model_manager
        .kv_chooser_for(
            &endpoint,
            cfg.block_size,
            Some(kv_router_config.clone()),
            None,
            WORKER_TYPE_DECODE,
            Some(cfg.model_name.clone()),
            false, // enable_eagle: raw vLLM does not expose an eagle config here
        )
        .await?;

    // Aggregated path: build the PrefillRouter but never activate it (no
    // activation endpoint is ever sent), so route_prefill always returns an
    // error and pick() falls back to decode-only. Disagg is a follow-up.
    let mut prefill_config = kv_router_config;
    prefill_config.router_track_active_blocks = false;
    let (_prefill_tx, prefill_rx) = tokio::sync::oneshot::channel();
    let prefill_router = PrefillRouter::new(
        prefill_rx,
        model_manager.clone(),
        RouterMode::KV,
        cfg.block_size,
        Some(prefill_config),
        None,
        false, // enforce_disagg: forced off in aggregated router-only mode
        cfg.model_name.clone(),
        ROUTER_ONLY_NAMESPACE.to_string(),
        false,
    );

    let k8s_namespace = std::env::var("POD_NAMESPACE").map_err(|_| {
        anyhow::anyhow!(
            "POD_NAMESPACE environment variable is not set. \
             Inject it via the downward API (fieldRef metadata.namespace) on the EPP pod."
        )
    })?;
    let reflector = Arc::new(
        RouterOnlyPodReflector::spawn(&k8s_namespace, &cfg.pod_selector, cfg.target_port).await?,
    );

    let republisher = Arc::new(
        KvRepublisher::new(
            &component,
            cfg.block_size,
            cfg.kv_event_port,
            cfg.kv_event_topic.clone(),
        )
        .await?,
    );

    if cfg.kv_events {
        spawn_listener_reconciler(reflector.clone(), republisher.clone());
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
        republisher,
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
/// Ready vLLM pod. Starts listeners for newly-Ready pods and cancels them when
/// pods leave the reflector's Ready set.
fn spawn_listener_reconciler(
    reflector: Arc<RouterOnlyPodReflector>,
    republisher: Arc<KvRepublisher>,
) {
    tokio::spawn(async move {
        let mut active: HashMap<u64, CancellationToken> = HashMap::new();
        loop {
            let ready: HashMap<u64, String> = reflector.ready_workers().into_iter().collect();

            // Start listeners for newly-Ready pods.
            for (worker_id, endpoint) in &ready {
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

            // Cancel listeners for pods that are no longer Ready.
            active.retain(|worker_id, token| {
                if ready.contains_key(worker_id) {
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
