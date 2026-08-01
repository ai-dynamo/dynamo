// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Engine-agnostic **direct-backend discoverability shim**.
//!
//! The direct-gRPC path does not run a Dynamo `Worker`: the frontend dials each
//! stock engine's gRPC server itself (see the per-engine `GrpcDispatch`). All a
//! replica needs is to be *discoverable* — a model card + a `TransportType::Grpc`
//! endpoint published in Dynamo discovery under its `DistributedRuntime` lease,
//! with a health gate that pulls the record when the engine dies.
//!
//! This crate owns that narrow responsibility as a thin shim on
//! `dynamo-runtime` + `dynamo-llm` — deliberately NOT on `backend-common`, and
//! NOT an `LLMEngine`/`Worker`. An engine contributes only a small
//! [`DirectBackend`] (connect / health / cleanup); the shim owns the DRT,
//! model-card build, endpoint registration, the hysteresis health loop, and
//! graceful shutdown.
//!
//! NOTE (v2 migration): the [`DirectBackend`] contract below is the committed
//! API. The `run_direct(backend, config)` driver — which replaces
//! `backend-common`'s `register_direct_orchestrator` — is the next step; until
//! it lands, engines keep using the engine-agnostic `--direct` registrar in
//! `backend-common`. `run_direct` must reimplement the model-card build
//! (`dynamo_llm::local_model`), `Endpoint::register_direct_endpoint_instance`,
//! and the 3-fail-unregister / 2-success-reregister health loop (with a
//! per-probe timeout) directly on runtime+llm.

use anyhow::Result;
use async_trait::async_trait;

/// Discovery key naming the direct dispatch provider for a model (e.g. `"trtllm"`).
pub const DIRECT_GRPC_ENDPOINT_KEY: &str = "direct_grpc_endpoint";

/// The facts a direct backend resolves at connect time — everything the shim
/// needs to publish the model card and advertise the gRPC endpoint.
#[derive(Clone, Debug)]
pub struct DirectRegistration {
    /// Provider name written to the model card's `runtime_data["direct_backend"]`
    /// (e.g. `"trtllm"` / `"vllm"` / `"sglang"`); the frontend uses it to pick the
    /// matching `GrpcDispatch`.
    pub backend: String,
    /// The gRPC address the frontend dials (advertised; may differ from the local
    /// endpoint the shim health-checks, for multi-node).
    pub grpc_endpoint: String,
    /// HF id or local path the frontend uses to build the tokenizer/template.
    pub model_path: String,
    /// Model context length, when the engine can report it (TRT-LLM `GetModelInfo`
    /// returns 0, so it comes from `--context-length`); fills a default `max_tokens`.
    pub context_length: Option<u32>,
}

/// An external engine reachable over gRPC that the shim makes discoverable.
///
/// This is the entire engine-facing contract for the direct path — no token
/// pipeline, no `generate`. `connect` resolves the registration facts; the shim
/// polls `health_check` on an interval and pulls the discovery record on failure
/// (re-adding on recovery); `cleanup` releases the client on shutdown.
#[async_trait]
pub trait DirectBackend: Send + Sync {
    /// Connect to the engine and resolve the model/endpoint facts to register.
    async fn connect(&self) -> Result<DirectRegistration>;

    /// Cheap liveness probe of the engine's gRPC. Drives the health gate.
    async fn health_check(&self) -> Result<()>;

    /// Release engine resources. Called once on shutdown.
    async fn cleanup(&self) -> Result<()>;
}
