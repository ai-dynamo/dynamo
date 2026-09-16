// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Static Dynamo KV-router provider for source-linked AIPerf simulation bundles.
//!
//! This crate is ordinary Rust composition. It exports no C ABI, loads no
//! shared libraries, and keeps engine and placement values under one Cargo
//! resolution.

use std::sync::Arc;

use aiperf_protocol::sidecar_input::BuiltinRunnerSidecarInputAdapterResolver;
use aiperf_runtime::application::Application;
use aiperf_runtime::dataset_input::BuiltinRunnerDatasetInputAdapterResolver;
use aiperf_runtime::execution_factories::native_execution_factories;
use aiperf_runtime::extensions::{
    AIPerfExtension, AIPerfRegistry, AIPerfRegistryFactory, BuiltinAIPerfRegistryFactory,
    ExtensionError,
};
use aiperf_runtime::graph_input::BuiltinRunnerGraphInputAdapterResolver;
use aiperf_simulate::aisimulate::{
    OfflineEngineConfig, OfflineEngineFactory, OfflinePlacement, OfflineTopology,
};
use aiperf_simulate::{AISimulateExtension, StaticSimulationProvider};
use aisimulate_core::engine::EngineConfig;
use aisimulate_core::replay::loadgen::{DynPlacement, SteppableAgg, SteppableReplay};
use aisimulate_core::replay::{ReplayEngineConfig, ReplayEngineFactory};
use anyhow::{Context, Result};
use dynamo_mocker::placement::{
    KvReplayMetadata, KvRouterConfig, KvRouterPlacement, MockEngineArgs, MockEngineArgsBuilder,
    RouterEventObservation,
};

const PROVIDER_ID: &str = "dynamo.kv_router";

type RouterObservation = RouterEventObservation;
type RouterMetadata = KvReplayMetadata;
type BoxedPlacementPolicy = DynPlacement<RouterObservation, RouterMetadata>;

/// Source-linked factory for offline engines placed by Dynamo's KV router.
#[derive(Debug, Clone, Copy, Default)]
pub struct DynamoKvRouterEngineFactory {
    _private: (),
}

fn selector_seed(config: &OfflineEngineConfig) -> Result<u64> {
    match config.placement {
        OfflinePlacement::KvRouter {
            selector_seed: Some(selector_seed),
        } => Ok(selector_seed),
        OfflinePlacement::KvRouter {
            selector_seed: None,
        } => anyhow::bail!(
            "Dynamo KV-router placement requires a deterministic selector seed in a static simulation bundle"
        ),
        OfflinePlacement::RoundRobin => anyhow::bail!(
            "Dynamo KV-router placement was requested but the configuration authors placement {:?}; the run report would describe routing that did not happen",
            config.placement
        ),
    }
}

fn ensure_supported(config: &OfflineEngineConfig) -> Result<u64> {
    let selector_seed = selector_seed(config)?;
    anyhow::ensure!(
        matches!(config.topology, OfflineTopology::Aggregated),
        "Dynamo KV-router placement supports the aggregated topology only, got {:?}",
        config.topology
    );
    anyhow::ensure!(
        !config.is_single_pass_engine(),
        "Dynamo KV-router placement cannot use the single-pass engine, which has no placement seam"
    );
    anyhow::ensure!(
        config.workers > 0,
        "offline aggregate workers must be positive"
    );
    Ok(selector_seed)
}

fn router_args(engine: &ReplayEngineConfig) -> Result<MockEngineArgs> {
    let rank: &EngineConfig = &engine.rank;
    MockEngineArgsBuilder::default()
        .block_size(rank.block_size)
        .num_gpu_blocks(rank.num_gpu_blocks)
        .enable_prefix_caching(rank.enable_prefix_caching)
        .max_num_batched_tokens(Some(rank.max_num_batched_tokens))
        .max_num_seqs(Some(rank.max_num_seqs))
        .dp_size(engine.dp_size)
        .build()
        .context("projecting the engine configuration onto the Dynamo KV router")
}

impl OfflineEngineFactory for DynamoKvRouterEngineFactory {
    fn validate(&self, config: &OfflineEngineConfig) -> Result<()> {
        ensure_supported(config).map(|_| ())
    }

    fn build(&self, config: &OfflineEngineConfig) -> Result<Box<dyn SteppableReplay>> {
        let seed = ensure_supported(config)?;
        let workers = config.workers;
        let replay_config = config.aggregate_replay_engine_config()?;
        let mock_args = router_args(&replay_config)?;

        let mut engine =
            SteppableAgg::<BoxedPlacementPolicy, RouterObservation, RouterMetadata>::with_placement(
                replay_config,
                &ReplayEngineFactory::new(),
                workers,
                move |dp_size, topology| {
                    anyhow::ensure!(
                        topology.len() == workers,
                        "runtime published {} topology entries for {workers} worker(s) at dp_size {dp_size}",
                        topology.len()
                    );
                    let placement = KvRouterPlacement::new(
                        &mock_args,
                        Some(KvRouterConfig {
                            router_queue_threshold: Some(0.0),
                            ..KvRouterConfig::default()
                        }),
                        None,
                        topology.len(),
                        Some(seed),
                    )
                    .context("constructing the Dynamo KV-router placement policy")?;
                    Ok(Box::new(placement) as BoxedPlacementPolicy)
                },
            )
            .context("building the Dynamo KV-router-backed steppable replay engine")?;
        engine.set_capture_per_request(config.capture_per_request);
        engine.set_sla_thresholds(config.sla);
        Ok(Box::new(engine))
    }
}

/// AIPerf extension that installs the `aisimulate` transport with Dynamo's
/// statically linked KV-router engine provider.
#[derive(Debug, Clone, Copy, Default)]
pub struct DynamoAISimulateExtension;

impl AIPerfExtension for DynamoAISimulateExtension {
    fn name(&self) -> &str {
        PROVIDER_ID
    }

    fn register(&self, registry: &mut AIPerfRegistry) -> Result<(), ExtensionError> {
        AISimulateExtension::with_provider(StaticSimulationProvider {
            id: PROVIDER_ID,
            version: env!("CARGO_PKG_VERSION"),
            bundle_identity: env!("CARGO_PKG_NAME"),
            engine_factory: Arc::new(DynamoKvRouterEngineFactory::default()),
        })
        .register(registry)
    }
}

/// Composition root for an AIPerf distribution linked with Dynamo's KV router.
///
/// Consumers pass this factory to the AIPerf application instead of manually
/// mutating a registry. The resulting immutable registry inventory therefore
/// always contains the `aisimulate` transport and the Dynamo provider together.
#[derive(Debug, Clone, Copy, Default)]
pub struct DynamoAIPerfRegistryFactory;

impl DynamoAIPerfRegistryFactory {
    /// Compose a complete production application around the Dynamo bundle.
    pub fn application(distribution_id: impl Into<String>) -> Result<Application> {
        Application::new(
            distribution_id,
            &Self,
            native_execution_factories(),
            Arc::new(BuiltinRunnerGraphInputAdapterResolver::new()),
            Arc::new(BuiltinRunnerDatasetInputAdapterResolver::new()),
            Arc::new(BuiltinRunnerSidecarInputAdapterResolver::new()),
        )
    }
}

impl AIPerfRegistryFactory for DynamoAIPerfRegistryFactory {
    fn build(&self) -> Result<AIPerfRegistry, ExtensionError> {
        BuiltinAIPerfRegistryFactory
            .build()?
            .with_builtin_extensions([&DynamoAISimulateExtension as &dyn AIPerfExtension])
    }
}
