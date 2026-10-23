// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Publishes one complete Agent snapshot; no inference or remote RPC endpoint.
use super::AgentClient;
use dynamo_runtime::discovery::{Discovery, DiscoverySpec};
use std::sync::Arc;
use std::time::Duration;
use tokio_util::sync::CancellationToken;
use tonic::Code;

pub(crate) async fn publish(
    discovery: Arc<dyn Discovery>,
    namespace: &str,
    agent: AgentClient,
    shutdown: CancellationToken,
) -> anyhow::Result<()> {
    let (payload, snapshot) = loop {
        let answer = tokio::select! {
            biased;
            _ = shutdown.cancelled() => return Ok(()),
            answer = agent.get_snapshot(Duration::from_secs(5)) => answer,
        };
        match answer {
            Ok(snapshot) => break snapshot,
            Err(error) if error.code() == Code::Unavailable => {
                tracing::debug!(%error, "Waiting for Agent topology")
            }
            Err(error) => return Err(error.into()),
        }
        tokio::select! {
            _ = shutdown.cancelled() => return Ok(()),
            _ = tokio::time::sleep(Duration::from_secs(1)) => {},
        }
    };
    let identity = snapshot.identity.expect("AgentClient validated identity");
    let spec = DiscoverySpec::LpxTopology {
        namespace: namespace.to_owned(),
        deployment: identity.deployment,
        allocation: identity.allocation,
        build: identity.build,
        payload,
    };
    // DWM persistence is atomic and independent of model compatibility and inference readiness.
    discovery.register(spec).await?;
    tracing::info!("Published complete LPX topology in discovery metadata");
    // Retain the published snapshot across Agent loss. Its owner pod bounds its lifetime.
    shutdown.cancelled().await;
    Ok(())
}
