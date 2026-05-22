// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `kvbmctl` client CLI for the disagg (conditional-disagg) feature.
//!
//! Surfaces the read-only hub endpoint (`/v1/features/disagg/instances`) as
//! `kvbmctl disagg instances` — the registered prefill/decode instance split.
//! Mirrors [`IndexerCli`](crate::features::indexer::cli::IndexerCli): a
//! zero-state [`FeatureCli`] that shares the feature's own route consts with
//! the server-side router.

use anyhow::Result;
use clap::{ArgMatches, Command};
use futures::future::BoxFuture;

use crate::client::HubClient;
use crate::features::cli::FeatureCli;
use crate::features::disagg::protocol::{ConditionalDisaggInstancesResponse, ROUTE_PREFIX, paths};
use crate::protocol::FeatureKey;

/// Zero-state CLI for the disagg feature.
pub struct DisaggCli;

impl FeatureCli for DisaggCli {
    fn key(&self) -> FeatureKey {
        FeatureKey::ConditionalDisagg
    }

    fn command(&self) -> Command {
        Command::new("disagg")
            .about("Query the hub's conditional-disagg (P/D) registry")
            .subcommand_required(true)
            .arg_required_else_help(true)
            .subcommand(
                Command::new("instances")
                    .about("List registered prefill/decode instances (GET /instances)"),
            )
    }

    fn run<'a>(
        &'a self,
        hub: &'a HubClient,
        matches: &'a ArgMatches,
    ) -> BoxFuture<'a, Result<serde_json::Value>> {
        Box::pin(async move {
            match matches.subcommand() {
                Some(("instances", _)) => {
                    let r: ConditionalDisaggInstancesResponse = hub
                        .get_json(&format!("/v1/features/{ROUTE_PREFIX}{}", paths::INSTANCES))
                        .await?;
                    Ok(serde_json::to_value(r)?)
                }
                _ => unreachable!("subcommand_required(true) guarantees a subcommand"),
            }
        })
    }
}
