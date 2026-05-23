// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `kvbmctl` client CLI for the KV indexer feature.
//!
//! Surfaces the read-only hub endpoints (`/v1/features/indexer/{config,
//! hashes/by_position/{pos}, query}`) as `kvbmctl get indexer
//! {config,by-pos,query}`. The win over hand-rolled curl is `query`: it takes a
//! decimal `u128` and packs it into the 16-big-endian-byte wire shape the hub
//! expects (`SequenceHash` = `PositionalLineageHash`, `serde_bytes_u128`), so
//! callers don't reproduce that encoding by hand.

use anyhow::{Context, Result};
use clap::{Arg, ArgMatches, Command};
use futures::future::BoxFuture;
use serde_json::json;

use crate::client::HubClient;
use crate::features::cli::FeatureCli;
use crate::features::indexer::protocol::ROUTE_PREFIX;
use crate::features::indexer::{
    ByPositionResponse, IndexerConfigResponse, InstancesResponse, QueryResponse,
};
use crate::protocol::FeatureKey;

/// Zero-state CLI for the KV indexer feature.
pub struct IndexerCli;

impl FeatureCli for IndexerCli {
    fn key(&self) -> FeatureKey {
        FeatureKey::Indexer
    }

    fn command(&self) -> Command {
        // The CLI command name and the hub HTTP namespace (`ROUTE_PREFIX`) are
        // both the feature key `indexer`, used for the URLs below.
        Command::new("indexer")
            .about("Query the hub's KV block index")
            .subcommand_required(true)
            .arg_required_else_help(true)
            .subcommand(Command::new("config").about("Show the indexer config (GET /config)"))
            .subcommand(Command::new("get-instances").about(
                "List instances registered to the indexer (GET /instances). \
                     Not all registered instances emit KV events.",
            ))
            .subcommand(
                Command::new("by-pos")
                    .about("List indexed blocks at a position bucket (GET /hashes/by_position)")
                    .arg(
                        Arg::new("pos")
                            .required(true)
                            .value_parser(clap::value_parser!(usize))
                            .help("Position bucket (0-based)"),
                    ),
            )
            .subcommand(
                Command::new("query")
                    .about("Resolve which instances hold a block hash (POST /query)")
                    .arg(
                        Arg::new("hash")
                            .required(true)
                            .help("Block hash as a decimal u128 (the hash_u128 from by-pos)"),
                    ),
            )
    }

    fn run<'a>(
        &'a self,
        hub: &'a HubClient,
        matches: &'a ArgMatches,
    ) -> BoxFuture<'a, Result<serde_json::Value>> {
        Box::pin(async move {
            let base = format!("/v1/features/{ROUTE_PREFIX}");
            match matches.subcommand() {
                Some(("config", _)) => {
                    let r: IndexerConfigResponse = hub.get_json(&format!("{base}/config")).await?;
                    Ok(serde_json::to_value(r)?)
                }
                Some(("get-instances", _)) => {
                    let r: InstancesResponse = hub.get_json(&format!("{base}/instances")).await?;
                    Ok(serde_json::to_value(r)?)
                }
                Some(("by-pos", sm)) => {
                    let pos = *sm.get_one::<usize>("pos").expect("pos is required");
                    let r: ByPositionResponse = hub
                        .get_json(&format!("{base}/hashes/by_position/{pos}"))
                        .await?;
                    Ok(serde_json::to_value(r)?)
                }
                Some(("query", sm)) => {
                    let hash_str = sm.get_one::<String>("hash").expect("hash is required");
                    let hash: u128 = hash_str.parse().with_context(|| {
                        format!("--hash must be a decimal u128, got {hash_str:?}")
                    })?;
                    // SequenceHash serializes as the 16 big-endian bytes of the
                    // u128; build that wire shape so the hub's byte-decoder
                    // accepts it (same encoding the kvindex smoke does in Python).
                    let body = json!({ "hashes": [hash.to_be_bytes().to_vec()] });
                    let r: QueryResponse = hub.post_json(&format!("{base}/query"), &body).await?;
                    Ok(serde_json::to_value(r)?)
                }
                _ => unreachable!("subcommand_required(true) guarantees a subcommand"),
            }
        })
    }
}
