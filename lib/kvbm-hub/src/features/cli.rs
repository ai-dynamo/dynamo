// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Client-side per-feature CLI surface for `kvbmctl`.
//!
//! Distinct from [`FeatureManager`](crate::features::FeatureManager), which is
//! *server-side* (holds the index, runs on the hub, constructed with server
//! args). `kvbmctl` is a **client**: it talks HTTP to a possibly-remote hub and
//! has no manager instance. So a feature's CLI surface — the clap subcommand it
//! contributes and the HTTP calls it makes — lives here, on a zero-state trait
//! that only needs a [`HubClient`].
//!
//! Each feature implements [`FeatureCli`] (next to its manager, e.g.
//! `features/indexer/cli.rs`). [`feature_clis`] is the registry `kvbmctl`
//! grafts under its `get` subcommand. Gated behind the `kvbmctl` cargo feature
//! so the default CPU-only hub build pulls in neither this module nor `clap`'s
//! use here beyond what the binary already needs.

use anyhow::Result;
use clap::{Arg, ArgMatches, Command};
use futures::future::BoxFuture;

use crate::client::HubClient;
use crate::protocol::FeatureKey;

/// A feature's `kvbmctl` command surface. Implementors are zero-state units —
/// all per-invocation state comes from the [`ArgMatches`] and the [`HubClient`].
pub trait FeatureCli: Send + Sync {
    /// The feature this CLI drives. Matches the server-side
    /// [`FeatureManager::key`](crate::features::FeatureManager::key).
    fn key(&self) -> FeatureKey;

    /// The clap subcommand this feature contributes under `kvbmctl get`. Named
    /// after the feature (e.g. `indexer`) with its own action subcommands.
    /// `kvbmctl` injects a `--hub` arg, so implementors must not declare one.
    fn command(&self) -> Command;

    /// Execute the matched subcommand against `hub`, returning a JSON value
    /// that `kvbmctl` pretty-prints. `matches` is the [`ArgMatches`] for this
    /// feature's subcommand (i.e. below `get <feature>`).
    fn run<'a>(
        &'a self,
        hub: &'a HubClient,
        matches: &'a ArgMatches,
    ) -> BoxFuture<'a, Result<serde_json::Value>>;
}

/// The `--hub` arg `kvbmctl` injects into every feature subcommand. Marked
/// `global` so it propagates to the action subcommands and may appear anywhere
/// after the feature name (e.g. `get indexer by-pos 0 --hub URL`). Falls
/// back to the `KVBMCTL_HUB` env var. Not `required` (a global+required combo is
/// awkward in clap); the dispatcher errors if it is absent.
pub fn hub_arg() -> Arg {
    Arg::new("hub")
        .long("hub")
        .global(true)
        .env("KVBMCTL_HUB")
        .value_name("URL")
        .help("Hub discovery base URL, e.g. http://hub-host:1337")
}

/// Registry of every feature CLI `kvbmctl` exposes. Add new features here.
pub fn feature_clis() -> Vec<Box<dyn FeatureCli>> {
    vec![
        Box::new(super::indexer::cli::IndexerCli),
        Box::new(super::disagg::cli::DisaggCli),
    ]
}
