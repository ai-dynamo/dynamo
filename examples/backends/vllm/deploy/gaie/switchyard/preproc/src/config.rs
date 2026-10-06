// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

//! Published runner configuration and Kubernetes pool bindings.
use std::{collections::BTreeMap, fs, path::Path};

use anyhow::{Context, Result, ensure};
use protocol::ModelId;
use serde::Deserialize;
use switchyard_runner::Runner;

// SDK 0.3.0 lacks catalog access; replace this projection when it exposes targets.
#[derive(Deserialize)]
struct TargetCatalog {
    targets: BTreeMap<String, Target>,
}

#[derive(Deserialize)]
struct Target {
    id: ModelId,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PoolBindings {
    pub default_route: String,
    pub targets: BTreeMap<String, String>,
}

pub struct Config {
    pub runner: Runner,
    pub default_route: String,
    pub target_models: BTreeMap<String, ModelId>,
}

impl Config {
    pub fn load(routes: impl AsRef<Path>, bindings: impl AsRef<Path>) -> Result<Self> {
        let routes = fs::read_to_string(routes.as_ref())
            .with_context(|| format!("read routing config {}", routes.as_ref().display()))?;
        let bindings = fs::read_to_string(bindings.as_ref())
            .with_context(|| format!("read pool bindings {}", bindings.as_ref().display()))?;
        Self::from_toml(&routes, &bindings)
    }

    pub fn from_toml(source: &str, bindings: &str) -> Result<Self> {
        // SDK parse/build access would allow validation of decision-only options
        // before runtime construction.
        let runner = Runner::from_toml(source)?;
        let catalog: TargetCatalog = toml::from_str(source)?;
        let mut target_models = BTreeMap::new();
        for (name, target) in catalog.targets {
            ensure!(
                dns_label(&name),
                "target {name}: expected a DNS label of at most 63 bytes"
            );
            let model = target.id;
            ensure!(
                model.as_str() != "auto",
                "target {name}: auto is reserved for the default route alias"
            );
            ensure!(
                !target_models.values().any(|id: &ModelId| id == &model),
                "target {name}: served model IDs must be unique for unambiguous pool selection"
            );
            target_models.insert(name, model);
        }
        ensure!(!target_models.is_empty(), "at least one target is required");
        for route in runner.models() {
            ensure!(
                !target_models.values().any(|model| model == route.id),
                "route {}: route IDs must not collide with served model IDs",
                route.id
            );
        }
        let bindings: PoolBindings = toml::from_str(bindings)?;
        ensure!(
            bindings.targets.len() == target_models.len()
                && target_models
                    .keys()
                    .all(|name| bindings.targets.contains_key(name)),
            "pool bindings must cover exactly the configured targets"
        );
        for (target, pool) in &bindings.targets {
            ensure!(
                dns_label(pool),
                "pool for {target}: expected a DNS label of at most 63 bytes"
            );
        }
        ensure!(
            runner.route(&bindings.default_route).is_some(),
            "default_route does not identify a configured route"
        );
        ensure!(
            bindings.default_route == "auto" || runner.route("auto").is_none(),
            "route id auto is reserved for default_route"
        );
        Ok(Self {
            runner,
            default_route: bindings.default_route,
            target_models,
        })
    }
}

fn dns_label(value: &str) -> bool {
    !value.is_empty()
        && value.len() <= 63
        && value
            .bytes()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == b'-')
        && value.as_bytes()[0].is_ascii_alphanumeric()
        && value.as_bytes()[value.len() - 1].is_ascii_alphanumeric()
}

#[cfg(test)]
mod tests {
    use super::*;

    const ROUTES: &str = include_str!("../config/routes.toml");
    const BINDINGS: &str = include_str!("../config/pool-bindings.toml");

    fn rejected(source: &str, bindings: &str, message: &str) {
        let error = Config::from_toml(source, bindings)
            .err()
            .expect("configuration must be rejected");
        assert!(
            error.to_string().contains(message),
            "unexpected error: {error}"
        );
    }

    #[test]
    fn native_sdk_options_are_not_filtered_by_proc() {
        let source = ROUTES.replace(
            "[targets.qwen-large]",
            "extra_body = { temperature = 0.9 }\n[targets.qwen-large]",
        );
        let source = format!("{source}\nvision = true\n");
        Config::from_toml(&source, BINDINGS).unwrap();
    }

    #[test]
    fn rejects_incomplete_and_ambiguous_pool_catalogs() {
        rejected(
            ROUTES,
            &BINDINGS.replace("qwen-large = \"qwen-large-pool\"", ""),
            "cover exactly",
        );
        rejected(
            ROUTES,
            &format!("{BINDINGS}\nextra = 'extra-pool'\n"),
            "cover exactly",
        );
        rejected(
            ROUTES,
            &BINDINGS.replace("qwen-large-pool", "pool with spaces"),
            "DNS label",
        );
        rejected(
            ROUTES,
            &BINDINGS.replace("default_route = \"auto\"", "default_route = \"missing\""),
            "default_route",
        );
        rejected(
            &ROUTES.replace("Qwen/Qwen3-1.7B", "Qwen/Qwen3-0.6B"),
            BINDINGS,
            "must be unique",
        );
        rejected(
            &ROUTES.replace("id = \"auto\"", "id = \"Qwen/Qwen3-0.6B\""),
            BINDINGS,
            "must not collide",
        );
        rejected(
            &ROUTES.replace("Qwen/Qwen3-0.6B", "auto"),
            BINDINGS,
            "auto is reserved",
        );
    }
}
