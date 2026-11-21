// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The in-tree plugin crates stay on the plugin API: outside their test modules they name no
//! Dynamo path other than `dynamo_kv_router::plugins::v1alpha1` and
//! `dynamo_kv_router::plugins::experimental`, and `dynamo-kv-router` is their only Dynamo
//! dependency. A plugin that needs something else needs it added to the plugin API first.

use std::fs;
use std::path::{Path, PathBuf};

const ALLOWED_PATHS: [&str; 2] = [
    "dynamo_kv_router::plugins::v1alpha1",
    "dynamo_kv_router::plugins::experimental",
];

/// The one path outside the plugin API a plugin crate may name, and the file that names it:
/// this crate's `default_registry` and `with_defaults` set the host's default policy, which
/// is a host action.
const HOST_EXCEPTION: (&str, &str) = (
    "src/lib.rs",
    "dynamo_kv_router::plugins::host::RouterPluginRegistryHostExt",
);

/// Plugin crates, relative to this crate. The EPP example is a routing host, not a plugin.
const PLUGIN_CRATES: [&str; 7] = [
    ".",
    "../catalog",
    "../../../examples/router/custom-policy-example/catalog",
    "../../../examples/router/custom-policy-example/soft-pin-repin",
    "../../../examples/router/custom-policy-example/disagg-filter-score-pick",
    "../../../examples/router/custom-policy-example/simple-filter-score-pick",
    "../../../examples/router/custom-policy-example/simple-stacked-score-pick",
];

fn rust_sources(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in fs::read_dir(dir).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// The paths rooted at a Dynamo crate that `line` names, such as `dynamo_scheduler::host`.
fn dynamo_paths(line: &str) -> impl Iterator<Item = &str> {
    line.match_indices("dynamo_").filter_map(|(start, _)| {
        let preceded_by_ident = line[..start]
            .chars()
            .next_back()
            .is_some_and(|c| c.is_alphanumeric() || c == '_');
        let path = &line[start..];
        let end = path
            .find(|c: char| !(c.is_alphanumeric() || c == '_' || c == ':'))
            .unwrap_or(path.len());
        let path = path[..end].trim_end_matches(':');
        (!preceded_by_ident && path.contains("::")).then_some(path)
    })
}

#[test]
fn plugin_crates_name_only_the_plugin_api() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut violations = Vec::new();

    for plugin_crate in PLUGIN_CRATES {
        let crate_dir = root.join(plugin_crate);

        let manifest = fs::read_to_string(crate_dir.join("Cargo.toml")).unwrap();
        let dependencies = manifest
            .split("\n[")
            .find(|section| section.starts_with("dependencies]"))
            .unwrap_or_default();
        for line in dependencies.lines() {
            let name = line.split(['=', '.']).next().unwrap_or_default().trim();
            if name.starts_with("dynamo-") && name != "dynamo-kv-router" {
                violations.push(format!("{plugin_crate}/Cargo.toml depends on {name}"));
            }
        }

        let mut sources = Vec::new();
        rust_sources(&crate_dir.join("src"), &mut sources);
        for source in sources {
            let text = fs::read_to_string(&source).unwrap();
            // Unit tests sit in a trailing `#[cfg(test)]` module and may name other crates.
            let code = text.split("#[cfg(test)]").next().unwrap_or_default();
            for (number, line) in code.lines().enumerate() {
                let line = line.split("//").next().unwrap_or_default();
                for path in dynamo_paths(line) {
                    let own_crate = path.starts_with("dynamo_router_policies::");
                    let allowed = ALLOWED_PATHS.iter().any(|allowed| {
                        path == *allowed || path.starts_with(&format!("{allowed}::"))
                    });
                    let excepted = plugin_crate == "."
                        && source.ends_with(HOST_EXCEPTION.0)
                        && path == HOST_EXCEPTION.1;
                    if !own_crate && !allowed && !excepted {
                        violations.push(format!(
                            "{}:{} names {path}",
                            source.strip_prefix(root).unwrap_or(&source).display(),
                            number + 1
                        ));
                    }
                }
            }
        }
    }

    assert!(
        violations.is_empty(),
        "plugin crates must stay on dynamo_kv_router::plugins::{{v1alpha1, experimental}}:\n{}",
        violations.join("\n")
    );
}
