// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! The in-tree plugin crates stay on the plugin API: outside their test modules they name no
//! Dynamo path other than `dynamo_kv_router::plugins::v1alpha1` and
//! `dynamo_kv_router::plugins::experimental`, and their only Dynamo dependencies are
//! `dynamo-kv-router` and each other, under any dependency key. A plugin that needs something
//! else needs it added to the plugin API first.

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

/// The paths rooted at a Dynamo crate that `line` names, such as `dynamo_scheduler::host`. A
/// path cut short by a brace (`dynamo_kv_router::{plugins::host::X}`) or renamed with `as`
/// (`use dynamo_kv_router as r;`) is returned up to that point, so it cannot pass as an
/// allowed path.
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
        let (path, rest) = path.split_at(end);
        let aliased = rest.trim_start().starts_with("as ");
        let rooted = path.contains("::") || aliased;
        (!preceded_by_ident && rooted).then_some(path.trim_end_matches(':'))
    })
}

/// The Dynamo packages a `[dependencies]` section names: the `package` of a renamed
/// dependency, otherwise its key.
fn dynamo_dependencies(dependencies: &str) -> Vec<&str> {
    dependencies
        .lines()
        .filter_map(|line| {
            let key = line.split(['=', '.']).next().unwrap_or_default().trim();
            let package = line
                .split_once("package")
                .and_then(|(_, rest)| rest.trim_start().strip_prefix('='))
                .and_then(|rest| rest.trim_start().strip_prefix('"'))
                .and_then(|rest| rest.split('"').next());
            let name = package.unwrap_or(key);
            name.starts_with("dynamo-").then_some(name)
        })
        .collect()
}

/// The section of `manifest` whose header is `[name]`.
fn manifest_section<'a>(manifest: &'a str, name: &str) -> &'a str {
    manifest
        .split("\n[")
        .find_map(|section| section.strip_prefix(name)?.strip_prefix(']'))
        .unwrap_or_default()
}

/// The `name` in a manifest's `[package]` section.
fn package_name(manifest: &str) -> String {
    manifest_section(manifest, "package")
        .lines()
        .find_map(|line| {
            let (key, value) = line.split_once('=')?;
            (key.trim() == "name").then(|| value.trim().trim_matches('"').to_owned())
        })
        .unwrap()
}

#[test]
fn the_scan_sees_renamed_dependencies_aliases_and_braced_imports() {
    let dependencies = r#"
dynamo-kv-router = { workspace = true }
scheduler = { package = "dynamo-scheduler", workspace = true }
dynamo-tokens.workspace = true
serde = { workspace = true }
"#;
    assert_eq!(
        dynamo_dependencies(dependencies),
        ["dynamo-kv-router", "dynamo-scheduler", "dynamo-tokens"]
    );

    let paths = |line| dynamo_paths(line).collect::<Vec<_>>();
    assert_eq!(
        paths("use dynamo_kv_router as router;"),
        ["dynamo_kv_router"]
    );
    assert_eq!(
        paths("use dynamo_kv_router::{plugins::host::RouterPluginRegistryHostExt};"),
        ["dynamo_kv_router"]
    );
    assert_eq!(
        paths("use dynamo_kv_router::plugins::v1alpha1::worker_selection::WorkerInputs;"),
        ["dynamo_kv_router::plugins::v1alpha1::worker_selection::WorkerInputs"]
    );
    assert!(paths("let dynamo_rows = 1;").is_empty());
}

#[test]
fn plugin_crates_name_only_the_plugin_api() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut violations = Vec::new();
    let manifests = PLUGIN_CRATES.map(|plugin_crate| {
        fs::read_to_string(root.join(plugin_crate).join("Cargo.toml")).unwrap()
    });
    // A plugin crate may depend on another plugin crate, as the example catalog does.
    let plugin_packages = manifests.each_ref().map(|manifest| package_name(manifest));

    for (plugin_crate, manifest) in PLUGIN_CRATES.into_iter().zip(&manifests) {
        let crate_dir = root.join(plugin_crate);

        for name in dynamo_dependencies(manifest_section(manifest, "dependencies")) {
            if name != "dynamo-kv-router" && !plugin_packages.iter().any(|p| p == name) {
                violations.push(format!("{plugin_crate}/Cargo.toml depends on {name}"));
            }
        }

        let mut sources = Vec::new();
        rust_sources(&crate_dir.join("src"), &mut sources);
        for source in sources {
            let text = fs::read_to_string(&source).unwrap();
            // Unit tests sit in a trailing `#[cfg(test)] mod` and may name other crates. A
            // `#[cfg(test)]` on a single item does not end the scan.
            let tests_at = text
                .match_indices("#[cfg(test)]")
                .find(|(at, marker)| text[at + marker.len()..].trim_start().starts_with("mod "))
                .map_or(text.len(), |(at, _)| at);
            let code = &text[..tests_at];
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
