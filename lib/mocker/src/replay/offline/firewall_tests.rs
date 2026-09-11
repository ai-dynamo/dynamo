// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fs;
use std::path::{Path, PathBuf};

fn rust_sources(root: &Path, sources: &mut Vec<PathBuf>) {
    assert!(
        root.is_dir(),
        "source firewall root does not exist: {}",
        root.display()
    );
    for entry in fs::read_dir(root).unwrap() {
        let path = entry.unwrap().path();
        if path.is_dir() {
            rust_sources(&path, sources);
        } else if path.extension().and_then(|extension| extension.to_str()) == Some("rs") {
            sources.push(path);
        }
    }
}

#[test]
fn offline_kv_router_crate_references_are_extension_owned() {
    let offline = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/replay/offline");
    let extensions = offline.join("extensions");
    let mut sources = Vec::new();
    rust_sources(&offline, &mut sources);
    assert!(
        !sources.is_empty(),
        "source firewall found no Rust sources under {}",
        offline.display()
    );
    for path in sources {
        if path.starts_with(&extensions) {
            continue;
        }
        let source = fs::read_to_string(&path).unwrap();
        assert!(
            !source.contains(concat!("dynamo_", "kv_router")),
            "{} directly depends on the KV-router crate outside the extension firewall",
            path.display()
        );
    }
}

/// The `placement` facade re-exports `KvRouterConfig`, naming the router crate
/// at the crate root -- outside the tree the firewall above scans. That is
/// sanctioned: a consumer would otherwise need its own direct dependency on
/// `dynamo-kv-router` just to tune scoring. Pinning it to exactly one line
/// keeps the exemption from growing into a second, unscanned surface.
#[test]
fn lib_root_names_the_kv_router_crate_exactly_once() {
    let lib = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/lib.rs");
    let source = fs::read_to_string(&lib).unwrap();
    // Code lines only: prose naming the crate creates no dependency, and the
    // facade's own doc has to be free to explain what it re-exports.
    let references = source
        .lines()
        .filter(|line| {
            let trimmed = line.trim_start();
            !trimmed.starts_with("//") && trimmed.contains(concat!("dynamo_", "kv_router"))
        })
        .collect::<Vec<_>>();
    // Built rather than written out, so this file does not itself contain the
    // literal the scan above rejects.
    let sanctioned = format!(
        "    pub use {}::config::KvRouterConfig;",
        concat!("dynamo_", "kv_router")
    );
    assert_eq!(
        references,
        vec![sanctioned.as_str()],
        "src/lib.rs may name the KV-router crate only in the sanctioned \
         `placement` facade re-export"
    );
}
