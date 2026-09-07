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
