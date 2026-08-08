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
fn replay_core_has_no_runtime_adapter_dependencies() {
    const FORBIDDEN: &[&str] = &[
        "crate::loadgen",
        "crate::scheduler",
        "OfflineWorkerState",
        "WorkloadDriver",
        "ReplayRouterMode",
        "extensions::",
    ];

    let core = Path::new(env!("CARGO_MANIFEST_DIR")).join("src/core");
    let mut sources = Vec::new();
    rust_sources(&core, &mut sources);
    assert!(
        !sources.is_empty(),
        "source firewall found no Rust sources under {}",
        core.display()
    );
    for path in sources {
        let source = fs::read_to_string(&path).unwrap();
        for forbidden in FORBIDDEN {
            assert!(
                !source.contains(forbidden),
                "{} crosses the replay core firewall with `{forbidden}`",
                path.display()
            );
        }
    }
}
