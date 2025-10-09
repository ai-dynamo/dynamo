// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::process::Command;
use vergen_gitcl::{Emitter, GitclBuilder};

fn main() -> anyhow::Result<()> {
    if has_cuda_toolkit() && !has_feature("cuda") && is_cuda_engine() {
        println!("cargo:warning=CUDA not enabled, re-run with `--features cuda`");
    }
    if is_mac() && !has_feature("metal") {
        println!("cargo:warning=Metal not enabled, re-run with `--features metal`");
    }

    let git_config = GitclBuilder::default()
        .describe(true, false, None)
        .build()?;

    Emitter::default().add_instructions(&git_config)?.emit()?;

    Ok(())
}

fn has_feature(s: &str) -> bool {
    env::var(format!("CARGO_FEATURE_{}", s.to_uppercase())).is_ok()
}

fn has_cuda_toolkit() -> bool {
    if let Ok(output) = Command::new("nvcc").arg("--version").output() {
        output.status.success()
    } else {
        false
    }
}

fn is_cuda_engine() -> bool {
    has_feature("mistralrs")
}

#[cfg(target_os = "macos")]
fn is_mac() -> bool {
    true
}

#[cfg(not(target_os = "macos"))]
fn is_mac() -> bool {
    false
}
