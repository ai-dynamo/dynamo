// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Generates the Python protocol modules for the media endpoints from the
//! Rust types.
//!
//! Usage (from the repository root):
//! ```bash
//! cargo run -p dynamo-llm --bin generate-media-protocols            # write the files
//! cargo run -p dynamo-llm --bin generate-media-protocols -- --check # fail when a file is stale
//! ```
//! The files go to `components/src/dynamo/common/protocols/`.

use std::process::ExitCode;
use std::thread;

use anyhow::Context as _;

use dynamo_llm::protocols::openai::pygen::{
    Modality, REGENERATE_COMMAND, check_committed, protocols_dir, render_module,
};

/// Stack size for the generator thread (8 MB). The utoipa schema derivation
/// recurses through nested types.
const GENERATOR_STACK_SIZE: usize = 8 * 1024 * 1024;

/// What the binary does with the generated text.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Mode {
    /// Write every module to its committed path.
    Write,
    /// Compare every module with its committed path. Write nothing.
    Check,
}

fn main() -> anyhow::Result<ExitCode> {
    let mode = match std::env::args().nth(1).as_deref() {
        None => Mode::Write,
        Some("--check") => Mode::Check,
        Some(other) => anyhow::bail!("unknown argument {other:?}; the only option is --check"),
    };
    let handle = thread::Builder::new()
        .stack_size(GENERATOR_STACK_SIZE)
        .spawn(move || run(mode))
        .context("spawn the generator thread")?;
    handle
        .join()
        .map_err(|e| anyhow::anyhow!("generator thread panicked: {e:?}"))?
}

fn run(mode: Mode) -> anyhow::Result<ExitCode> {
    let mut stale = 0;
    for modality in Modality::ALL {
        match mode {
            Mode::Write => {
                let path = protocols_dir().join(modality.python_file_name());
                std::fs::write(&path, render_module(modality))
                    .with_context(|| format!("write {}", path.display()))?;
                println!("wrote {}", path.display());
            }
            Mode::Check => {
                if let Some(drift) = check_committed(modality)? {
                    eprintln!("{drift}");
                    stale += 1;
                }
            }
        }
    }
    if stale == 0 {
        return Ok(ExitCode::SUCCESS);
    }
    eprintln!("Regenerate: {REGENERATE_COMMAND}");
    Ok(ExitCode::FAILURE)
}
