// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Inspect host GPU / NUMA / CPU topology and per-GPU CPU slices.
//!
//! Pure analysis — no allocation, no `move_pages`. Complements the
//! `validate_numa_placement` binary, which exercises actual allocation and
//! validates page residency.
//!
//! # Usage
//! ```bash
//! cargo run -p dynamo-memory --bin inspect_resources
//! cargo run -p dynamo-memory --bin inspect_resources -- --visible-only
//! ```

use std::process;

use dynamo_memory::resources::{Resources, SlicingMode};

fn main() {
    let mut mode = SlicingMode::AssumeAllBusy;

    for arg in std::env::args().skip(1) {
        match arg.as_str() {
            "--visible-only" => mode = SlicingMode::VisibleOnly,
            "--help" | "-h" => {
                eprintln!(
                    "Usage: inspect_resources [--visible-only]\n\n\
                     Options:\n  \
                     --visible-only   Slice CPU sets across only CUDA-visible GPUs.\n  \
                     -h, --help       Show this help.\n"
                );
                process::exit(0);
            }
            other => {
                eprintln!("unknown argument: {other}");
                process::exit(2);
            }
        }
    }

    let resources = Resources::discover_with(mode);
    print!("{resources}");
}
