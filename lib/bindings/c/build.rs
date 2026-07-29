// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::fs;
use std::path::Path;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_dir = env::var("OUT_DIR").unwrap();

    let bindings = cbindgen::generate(&crate_dir).expect("Unable to generate bindings");

    // Primary output: write to OUT_DIR (inside target/) so the header survives
    // Docker BuildKit cache mounts across rebuilds.
    let out_dir_header = Path::new(&out_dir).join("llm_engine.h");
    bindings.write_to_file(&out_dir_header);

    // Convenience copy: write to source tree for local development workflows
    // (e.g. `make build` which expects the header under the crate directory).
    let src_tree_header = Path::new(&crate_dir)
        .join("include")
        .join("nvidia")
        .join("dynamo_llm")
        .join("llm_engine.h");
    fs::create_dir_all(src_tree_header.parent().unwrap()).ok();
    fs::copy(&out_dir_header, &src_tree_header).ok();
}
