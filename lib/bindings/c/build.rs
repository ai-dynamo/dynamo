// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::path::Path;

fn main() {
    let crate_dir = env::var("CARGO_MANIFEST_DIR").unwrap();

    let header_path = Path::new(&crate_dir)
        .join("include")
        .join("nvidia")
        .join("dynamo_llm")
        .join("llm_engine.h");

    cbindgen::generate(crate_dir)
        .expect("Unable to generate bindings")
        .write_to_file(header_path);
}
