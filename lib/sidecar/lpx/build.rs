// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Compiles the client and server of the LPX topology contract
//! (`lpx.topology.v1`), whose canonical copy is in `proto/`.

use std::env;
use std::path::PathBuf;

/// The contract's files, relative to `proto/`. Listed explicitly so a new file
/// is added deliberately rather than picked up by a glob.
const PROTOS: &[&str] = &["lpx/topology/v1/topology.proto"];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR")?).join("proto");
    let paths: Vec<PathBuf> = PROTOS.iter().map(|name| proto_dir.join(name)).collect();

    tonic_build::configure().compile_protos(&paths, &[proto_dir.as_path()])?;

    for path in &paths {
        println!("cargo:rerun-if-changed={}", path.display());
    }
    Ok(())
}
