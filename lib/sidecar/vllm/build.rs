// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::fs;
use std::io::Read;
use std::path::{Path, PathBuf};

use sha2::{Digest, Sha256};

const BSR_COMMIT: &str = "7726adbdafb34bda85e25c8fc5e192f4";
const PROTOS: [(&str, &str); 2] = [
    (
        "inference.proto",
        "6152c306583166ecd691c9c715cab950523e8d1ed2db3dc2bcb538f6ca90e56f",
    ),
    (
        "control.proto",
        "390c88e94f1b68421c54c6d9440f2088d2709a432549c7a0fe94d35ce7b37476",
    ),
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let proto_dir = match env::var_os("DYNAMO_VLLM_PROTO_DIR") {
        Some(path) => PathBuf::from(path),
        None => {
            let path = PathBuf::from(env::var_os("OUT_DIR").ok_or("OUT_DIR is not set")?)
                .join(format!("vllm-proto-{BSR_COMMIT}"));
            fs::create_dir_all(&path)?;
            for (name, _) in PROTOS {
                let destination = path.join(name);
                if !destination.exists() {
                    download(name, &destination)?;
                }
            }
            path
        }
    };

    for (name, checksum) in PROTOS {
        let path = proto_dir.join(name);
        verify(&path, checksum)?;
        println!("cargo:rerun-if-changed={}", path.display());
    }

    tonic_build::configure()
        .protoc_arg("--experimental_allow_proto3_optional")
        .compile_protos(
            &[
                proto_dir.join("inference.proto"),
                proto_dir.join("control.proto"),
            ],
            &[proto_dir],
        )?;
    println!("cargo:rerun-if-env-changed=DYNAMO_VLLM_PROTO_DIR");
    Ok(())
}

fn download(name: &str, destination: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let url = format!("https://buf.build/vllm-project/vllm/raw/{BSR_COMMIT}/-/{name}");
    let mut bytes = Vec::new();
    ureq::get(&url)
        .call()?
        .into_reader()
        .read_to_end(&mut bytes)?;
    fs::write(destination, bytes)?;
    Ok(())
}

fn verify(path: &Path, expected: &str) -> Result<(), Box<dyn std::error::Error>> {
    let bytes = fs::read(path)?;
    let actual = format!("{:x}", Sha256::digest(bytes));
    if actual != expected {
        return Err(format!(
            "vLLM proto checksum mismatch for {}: expected {expected}, got {actual}",
            path.display()
        )
        .into());
    }
    Ok(())
}
