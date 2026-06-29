// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut include_dirs = vec!["proto".to_string()];

    // Well-known Google protobuf types (struct.proto, duration.proto, etc.)
    // live in /usr/include when libprotobuf-dev is installed.
    if std::path::Path::new("/usr/include/google/protobuf").exists() {
        include_dirs.push("/usr/include".to_string());
    }

    let includes: Vec<&str> = include_dirs.iter().map(|s| s.as_str()).collect();

    tonic_build::configure().compile_protos(
        &[
            "proto/envoy/config/core/v3/base.proto",
            "proto/envoy/type/v3/http_status.proto",
            "proto/envoy/extensions/filters/http/ext_proc/v3/processing_mode.proto",
            "proto/envoy/service/ext_proc/v3/external_processor.proto",
        ],
        &includes,
    )?;
    Ok(())
}
