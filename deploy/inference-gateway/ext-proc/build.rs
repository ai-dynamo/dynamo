// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tonic_build::configure().compile_protos(
        &[
            "proto/envoy/config/core/v3/base.proto",
            "proto/envoy/type/v3/http_status.proto",
            "proto/envoy/extensions/filters/http/ext_proc/v3/processing_mode.proto",
            "proto/envoy/service/ext_proc/v3/external_processor.proto",
        ],
        &["proto"],
    )?;
    Ok(())
}
