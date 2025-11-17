// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Only compile proto files if grpc feature is enabled
    #[cfg(feature = "grpc")]
    {
        tonic_build::compile_protos("proto/nova.proto")?;
    }
    Ok(())
}
