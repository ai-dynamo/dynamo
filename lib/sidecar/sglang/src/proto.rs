// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#![allow(clippy::all)]
#![allow(missing_docs)]

tonic::include_proto!("sglang.api.v1");

pub(crate) fn is_zero(value: &u32) -> bool {
    *value == 0
}

pub mod encoder {
    tonic::include_proto!("sglang.grpc.encoder");
}
