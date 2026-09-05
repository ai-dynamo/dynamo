// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! WAN gRPC transport for Relay pool publications.

#[cfg(test)]
mod conformance;
mod grpc;
mod identity;
mod load;
mod server;
mod source;

pub(crate) use server::KvDcRelayTransport;
pub(crate) use source::WanPublicationSource;
