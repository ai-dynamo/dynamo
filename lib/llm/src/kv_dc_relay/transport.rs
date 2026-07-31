// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Mutually authenticated WAN transport for Relay pool publications.

#[cfg(test)]
mod conformance;
mod grpc;
mod identity;
mod load;
mod metrics;
mod server;
mod source;
#[cfg(test)]
mod test_support;

pub(crate) use server::KvDcRelayTransport;
pub(crate) use source::WanPublicationSource;
