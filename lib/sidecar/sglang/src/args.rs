// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Command-line arguments for the SGLang sidecar.

use std::num::NonZeroUsize;

use dynamo_sidecar_common::SidecarArgs;

pub(crate) const DEFAULT_NATIVE_HTTP_MAX_EVENT_BYTES: NonZeroUsize =
    NonZeroUsize::new(512 * 1024 * 1024).unwrap();

/// Parsed sidecar arguments.
#[derive(clap::Parser, Debug, Clone)]
#[command(
    name = "dynamo-sglang-sidecar",
    about = "Dynamo sidecar for an out-of-process SGLang native gRPC server."
)]
pub struct Args {
    #[command(flatten)]
    pub sidecar: SidecarArgs,

    /// Reachable host that decode workers use to connect to a prefill worker's
    /// SGLang disaggregation bootstrap port. By default this is derived from
    /// SGLang's concrete `host`, then `dist_init_addr`, then a routable local
    /// address. This is required when discovery exposes only loopback or
    /// wildcard addresses.
    #[arg(long, env = "SGLANG_DISAGGREGATION_BOOTSTRAP_HOST")]
    pub bootstrap_host: Option<String>,

    /// Maximum size, in bytes, of one event line in SGLang's native `/generate`
    /// stream. Each in-flight request may buffer up to this much.
    #[arg(
        long,
        env = "DYN_SGLANG_NATIVE_HTTP_MAX_EVENT_BYTES",
        default_value_t = DEFAULT_NATIVE_HTTP_MAX_EVENT_BYTES
    )]
    pub native_http_max_event_bytes: NonZeroUsize,
}
