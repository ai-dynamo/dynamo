// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Command-line arguments for the SGLang sidecar.

use std::num::{NonZeroU64, NonZeroUsize};

use dynamo_sidecar_common::SidecarArgs;

#[derive(clap::Args, Debug, Clone)]
pub struct MetadataUploadArgs {
    /// Maximum number of OpenDAL operators retained by the metadata upload LRU.
    #[arg(
        long = "metadata-upload-operator-cache-capacity",
        env = "DYN_SGLANG_METADATA_UPLOAD_OPERATOR_CACHE_CAPACITY",
        default_value = "64"
    )]
    pub operator_cache_capacity: NonZeroUsize,

    /// OpenDAL control-operation timeout in seconds.
    #[arg(
        long = "metadata-upload-timeout-secs",
        env = "DYN_SGLANG_METADATA_UPLOAD_TIMEOUT_SECS",
        default_value = "3"
    )]
    pub timeout_secs: NonZeroU64,

    /// Timeout in seconds for each OpenDAL I/O attempt.
    #[arg(
        long = "metadata-upload-io-timeout-secs",
        env = "DYN_SGLANG_METADATA_UPLOAD_IO_TIMEOUT_SECS",
        default_value = "3"
    )]
    pub io_timeout_secs: NonZeroU64,

    /// Maximum attempts for a retryable OpenDAL operation.
    #[arg(
        long = "metadata-upload-retry-max-times",
        env = "DYN_SGLANG_METADATA_UPLOAD_RETRY_MAX_TIMES",
        default_value = "3"
    )]
    pub retry_max_times: NonZeroUsize,

    /// Initial exponential retry delay in milliseconds.
    #[arg(
        long = "metadata-upload-retry-min-delay-ms",
        env = "DYN_SGLANG_METADATA_UPLOAD_RETRY_MIN_DELAY_MS",
        default_value_t = 100
    )]
    pub retry_min_delay_ms: u64,

    /// Maximum exponential retry delay in milliseconds.
    #[arg(
        long = "metadata-upload-retry-max-delay-ms",
        env = "DYN_SGLANG_METADATA_UPLOAD_RETRY_MAX_DELAY_MS",
        default_value_t = 1_000
    )]
    pub retry_max_delay_ms: u64,

    /// Exponential retry multiplier; must be at least 1.0.
    #[arg(
        long = "metadata-upload-retry-factor",
        env = "DYN_SGLANG_METADATA_UPLOAD_RETRY_FACTOR",
        default_value_t = 2.0
    )]
    pub retry_factor: f32,

    /// Add random jitter to exponential retry delays.
    #[arg(
        long = "metadata-upload-retry-jitter",
        env = "DYN_SGLANG_METADATA_UPLOAD_RETRY_JITTER",
        default_value_t = true,
        action = clap::ArgAction::Set
    )]
    pub retry_jitter: bool,
}

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

    #[command(flatten)]
    pub metadata_upload: MetadataUploadArgs,
}

#[cfg(test)]
mod tests {
    use clap::{CommandFactory, Parser};

    use super::Args;

    #[test]
    fn metadata_upload_policy_is_configurable_by_cli_and_env() {
        let args = Args::try_parse_from([
            "dynamo-sglang-sidecar",
            "--grpc-endpoint",
            "127.0.0.1:30000",
            "--metadata-upload-operator-cache-capacity",
            "8",
            "--metadata-upload-timeout-secs",
            "11",
            "--metadata-upload-io-timeout-secs",
            "12",
            "--metadata-upload-retry-max-times",
            "4",
            "--metadata-upload-retry-min-delay-ms",
            "20",
            "--metadata-upload-retry-max-delay-ms",
            "200",
            "--metadata-upload-retry-factor",
            "1.5",
            "--metadata-upload-retry-jitter=false",
        ])
        .unwrap();
        let policy = args.metadata_upload;
        assert_eq!(policy.operator_cache_capacity.get(), 8);
        assert_eq!(policy.timeout_secs.get(), 11);
        assert_eq!(policy.io_timeout_secs.get(), 12);
        assert_eq!(policy.retry_max_times.get(), 4);
        assert_eq!(policy.retry_min_delay_ms, 20);
        assert_eq!(policy.retry_max_delay_ms, 200);
        assert_eq!(policy.retry_factor, 1.5);
        assert!(!policy.retry_jitter);

        let command = Args::command();
        for (id, env) in [
            (
                "operator_cache_capacity",
                "DYN_SGLANG_METADATA_UPLOAD_OPERATOR_CACHE_CAPACITY",
            ),
            ("timeout_secs", "DYN_SGLANG_METADATA_UPLOAD_TIMEOUT_SECS"),
            (
                "io_timeout_secs",
                "DYN_SGLANG_METADATA_UPLOAD_IO_TIMEOUT_SECS",
            ),
            (
                "retry_max_times",
                "DYN_SGLANG_METADATA_UPLOAD_RETRY_MAX_TIMES",
            ),
            (
                "retry_min_delay_ms",
                "DYN_SGLANG_METADATA_UPLOAD_RETRY_MIN_DELAY_MS",
            ),
            (
                "retry_max_delay_ms",
                "DYN_SGLANG_METADATA_UPLOAD_RETRY_MAX_DELAY_MS",
            ),
            ("retry_factor", "DYN_SGLANG_METADATA_UPLOAD_RETRY_FACTOR"),
            ("retry_jitter", "DYN_SGLANG_METADATA_UPLOAD_RETRY_JITTER"),
        ] {
            let arg = command
                .get_arguments()
                .find(|arg| arg.get_id() == id)
                .unwrap();
            assert_eq!(arg.get_env().unwrap(), env);
        }
    }
}
