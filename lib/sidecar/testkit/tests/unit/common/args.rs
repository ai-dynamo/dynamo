// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use clap::Parser;

use super::SidecarArgs;

#[derive(Parser)]
struct TestArgs {
    #[command(flatten)]
    sidecar: SidecarArgs,
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn parses_defaults_and_overrides() {
        let defaults = TestArgs::try_parse_from(["test", "--grpc-endpoint", "127.0.0.1:50051"])
            .expect("parse defaults");
        let config = defaults.sidecar.grpc.config();
        assert_eq!(config.connections.get(), 8);
        assert_eq!(config.connect_attempt_timeout, Duration::from_secs(30));
        assert_eq!(config.retry_interval, Duration::from_secs(1));
        assert_eq!(config.startup_deadline, Duration::from_secs(1800));

        let overrides = TestArgs::try_parse_from([
            "test",
            "--grpc-endpoint",
            "127.0.0.1:50051",
            "--grpc-connections",
            "2",
            "--grpc-connect-attempt-timeout-secs",
            "7",
            "--grpc-retry-interval-secs",
            "3",
            "--grpc-startup-deadline-secs",
            "11",
        ])
        .expect("parse overrides");
        let config = overrides.sidecar.grpc.config();
        assert_eq!(config.connections.get(), 2);
        assert_eq!(config.connect_attempt_timeout, Duration::from_secs(7));
        assert_eq!(config.retry_interval, Duration::from_secs(3));
        assert_eq!(config.startup_deadline, Duration::from_secs(11));
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn rejects_zero_values() {
        for flag in [
            "--grpc-connections",
            "--grpc-connect-attempt-timeout-secs",
            "--grpc-retry-interval-secs",
            "--grpc-startup-deadline-secs",
        ] {
            assert!(
                TestArgs::try_parse_from(["test", "--grpc-endpoint", "127.0.0.1:50051", flag, "0",])
                    .is_err()
            );
        }
    }
}
