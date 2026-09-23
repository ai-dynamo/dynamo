// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::{GrpcEndpoint, HttpEndpoint};

const ARGUMENT: &str = "--test-endpoint";

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn normalizes_plaintext_endpoints() {
        assert_eq!(
            GrpcEndpoint::parse(" 127.0.0.1:50051 ", ARGUMENT)
                .unwrap()
                .as_str(),
            "http://127.0.0.1:50051"
        );
        assert_eq!(
            GrpcEndpoint::parse("http://server:50051", ARGUMENT)
                .unwrap()
                .as_str(),
            "http://server:50051"
        );
        assert_eq!(
            GrpcEndpoint::parse("grpc://server:50051", ARGUMENT)
                .unwrap()
                .as_str(),
            "http://server:50051"
        );
        let ipv6 = GrpcEndpoint::parse("http://[2001:db8::1]:50051", ARGUMENT).unwrap();
        assert_eq!(ipv6.as_str(), "http://[2001:db8::1]:50051");
        assert_eq!(ipv6.authority_host(), "[2001:db8::1]");
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn rejects_unsupported_or_ambiguous_endpoints() {
        for endpoint in [
            "",
            " ",
            "http://",
            "grpc://",
            "https://server",
            "other://server",
            "http://user:password@server:50051",
            "http://server:50051/path",
            "http://server:50051?token=secret",
            "http://server:50051#fragment",
        ] {
            assert!(GrpcEndpoint::parse(endpoint, ARGUMENT).is_err());
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn derives_http_endpoint_from_grpc_host() {
        let grpc = GrpcEndpoint::parse("http://server:30001", ARGUMENT).unwrap();
        let http = HttpEndpoint::from_grpc(&grpc, 30000).unwrap();
        assert_eq!(http.as_str(), "http://server:30000/");
        assert_eq!(
            http.with_path("/generate").as_str(),
            "http://server:30000/generate"
        );

        let grpc = GrpcEndpoint::parse("http://[2001:db8::1]:30001", ARGUMENT).unwrap();
        let http = HttpEndpoint::from_grpc(&grpc, 30000).unwrap();
        assert_eq!(http.as_str(), "http://[2001:db8::1]:30000/");
        assert!(HttpEndpoint::from_grpc(&grpc, 0).is_err());
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn parses_http_endpoints_with_path_prefixes() {
        for endpoint in [
            "http://worker:8120",
            "https://worker.example.com",
            "https://worker.example.com/admin/v1",
        ] {
            assert!(HttpEndpoint::parse(endpoint, "--http-endpoint").is_ok());
        }
    }
}

sidecar_test! {
    lane: pre_merge;
    #[test]
    fn rejects_invalid_http_endpoints() {
        for endpoint in [
            "",
            "worker:8120",
            "grpc://worker:8120",
            "https:///admin",
            "HTTP:///admin",
            "https://user:token@worker.example.com/admin",
            "https://worker.example.com/admin?token=secret",
            "https://worker.example.com/admin#fragment",
        ] {
            assert!(HttpEndpoint::parse(endpoint, "--http-endpoint").is_err());
        }
    }
}
