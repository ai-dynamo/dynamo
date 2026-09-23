// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

macro_rules! sidecar_shared_tests {
    (args) => {
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
    };
    (endpoint) => {
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
    };
    (transport) => {
        use std::cell::Cell;
        use std::io;
        use std::num::NonZeroUsize;
        use std::time::Duration;

        use crate::GrpcTransportConfig;
        use crate::transport::connect_pool_with;
        use dynamo_backend_common::{BackendError, ErrorType};
        use tokio::time::{Instant, sleep};

        fn config() -> GrpcTransportConfig {
            GrpcTransportConfig {
                connections: NonZeroUsize::new(3).unwrap(),
                connect_attempt_timeout: Duration::from_millis(80),
                retry_interval: Duration::from_millis(10),
                startup_deadline: Duration::from_millis(100),
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[tokio::test(start_paused = true)]
            async fn failed_connection_retries_then_pool_contains_every_slot() {
                let attempts = Cell::new(0);
                let started = Instant::now();
                let channels = connect_pool_with("test", "in-memory", config(), false, |slot, timeout| {
                    assert_eq!(timeout, Duration::from_millis(80));
                    let attempt = attempts.get();
                    attempts.set(attempt + 1);
                    async move {
                        if attempt == 0 {
                            Err(io::Error::other("initial failure"))
                        } else {
                            Ok(slot)
                        }
                    }
                })
                .await
                .unwrap();
                assert_eq!(channels, vec![1, 2, 3]);
                assert_eq!(attempts.get(), 4);
                assert_eq!(started.elapsed(), Duration::from_millis(10));
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[tokio::test(start_paused = true)]
            async fn later_pool_slots_share_the_original_deadline() {
                let observed = std::cell::RefCell::new(Vec::new());
                let started = Instant::now();
                let error = connect_pool_with("test", "in-memory", config(), false, |slot, timeout| {
                    observed.borrow_mut().push((slot, timeout));
                    async move {
                        if slot == 1 {
                            sleep(Duration::from_millis(70)).await;
                            Ok(slot)
                        } else {
                            std::future::pending::<Result<usize, io::Error>>().await
                        }
                    }
                })
                .await
                .unwrap_err();
                assert_eq!(started.elapsed(), Duration::from_millis(100));
                assert_eq!(
                    *observed.borrow(),
                    vec![
                        (1, Duration::from_millis(80)),
                        (2, Duration::from_millis(30)),
                        (3, Duration::from_millis(30))
                    ]
                );
                assert_eq!(
                    error.error_type(),
                    ErrorType::Backend(BackendError::CannotConnect)
                );
                assert!(error.to_string().contains("pool slot 2"));
                assert!(error.to_string().contains("exceeded the startup deadline"));
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[tokio::test(start_paused = true)]
            async fn retry_wait_is_capped_and_last_failure_is_preserved() {
                let mut transport = config();
                transport.retry_interval = Duration::from_secs(1);
                let started = Instant::now();
                let error = connect_pool_with("test", "in-memory", transport, false, |_, _| async {
                    Err::<(), _>(io::Error::other("peer rejected connection"))
                })
                .await
                .unwrap_err();
                assert_eq!(started.elapsed(), Duration::from_millis(100));
                assert!(error.to_string().contains("peer rejected connection"));
                assert!(error.to_string().contains("after 1 attempts"));
            }
        }
    };
    (errors) => {
        use dynamo_backend_common::{BackendError, ErrorType};

        use super::status_to_dynamo;

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn maps_transport_statuses_to_backend_errors() {
                for (code, expected) in [
                    (tonic::Code::InvalidArgument, BackendError::InvalidArgument),
                    (tonic::Code::NotFound, BackendError::InvalidArgument),
                    (tonic::Code::OutOfRange, BackendError::InvalidArgument),
                    (
                        tonic::Code::FailedPrecondition,
                        BackendError::InvalidArgument,
                    ),
                    (tonic::Code::AlreadyExists, BackendError::InvalidArgument),
                    (tonic::Code::Unknown, BackendError::Unknown),
                    (tonic::Code::Unimplemented, BackendError::Unknown),
                    (tonic::Code::ResourceExhausted, BackendError::Unknown),
                    (tonic::Code::PermissionDenied, BackendError::Unknown),
                    (tonic::Code::Unauthenticated, BackendError::Unknown),
                    (tonic::Code::Aborted, BackendError::Unknown),
                    (tonic::Code::DataLoss, BackendError::Unknown),
                    (tonic::Code::Ok, BackendError::Unknown),
                    (tonic::Code::Unavailable, BackendError::CannotConnect),
                    (tonic::Code::Cancelled, BackendError::Cancelled),
                    (
                        tonic::Code::DeadlineExceeded,
                        BackendError::ConnectionTimeout,
                    ),
                    (tonic::Code::Internal, BackendError::Unknown),
                ] {
                    let error = status_to_dynamo("Test", tonic::Status::new(code, "failure"));
                    assert_eq!(error.error_type(), ErrorType::Backend(expected));
                    assert!(error.to_string().contains("Test: failure"));
                    assert!(error.to_string().contains(&format!("{code:?}")));
                    #[cfg(feature = "tonic-v14")]
                    {
                        let v14 = crate::error::status_to_dynamo_v14(
                            "Test",
                            tonic_v14::Status::new(tonic_v14::Code::from_i32(code as i32), "failure"),
                        );
                        assert_eq!(v14.error_type(), error.error_type());
                        assert_eq!(v14.to_string(), error.to_string());
                    }
                }
            }
        }
    };
    (model) => {
        use super::unit_support::model as backend;

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn model_identity_and_limits_are_preserved() {
                let config = backend::engine_config();
                assert_eq!(config.model, "model-source");
                assert_eq!(config.served_model_name.as_deref(), Some("served-model"));
                let llm = config.llm.expect("LLM registration");
                assert_eq!(llm.context_length, Some(8192));
                assert_eq!(llm.kv_cache_block_size, Some(16));
                assert_eq!(llm.max_num_seqs, Some(128));
                assert_eq!(llm.max_num_batched_tokens, Some(2048));
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn missing_optional_limits_are_not_invented() {
                let llm = backend::without_optional_limits().llm.expect("LLM registration");
                assert_eq!(
                    (
                        llm.context_length,
                        llm.kv_cache_block_size,
                        llm.max_num_seqs,
                        llm.max_num_batched_tokens,
                    ),
                    (None, None, None, None)
                );
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn logical_block_size_and_per_rank_capacity_are_registered() {
                let llm = backend::logical_block_size_and_capacity()
                    .llm
                    .expect("LLM registration");
                assert_eq!(llm.kv_cache_block_size, Some(64));
                assert_eq!(llm.total_kv_blocks, Some(2048));
                assert_eq!(llm.data_parallel_size, Some(2));
                assert_eq!(llm.data_parallel_start_rank, Some(0));
            }
        }
    };
    (worker) => {
        use super::unit_support::worker as backend;
        use crate::unit_fixtures::minimal_request;
        use dynamo_backend_common::{BackendError, ErrorType, GenerateContext, LLMEngine, WorkerConfig};

        pub(super) fn assert_worker_options(config: &WorkerConfig, mode: &str, component: &str) {
            assert_eq!(config.namespace, "test-namespace");
            assert_eq!(config.component, component);
            assert_eq!(config.endpoint, "tokens");
            assert_eq!(
                config.custom_jinja_template.as_deref(),
                Some(std::path::Path::new("local-template.jinja"))
            );
            assert_eq!(config.model_name, "model-source");
            assert_eq!(config.served_model_name.as_deref(), Some("served-model"));
            assert!(config.enable_kv_routing);
            assert_eq!(
                config.disaggregation_mode.as_str(),
                if mode == "aggregated" { "agg" } else { mode }
            );
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn worker_options_and_model_identity_are_preserved() {
                for (mode, component) in [
                    ("aggregated", "configured-component"),
                    ("prefill", "prefill"),
                    ("decode", "backend"),
                ] {
                    let (_, config) = backend::worker(mode);
                    assert_worker_options(&config, mode, component);
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[tokio::test]
            async fn unstarted_generation_fails_and_cleanup_is_idempotent() {
                let (engine, _) = backend::worker("aggregated");
                let context = GenerateContext::new(dynamo_backend_common::testing::mock_context(), None);
                let error = engine
                    .generate(minimal_request(), context)
                    .await
                    .err()
                    .expect("unstarted engine");
                assert_eq!(error.error_type(), ErrorType::Backend(BackendError::EngineShutdown));
                engine.cleanup().await.unwrap();
                engine.cleanup().await.unwrap();
                assert!(backend::is_cancelled(&engine));
            }
        }
    };
    (requests) => {
        use super::unit_support::requests as backend;
        use crate::unit_fixtures::minimal_request;
        use dynamo_backend_common::engine::RoutingHints;
        use dynamo_backend_common::{BackendError, DisaggregationMode, ErrorType};

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn oversized_logprob_counts_are_rejected() {
                for prompt in [false, true] {
                    let mut request = minimal_request();
                    let count = i32::MAX as u32 + 1;
                    if prompt {
                        request.output_options.prompt_logprobs = Some(count);
                    } else {
                        request.output_options.logprobs = Some(count);
                    }
                    let error = backend::lower_request(request)
                        .expect_err("oversized logprob count");
                    assert!(error.to_string().contains("fit in i32"));
                    assert_eq!(
                        error.error_type(),
                        ErrorType::Backend(BackendError::InvalidArgument)
                    );
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn selected_lora_adapter_is_forwarded() {
                let mut request = minimal_request();
                request.routing = Some(RoutingHints {
                    lora_name: Some("adapter-a".into()),
                    ..Default::default()
                });
                let selected = backend::selected_lora(request).unwrap();
                assert_eq!(selected.as_deref(), Some("adapter-a"));
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn prefill_rank_overrides_decode_rank_with_fallback() {
                let mut request = minimal_request();
                request.routing = Some(RoutingHints {
                    dp_rank: Some(5),
                    prefill_dp_rank: Some(3),
                    ..Default::default()
                });
                for (mode, expected) in [
                    (DisaggregationMode::Aggregated, Some(5)),
                    (DisaggregationMode::Decode, Some(5)),
                    (DisaggregationMode::Prefill, Some(3)),
                ] {
                    assert_eq!(backend::routed_rank(&request, mode), expected);
                }
                request.routing.as_mut().unwrap().prefill_dp_rank = None;
                assert_eq!(
                    backend::routed_rank(&request, DisaggregationMode::Prefill),
                    Some(5)
                );
            }
        }
    };
    (responses) => {
        use super::unit_support::responses as backend;
        use crate::unit_fixtures::minimal_request;
        use dynamo_backend_common::FinishReason;

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn prompt_metadata_preserves_opt_in_positions_and_values() {
                for has_prompt_logprobs in [false, true] {
                    let mut request = minimal_request();
                    request.output_options.prompt_logprobs = has_prompt_logprobs.then_some(1);
                    let result = backend::convert_prompt_logprobs(
                        &request,
                        &[0.0, -0.25, -0.5],
                        &[vec![], vec![(23, -0.75)], vec![]],
                    ).unwrap();
                    if !has_prompt_logprobs {
                        assert!(result.is_none());
                        continue;
                    }
                    let data = result.unwrap();
                    let positions = data["prompt_logprobs"].as_array().unwrap();
                    assert_eq!(positions.len(), 3);
                    assert!(positions[0].is_null());
                    for (position, expected) in [
                        (1, vec![("22", -0.25), ("23", -0.75)]),
                        (2, vec![("33", -0.5)]),
                    ] {
                        let entries = positions[position].as_object().unwrap();
                        assert_eq!(entries.len(), expected.len());
                        for (token, probability) in expected {
                            assert_eq!(entries[token]["logprob"].as_f64(), Some(probability));
                        }
                    }
                }
            }
        }

        sidecar_test! {
            lane: pre_merge;
            #[test]
            fn terminal_reasons_are_preserved() {
                for expected in [
                    FinishReason::Stop,
                    FinishReason::Length,
                    FinishReason::Cancelled,
                ] {
                    let result = backend::convert_terminal(&minimal_request(), expected.clone()).unwrap();
                    assert_eq!(result.finish_reason, Some(expected));
                }
            }
        }
    };
}
