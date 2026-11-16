// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::test_fixtures::AGG;

/// Argument parsing decides where each worker registers. A disaggregated leg
/// that lands on the operator-configured component instead of its role
/// component is invisible to the frontend's prefill router, and no amount of
/// request-level testing would show it.
#[test]
fn parsed_arguments_map_onto_the_worker_registration() {
    let parse = |extra: &[&str]| {
        let mut argv = vec![
            "dynamo-trtllm-sidecar".to_string(),
            "--grpc-endpoint".to_string(),
            "127.0.0.1:50051".to_string(),
            "--model-path".to_string(),
            "model-source".to_string(),
        ];
        argv.extend(extra.iter().map(|arg| arg.to_string()));
        TrtllmSidecarEngine::from_args(argv)
    };

    let (_, aggregated) = parse(&[
        "--component",
        "operator-chosen",
        "--dyn-default-thinking-mode",
        "disabled",
    ])
    .expect("aggregated parses");
    assert_eq!(aggregated.component, "operator-chosen");
    assert_eq!(aggregated.disaggregation_mode, AGG);
    assert_eq!(aggregated.model_name, "model-source");
    assert_eq!(
        aggregated.default_thinking_mode.as_deref(),
        Some("disabled")
    );
    assert!(
        aggregated.enable_kv_routing,
        "KV routing wiring is enabled; discovered sources control event publication"
    );

    for (mode, expected) in [
        (DisaggregationMode::Prefill, "prefill"),
        (DisaggregationMode::Decode, "backend"),
    ] {
        let (_, config) = parse(&[
            "--component",
            "operator-chosen",
            "--disaggregation-mode",
            &mode.to_string(),
        ])
        .expect("a disaggregated leg parses");
        assert_eq!(
            config.component,
            mode.discovery_component(),
            "{mode} must register under its own component, not the configured one"
        );
        assert_eq!(config.component, expected);
        assert_eq!(config.disaggregation_mode, mode);
    }

    for rejected in [
        vec!["--disaggregation-mode", "encode"],
        vec!["--route-to-encoder"],
    ] {
        assert!(
            parse(&rejected).is_err(),
            "{rejected:?} is not supported by this sidecar"
        );
    }
    assert!(
        TrtllmSidecarEngine::from_args(vec![
            "dynamo-trtllm-sidecar".to_string(),
            "--grpc-endpoint".to_string(),
            "127.0.0.1:50051".to_string(),
            "--model-path".to_string(),
            "   ".to_string(),
        ])
        .is_err(),
        "an empty model path has nothing to tokenize with"
    );
}
