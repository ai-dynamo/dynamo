// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::protocols::EndpointId;

#[test]
fn test_endpoint_namespace_parsing() {
    // Test Endpoint creation from string with namespace
    let endpoint1 = EndpointId::from("vllm-agg.backend.generate");
    assert_eq!(endpoint1.namespace, "vllm-agg");
    assert_eq!(endpoint1.component, "backend");
    assert_eq!(endpoint1.name, "generate");

    let endpoint2 = EndpointId::from("global.frontend.http");
    assert_eq!(endpoint2.namespace, "global");
    assert_eq!(endpoint2.component, "frontend");
    assert_eq!(endpoint2.name, "http");

    // Test with forward slash separator
    let endpoint3 = EndpointId::from("sglang-prod/backend/generate");
    assert_eq!(endpoint3.namespace, "sglang-prod");
    assert_eq!(endpoint3.component, "backend");
    assert_eq!(endpoint3.name, "generate");
}
