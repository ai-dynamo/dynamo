// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn native_abort_preserves_retry_status_and_cancels_pending_peer() {
    for status in [429, 503] {
        let engine = TestEngine::new(EngineBehavior::NativeAbort {
            status: Some(status),
            branch: 1,
        });
        let (response, parent, semaphore) = tokio::time::timeout(
            Duration::from_secs(2),
            dispatch_request(engine.clone(), 3, "native-abort"),
        )
        .await
        .expect("native abort must not wait for stream closure or its pending peer");
        let expected = if status == 429 {
            super::super::super::error::overload_status_code()
        } else {
            StatusCode::SERVICE_UNAVAILABLE
        };
        assert_eq!(response.status(), expected);
        assert_eq!(response.headers()["retry-after"], "1");
        assert_eq!(
            response
                .extensions()
                .get::<super::super::DecisionCapacityRejection>()
                .is_some(),
            status == 429
        );
        let body = response_body(response).await;
        assert_eq!(
            body["code"],
            if status == 429 {
                "rate_limit_exceeded"
            } else {
                "server_error"
            }
        );
        assert!(body.get("answers").is_none());
        assert!(!body.to_string().contains("private worker detail"));
        assert!(parent.is_killed());
        assert_eq!(semaphore.available_permits(), 3);
        let observations = engine.observations.lock().unwrap();
        let peer = observations.iter().find(|o| o.index == 2).unwrap();
        assert!(peer.request_context.is_killed());
        assert!(peer.stream_context.is_killed());
    }
}

#[tokio::test]
async fn unknown_native_abort_is_unavailable_not_client_error_or_cancellation() {
    for status in [None, Some(400), Some(499), Some(500)] {
        let engine = TestEngine::new(EngineBehavior::NativeAbort { status, branch: 0 });
        let (response, parent, semaphore) = tokio::time::timeout(
            Duration::from_secs(2),
            dispatch_request(engine, 2, "native-abort"),
        )
        .await
        .expect("native abort must fail without waiting for stream closure");
        assert_eq!(response.status(), StatusCode::SERVICE_UNAVAILABLE);
        assert!(!response.headers().contains_key("retry-after"));
        assert!(
            response
                .extensions()
                .get::<super::super::DecisionCapacityRejection>()
                .is_none()
        );
        let body = response_body(response).await;
        assert_eq!(body["detail"], "SGLang scoring worker is unavailable");
        assert!(body.get("answers").is_none());
        assert!(!body.to_string().contains("private worker detail"));
        assert!(parent.is_killed());
        assert_eq!(semaphore.available_permits(), 2);
    }
}
