// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Integration tests for the S3 audit sink.
//!
//! These tests are `#[ignore]`'d by default because they require a live
//! S3-compatible endpoint (LocalStack or MinIO).
//!
//! Recommended setup:
//! ```bash
//! docker run --rm -d -p 4566:4566 --name localstack localstack/localstack
//! aws --endpoint-url http://localhost:4566 \
//!     s3 mb s3://dynamo-audit-test
//! ```
//!
//! Run:
//! ```bash
//! AWS_ACCESS_KEY_ID=test \
//! AWS_SECRET_ACCESS_KEY=test \
//! cargo test --test audit_s3_integration -- --ignored --nocapture
//! ```

#[cfg(test)]
mod tests {
    use std::io::Read;
    use std::sync::Arc;
    use std::time::Duration;

    use aws_sdk_s3::Client;
    use dynamo_llm::audit::handle::AuditHandle;
    use dynamo_llm::audit::{bus, sink};
    use dynamo_llm::protocols::openai::chat_completions::{
        NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
    };
    use flate2::read::MultiGzDecoder;
    use temp_env::async_with_vars;
    use tokio_util::sync::CancellationToken;

    const ENDPOINT: &str = "http://localhost:4566";
    const REGION: &str = "us-east-1";
    const BUCKET: &str = "dynamo-audit-test";

    fn create_test_request(model: &str, store: bool) -> NvCreateChatCompletionRequest {
        let json = serde_json::json!({
            "model": model,
            "messages": [{"role": "user", "content": "test message"}],
            "store": store
        });
        serde_json::from_value(json).expect("Failed to create test request")
    }

    fn create_test_response(model: &str, content: &str) -> NvCreateChatCompletionResponse {
        let json = serde_json::json!({
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 1234567890,
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop"
            }]
        });
        serde_json::from_value(json).expect("Failed to create test response")
    }

    async fn build_localstack_client() -> Client {
        let sdk_config = aws_config::defaults(aws_config::BehaviorVersion::latest())
            .region(aws_sdk_s3::config::Region::new(REGION))
            .load()
            .await;
        let s3_config = aws_sdk_s3::config::Builder::from(&sdk_config)
            .endpoint_url(ENDPOINT)
            .force_path_style(true)
            .build();
        Client::from_conf(s3_config)
    }

    async fn ensure_bucket(client: &Client) {
        // Idempotent: ignore "already owned by you" / "exists".
        let _ = client.create_bucket().bucket(BUCKET).send().await;
    }

    async fn empty_bucket(client: &Client) {
        let mut continuation_token = None;
        loop {
            let mut req = client.list_objects_v2().bucket(BUCKET);
            if let Some(token) = continuation_token.as_ref() {
                req = req.continuation_token(token);
            }
            let resp = match req.send().await {
                Ok(r) => r,
                Err(_) => return,
            };
            for obj in resp.contents() {
                if let Some(key) = obj.key() {
                    let _ = client.delete_object().bucket(BUCKET).key(key).send().await;
                }
            }
            match resp.next_continuation_token() {
                Some(token) => continuation_token = Some(token.to_string()),
                None => break,
            }
        }
    }

    async fn list_keys(client: &Client, prefix: &str) -> Vec<String> {
        let mut keys = Vec::new();
        let mut continuation_token = None;
        loop {
            let mut req = client.list_objects_v2().bucket(BUCKET).prefix(prefix);
            if let Some(token) = continuation_token.as_ref() {
                req = req.continuation_token(token);
            }
            let Ok(resp) = req.send().await else {
                break;
            };
            for obj in resp.contents() {
                if let Some(k) = obj.key() {
                    keys.push(k.to_string());
                }
            }
            match resp.next_continuation_token() {
                Some(token) => continuation_token = Some(token.to_string()),
                None => break,
            }
        }
        keys.sort();
        keys
    }

    async fn fetch_decompressed(client: &Client, key: &str) -> String {
        let resp = client
            .get_object()
            .bucket(BUCKET)
            .key(key)
            .send()
            .await
            .expect("get_object");
        let bytes = resp.body.collect().await.expect("body").into_bytes();
        let mut decoder = MultiGzDecoder::new(&bytes[..]);
        let mut content = String::new();
        decoder
            .read_to_string(&mut content)
            .expect("decompress segment");
        content
    }

    fn emit_simple_audit(prefix: &str) {
        let request = create_test_request("nemotron", true);
        let request_id = format!("{prefix}-{}", uuid::Uuid::new_v4());
        let mut handle: AuditHandle =
            dynamo_llm::audit::handle::create_handle(&request, &request_id).expect("audit handle");
        handle.set_request(Arc::new(request.clone()));
        handle.set_response(Arc::new(create_test_response("nemotron", "audit body")));
        handle.emit();
    }

    fn init_tracing_for_test() {
        let _ = tracing_subscriber::fmt()
            .with_env_filter(
                tracing_subscriber::EnvFilter::try_from_default_env()
                    .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("debug")),
            )
            .with_test_writer()
            .try_init();
    }

    #[tokio::test]
    #[ignore]
    async fn test_audit_s3_basic_flow() {
        init_tracing_for_test();
        let client = build_localstack_client().await;
        ensure_bucket(&client).await;
        empty_bucket(&client).await;

        let prefix = format!("test-basic-{}", uuid::Uuid::new_v4());
        async_with_vars(
            [
                ("DYN_AUDIT_SINKS", Some("s3")),
                ("DYN_AUDIT_S3_BUCKET", Some(BUCKET)),
                ("DYN_AUDIT_S3_PREFIX", Some(prefix.as_str())),
                ("DYN_AUDIT_S3_REGION", Some(REGION)),
                ("DYN_AUDIT_S3_ENDPOINT_URL", Some(ENDPOINT)),
                ("DYN_AUDIT_S3_ROLL_INTERVAL_MS", Some("500")),
                ("DYN_AUDIT_S3_INSTANCE_ID", Some("test-pod-basic")),
                ("DYN_AUDIT_DEPLOYMENT", Some("test-dgd-basic")),
                // MinIO doesn't accept SSE-S3 in default config. Disable
                // server-side encryption for this LocalStack/MinIO test.
                // Production deployments leave the env unset (default AES256)
                // or set aws:kms.
                ("DYN_AUDIT_S3_SSE", Some("none")),
                // AWS credentials are NOT overridden here. The test
                // expects the outer shell to provide AWS_ACCESS_KEY_ID
                // and AWS_SECRET_ACCESS_KEY matching the LocalStack /
                // MinIO instance under test (see module-level docs).
            ],
            async {
                let shutdown = CancellationToken::new();
                bus::init(100);
                sink::spawn_workers_from_env(shutdown.clone())
                    .await
                    .unwrap();
                tokio::time::sleep(Duration::from_millis(100)).await;

                emit_simple_audit("req");
                emit_simple_audit("req");

                // Wait long enough for the 500ms roll_interval to fire and
                // upload at least one segment.
                tokio::time::sleep(Duration::from_millis(900)).await;

                shutdown.cancel();
                tokio::time::sleep(Duration::from_millis(300)).await;

                let keys = list_keys(&client, &prefix).await;
                assert!(
                    !keys.is_empty(),
                    "expected at least one segment under prefix {prefix}"
                );

                let segments =
                    futures::future::join_all(keys.iter().map(|k| fetch_decompressed(&client, k)))
                        .await;
                let combined = segments.concat();
                let line_count = combined.lines().count();
                assert_eq!(
                    line_count,
                    2,
                    "expected 2 NDJSON lines across {} segment(s)",
                    keys.len()
                );

                // Check that records carry deployment + emitted_at_unix_ms.
                let first: serde_json::Value =
                    serde_json::from_str(combined.lines().next().unwrap()).unwrap();
                assert_eq!(first["event"]["deployment"], "test-dgd-basic");
                assert!(first["event"]["emitted_at_unix_ms"].as_i64().is_some());

                // Object keys should include the deployment segment.
                assert!(
                    keys.iter().all(|k| k.contains("/test-dgd-basic/")),
                    "expected deployment in key path: {keys:?}"
                );
            },
        )
        .await;
    }

    // Note: only one #[tokio::test] in this file because the audit
    // policy is cached behind a OnceLock and would leak across cases
    // run in the same test binary. Force-logging behavior is exercised
    // by unit tests in `audit::handle::tests` instead.
}
