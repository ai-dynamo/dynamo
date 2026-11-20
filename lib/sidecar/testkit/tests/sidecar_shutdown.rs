// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// A non-serving sidecar task has no engine Worker to join runtime teardown.
#[test]
fn sidecar_returns_an_error_when_teardown_exhausts_the_deadline() {
    use dynamo_runtime::DistributedRuntime;
    use dynamo_runtime::engine::AsyncEngineContextProvider;
    use dynamo_runtime::pipeline::Context;

    temp_env::with_vars(
        [
            ("DYN_WORKER_SHUTDOWN_TOTAL_TIMEOUT_SECS", Some("1")),
            ("DYN_TCP_RESPONSE_STREAM_HOST", Some("127.0.0.1")),
        ],
        || {
            // An open response keeps the last runtime's Velo service draining.
            let mut response = None;
            let result = dynamo_sidecar_common::run_task(|runtime, _shutdown| async {
                let drt = DistributedRuntime::new(
                    runtime,
                    dynamo_runtime::distributed::DistributedConfig::process_local(),
                )
                .await?;
                response = Some(
                    drt.velo_response_service()
                        .await?
                        .register_response(Context::new(()).context(), false)?,
                );
                Ok(())
            });
            assert!(
                response.is_some(),
                "could not open the response stream: {result:?}"
            );
            assert!(
                result
                    .unwrap_err()
                    .to_string()
                    .contains("shutdown deadline")
            );
        },
    );
}
