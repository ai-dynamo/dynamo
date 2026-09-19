// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::{BackendError, DisaggregationMode, ErrorType, FinishReason, LLMEngine};
use dynamo_runtime::pipeline::{AsyncEngine, AsyncEngineContextProvider};
use dynamo_sidecar_testkit::{
    assert::terminal,
    bounded,
    control::{Controller, Event, OpenAction, RequestPlan, StreamAction, StreamFault, StreamPoint},
};
use futures::StreamExt;

use crate::lifecycle::healthy;
use crate::process::{Environment, ProcessFixture, outputs};
use crate::support::FixtureConfig;

fn hold_after_token() -> RequestPlan {
    RequestPlan {
        open: OpenAction::Continue,
        stream: Some(StreamFault {
            at: StreamPoint::TokenResponse(1),
            action: StreamAction::Continue,
            pause: true,
        }),
    }
}

pub async fn worker_cancel_and_consumer_drop_release_only_the_target<F: ProcessFixture>() {
    let env = Environment::new().await;
    let control = Controller::default();
    let mut peer = F::start(
        control.clone(),
        FixtureConfig {
            model: env.model.clone(),
            speedup_ratio: 0.1,
            ..Default::default()
        },
    )
    .await;
    let mut child = env.spawn::<F>(&peer.endpoint(), DisaggregationMode::Aggregated, 5);
    let router = env.ready("backend").await;

    for (id, explicit, waiting_headers) in [
        ("cancel-open", true, true),
        ("cancel-stream", true, false),
        ("drop-stream", false, false),
    ] {
        let plan = if waiting_headers {
            RequestPlan {
                open: OpenAction::Hold,
                stream: None,
            }
        } else {
            hold_after_token()
        };
        let target = control.request(id, plan);
        let request = env.request(id, 10_000);
        let context = request.context();
        let generation = router.generate(request);
        tokio::pin!(generation);
        let mut stream = bounded("target request ingress", async {
            tokio::select! {
                result = &mut generation => Some(result.unwrap()),
                _ = target.wait(Event::Received) => None,
            }
        })
        .await;
        bounded("target native acceptance", target.wait(Event::Received)).await;
        if !waiting_headers {
            if stream.is_none() {
                stream = Some(
                    bounded("target response headers", &mut generation)
                        .await
                        .unwrap(),
                );
            }
            let first = bounded("target first token", stream.as_mut().unwrap().next())
                .await
                .unwrap()
                .into_data()
                .unwrap()
                .unwrap();
            assert_eq!(first.token_ids.len(), 1);
            bounded("target stream checkpoint", target.wait(Event::Checkpoint)).await;
            peer.scheduler_active().await;
        }
        let other_id = format!("other-{id}");
        let other = control.request(&other_id, hold_after_token());
        let mut other_stream = router.generate(env.request(&other_id, 3)).await.unwrap();
        let other_first = bounded("independent first token", other_stream.next())
            .await
            .unwrap()
            .into_data()
            .unwrap()
            .unwrap();
        bounded(
            "independent stream checkpoint",
            other.wait(Event::Checkpoint),
        )
        .await;
        if explicit {
            context.stop_generating();
            let cancelled = if let Some(stream) = stream {
                outputs(stream).await
            } else {
                match bounded("cancel pending response headers", &mut generation).await {
                    Ok(stream) => outputs(stream).await,
                    Err(error) => {
                        assert!(
                            dynamo_runtime::error::match_error_chain(
                                error.as_ref(),
                                &[
                                    ErrorType::Cancelled,
                                    ErrorType::Backend(BackendError::Cancelled)
                                ],
                                &[]
                            ),
                            "{error}"
                        );
                        Vec::new()
                    }
                }
            };
            assert!(
                cancelled
                    .iter()
                    .filter_map(|output| output.as_ref().ok())
                    .all(|output| !matches!(
                        output.finish_reason,
                        Some(FinishReason::Stop | FinishReason::Length)
                    ))
            );
        } else {
            assert!(
                !context.is_stopped(),
                "consumer drop must not call explicit cancellation"
            );
            drop(stream);
        }
        bounded("target remote release", target.wait(Event::Dropped)).await;
        assert!(!other.reached(Event::Dropped));
        assert_eq!(other.tokens().len(), 1);
        other.release();
        let mut other_outputs = vec![Ok(other_first)];
        other_outputs.extend(outputs(other_stream).await);
        terminal(other_outputs, &other.tokens(), 4, FinishReason::Length);
        peer.scheduler_idle().await;
        healthy::<F>(&env, &router, &control, &format!("recovery-{id}")).await;
    }
    child.shutdown().await;
    env.withdrawn("backend", &router).await;
    peer.shutdown().await;
}

pub async fn sigterm_withdraws_worker_and_releases_active_native_request<F: ProcessFixture>() {
    let env = Environment::new().await;
    let control = Controller::default();
    let mut peer = F::start(
        control.clone(),
        FixtureConfig {
            model: env.model.clone(),
            speedup_ratio: 0.1,
            ..Default::default()
        },
    )
    .await;
    let mut child = env.spawn_with_grace::<F>(&peer.endpoint(), DisaggregationMode::Aggregated);
    let router = env.ready("backend").await;
    let handle = control.request("shutdown-active", hold_after_token());
    let mut stream = router
        .generate(env.request("shutdown-active", 10_000))
        .await
        .unwrap();
    bounded("shutdown first token", stream.next())
        .await
        .unwrap()
        .into_data()
        .unwrap()
        .unwrap();
    bounded("shutdown native checkpoint", handle.wait(Event::Checkpoint)).await;
    peer.scheduler_active().await;
    child.signal(libc::SIGTERM);
    env.withdrawn("backend", &router).await;
    assert!(child.is_running(), "withdrawal must precede process exit");
    assert!(
        !handle.reached(Event::Dropped),
        "withdrawal must precede engine cleanup"
    );
    peer.scheduler_active().await;
    let tail = outputs(stream).await;
    assert!(
        tail.iter()
            .filter_map(|item| item.as_ref().ok())
            .all(|item| !matches!(
                item.finish_reason,
                Some(FinishReason::Stop | FinishReason::Length)
            ))
    );
    bounded("shutdown remote release", handle.wait(Event::Dropped)).await;
    peer.scheduler_idle().await;
    assert!(child.exit().await.success(), "{}", child.logs());
    let independent = peer.engine().await;
    independent.start(0).await.unwrap();
    independent.cleanup().await.unwrap();
    peer.shutdown().await;
}
