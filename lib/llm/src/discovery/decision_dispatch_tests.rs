// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use dynamo_runtime::{
    DistributedRuntime, Runtime,
    component::{Client, Instance},
    engine::{AsyncEngine, AsyncEngineContextProvider, Data},
    pipeline::{
        AddressedRequest, Context, Error, ManyIn, ManyOut, PushRouter, PushRouterOptions,
        ResponseStream, RouterMode, SingleIn, StreamingDispatch,
    },
    protocols::maybe_error::MaybeError,
};
use serde::{Deserialize, Serialize};
use std::sync::atomic::{AtomicUsize, Ordering};
use tokio::sync::watch;

struct RecordingTransport(Arc<AtomicUsize>);

pub(crate) async fn assert_native_dispatch(client: &Client, context: &DecisionContext) {
    use crate::protocols::{
        Annotated,
        decision_read::{DecisionReadRequest, DecisionReadResponse},
    };
    let calls = Arc::new(AtomicUsize::new(0));
    let router = PushRouter::<DecisionReadRequest, Annotated<DecisionReadResponse>>::from_client_with_options(
        client.clone(), RouterMode::RoundRobin,
        PushRouterOptions { dispatch: Some(crate::discovery::DecisionDispatch::with_dispatch(Arc::new(RecordingTransport(calls.clone())), crate::discovery::GuardPolicy::Required)), ..crate::kv_router::plugins::push_router_options() },
    ).await.unwrap();
    let mut request = Context::new(DecisionReadRequest {
        decision_read: crate::protocols::decision_native::NativeDecisionRequestV1 {
            protocol_version: 1,
            profile: context.profile(),
            model: "model".into(),
            input: serde_json::json!("state"),
            questions: vec![],
            temperature: 1.0,
            cache_salt: "salt".into(),
            max_input_tokens: 100,
            max_tokenization_work: 100,
            max_encoding_bytes: 100,
        },
    });
    context.attach_dispatch_guard(&mut request);
    let _response = router.generate(request).await.unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
}

#[test]
fn decision_branches_share_one_owned_dispatch_fence() {
    let (ordinary, admitted) = watch::channel(vec![1]);
    let (publisher, fleet) =
        test_support::fleet(Some((DecisionExecutionProfile::VllmCausalV1, 26)), vec![1]);
    let context = test_support::context_for_test(fleet, admitted);
    let receivers = (ordinary.receiver_count(), publisher.receiver_count());
    let requests: Vec<_> = (0..3)
        .map(|_| {
            let mut request = Context::new(());
            context.attach_dispatch_guard(&mut request);
            request
        })
        .collect();
    assert_eq!(
        (ordinary.receiver_count(), publisher.receiver_count()),
        receivers,
        "branch guards must share the request's owned fence, not clone its watches"
    );
    assert_eq!(requests.len(), 3);
    context.validate_dispatch(Some(1)).unwrap();
}

#[async_trait::async_trait]
impl<T, U> StreamingDispatch<T, U> for RecordingTransport
where
    T: Data + Serialize,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    async fn generate(&self, request: SingleIn<AddressedRequest<T>>) -> Result<ManyOut<U>, Error> {
        self.0.fetch_add(1, Ordering::SeqCst);
        Ok(ResponseStream::new(
            Box::pin(futures::stream::empty()),
            request.context(),
        ))
    }

    async fn generate_bidirectional(
        &self,
        _: Instance,
        _: String,
        _: ManyIn<T>,
    ) -> Result<ManyOut<U>, Error> {
        unreachable!("decisions use unary dispatch")
    }
}

async fn assert_stalled_profile_transition<T, U>(
    client: &Client,
    context: &DecisionContext,
    qualification: &watch::Sender<DecisionAvailability>,
    worker: u64,
    next: DecisionQualification,
    payload: T,
) where
    T: Data + Serialize + Clone,
    U: Data + for<'de> Deserialize<'de> + MaybeError,
{
    context.validate_dispatch(Some(worker)).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let router = Arc::new(
        PushRouter::<T, U>::from_client_with_options(
            client.clone(),
            RouterMode::Direct,
            PushRouterOptions {
                dispatch: Some(crate::discovery::DecisionDispatch::with_dispatch(
                    Arc::new(RecordingTransport(calls.clone())),
                    crate::discovery::GuardPolicy::Required,
                )),
                fault_detection: true,
                ..PushRouterOptions::default()
            },
        )
        .await
        .unwrap(),
    );
    let mut request = Context::new(payload.clone());
    context.attach_dispatch_guard(&mut request);
    // Routing/preprocessing can transfer the envelope while retaining its local registry.
    let (wire, envelope) = request.transfer(());
    let request = envelope.map(|()| wire);
    let (resume, paused) = tokio::sync::oneshot::channel();
    let pending_router = router.clone();
    let pending = tokio::spawn(async move {
        paused.await.unwrap();
        pending_router.direct(request, worker).await.err()
    });
    qualification.send_replace(crate::discovery::DecisionAvailability::Unadvertised);
    qualification.send_replace(DecisionAvailability::qualified(next, vec![worker]));
    assert!(client.is_instance_live(worker));
    resume.send(()).unwrap();
    let result = pending.await.unwrap();
    assert_eq!(
        calls.load(Ordering::SeqCst),
        0,
        "stale qualification must never reach transport"
    );
    let error = result.expect("stale qualification must fail before transport");
    assert!(dynamo_runtime::error::match_error_chain(
        error.as_ref(),
        &[dynamo_runtime::error::ErrorType::Unavailable],
        &[]
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    assert!(
        client.instance_ids_avail().contains(&worker),
        "stale request-scoped fences must not inhibit the ordinary worker"
    );

    let mut fresh = context.clone();
    (fresh.profile, fresh.max_candidates) = next;
    fresh.fence = Arc::new(DispatchFence {
        fleet: context.fence.fleet.clone(),
        admission: qualification.borrow().admission().unwrap().clone(),
        admitted_ids: context.fence.admitted_ids.clone(),
        cancellation: context.fence.cancellation.clone(),
    });
    let mut request = Context::new(payload.clone());
    fresh.attach_dispatch_guard(&mut request);
    assert!(router.direct(request, worker).await.is_ok());
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(
        router
            .direct(Context::new(payload.clone()), worker)
            .await
            .is_err(),
        "private transport must reject a missing guard"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    let ordinary = PushRouter::<T, U>::from_client_with_options(
        client.clone(),
        RouterMode::Direct,
        PushRouterOptions {
            dispatch: Some(crate::discovery::DecisionDispatch::with_dispatch(
                Arc::new(RecordingTransport(calls.clone())),
                crate::discovery::GuardPolicy::Optional,
            )),
            fault_detection: false,
            ..Default::default()
        },
    )
    .await
    .unwrap();
    assert!(
        ordinary.direct(Context::new(payload), worker).await.is_ok(),
        "ordinary requests need no private decision guard"
    );
    assert_eq!(calls.load(Ordering::SeqCst), 2);
}

#[tokio::test]
async fn decision_dispatch_fences_reopened_ids_for_token_and_native_profiles() {
    if crate::test_utils::run_isolated(
        concat!(
            module_path!(),
            "::decision_dispatch_fences_reopened_ids_for_token_and_native_profiles"
        ),
        &[],
    ) {
        return;
    }
    use crate::decision_profile::DecisionExecutionProfile as Profile;
    use crate::protocols::{
        Annotated,
        common::{llm_backend::LLMEngineOutput, preprocessor::PreprocessedRequest},
        decision_read::{DecisionReadRequest, DecisionReadResponse},
    };
    let runtime = Runtime::from_current().unwrap();
    let drt = DistributedRuntime::new(
        runtime.clone(),
        dynamo_runtime::distributed::DistributedConfig::process_local(),
    )
    .await
    .unwrap();
    let endpoint = drt
        .namespace("decision-dispatch")
        .unwrap()
        .component("worker")
        .unwrap()
        .endpoint("generate");
    let discovered = endpoint.client().await.unwrap();
    endpoint.register_endpoint_instance().await.unwrap();
    let worker = discovered.wait_for_instances().await.unwrap()[0].id();
    let (_ordinary, admitted) = watch::channel(vec![worker]);
    let (qualifications, fleet) = test_support::fleet(None, vec![worker]);
    let client = discovered
        .with_admitted_instances_and_cancellation(admitted.clone(), runtime.child_token());
    for (initial, next) in [
        ((Profile::SglangSerialV1, 255), (Profile::VllmCausalV1, 26)),
        ((Profile::SglangSerialV1, 255), (Profile::VllmLiquidV1, 26)),
        ((Profile::VllmLiquidV1, 26), (Profile::VllmDiffusionV1, 26)),
        ((Profile::VllmLiquidV1, 26), (Profile::VllmLiquidV1, 3)),
        ((Profile::VllmLiquidV1, 26), (Profile::VllmLiquidV1, 26)),
    ] {
        test_support::publish(&qualifications, Some(initial), vec![worker]);
        let context = DecisionContext {
            canonical_model: "model".into(),
            card: ModelDeploymentCard::with_name_only("model"),
            fence: Arc::new(DispatchFence {
                admitted_ids: admitted.clone(),
                cancellation: runtime.child_token(),
                fleet: fleet.clone(),
                admission: qualifications.borrow().admission().unwrap().clone(),
            }),
            profile: initial.0,
            max_candidates: initial.1,
        };
        if matches!(initial.0.kind(), DecisionProfileKind::Token(_)) {
            let payload = PreprocessedRequest::builder()
                .model("model".into())
                .token_ids(vec![1])
                .stop_conditions(Default::default())
                .sampling_options(Default::default())
                .output_options(Default::default())
                .build()
                .unwrap();
            assert_stalled_profile_transition::<_, Annotated<LLMEngineOutput>>(
                &client,
                &context,
                &qualifications,
                worker,
                next,
                payload,
            )
            .await;
        } else {
            let payload = DecisionReadRequest {
                decision_read: crate::protocols::decision_native::NativeDecisionRequestV1 {
                    protocol_version: 1,
                    profile: initial.0,
                    model: "model".into(),
                    input: serde_json::json!("state"),
                    questions: vec![],
                    temperature: 1.0,
                    cache_salt: "salt".into(),
                    max_input_tokens: 100,
                    max_tokenization_work: 100,
                    max_encoding_bytes: 100,
                },
            };
            assert_stalled_profile_transition::<_, Annotated<DecisionReadResponse>>(
                &client,
                &context,
                &qualifications,
                worker,
                next,
                payload,
            )
            .await;
        }
    }
    runtime.shutdown();
}
