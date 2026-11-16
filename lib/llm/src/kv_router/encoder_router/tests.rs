// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    collections::{HashMap, HashSet},
    sync::Mutex,
    time::Duration,
};

use futures::{FutureExt, stream};
use serde_json::json;

use dynamo_runtime::{
    DistributedRuntime, Runtime,
    discovery::EventTransportKind,
    distributed::{DiscoveryBackend, DistributedConfig, RequestPlaneMode},
    engine::{AsyncEngine, AsyncEngineContextProvider},
    pipeline::{Error, ResponseStream, context::Controller, network::Ingress},
    storage::kv,
};

use crate::discovery::CommittedWorkerSetTarget;
use crate::model_card::ModelDeploymentCard;
use crate::protocols::common::preprocessor::{
    ENCODER_RESULT_RELEASE_SUPPORTED, MultimodalData, RELEASE_ENCODER_RESULT,
};
use crate::protocols::common::{OutputOptions, SamplingOptions, StopConditions};

use super::*;

mod cleanup;
mod media;
mod membership;
mod support;

use support::*;

#[tokio::test]
async fn binding_availability_tracks_local_inhibition_and_cancellation() {
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
    use std::collections::HashSet;

    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let endpoint = distributed
        .namespace("encoder-availability")
        .unwrap()
        .component("workers")
        .unwrap()
        .endpoint("generate");
    let router = EncoderRouter::disabled();
    assert!(
        router
            .available_worker_ids_for("legacy", &endpoint.id())
            .is_none()
    );
    let binding = Arc::new(
        EncoderRouter::build(
            WorkerSetTarget::Legacy(endpoint.clone()),
            CancellationToken::new(),
        )
        .await
        .unwrap(),
    );
    binding
        .router
        .client
        .override_discovered_instances(vec![1, 2]);
    router.binding.store(Some(binding.clone()));
    router
        .lifecycle
        .store(EncoderLifecycleState::Active as u8, Ordering::Release);
    assert_eq!(
        router.available_worker_ids_for("legacy", &endpoint.id()),
        Some(HashSet::from([1, 2]))
    );
    binding.router.client.report_instance_down(1);
    assert_eq!(
        router.available_worker_ids_for("legacy", &endpoint.id()),
        Some(HashSet::from([2]))
    );
    let mut other = endpoint.id();
    other.name = "other".into();
    assert!(router.available_worker_ids_for("legacy", &other).is_none());
    router.cancel_token.cancel();
    assert_eq!(
        router.available_worker_ids_for("legacy", &endpoint.id()),
        Some(HashSet::new())
    );
    runtime.shutdown();
}

#[tokio::test]
async fn binding_availability_waits_for_rebind_publication() {
    use dynamo_runtime::{DistributedRuntime, Runtime, distributed::DistributedConfig};
    use std::{sync::mpsc, time::Duration};

    let runtime = Runtime::from_current().unwrap();
    let distributed = DistributedRuntime::new(runtime.clone(), DistributedConfig::process_local())
        .await
        .unwrap();
    let component = distributed
        .namespace("encoder-rebind")
        .unwrap()
        .component("workers")
        .unwrap();
    let original = component.endpoint("original");
    let replacement = Arc::new(
        EncoderRouter::build(
            WorkerSetTarget::Legacy(component.endpoint("replacement")),
            CancellationToken::new(),
        )
        .await
        .unwrap(),
    );
    let router = EncoderRouter::disabled();
    let binding = Arc::new(
        EncoderRouter::build(
            WorkerSetTarget::Legacy(original.clone()),
            CancellationToken::new(),
        )
        .await
        .unwrap(),
    );
    binding.router.client.override_discovered_instances(vec![1]);
    router.binding.store(Some(binding));
    router
        .lifecycle
        .store(EncoderLifecycleState::Active as u8, Ordering::Release);
    let mut target = router.target.lock();
    *target = Some(WorkerSetTargetId::Legacy(original.id()));
    let (started_tx, started_rx) = mpsc::channel();
    let (result_tx, result_rx) = mpsc::channel();
    let reader_router = router.clone();
    let reader = std::thread::spawn(move || {
        started_tx.send(()).unwrap();
        result_tx
            .send(reader_router.available_worker_ids_for("legacy", &original.id()))
            .unwrap();
    });
    started_rx.recv_timeout(Duration::from_secs(5)).unwrap();
    assert!(matches!(
        result_rx.recv_timeout(Duration::from_millis(100)),
        Err(mpsc::RecvTimeoutError::Timeout)
    ));
    *target = Some(replacement.target_id.clone());
    router.binding.store(Some(replacement));
    router
        .lifecycle
        .store(EncoderLifecycleState::Active as u8, Ordering::Release);
    drop(target);
    assert_eq!(
        result_rx.recv_timeout(Duration::from_secs(5)).unwrap(),
        None
    );
    reader.join().unwrap();
    runtime.shutdown();
}

#[tokio::test]
async fn pending_activation_does_not_keep_router_alive() {
    let router = EncoderRouter::new("model".into(), "namespace".into());
    let weak = Arc::downgrade(&router);

    drop(router);

    assert!(weak.upgrade().is_none());
}

#[tokio::test]
async fn required_encoder_pending_error_is_migratable() {
    for state in [
        EncoderLifecycleState::Pending,
        EncoderLifecycleState::Active,
    ] {
        let router = EncoderRouter::disabled();
        router.set_required(true);
        router.lifecycle.store(state as u8, Ordering::Release);
        let downstream = Arc::new(CaptureEngine::default());
        let error = router
            .generate(SingleIn::new(multimodal_request()), downstream.clone())
            .await
            .unwrap_err();
        assert!(dynamo_runtime::error::match_error_chain(
            error.as_ref(),
            &[dynamo_runtime::error::ErrorType::WorkerUnavailable],
            &[]
        ));
        assert!(crate::migration::is_migratable(error.as_ref()));
        assert!(downstream.request.lock().unwrap().is_none());
    }
}

#[tokio::test]
async fn query_annotation_only_bypasses_encoder_for_kv_lookup() {
    for state in [
        EncoderLifecycleState::Pending,
        EncoderLifecycleState::Active,
        EncoderLifecycleState::Unavailable,
    ] {
        for mode in [RouterMode::KV, RouterMode::RoundRobin, RouterMode::Random] {
            let mut router = EncoderRouter::disabled();
            Arc::get_mut(&mut router).unwrap().router_mode = mode;
            router.set_required(true);
            router.lifecycle.store(state as u8, Ordering::Release);
            for kind in ["image_url", "video_url", "audio_url"] {
                let mut request = media_request(kind);
                request.annotations.push("query_instance_id:".into());
                let downstream = Arc::new(CaptureEngine::default());
                let result = router
                    .generate(SingleIn::new(request), downstream.clone())
                    .await;
                if !mode.is_kv_routing() {
                    assert!(result.is_err(), "executing routes must require an encoder");
                    assert!(downstream.request.lock().unwrap().is_none());
                    continue;
                }
                drop(result.unwrap());
                let seen = downstream.request.lock().unwrap();
                assert!(seen.as_ref().unwrap().encoder_result.is_none());
                assert!(
                    seen.as_ref()
                        .unwrap()
                        .multi_modal_data
                        .as_ref()
                        .unwrap()
                        .contains_key(kind)
                );
            }
        }
    }
}

#[tokio::test]
async fn unavailable_required_encoder_rejects_media_but_passes_text() {
    let router = EncoderRouter::disabled();
    router.set_required(true);
    for modality in ["image_url", "video_url", "audio_url"] {
        let downstream = Arc::new(CaptureEngine::default());
        let error = router
            .generate(SingleIn::new(media_request(modality)), downstream.clone())
            .await
            .unwrap_err();
        assert!(dynamo_runtime::error::match_error_chain(
            error.as_ref(),
            &[dynamo_runtime::error::ErrorType::WorkerUnavailable],
            &[],
        ));
        assert!(downstream.request.lock().unwrap().is_none());
    }
    let downstream = Arc::new(CaptureEngine::default());
    let mut text = multimodal_request();
    text.multi_modal_data = None;
    router
        .generate(SingleIn::new(text), downstream.clone())
        .await
        .unwrap();
    assert!(downstream.request.lock().unwrap().is_some());
}

#[tokio::test]
async fn encoder_failure_falls_back_to_downstream() {
    let router = EncoderRouter::disabled();
    router
        .lifecycle
        .store(EncoderLifecycleState::Active as u8, Ordering::Release);
    let downstream = Arc::new(CaptureEngine::default());
    let next: ServerStreamingEngine<PreprocessedRequest, Annotated<LLMEngineOutput>> =
        downstream.clone();

    let _response = router
        .generate(SingleIn::new(multimodal_request()), next)
        .await
        .expect("encoder failure should fall through to downstream");

    let request = downstream
        .request
        .lock()
        .unwrap()
        .take()
        .expect("downstream must receive the original request");
    assert!(request.encoder_result.is_none());
    assert!(request.multi_modal_data.is_some());
}

#[tokio::test]
async fn consumes_object_shaped_encode_terminal() {
    let output =
        LLMEngineOutput::encode_terminal(json!({"schema_version": 1}).as_object().unwrap().clone());
    let (result, _) =
        EncoderRouter::consume_encode_stream(stream_of(vec![Annotated::from_data(output)]), None)
            .await
            .unwrap();
    assert_eq!(result, json!({"schema_version": 1}));
}

#[tokio::test]
async fn rejects_terminal_without_encoder_result() {
    let result = EncoderRouter::consume_encode_stream(
        stream_of(vec![Annotated::from_data(LLMEngineOutput::stop())]),
        None,
    )
    .await;
    assert!(result.is_err());
}
