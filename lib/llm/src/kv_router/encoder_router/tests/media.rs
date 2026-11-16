// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[tokio::test]
async fn generic_encoder_accepts_video_and_audio() {
    if isolated(concat!(
        module_path!(),
        "::generic_encoder_accepts_video_and_audio"
    ))
    .await
    {
        return;
    }
    let fixture = Fixture::new().await;
    fixture.router.set_required(true);
    for kind in ["video_url", "audio_url"] {
        let downstream = Arc::new(CaptureEngine::default());
        drop(
            fixture
                .router
                .generate(SingleIn::new(media_request(kind)), downstream.clone())
                .await
                .unwrap(),
        );
        let seen = downstream.request.lock().unwrap();
        let seen = seen.as_ref().unwrap();
        assert_eq!(
            seen.encoder_result.as_ref().unwrap()["worker_id"],
            fixture.ids[0]
        );
        assert!(seen.multi_modal_data.as_ref().unwrap().contains_key(kind));
    }
    fixture.shutdown().await;
}

#[tokio::test]
async fn encoder_errors_remain_typed_and_optional_requests_can_fall_inline() {
    if isolated(concat!(
        module_path!(),
        "::encoder_errors_remain_typed_and_optional_requests_can_fall_inline"
    ))
    .await
    {
        return;
    }
    let fixture = Fixture::new().await;
    let router = fixture.router.clone();
    let ids = fixture.ids.clone();
    router.set_required(true);
    let downstream = Arc::new(CaptureEngine::default());
    let error = router
        .generate(
            SingleIn::new(media_request("unsupported_url")),
            downstream.clone(),
        )
        .await
        .unwrap_err();
    assert_eq!(
        error.downcast_ref::<DynamoError>().unwrap().class(),
        ErrorClass::InvalidRequest
    );
    assert!(error.to_string().contains("Required encoder hop failed"));
    assert!(format!("{error:#}").contains("encoder rejected request"));
    assert!(downstream.request.lock().unwrap().is_none());
    let downstream = Arc::new(CaptureEngine::default());
    let mut invalid = multimodal_request();
    invalid.sampling_options.n = Some(2);
    let error = router
        .generate(SingleIn::new(invalid.clone()), downstream.clone())
        .await
        .unwrap_err();
    assert_eq!(
        error.downcast_ref::<DynamoError>().unwrap().class(),
        ErrorClass::InvalidRequest
    );
    assert!(downstream.request.lock().unwrap().is_none());
    for (n, expected, migratable) in [
        (4, dynamo_runtime::error::ErrorType::ResourceExhausted, true),
        (6, dynamo_runtime::error::ErrorType::Cancelled, false),
    ] {
        let mut request = multimodal_request();
        request.sampling_options.n = Some(n);
        let error = router
            .generate(SingleIn::new(request), downstream.clone())
            .await
            .unwrap_err();
        assert!(
            dynamo_runtime::error::match_error_chain(error.as_ref(), &[expected], &[]),
            "n={n}: {error:#}"
        );
        if n == 4 {
            let typed = error.downcast_ref::<DynamoError>().unwrap();
            assert_eq!(typed.class(), ErrorClass::CapacityExhausted);
            assert_eq!(typed.reason().as_str(), "capacity.worker_overloaded");
        }
        assert_eq!(
            crate::migration::is_migratable(error.as_ref()),
            migratable,
            "n={n}: {error:#}"
        );
        assert!(downstream.request.lock().unwrap().is_none());
    }
    let binding = router.binding.load_full().unwrap();
    binding.router.client.set_overloaded_instances(&[ids[0]]);
    let error = router
        .generate(SingleIn::new(multimodal_request()), downstream.clone())
        .await
        .unwrap_err();
    assert!(dynamo_runtime::error::match_error_chain(
        error.as_ref(),
        &[dynamo_runtime::error::ErrorType::ResourceExhausted],
        &[]
    ));
    assert!(downstream.request.lock().unwrap().is_none());
    binding.router.client.set_overloaded_instances(&[]);
    router.set_required(false);
    let response = router
        .generate(SingleIn::new(invalid), downstream.clone())
        .await
        .unwrap();
    drop(response);
    assert!(
        downstream
            .request
            .lock()
            .unwrap()
            .as_ref()
            .unwrap()
            .encoder_result
            .is_none()
    );

    // A worker-unavailable stream quarantines its producer. Check it last so
    // the real transport's health tracking cannot poison subsequent fixtures.
    router.set_required(true);
    let downstream = Arc::new(CaptureEngine::default());
    let mut request = multimodal_request();
    request.sampling_options.n = Some(5);
    let error = router
        .generate(SingleIn::new(request), downstream.clone())
        .await
        .unwrap_err();
    assert!(
        dynamo_runtime::error::match_error_chain(
            error.as_ref(),
            &[dynamo_runtime::error::ErrorType::WorkerUnavailable],
            &[]
        ),
        "{error:#}"
    );
    assert!(crate::migration::is_migratable(error.as_ref()), "{error:#}");
    assert!(downstream.request.lock().unwrap().is_none());

    drop(router);
    fixture.shutdown().await;
}

#[tokio::test]
async fn optional_encoder_skips_unsupported_media_as_a_whole() {
    if isolated(concat!(
        module_path!(),
        "::optional_encoder_skips_unsupported_media_as_a_whole"
    ))
    .await
    {
        return;
    }
    let mut card = ModelDeploymentCard::with_name_only("model");
    card.runtime_config
        .runtime_data
        .insert(ENCODER_SUPPORTED_MODALITIES.into(), json!(["image_url"]));
    let fixture = Fixture::with_card(card).await;
    let router = &fixture.router;
    for required in [false, true] {
        router.set_required(required);
        for kind in ["video_url", "audio_url"] {
            let mut input = media_request(kind);
            // Mixed inputs must stay inline as a whole, never partially encoded.
            input.multi_modal_data.as_mut().unwrap().insert(
                "image_url".into(),
                multimodal_request()
                    .multi_modal_data
                    .unwrap()
                    .remove("image_url")
                    .unwrap(),
            );
            let downstream = Arc::new(CaptureEngine::default());
            let response = router
                .generate(SingleIn::new(input), downstream.clone())
                .await;
            drop(response.unwrap());
            assert_eq!(
                downstream
                    .request
                    .lock()
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .encoder_result
                    .is_some(),
                required,
                "optional hints must not reject required traffic accepted by the producer",
            );
        }
    }

    fixture.shutdown().await;
}
