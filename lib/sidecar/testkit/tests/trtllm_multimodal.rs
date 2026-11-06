// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::atomic::Ordering;
use std::time::Duration;

use dynamo_backend_common::testing::mock_context;
use dynamo_backend_common::{
    FinishReason, GenerateContext, LLMEngine, MultimodalData, PreprocessedRequest,
};
use dynamo_sidecar_common::EngineBootstrapResult;
use dynamo_sidecar_testkit::bounded;
use dynamo_sidecar_testkit::control::{Controller, RequestPlan};
use dynamo_sidecar_testkit::fixtures::{collect, request};
use dynamo_trtllm_sidecar::TrtllmSidecarEngine;
use dynamo_trtllm_sidecar::proto as pb;

#[allow(dead_code)]
mod support;
use support::trtllm::{Adapter, Fixture};
use support::{FixtureConfig, ProcessFixture, SidecarFixture};

fn image_request() -> PreprocessedRequest {
    let mut request = request("mocker-model", vec![11, 22, 33], 1);
    request.stop_conditions.max_tokens = None;
    request.multi_modal_data = Some(HashMap::from([(
        "image_url".into(),
        vec![
            MultimodalData::RawUrl("https://example.com/first.png".into()),
            MultimodalData::RawUrl("data:image/png;base64,aW1hZ2U=".into()),
        ],
    )]));
    request.multi_modal_uuids = Some(HashMap::from([(
        "image_url".into(),
        vec![Some("first".into()), None],
    )]));
    request
}

fn video_request() -> PreprocessedRequest {
    let mut req = image_request();
    req.multi_modal_data = Some(HashMap::from([(
        "video_url".into(),
        vec![
            MultimodalData::RawUrl("https://example.com/first.mp4".into()),
            MultimodalData::RawUrl("data:video/mp4;base64,dmlkZW8=".into()),
        ],
    )]));
    req.multi_modal_uuids = Some(HashMap::from([(
        "video_url".into(),
        vec![Some("first-video".into()), None],
    )]));
    req
}

fn finished() -> pb::GenerateResponse {
    pb::GenerateResponse {
        event: Some(pb::generate_response::Event::Finished(
            pb::GenerationFinished {
                output_index: Some(0),
                reason: pb::FinishReason::Stop as i32,
                ..Default::default()
            },
        )),
        usage: Some(pb::Usage {
            prompt_tokens: 131,
            total_tokens: 131,
            ..Default::default()
        }),
        ..Default::default()
    }
}

fn advertise_media(fixture: &Fixture, modalities: &[&str]) {
    *fixture.discovery.model_info.lock().unwrap() = Some(pb::ModelInfo {
        model_id: "mocker-model".into(),
        served_model_name: "mocker-model".into(),
        max_context_length: Some(32768),
        supports_multimodal: Some(true),
        supports_text_input: Some(true),
        supports_token_ids_input: Some(true),
        extra: Some(
            dynamo_sidecar_common::json_to_struct(
                serde_json::json!({"trtllm_supported_modalities": modalities}),
                "test capabilities",
            )
            .unwrap(),
        ),
        ..Default::default()
    });
}

async fn wait_for_calls(fixture: &Fixture, count: usize) {
    bounded("Control discovery reached", async {
        while fixture.discovery.calls.load(Ordering::SeqCst) < count {
            tokio::task::yield_now().await;
        }
    })
    .await;
}

#[tokio::test]
async fn images_preserve_order_uuids_and_engine_expanded_usage() {
    let control = Controller::<Adapter>::default();
    let mut fixture = Fixture::start(control.clone(), FixtureConfig::default()).await;
    advertise_media(&fixture, &["image"]);
    let engine = fixture.engine().await;
    engine.start(0).await.unwrap();
    let ctx = mock_context();
    let handle = control.request(ctx.id(), RequestPlan::default());
    fixture.respond(ctx.id(), vec![finished()]);
    let outputs = collect(&engine, image_request(), GenerateContext::new(ctx, None)).await;
    let output = outputs.last().unwrap().as_ref().unwrap();
    assert_eq!(output.finish_reason, Some(FinishReason::Stop));
    assert_eq!(output.completion_usage.as_ref().unwrap().prompt_tokens, 131);
    let native = handle.native_request().unwrap();
    assert_eq!(native.stopping.unwrap().max_tokens, None);
    assert_eq!(native.media.len(), 2);
    assert!(
        native
            .media
            .iter()
            .all(|item| item.modality == pb::Modality::Image as i32)
    );
    assert_eq!(native.media[0].uuid, "first");
    assert_eq!(native.media[1].uuid, "");
    assert_eq!(
        native.media[0].source,
        Some(pb::media_item::Source::Url(
            "https://example.com/first.png".into()
        ))
    );
    assert_eq!(
        native.media[1].source,
        Some(pb::media_item::Source::DataUri(
            "data:image/png;base64,aW1hZ2U=".into()
        ))
    );
    engine.cleanup().await.unwrap();
    fixture.shutdown().await;
}

#[tokio::test]
async fn formatter_regrouped_tool_media_reaches_openengine_in_rendered_order() {
    use dynamo_llm::common::checked_file::CheckedFile;
    use dynamo_llm::model_card::{ModelDeploymentCard, ModelInfoType, TokenizerKind};
    use dynamo_llm::preprocessor::OpenAIPreprocessor;
    use dynamo_llm::protocols::openai::chat_completions::NvCreateChatCompletionRequest;

    let model_path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../llm/tests/data/sample-models/mock-llama-3.1-8b-instruct");
    let mut card = ModelDeploymentCard::load_from_disk(model_path, None).unwrap();
    let root = tempfile::tempdir().unwrap();
    let config = root.path().join("config.json");
    std::fs::write(
        &config,
        r#"{"architectures":[],"model_type":"","eos_token_id":128009}"#,
    )
    .unwrap();
    card.model_info = Some(ModelInfoType::HfConfigJson(
        CheckedFile::from_disk(&config).unwrap(),
    ));
    let tokenizer = root.path().join("tokenizer.json");
    std::fs::write(&tokenizer, r#"{"version":"1.0","truncation":null,"padding":null,"added_tokens":[],"normalizer":null,"pre_tokenizer":{"type":"Whitespace"},"post_processor":null,"decoder":null,"model":{"type":"WordLevel","vocab":{"[UNK]":0,"Result":1,"A":2,"B":3},"unk_token":"[UNK]"}}"#).unwrap();
    card.tokenizer = Some(TokenizerKind::HfTokenizerJson(
        CheckedFile::from_disk(&tokenizer).unwrap(),
    ));
    card.display_name = "DeepSeek-V4.1-Flash".into();
    let preprocessor = OpenAIPreprocessor::new(card).unwrap();
    // The real formatter regroups tool replies by call order, not arrival order.
    let request: NvCreateChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model":"mocker-model", "max_tokens":16, "messages":[
            {"role":"assistant","tool_calls":[
                {"id":"a","type":"function","function":{"name":"image","arguments":"{}"}},
                {"id":"b","type":"function","function":{"name":"image","arguments":"{}"}}
            ]},
            {"role":"tool","tool_call_id":"b","content":[
                {"type":"image_url","image_url":{"url":"https://example.com/b.png"},"uuid":"b"},
                {"type":"text","text":"Result B"}
            ]},
            {"role":"tool","tool_call_id":"a","content":[
                {"type":"image_url","image_url":{"url":"https://example.com/a.png"},"uuid":"a"},
                {"type":"text","text":"Result A"}
            ]}
        ]
    }))
    .unwrap();
    let (mut prepared, _, _) = preprocessor
        .preprocess_request(&request, None)
        .await
        .unwrap();
    let prompt = prepared.extra_args.as_ref().unwrap()["formatted_prompt"]
        .as_str()
        .unwrap();
    assert!(prompt.find("Result A").unwrap() < prompt.find("Result B").unwrap());
    assert_eq!(prompt.matches("<｜deepseek_image｜>").count(), 2);
    let expected_tokens = prepared.token_ids.clone();
    assert!(!expected_tokens.is_empty());
    prepared.model = "mocker-model".into();

    let control = Controller::<Adapter>::default();
    let mut fixture = Fixture::start(control.clone(), FixtureConfig::default()).await;
    advertise_media(&fixture, &["image"]);
    let engine = fixture.engine().await;
    engine.start(0).await.unwrap();
    let ctx = mock_context();
    let handle = control.request(ctx.id(), RequestPlan::default());
    fixture.respond(ctx.id(), vec![finished()]);
    let outputs = collect(&engine, prepared, GenerateContext::new(ctx, None)).await;
    assert_eq!(
        outputs.last().unwrap().as_ref().unwrap().finish_reason,
        Some(FinishReason::Stop)
    );
    let native = handle.native_request().unwrap();
    assert_eq!(
        native.input,
        Some(pb::generate_request::Input::TokenIds(pb::TokenIds {
            ids: expected_tokens.to_vec()
        }))
    );
    assert_eq!(native.media.len(), 2);
    for (item, id) in native.media.iter().zip(["a", "b"]) {
        assert_eq!(item.modality, pb::Modality::Image as i32);
        assert_eq!(item.uuid, id);
        assert_eq!(
            item.source,
            Some(pb::media_item::Source::Url(format!(
                "https://example.com/{id}.png"
            )))
        );
    }
    engine.cleanup().await.unwrap();
    fixture.shutdown().await;
}

#[tokio::test]
async fn both_media_capability_signals_are_required() {
    for video in [false, true] {
        for (portable, advertised) in [(true, false), (false, true)] {
            let control = Controller::<Adapter>::default();
            let mut fixture = Fixture::start(control.clone(), FixtureConfig::default()).await;
            let mut modalities = vec![if video { "image" } else { "video" }];
            if advertised {
                modalities.push(if video { "video" } else { "image" });
            }
            advertise_media(&fixture, &modalities);
            fixture
                .discovery
                .model_info
                .lock()
                .unwrap()
                .as_mut()
                .unwrap()
                .supports_multimodal = Some(portable);
            let engine = fixture.engine().await;
            engine.start(0).await.unwrap();
            let ctx = mock_context();
            let handle = control.request(ctx.id(), RequestPlan::default());
            assert!(
                engine
                    .generate(
                        if video {
                            video_request()
                        } else {
                            image_request()
                        },
                        GenerateContext::new(ctx, None)
                    )
                    .await
                    .is_err()
            );
            assert!(handle.native_request().is_none());
            engine.cleanup().await.unwrap();
            fixture.shutdown().await;
        }
    }
}

#[tokio::test]
async fn videos_recover_discovery_and_preserve_sources_uuids_and_usage() {
    let control = Controller::<Adapter>::default();
    let mut fixture = Fixture::start(control.clone(), FixtureConfig::default()).await;
    fixture.discovery.unavailable.store(true, Ordering::SeqCst);
    advertise_media(&fixture, &["video"]);
    let mut startup = tokio::spawn(bootstrap(&fixture, Some(32768), 5));
    wait_for_calls(&fixture, 2).await;
    assert!(!startup.is_finished());
    fixture.discovery.unavailable.store(false, Ordering::SeqCst);
    let (engine, _) = bounded("recover video startup discovery", &mut startup)
        .await
        .unwrap()
        .unwrap();
    engine.start(0).await.unwrap();
    let calls = fixture.discovery.calls.load(Ordering::SeqCst);
    fixture.discovery.unavailable.store(true, Ordering::SeqCst);
    for _ in 0..2 {
        let ctx = mock_context();
        let handle = control.request(ctx.id(), RequestPlan::default());
        fixture.respond(ctx.id(), vec![finished()]);
        let outputs = collect(&engine, video_request(), GenerateContext::new(ctx, None)).await;
        let output = outputs.last().unwrap().as_ref().unwrap();
        assert_eq!(output.finish_reason, Some(FinishReason::Stop));
        assert_eq!(output.completion_usage.as_ref().unwrap().prompt_tokens, 131);
        let native = handle.native_request().unwrap();
        assert_eq!(native.stopping.unwrap().max_tokens, None);
        assert_eq!(native.media.len(), 2);
        assert!(
            native
                .media
                .iter()
                .all(|item| item.modality == pb::Modality::Video as i32)
        );
        assert_eq!(native.media[0].uuid, "first-video");
        assert_eq!(native.media[1].uuid, "");
        assert_eq!(
            native.media[0].source,
            Some(pb::media_item::Source::Url(
                "https://example.com/first.mp4".into()
            ))
        );
        assert_eq!(
            native.media[1].source,
            Some(pb::media_item::Source::DataUri(
                "data:video/mp4;base64,dmlkZW8=".into()
            ))
        );
    }
    assert_eq!(fixture.discovery.calls.load(Ordering::SeqCst), calls);
    engine.cleanup().await.unwrap();
    fixture.shutdown().await;
}

fn bootstrap(
    fixture: &Fixture,
    context_length: Option<u32>,
    deadline_secs: u32,
) -> impl std::future::Future<Output = EngineBootstrapResult<TrtllmSidecarEngine>> + use<> {
    let mut args = vec![
        "dynamo-trtllm-sidecar".into(),
        "--grpc-endpoint".into(),
        fixture.endpoint(),
        "--model-path".into(),
        "mocker-model".into(),
        "--grpc-startup-deadline-secs".into(),
        deadline_secs.to_string(),
    ];
    if let Some(context_length) = context_length {
        args.extend(["--context-length".into(), context_length.to_string()]);
    }
    TrtllmSidecarEngine::try_from_args_async(args).unwrap().1
}

#[tokio::test]
async fn startup_recovers_media_capabilities_before_registration() {
    for omit_window in [false, true] {
        let control = Controller::<Adapter>::default();
        let mut fixture = Fixture::start(control.clone(), FixtureConfig::default()).await;
        advertise_media(&fixture, &["image", "video"]);
        fixture
            .discovery
            .omit_context_length
            .store(omit_window, Ordering::SeqCst);
        fixture.discovery.unavailable.store(true, Ordering::SeqCst);
        let mut startup = tokio::spawn(bootstrap(&fixture, Some(32768), 5));
        wait_for_calls(&fixture, 2).await;
        assert!(
            !startup.is_finished(),
            "unresolved capabilities must prevent registration"
        );
        fixture.discovery.unavailable.store(false, Ordering::SeqCst);
        let (engine, _) = bounded("recover startup discovery", &mut startup)
            .await
            .unwrap()
            .unwrap();
        let config = engine.start(0).await.unwrap();
        assert_eq!(config.llm.unwrap().context_length, Some(32768));
        let calls = fixture.discovery.calls.load(Ordering::SeqCst);
        // Serving must use startup metadata, including image support when the
        // engine supplied no window. No Generate may rediscover capabilities.
        fixture.discovery.unavailable.store(true, Ordering::SeqCst);
        for req in [image_request(), video_request()] {
            let ctx = mock_context();
            let handle = control.request(ctx.id(), RequestPlan::default());
            if omit_window {
                let error = engine
                    .generate(req, GenerateContext::new(ctx, None))
                    .await
                    .err()
                    .expect("media still needs a reported engine window");
                assert!(
                    error.to_string().contains("--context-length to match"),
                    "{error}"
                );
                assert!(handle.native_request().is_none());
            } else {
                fixture.respond(ctx.id(), vec![finished()]);
                let outputs = collect(&engine, req, GenerateContext::new(ctx, None)).await;
                assert_eq!(
                    outputs.last().unwrap().as_ref().unwrap().finish_reason,
                    Some(FinishReason::Stop)
                );
                assert_eq!(handle.native_request().unwrap().media.len(), 2);
            }
        }
        assert_eq!(fixture.discovery.calls.load(Ordering::SeqCst), calls);
        engine.cleanup().await.unwrap();
        fixture.shutdown().await;
    }
}

#[tokio::test]
async fn discovery_model_and_authentication_errors_fail_startup_without_retry() {
    for configured in [None, Some(32768)] {
        for code in [
            tonic::Code::InvalidArgument,
            tonic::Code::NotFound,
            tonic::Code::Unauthenticated,
            tonic::Code::PermissionDenied,
        ] {
            let mut fixture =
                Fixture::start(Controller::<Adapter>::default(), FixtureConfig::default()).await;
            *fixture.discovery.error.lock().unwrap() = Some(code);
            let result = bounded(
                "terminal discovery error",
                bootstrap(&fixture, configured, 60),
            )
            .await;
            let error = result
                .err()
                .expect("model/authentication errors must prevent registration");
            assert!(
                error.to_string().contains("model discovery rejected"),
                "{error}"
            );
            assert!(error.to_string().contains(&format!("{code:?}")), "{error}");
            assert_eq!(fixture.discovery.calls.load(Ordering::SeqCst), 1);
            fixture.shutdown().await;
        }
    }
}

#[tokio::test]
async fn startup_deadline_bounds_retrying_and_stalled_discovery() {
    for configured in [None, Some(32768)] {
        for held in [false, true] {
            let mut fixture =
                Fixture::start(Controller::<Adapter>::default(), FixtureConfig::default()).await;
            fixture.discovery.unavailable.store(true, Ordering::SeqCst);
            fixture.discovery.hold.store(held, Ordering::SeqCst);
            let result =
                tokio::time::timeout(Duration::from_secs(4), bootstrap(&fixture, configured, 2))
                    .await
                    .expect("discovery must respect the startup deadline");
            let error = result
                .err()
                .expect("unresolved capabilities must prevent registration");
            assert!(error.to_string().contains("startup deadline"), "{error}");
            let calls = fixture.discovery.calls.load(Ordering::SeqCst);
            if held {
                assert_eq!(calls, 1);
            } else {
                assert!(calls >= 2);
            }
            fixture.release_discovery();
            fixture.shutdown().await;
        }
    }
}

#[tokio::test]
async fn shutdown_cancels_retrying_and_stalled_bootstrap_discovery() {
    for held in [false, true] {
        let mut fixture =
            Fixture::start(Controller::<Adapter>::default(), FixtureConfig::default()).await;
        fixture.discovery.unavailable.store(true, Ordering::SeqCst);
        fixture.discovery.hold.store(held, Ordering::SeqCst);
        let shutdown = tokio_util::sync::CancellationToken::new();
        let cancel = shutdown.clone();
        let startup = bootstrap(&fixture, Some(32768), 60);
        // Mirrors the shared runner's cancellation of the bootstrap future.
        let task = tokio::spawn(async move {
            tokio::select! {
                biased;
                _ = shutdown.cancelled() => {},
                _ = startup => panic!("unresolved discovery must not register a worker"),
            }
        });
        wait_for_calls(&fixture, 1).await;
        cancel.cancel();
        bounded("shutdown cancels bootstrap", task).await.unwrap();
        let calls = fixture.discovery.calls.load(Ordering::SeqCst);
        fixture.release_discovery();
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert_eq!(fixture.discovery.calls.load(Ordering::SeqCst), calls);
        fixture.shutdown().await;
    }
}

#[tokio::test]
async fn media_window_override_is_rejected_before_dispatch() {
    let control = Controller::<Adapter>::default();
    let mut fixture = Fixture::start(control.clone(), FixtureConfig::default()).await;
    advertise_media(&fixture, &["image", "video"]);
    let engine = fixture.engine_with_context_length(4096);
    engine.start(0).await.unwrap();
    for (video, max_tokens) in [
        (false, None),
        (false, Some(1)),
        (true, None),
        (true, Some(1)),
    ] {
        let ctx = mock_context();
        let handle = control.request(ctx.id(), RequestPlan::default());
        let mut req = if video {
            video_request()
        } else {
            image_request()
        };
        req.stop_conditions.max_tokens = max_tokens;
        let result = engine.generate(req, GenerateContext::new(ctx, None)).await;
        let error = result.err().expect("window mismatch must fail");
        assert!(error.to_string().contains("--context-length to match"));
        assert!(handle.native_request().is_none());
    }
    engine.cleanup().await.unwrap();
    fixture.shutdown().await;
}

#[tokio::test]
async fn text_only_audio_preserves_whisper_task_prompt_with_empty_token_ids() {
    let control = Controller::<Adapter>::default();
    let mut fixture = Fixture::start(control.clone(), FixtureConfig::default()).await;
    advertise_media(&fixture, &["audio"]);
    fixture
        .discovery
        .model_info
        .lock()
        .unwrap()
        .as_mut()
        .unwrap()
        .supports_token_ids_input = Some(false);
    let engine = fixture.engine().await;
    engine.start(0).await.unwrap();
    let uri = "data:audio/wav;base64,YXVkaW8=";
    for prompt in [
        "",
        "<|startoftranscript|><|fr|><|translate|><|notimestamps|>",
    ] {
        let mut req = request("mocker-model", vec![], 16);
        req.extra_args = Some(serde_json::json!({"formatted_prompt": prompt}));
        req.multi_modal_data = Some(HashMap::from([(
            "audio_url".into(),
            vec![MultimodalData::RawUrl(uri.into())],
        )]));
        let ctx = mock_context();
        let handle = control.request(ctx.id(), RequestPlan::default());
        fixture.respond(ctx.id(), vec![finished()]);
        let outputs = collect(&engine, req, GenerateContext::new(ctx, None)).await;
        assert_eq!(
            outputs.last().unwrap().as_ref().unwrap().finish_reason,
            Some(FinishReason::Stop)
        );
        let native = handle.native_request().unwrap();
        assert_eq!(
            native.input,
            Some(pb::generate_request::Input::Prompt(prompt.into()))
        );
        assert_eq!(native.media.len(), 1);
        assert_eq!(native.media[0].modality, pb::Modality::Audio as i32);
        assert_eq!(
            native.media[0].source,
            Some(pb::media_item::Source::DataUri(uri.into()))
        );
    }
    assert_eq!(fixture.discovery.calls.load(Ordering::SeqCst), 1);
    engine.cleanup().await.unwrap();
    fixture.shutdown().await;
}

#[test]
fn remote_url_policy_is_enforced_before_engine_dispatch() {
    // Isolate startup environment changes from the other parallel tests.
    const CHILD: &str = "DYNAMO_TEST_REMOTE_MEDIA_POLICY";
    if std::env::var_os(CHILD).is_none() {
        for value in [Some("0"), Some("true"), None] {
            let mut command = std::process::Command::new(std::env::current_exe().unwrap());
            match value {
                Some(value) => {
                    command.env("DYN_MM_ALLOW_REMOTE_URLS", value);
                }
                None => {
                    command.env_remove("DYN_MM_ALLOW_REMOTE_URLS");
                }
            }
            let output = command
                .args([
                    "--exact",
                    "remote_url_policy_is_enforced_before_engine_dispatch",
                    "--nocapture",
                ])
                .env(CHILD, "1")
                .env("DYN_MM_ALLOW_INTERNAL", "1")
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "policy {value:?}:\n{}\n{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr)
            );
            assert!(
                String::from_utf8_lossy(&output.stdout).contains("1 passed; 0 failed"),
                "the child must execute the policy test"
            );
        }
        return;
    }

    let allow_remote = std::env::var("DYN_MM_ALLOW_REMOTE_URLS")
        .map(|value| dynamo_runtime::config::parse_bool(&value).unwrap())
        .unwrap_or(true);
    tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .unwrap()
        .block_on(check_remote_url_policy(allow_remote));
}

async fn check_remote_url_policy(allow_remote: bool) {
    let control = Controller::<Adapter>::default();
    let mut fixture = Fixture::start(control.clone(), FixtureConfig::default()).await;
    advertise_media(&fixture, &["image", "video", "audio"]);
    let engine = fixture.engine().await;
    engine.start(0).await.unwrap();
    for (key, media) in [
        (
            "image_url",
            MultimodalData::RawUrl("http://example.com/image.png".into()),
        ),
        (
            "image_url",
            MultimodalData::RawUrl("HTTPS://example.com/image.png".into()),
        ),
        (
            "image_url",
            MultimodalData::Url("https://example.com/image.png".parse().unwrap()),
        ),
        (
            "image_url",
            MultimodalData::RawUrl("data:image/png;base64,aW1hZ2U=".into()),
        ),
        (
            "video_url",
            MultimodalData::RawUrl("http://example.com/video.mp4".into()),
        ),
        (
            "video_url",
            MultimodalData::RawUrl("HTTPS://example.com/video.mp4".into()),
        ),
        (
            "video_url",
            MultimodalData::Url("https://example.com/video.mp4".parse().unwrap()),
        ),
        (
            "video_url",
            MultimodalData::RawUrl("data:video/mp4;base64,dmlkZW8=".into()),
        ),
        (
            "audio_url",
            MultimodalData::RawUrl("http://example.com/audio.wav".into()),
        ),
        (
            "audio_url",
            MultimodalData::RawUrl("HTTPS://example.com/audio.wav".into()),
        ),
        (
            "audio_url",
            MultimodalData::Url("https://example.com/audio.wav".parse().unwrap()),
        ),
        (
            "audio_url",
            MultimodalData::RawUrl("data:audio/wav;base64,YXVkaW8=".into()),
        ),
    ] {
        let inline = matches!(&media, MultimodalData::RawUrl(url) if url.starts_with("data:"));
        let mut req = image_request();
        req.multi_modal_data = Some(HashMap::from([(key.into(), vec![media])]));
        req.multi_modal_uuids = None;
        let ctx = mock_context();
        let handle = control.request(ctx.id(), RequestPlan::default());
        if allow_remote || inline {
            fixture.respond(ctx.id(), vec![finished()]);
            let outputs = collect(&engine, req, GenerateContext::new(ctx, None)).await;
            assert_eq!(
                outputs.last().unwrap().as_ref().unwrap().finish_reason,
                Some(FinishReason::Stop)
            );
            assert_eq!(handle.native_request().unwrap().media.len(), 1);
        } else {
            let error = engine
                .generate(req, GenerateContext::new(ctx, None))
                .await
                .err()
                .expect("remote URL must fail");
            assert_eq!(
                error.error_type(),
                dynamo_backend_common::ErrorType::Backend(
                    dynamo_backend_common::BackendError::InvalidArgument,
                )
            );
            assert!(
                error.to_string().contains("DYN_MM_ALLOW_REMOTE_URLS=0"),
                "{error}"
            );
            assert!(handle.native_request().is_none());
        }
    }
    engine.cleanup().await.unwrap();
    fixture.shutdown().await;
}
