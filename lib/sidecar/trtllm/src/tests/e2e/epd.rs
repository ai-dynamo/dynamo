// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::client::{ModelLimits, TrtllmClient};
use crate::convert::build_generate_request;
use dynamo_backend_common::LLMEngineOutput;

use super::*;

const NIXL_BACKEND: &str = "trtllm-nixl-encoder-v1";
const TEST_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ImageResult {
    image_uri: String,
    image_uuid: Option<String>,
    token_ids: Vec<u32>,
    prompt_tokens: u32,
    completion_tokens: u32,
    cached_tokens: u32,
    encoder_transfer_backend: Option<String>,
    prefill_prompt_tokens: Option<u32>,
    replay_error: Option<String>,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct RunResult {
    model: String,
    cases: Vec<ImageResult>,
}

fn required(name: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|value| !value.trim().is_empty())
        .unwrap_or_else(|| panic!("set {name} when explicitly running the EPD engine tests"))
}

fn image_requests(model: &str) -> Vec<PreprocessedRequest> {
    let images: Vec<String> = serde_json::from_slice(
        &std::fs::read(required("TRTLLM_E2E_EPD_IMAGES"))
            .expect("read opposite-color image fixture"),
    )
    .expect("TRTLLM_E2E_EPD_IMAGES must contain a JSON array of two image data URIs");
    assert_eq!(images.len(), 2, "EPD requires both opposite-color images");
    assert_ne!(images[0], images[1], "opposite images must differ");
    images
        .into_iter()
        .enumerate()
        .map(|(index, image)| {
            assert!(image.starts_with("data:image/png;base64,"));
            let mut request = PreprocessedRequest::builder()
                .model(model.to_owned())
                // Qwen3-VL's existing real-image test prompt: one image and a color question.
                .token_ids(vec![
                    151644, 872, 198, 151652, 151655, 151653, 675, 279, 7987, 504, 2115, 311, 1290,
                    13, 151645, 198, 151644, 77091, 198,
                ])
                .stop_conditions(StopConditions {
                    max_tokens: Some(64),
                    ..Default::default()
                })
                .sampling_options(SamplingOptions {
                    temperature: Some(0.0),
                    ..Default::default()
                })
                .output_options(OutputOptions::default())
                .build()
                .expect("Qwen3-VL image request");
            request.multi_modal_data = Some(HashMap::from([(
                "image_url".into(),
                vec![MultimodalData::RawUrl(image)],
            )]));
            request.multi_modal_uuids = Some(HashMap::from([(
                "image_url".into(),
                vec![(index == 1).then(|| "epd-opposite-colors".into())],
            )]));
            request
        })
        .collect()
}

fn result(request: &PreprocessedRequest, outputs: &[LLMEngineOutput]) -> ImageResult {
    let terminal = outputs
        .iter()
        .find(|output| output.finish_reason.is_some())
        .expect("generation must produce a terminal chunk");
    let usage = terminal.completion_usage.as_ref().expect("terminal usage");
    let token_ids: Vec<u32> = outputs
        .iter()
        .flat_map(|output| output.token_ids.iter().copied())
        .collect();
    assert!(!token_ids.is_empty(), "image-conditioned output is empty");
    assert!(usage.completion_tokens > 0);
    assert!(
        usage.prompt_tokens > request.token_ids.len() as u32,
        "terminal usage must report the authoritative expanded image prompt"
    );
    assert_eq!(
        usage
            .prompt_tokens_details
            .as_ref()
            .and_then(|details| details.cached_tokens),
        Some(0),
        "nightly EPD runs disable prefix caching and require authoritative zero cache usage"
    );
    let MultimodalData::RawUrl(image_uri) =
        &request.multi_modal_data.as_ref().unwrap()["image_url"][0]
    else {
        panic!("expected the original image data URI");
    };
    ImageResult {
        image_uri: image_uri.clone(),
        image_uuid: request.multi_modal_uuids.as_ref().unwrap()["image_url"][0].clone(),
        token_ids,
        prompt_tokens: usage.prompt_tokens,
        completion_tokens: usage.completion_tokens,
        cached_tokens: 0,
        encoder_transfer_backend: None,
        prefill_prompt_tokens: None,
        replay_error: None,
    }
}

fn write_result(model: &str, cases: Vec<ImageResult>) {
    std::fs::write(
        required("TRTLLM_E2E_RESULT"),
        serde_json::to_vec_pretty(&RunResult {
            model: model.into(),
            cases,
        })
        .expect("serialize EPD results"),
    )
    .expect("write EPD results");
}

pub(super) async fn baseline(engine: &TrtllmSidecarEngine, model: &str) {
    tokio::time::timeout(TEST_TIMEOUT, async {
        let mut cases = Vec::new();
        for request in image_requests(model) {
            cases.push(result(&request, &collect(engine, request.clone()).await));
        }
        write_result(model, cases);
    })
    .await
    .expect("aggregated EPD baseline timed out");
}

async fn worker(
    endpoint: &str,
    model: &str,
    mode: DisaggregationMode,
) -> (TrtllmSidecarEngine, ModelLimits) {
    let endpoint = GrpcEndpoint::parse(endpoint, "--grpc-endpoint").expect("valid EPD endpoint");
    let client = TrtllmClient::connect(&endpoint, transport(1))
        .await
        .expect("connect real EPD engine");
    let info = client
        .wait_for_model_info(
            model,
            None,
            tokio::time::Instant::now() + Duration::from_secs(30),
            Duration::from_millis(100),
        )
        .await
        .expect("read real Control.GetModelInfo for EPD negotiation");
    assert_eq!(
        info.limits.encoder_only,
        mode.is_encode(),
        "official runtime lacks the required negotiated encoder-only role"
    );
    assert_eq!(
        info.limits.encoder_consumer,
        !mode.is_encode(),
        "official runtime must advertise a vision-disabled encoder-handoff consumer"
    );
    let mut args = vec![
        "sidecar".into(),
        "--grpc-endpoint".into(),
        endpoint.to_string(),
        "--model-path".into(),
        model.into(),
        "--disaggregation-mode".into(),
        mode.to_string(),
    ];
    if !mode.is_encode() && !mode.is_decode() {
        args.push("--route-to-encoder".into());
    }
    let engine = TrtllmSidecarEngine::from_args(args)
        .expect("configure real EPD worker")
        .0;
    engine.start(0).await.expect("start real EPD worker");
    (engine, info.limits)
}

async fn reject_replay(endpoint: &str, mut request: pb::GenerateRequest) -> String {
    static REPLAY_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
    request.request_id = format!(
        "replay-{}",
        REPLAY_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    assert!(matches!(
        request.encoder.as_ref().unwrap().mode,
        Some(pb::encoder_options::Mode::Handoff(_))
    ));
    assert!(request.kv.as_ref().is_none_or(|kv| kv.session.is_none()));
    let mut inference = pb::inference_client::InferenceClient::connect(endpoint.to_owned())
        .await
        .expect("connect the consumer inference service");
    let status = match inference.generate(request).await {
        Err(status) => status,
        Ok(response) => response
            .into_inner()
            .message()
            .await
            .expect_err("replay must fail before emitting an inference event"),
    };
    assert_eq!(status.code(), tonic::Code::InvalidArgument);
    assert!(
        status.message().contains("expired or was already consumed"),
        "replay must reject the spent export: {status}"
    );
    status.to_string()
}

async fn run(disaggregated: bool) {
    let model = required("TRTLLM_E2E_MODEL");
    let baseline: RunResult = serde_json::from_slice(
        &std::fs::read(required("TRTLLM_E2E_EPD_BASELINE"))
            .expect("read sequential aggregated baseline"),
    )
    .expect("aggregated baseline must contain model and two cases");
    assert_eq!(
        baseline.model, model,
        "baseline must use the same checkpoint"
    );
    assert_eq!(baseline.cases.len(), 2);
    let encoder_endpoint = required("TRTLLM_E2E_ENCODER_ENDPOINT");
    let (encoder, encoder_limits) =
        worker(&encoder_endpoint, &model, DisaggregationMode::Encode).await;
    let mode = if disaggregated { PREFILL } else { AGG };
    let consumer_endpoint = required("TRTLLM_E2E_PREFILL_ENDPOINT");
    let (consumer, consumer_limits) = worker(&consumer_endpoint, &model, mode).await;
    let decode = if disaggregated {
        Some(
            worker(
                &required("TRTLLM_E2E_DECODE_ENDPOINT"),
                &model,
                DisaggregationMode::Decode,
            )
            .await,
        )
    } else {
        None
    };
    let mut cases = Vec::new();
    for (mut request, expected) in image_requests(&model).into_iter().zip(baseline.cases) {
        let encoder_wire = build_generate_request(
            request.clone(),
            "e",
            &model,
            Some(encoder_limits),
            DisaggregationMode::Encode,
        )
        .unwrap();
        assert!(matches!(
            encoder_wire.encoder.as_ref().unwrap().mode,
            Some(pb::encoder_options::Mode::EncodeOnly(_))
        ));
        let encoded = collect(&encoder, request.clone()).await;
        assert_eq!(encoded.len(), 1, "EncodeReady is the sole encoder terminal");
        assert!(encoded[0].token_ids.is_empty());
        assert_eq!(encoded[0].finish_reason, Some(FinishReason::Stop));
        let handoff = encoded[0]
            .encoder_result
            .clone()
            .expect("EncodeReady must carry its opaque reference");
        assert_eq!(handoff["transfer_backend"], NIXL_BACKEND);
        assert!(
            !handoff["opaque_state"]
                .as_array()
                .expect("opaque reference bytes")
                .is_empty()
        );
        request.encoder_result = Some(handoff);
        let consumer_wire = build_generate_request(
            request.clone(),
            "consumer",
            &model,
            Some(consumer_limits),
            mode,
        )
        .unwrap();
        assert_eq!(
            encoder_wire.media, consumer_wire.media,
            "encoder and consumer image envelopes must match, including UUIDs"
        );
        assert_eq!(encoder_wire.input, consumer_wire.input);
        let outputs = collect(&consumer, request.clone()).await;
        let replay_error = reject_replay(&consumer_endpoint, consumer_wire.clone()).await;
        let mut prefill_prompt_tokens = None;
        let outputs = if let Some((decode, limits)) = &decode {
            assert!(outputs.iter().all(|output| output.token_ids.is_empty()));
            let terminal = outputs.last().expect("prefill terminal");
            assert_eq!(terminal.finish_reason, Some(FinishReason::Length));
            let usage = terminal
                .completion_usage
                .as_ref()
                .expect("prefill expanded usage");
            assert_eq!(
                usage
                    .prompt_tokens_details
                    .as_ref()
                    .and_then(|details| details.cached_tokens),
                Some(0),
                "prefill must report zero prefix reuse when caching is disabled"
            );
            prefill_prompt_tokens = Some(usage.prompt_tokens);
            request.prefill_result = Some(PrefillResult {
                disaggregated_params: terminal
                    .disaggregated_params
                    .clone()
                    .expect("PrefillReady must carry the native KV session"),
                prompt_tokens_details: usage.prompt_tokens_details.clone(),
            });
            assert_eq!(
                consumer_wire.extra.as_ref().unwrap().fields[crate::disagg::REQUEST_TYPE_KEY].kind,
                Some(prost_types::value::Kind::StringValue(
                    crate::disagg::CONTEXT_ONLY.into()
                ))
            );
            let wire = build_generate_request(
                request.clone(),
                "d",
                &model,
                Some(*limits),
                DisaggregationMode::Decode,
            )
            .unwrap();
            assert!(
                request.encoder_result.is_some(),
                "retain the spent encoder reference through decode"
            );
            assert!(
                wire.encoder.is_none(),
                "D must consume KV without replaying the encoder reference"
            );
            assert!(wire.kv.unwrap().session.is_some());
            assert_eq!(wire.media, encoder_wire.media);
            assert_eq!(wire.input, encoder_wire.input);
            collect(decode, request.clone()).await
        } else {
            outputs
        };
        let mut actual = result(&request, &outputs);
        assert_eq!(actual.image_uri, expected.image_uri);
        assert_eq!(actual.image_uuid, expected.image_uuid);
        assert_eq!(
            actual.token_ids, expected.token_ids,
            "EPD output must match the aggregated baseline for each image"
        );
        assert_eq!(actual.prompt_tokens, expected.prompt_tokens);
        assert_eq!(actual.completion_tokens, expected.completion_tokens);
        assert_eq!(actual.cached_tokens, expected.cached_tokens);
        if let Some(prompt_tokens) = prefill_prompt_tokens {
            assert_eq!(prompt_tokens, actual.prompt_tokens);
        }
        actual.encoder_transfer_backend = Some(NIXL_BACKEND.into());
        actual.prefill_prompt_tokens = prefill_prompt_tokens;
        actual.replay_error = Some(replay_error);
        cases.push(actual);
    }
    write_result(&model, cases);
    encoder.cleanup().await.unwrap();
    consumer.cleanup().await.unwrap();
    if let Some((decode, _)) = decode {
        decode.cleanup().await.unwrap();
    }
}

#[tokio::test]
#[ignore = "requires real NIXL encoder and vision-disabled PD engines plus an aggregated baseline"]
async fn e2e_real_openengine_image_epd() {
    tokio::time::timeout(TEST_TIMEOUT, run(false))
        .await
        .expect("real E+PD test timed out");
}

#[tokio::test]
#[ignore = "requires real NIXL encoder/P/D engines plus an aggregated baseline"]
async fn e2e_real_openengine_image_epd_disaggregated() {
    tokio::time::timeout(TEST_TIMEOUT, run(true))
        .await
        .expect("real E+P+D test timed out");
}
