// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_backend_common::LLMEngineOutput;

use super::*;

const TEST_TIMEOUT: Duration = Duration::from_secs(300);

#[derive(Debug, serde::Serialize, serde::Deserialize)]
struct ImageResult {
    image_uri: String,
    image_uuid: Option<String>,
    token_ids: Vec<u32>,
    prompt_tokens: u32,
    completion_tokens: u32,
    cached_tokens: u32,
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

async fn worker(endpoint: &str, model: &str, mode: DisaggregationMode) -> TrtllmSidecarEngine {
    let mut args = vec![
        "sidecar".into(),
        "--grpc-endpoint".into(),
        endpoint.into(),
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
    engine
}

async fn reject_replay(consumer: &TrtllmSidecarEngine, request: PreprocessedRequest) -> String {
    let context = dynamo_backend_common::testing::mock_context();
    let error = match consumer
        .generate(request, GenerateContext::new(context, None))
        .await
    {
        Err(error) => error,
        Ok(mut stream) => stream
            .next()
            .await
            .expect("replay must fail before emitting inference output")
            .expect_err("a consumed encoder reference must not be reusable"),
    };
    assert_eq!(
        error.error_type(),
        ErrorType::Backend(dynamo_backend_common::BackendError::InvalidArgument)
    );
    error.to_string()
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
    let encoder = worker(&encoder_endpoint, &model, DisaggregationMode::Encode).await;
    let mode = if disaggregated { PREFILL } else { AGG };
    let consumer_endpoint = required("TRTLLM_E2E_PREFILL_ENDPOINT");
    let consumer = worker(&consumer_endpoint, &model, mode).await;
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
    let mut probe = image_requests(&model).remove(0);
    probe.is_probe = true;
    let probe_outputs = collect(&encoder, probe).await;
    assert_eq!(probe_outputs.len(), 1);
    assert_eq!(probe_outputs[0].finish_reason, Some(FinishReason::Stop));
    assert!(probe_outputs[0].token_ids.is_empty());
    assert!(probe_outputs[0].encoder_result.is_none());
    let mut cases = Vec::new();
    for (mut request, expected) in image_requests(&model).into_iter().zip(baseline.cases) {
        let encoded = collect(&encoder, request.clone()).await;
        assert_eq!(
            encoded.len(),
            1,
            "Encode produces a single encoder terminal"
        );
        assert!(encoded[0].token_ids.is_empty());
        assert_eq!(encoded[0].finish_reason, Some(FinishReason::Stop));
        request.encoder_result = Some(
            encoded[0]
                .encoder_result
                .as_deref()
                .cloned()
                .expect("Encode must carry its opaque reference"),
        );
        let outputs = collect(&consumer, request.clone()).await;
        let replay_error = reject_replay(&consumer, request.clone()).await;
        let mut prefill_prompt_tokens = None;
        let outputs = if let Some(decode) = &decode {
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
                    .as_deref()
                    .cloned()
                    .expect("PrefillReady must carry the native KV session"),
                prompt_tokens_details: usage.prompt_tokens_details.clone(),
            });
            // Keep the spent reference on D's request; only its staged KV may be replayed.
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
        actual.prefill_prompt_tokens = prefill_prompt_tokens;
        actual.replay_error = Some(replay_error);
        cases.push(actual);
    }
    write_result(&model, cases);
    encoder.cleanup().await.unwrap();
    consumer.cleanup().await.unwrap();
    if let Some(decode) = decode {
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
