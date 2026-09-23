// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CPU-only integration smoke through Dynamo's public live-request boundary.
//! Run with `cargo test -p dynamo-mocker --test state_cache_live -- --nocapture`.
//! No model, tokenizer, distributed runtime, KV-event publisher, or GPU is needed.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use dynamo_mocker::common::protocols::{
    DirectRequest, ForwardPassSnapshot, FpmPublisher, FpmSink, MockEngineArgs,
};
use dynamo_mocker::live::{LiveEngine, LiveEngineConfig};
use rstest::rstest;
use serde_json::json;
use uuid::Uuid;

const PROMPT_TOKENS: usize = 24_300;
const COLD_STATE_PREFILL_ENDPOINTS: &[u64] = &[7_680, 15_360, 23_040, 24_192, 24_300];

#[derive(Default)]
struct CompletedPasses(Mutex<Vec<ForwardPassSnapshot>>);

impl FpmSink for CompletedPasses {
    fn publish(&self, snapshot: ForwardPassSnapshot) -> anyhow::Result<()> {
        self.0.lock().unwrap().push(snapshot);
        Ok(())
    }
}

impl CompletedPasses {
    fn take_prefill_chunks(&self) -> Vec<u64> {
        std::mem::take(&mut *self.0.lock().unwrap())
            .into_iter()
            .map(|pass| pass.sum_prefill_tokens)
            .filter(|tokens| *tokens > 0)
            .collect()
    }
}

fn engine_args(with_state_cache: bool) -> MockEngineArgs {
    // These are artificial per-rank byte counts, not measured model geometry.
    // Exercise JSON deserialization as well as the live/native adapter.
    let mut config = json!({
        "engine_type": "vllm",
        "worker_type": "aggregated",
        "block_size": 1536,
        "num_gpu_blocks": 512,
        "max_model_len": 32768,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 8192,
        "enable_prefix_caching": true,
        "enable_chunked_prefill": true,
        "speedup_ratio": 0.0,
        "timing_model": {"type": "fixed", "prefill_ms": 1.0, "decode_ms": 1.0}
    });
    if with_state_cache {
        config["kv_cache_bytes_per_token"] = json!(16);
        config["state_cache"] = json!({"bytes_per_request": 24576});
        config["prefix_match_unit"] = json!(128);
        config["enable_kv_events"] = json!(false);
    }
    MockEngineArgs::from_json_str(&config.to_string()).unwrap()
}

async fn submit_and_collect(engine: &LiveEngine, tokens: Vec<u32>, id: u128) -> usize {
    let expected_outputs = vec![50_000 + id as u32, 60_000 + id as u32];
    let mut request = engine
        .submit(DirectRequest {
            tokens,
            max_output_tokens: expected_outputs.len(),
            output_token_ids: Some(expected_outputs.clone()),
            uuid: Some(Uuid::from_u128(id)),
            ..Default::default()
        })
        .await
        .unwrap();
    let mut output_tokens = Vec::new();
    let mut cached_tokens = None;
    let mut completed = false;
    while let Some(output) = request.recv().await {
        assert!(!output.rejected, "live request {id} was rejected");
        if let Some(cached) = output.cached_tokens {
            assert!(cached_tokens.replace(cached).is_none());
        }
        if let Some(token) = output.token_id {
            output_tokens.push(token);
        }
        if output.completed {
            completed = true;
            break;
        }
    }
    assert!(completed, "live request {id} closed without completion");
    assert_eq!(output_tokens, expected_outputs);
    assert!(request.recv().await.is_none());
    cached_tokens.expect("first output must report admission cache reuse")
}

async fn run_serial_pair(
    with_state_cache: bool,
    shared_prefix_tokens: usize,
    expected_reused_tokens: usize,
    expected_committed_prefill_tokens: u64,
    expected_cold_endpoints: &[u64],
) {
    tokio::time::timeout(Duration::from_secs(10), async {
        let passes = Arc::new(CompletedPasses::default());
        let engine = LiveEngine::start_with_config(
            engine_args(with_state_cache),
            0,
            LiveEngineConfig {
                fpm_publisher: FpmPublisher::new(Some(Arc::clone(&passes) as Arc<dyn FpmSink>)),
                ..LiveEngineConfig::default()
            },
        )
        .unwrap();

        let cold_prompt = (0..PROMPT_TOKENS as u32).collect::<Vec<_>>();
        assert_eq!(submit_and_collect(&engine, cold_prompt.clone(), 1).await, 0);
        // LiveEngine publishes the completed pass's FPM before its output.
        // This sink sees every pass (there is no telemetry sampling), so the
        // sum below measures committed work rather than inferring it from reuse.
        let cold_chunks = passes.take_prefill_chunks();
        let cold_endpoints = cold_chunks
            .iter()
            .scan(0, |total, chunk| {
                *total += chunk;
                Some(*total)
            })
            .collect::<Vec<_>>();
        assert_eq!(cold_endpoints, expected_cold_endpoints);

        let mut second_prompt = cold_prompt[..shared_prefix_tokens].to_vec();
        second_prompt
            .extend((shared_prefix_tokens..PROMPT_TOKENS).map(|index| 100_000 + index as u32));
        assert_eq!(
            submit_and_collect(&engine, second_prompt, 2).await,
            expected_reused_tokens,
        );
        engine.shutdown().await.unwrap();

        let warm_chunks = passes.take_prefill_chunks();
        let committed_prefill_tokens = cold_chunks.iter().chain(&warm_chunks).sum::<u64>();
        assert_eq!(committed_prefill_tokens, expected_committed_prefill_tokens);
        assert_eq!(
            warm_chunks.iter().sum::<u64>(),
            (PROMPT_TOKENS - expected_reused_tokens) as u64,
        );
        assert_eq!(engine.active_request_count(), 0);
        println!(
            "state_cache={with_state_cache}, shared_prefix_tokens={shared_prefix_tokens}, \
             cached_tokens={expected_reused_tokens}, cold_prefill_endpoints={cold_endpoints:?}, \
             second_prefill_chunks={warm_chunks:?}, \
             committed_prefill_tokens={committed_prefill_tokens}, completed_requests=2, \
             total_output_tokens=4"
        );
    })
    .await
    .expect("two serial live requests must complete promptly");
}

#[rstest]
#[case::partial_checkpoint_hit(24_192, 24_192, 24_408)]
#[case::last_full_chunk_checkpoint(23_040, 23_040, 25_560)]
#[case::first_chunk_checkpoint_was_released(7_680, 0, 48_600)]
#[case::second_chunk_checkpoint_was_released(15_360, 0, 48_600)]
#[case::interior_physical_boundary_has_no_state(21_504, 0, 48_600)]
#[case::between_checkpoints(23_700, 23_040, 25_560)]
#[case::one_token_before_checkpoint(24_191, 23_040, 25_560)]
#[case::no_shared_prefix(0, 0, 48_600)]
#[tokio::test]
async fn state_cache_live_retained_checkpoints(
    #[case] shared_prefix_tokens: usize,
    #[case] expected_reused_tokens: usize,
    #[case] expected_committed_prefill_tokens: u64,
) {
    run_serial_pair(
        true,
        shared_prefix_tokens,
        expected_reused_tokens,
        expected_committed_prefill_tokens,
        COLD_STATE_PREFILL_ENDPOINTS,
    )
    .await;
}

#[tokio::test]
async fn state_cache_live_legacy_token_only_path_is_unchanged() {
    // No new options are set: full physical blocks remain reusable, while
    // prefill uses the original token budget without checkpoint alignment.
    run_serial_pair(false, 24_192, 23_040, 25_560, &[8_192, 16_384, 24_300]).await;
}
