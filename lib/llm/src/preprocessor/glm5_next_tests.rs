// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;
use crate::common::checked_file::CheckedFile;
use crate::model_card::ModelInfoType;
use dynamo_kv_router::protocols::pad_value_for_mm_hash;
use dynamo_multimodal::processor::MediaMetadata;
use dynamo_multimodal::{registry, token_layout};

fn preprocessor(model_dir: &std::path::Path) -> Arc<OpenAIPreprocessor> {
    // Reuse the local mock tokenizer; only the routing metadata is GLM. This
    // exercises startup selection without downloading a tokenizer or weights.
    let mut mdc = ModelDeploymentCard::load_from_disk(
        "tests/data/sample-models/mock-llama-3.1-8b-instruct",
        None,
    )
    .unwrap();
    mdc.model_info = Some(ModelInfoType::HfConfigJson(
        CheckedFile::from_disk(model_dir.join("config.json")).unwrap(),
    ));
    mdc.kv_cache_block_size = 16;
    OpenAIPreprocessor::new(mdc).unwrap()
}

#[test]
fn glm_routing_matches_shared_layout_and_keeps_delimiters() {
    let dir = mm_routing::glm5_next::tests::model_dir();
    let preprocessor = preprocessor(dir.path());
    assert_eq!(preprocessor.routing_image_token_id, Some(154854));
    let ids = [7, 154830, 154854, 154831, 8, 154830, 154854, 154831, 9];
    let entries = [
        MmRoutingEntry::Image {
            mm_hash: 0x1234,
            width: 448,
            height: 448,
        },
        MmRoutingEntry::Image {
            mm_hash: 0x5678,
            width: 13,
            height: 15,
        },
    ];
    let routing = preprocessor
        .build_mm_exact_routing_info(&entries, &ids, None)
        .unwrap();
    let single = preprocessor
        .build_mm_exact_routing_info(&entries[..1], &ids[..5], None)
        .unwrap();
    assert_eq!(single.expanded_prompt_len, 260);
    assert_eq!(&single.routing_token_ids[..2], &[7, 154830]);
    assert_eq!(&single.routing_token_ids[258..260], &[154831, 8]);

    let planner =
        registry::build_processor(registry::spec_from_model_dir(dir.path()).unwrap()).unwrap();
    let metadata = [
        MediaMetadata::Image {
            width: 448,
            height: 448,
        },
        MediaMetadata::Image {
            width: 13,
            height: 15,
        },
    ];
    let counts: Vec<_> = metadata
        .iter()
        .map(|m| planner.num_media_tokens(m).unwrap())
        .collect();
    assert_eq!(counts, [256, 20]);
    let src: Vec<_> = ids.iter().map(|&id| i32::try_from(id).unwrap()).collect();
    let layout = planner.layout_from_metadata(&src, &metadata).unwrap();
    let shared = token_layout::apply_layout(&src, &layout, &counts, 4096).unwrap();
    let mut expected: Vec<_> = shared
        .input_ids
        .iter()
        .map(|&id| u32::try_from(id).unwrap())
        .collect();
    for (ranges, entry) in shared.feature_ranges.iter().zip(&entries) {
        for range in ranges {
            expected[range.start as usize..range.end as usize]
                .fill(pad_value_for_mm_hash(entry.mm_hash()));
        }
    }
    assert_eq!(routing.expanded_prompt_len, 283);
    assert_eq!(&routing.routing_token_ids[..283], expected);
    assert_eq!(routing.routing_token_ids.len(), 288);
    assert!(routing.block_mm_infos.is_empty());

    // Same dimensions with different image content must change routing keys.
    let other = [
        MmRoutingEntry::Image {
            mm_hash: 0x9abc,
            width: 448,
            height: 448,
        },
        entries[1].clone(),
    ];
    let changed = preprocessor
        .build_mm_exact_routing_info(&other, &ids, None)
        .unwrap();
    assert_ne!(
        &routing.routing_token_ids[2..258],
        &changed.routing_token_ids[2..258]
    );
    assert_eq!(
        &routing.routing_token_ids[258..],
        &changed.routing_token_ids[258..]
    );
}

#[test]
fn glm_count_and_layout_failures_disable_exact_routing() {
    let dir = mm_routing::glm5_next::tests::model_dir();
    let preprocessor = preprocessor(dir.path());
    for (width, height) in [(0, 448), (448, 0), (u32::MAX, u32::MAX)] {
        let entries = [MmRoutingEntry::Image {
            mm_hash: 1,
            width,
            height,
        }];
        assert!(
            preprocessor
                .build_mm_exact_routing_info(&entries, &[154854], None)
                .is_none()
        );
    }
    let entries = [MmRoutingEntry::Image {
        mm_hash: 1,
        width: 448,
        height: 448,
    }];
    for ids in [
        vec![],
        vec![154854, 154854],
        vec![154854, 154855],
        vec![154854, u32::MAX],
    ] {
        assert!(
            preprocessor
                .build_mm_exact_routing_info(&entries, &ids, None)
                .is_none()
        );
    }
}

#[tokio::test]
async fn glm_gather_installs_routing_and_hashes_atomically() {
    let image_url = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg==";
    for case in [
        "valid",
        "count_failure",
        "layout_failure",
        "video",
        "override",
        "uuid",
        "unsupported_config",
    ] {
        let dir = mm_routing::glm5_next::tests::model_dir();
        if matches!(case, "count_failure" | "unsupported_config") {
            let path = dir.path().join("processor_config.json");
            let mut config: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
            if case == "count_failure" {
                // A valid spec whose expanded canvas overflows the planner's
                // u32 feature-token count, even for this one-pixel input.
                config["image_processor"]["patch_expand_factor"] = 65536.into();
            } else {
                config["image_processor"]["do_resize"] = false.into();
            }
            std::fs::write(path, config.to_string()).unwrap();
        }
        let preprocessor = preprocessor(dir.path());
        if case == "count_failure" {
            assert!(
                preprocessor
                    .image_token_counter
                    .as_ref()
                    .unwrap()
                    .count_tokens(1, 1)
                    .is_err()
            );
        }
        let mut body = serde_json::json!({
            "model": "custom-checkpoint",
            "messages": [{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": image_url}},
                {"type": "image_url", "image_url": {"url": image_url}}
            ]}]
        });
        if case == "override" {
            body["mm_processor_kwargs"] = serde_json::json!({"max_image_tokens": 64});
        }
        if case == "uuid" {
            body["messages"][0]["content"][0]["uuid"] = "external-cache-key".into();
        }
        if case == "video" {
            body["messages"][0]["content"].as_array_mut().unwrap().push(
                serde_json::json!({"type": "video_url", "video_url": {"url": "https://example.com/video.mp4"}}),
            );
        }
        let request: NvCreateChatCompletionRequest = serde_json::from_value(body).unwrap();
        let mut ids = vec![7, 154830, 154854, 154831, 8, 154830, 154854, 154831, 9];
        if case == "layout_failure" {
            ids.push(154855);
        }
        let mut builder = PreprocessedRequest::builder();
        builder
            .model("custom-checkpoint".to_string())
            .token_ids(ids.clone())
            .stop_conditions(Default::default())
            .sampling_options(Default::default())
            .output_options(Default::default());
        let (_, image_tokens) = preprocessor
            .gather_multi_modal_data_with_image_tokens(&request, &mut builder, None, &ids)
            .await
            .unwrap();
        let result = builder.build().unwrap();
        assert_eq!(
            result.token_ids.as_ref(),
            &ids,
            "{case}: backend prompt must remain unchanged"
        );
        assert_eq!(
            result.multi_modal_data.as_ref().unwrap()["image_url"].len(),
            2
        );
        let extra = result.extra_args.as_ref().unwrap();
        assert_eq!(result.mm_routing_info.is_some(), case == "valid", "{case}");
        assert_eq!(extra.get("mm_hashes").is_some(), case == "valid", "{case}");
        assert!(extra.get("mm_hashes_by_modality").is_none(), "{case}");
        if case == "valid" {
            let hash = format!("{:016x}", OpenAIPreprocessor::hash_image_url(image_url));
            assert_eq!(extra["mm_hashes"], serde_json::json!([hash, hash]));
            assert_eq!(image_tokens, Some(32));
        } else if matches!(case, "count_failure" | "override" | "unsupported_config") {
            assert_eq!(
                image_tokens, None,
                "{case}: failed counts must not become zero"
            );
        }
    }
}
