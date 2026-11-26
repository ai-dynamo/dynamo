// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Image-token accounting and placeholder resolution for Inkling 2 checkpoints.

use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;

use crate::protocols::TokenIdType;

use super::config::read_json;
use super::image::{ImagePromptKind, ImageRoutingBackend};

const IMAGE_PROCESSOR_TYPE: &str = "Inkling2ImageProcessor";
const PROCESSOR_CLASS: &str = "Inkling2Processor";

pub(super) fn supports_model_type(model_type: Option<&str>) -> bool {
    matches!(model_type, Some("inkling2" | "inkling2_loop"))
}

pub(super) fn is_inkling2_config(config: &Value) -> bool {
    supports_model_type(config.get("model_type").and_then(Value::as_str))
}

pub(super) struct Inkling2ImageRoutingBackend {
    patch_width: u32,
    patch_height: u32,
}

impl Inkling2ImageRoutingBackend {
    pub(super) fn try_new(model_dir: &Path) -> Result<Self> {
        let processor_cfg_path = model_dir.join("processor_config.json");
        let image_processor = if processor_cfg_path.exists() {
            let root = read_json(model_dir, "processor_config.json")?;
            if let Some(cls) = root.get("processor_class").and_then(Value::as_str) {
                anyhow::ensure!(
                    cls == PROCESSOR_CLASS,
                    "mm-routing: unsupported Inkling 2 processor_class {cls:?}; expected {PROCESSOR_CLASS:?}"
                );
            }
            root.get("image_processor")
                .cloned()
                .context("mm-routing: missing image_processor in processor_config.json")?
        } else {
            read_json(model_dir, "preprocessor_config.json")?
        };

        let proc_type = image_processor
            .get("image_processor_type")
            .and_then(Value::as_str)
            .context("mm-routing: missing Inkling 2 image_processor_type")?;
        anyhow::ensure!(
            proc_type == IMAGE_PROCESSOR_TYPE,
            "mm-routing: unsupported Inkling 2 image_processor_type {proc_type:?}; expected {IMAGE_PROCESSOR_TYPE:?}"
        );

        let do_resize = image_processor
            .get("do_resize")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        anyhow::ensure!(
            !do_resize,
            "mm-routing: Inkling 2 do_resize=true is unsupported"
        );

        let size = image_processor
            .get("size")
            .and_then(Value::as_object)
            .context("mm-routing: missing Inkling 2 image_processor.size object")?;
        let patch_width = required_positive_u32(size.get("width"), "size.width")?;
        let patch_height = required_positive_u32(size.get("height"), "size.height")?;

        Ok(Self {
            patch_width,
            patch_height,
        })
    }
}

impl ImageRoutingBackend for Inkling2ImageRoutingBackend {
    fn count_tokens(&self, width: u32, height: u32) -> Result<usize> {
        anyhow::ensure!(
            width > 0 && height > 0,
            "mm-routing: Inkling 2 image dimensions must be positive, got {width}x{height}"
        );
        let rows = usize::try_from(height.div_ceil(self.patch_height))
            .context("mm-routing: Inkling 2 patch row count overflow")?;
        let cols = usize::try_from(width.div_ceil(self.patch_width))
            .context("mm-routing: Inkling 2 patch column count overflow")?;
        rows.checked_mul(cols)
            .context("mm-routing: Inkling 2 image token count overflow")
    }

    fn routing_prompt_kind(&self) -> Option<ImagePromptKind> {
        Some(ImagePromptKind::RepeatedPad)
    }
}

pub(super) fn resolve_image_token_id(
    model_dir: &Path,
    config: &Value,
    tokenizer_config: Option<&Value>,
) -> Result<TokenIdType> {
    if let Some(raw_id) = config.get("image_token_id") {
        return raw_id
            .as_u64()
            .and_then(|id| TokenIdType::try_from(id).ok())
            .context("mm-routing: invalid Inkling 2 image_token_id in config.json");
    }

    let processor_config = read_json(model_dir, "processor_config.json")?;
    if let Some(cls) = processor_config
        .get("processor_class")
        .and_then(Value::as_str)
    {
        anyhow::ensure!(
            cls == PROCESSOR_CLASS,
            "mm-routing: unsupported Inkling 2 processor_class {cls:?}; expected {PROCESSOR_CLASS:?}"
        );
    }
    let image_token = processor_config
        .get("image_token")
        .and_then(Value::as_str)
        .filter(|s| !s.is_empty())
        .context("mm-routing: missing or empty image_token in processor_config.json")?;

    let owned_tc;
    let tc = match tokenizer_config {
        Some(tc) => Some(tc),
        None => {
            owned_tc = read_json(model_dir, "tokenizer_config.json").ok();
            owned_tc.as_ref()
        }
    };
    if let Some(id) = tc.and_then(|cfg| lookup_in_added_tokens_decoder(cfg, image_token)) {
        return Ok(id);
    }

    let tokenizer_json = read_json(model_dir, "tokenizer.json")?;
    if let Some(id) = lookup_in_tokenizer_json(&tokenizer_json, image_token) {
        return Ok(id);
    }

    bail!("mm-routing: Inkling 2 image_token {image_token:?} not found in tokenizer files")
}

fn required_positive_u32(value: Option<&Value>, field: &str) -> Result<u32> {
    let parsed = value
        .and_then(Value::as_u64)
        .and_then(|n| u32::try_from(n).ok())
        .filter(|&n| n > 0)
        .ok_or_else(|| anyhow!("mm-routing: missing or non-positive Inkling 2 {field}"))?;
    Ok(parsed)
}

fn lookup_in_added_tokens_decoder(tokenizer_config: &Value, token: &str) -> Option<TokenIdType> {
    let decoder = tokenizer_config.get("added_tokens_decoder")?.as_object()?;
    for (id_str, entry) in decoder {
        if entry.get("content").and_then(Value::as_str) == Some(token) {
            return id_str.parse::<TokenIdType>().ok();
        }
    }
    None
}

fn lookup_in_tokenizer_json(tokenizer_json: &Value, token: &str) -> Option<TokenIdType> {
    if let Some(added_tokens) = tokenizer_json.get("added_tokens").and_then(Value::as_array) {
        for entry in added_tokens {
            if entry.get("content").and_then(Value::as_str) == Some(token) {
                return entry
                    .get("id")
                    .and_then(Value::as_u64)
                    .and_then(|id| TokenIdType::try_from(id).ok());
            }
        }
    }
    tokenizer_json
        .get("model")
        .and_then(|m| m.get("vocab"))
        .and_then(|v| v.get(token))
        .and_then(Value::as_u64)
        .and_then(|id| TokenIdType::try_from(id).ok())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::common::checked_file::CheckedFile;
    use crate::model_card::ModelInfoType;
    use crate::preprocessor::mm_routing::image::{
        ImageRoutingProcessor, resolve_exact_routing_image_token_id, resolve_routing_tokens,
    };
    use crate::preprocessor::{MmRoutingEntry, ModelDeploymentCard, OpenAIPreprocessor};

    fn write_inkling2_checkpoint(dir: &Path, model_type: &str, with_tokenizer_config: bool) {
        std::fs::write(
            dir.join("config.json"),
            serde_json::json!({
                "architectures": ["Inkling2LoopForConditionalGeneration"],
                "model_type": model_type,
                "eos_token_id": 200006
            })
            .to_string(),
        )
        .unwrap();
        std::fs::write(
            dir.join("processor_config.json"),
            serde_json::json!({
                "image_processor": {
                    "image_processor_type": "Inkling2ImageProcessor",
                    "size": {"height": 40, "width": 40}
                },
                "image_token": "<|unused_200054|>",
                "processor_class": "Inkling2Processor"
            })
            .to_string(),
        )
        .unwrap();
        if with_tokenizer_config {
            std::fs::write(
                dir.join("tokenizer_config.json"),
                serde_json::json!({
                    "added_tokens_decoder": {
                        "200005": {"content": "<|content_image|>", "special": true},
                        "200054": {"content": "<|unused_200054|>", "special": true}
                    }
                })
                .to_string(),
            )
            .unwrap();
        } else {
            std::fs::write(
                dir.join("tokenizer.json"),
                serde_json::json!({
                    "added_tokens": [
                        {"id": 200005, "content": "<|content_image|>", "special": true},
                        {"id": 200054, "content": "<|unused_200054|>", "special": true}
                    ],
                    "model": {"vocab": {}}
                })
                .to_string(),
            )
            .unwrap();
        }
    }

    #[test]
    fn inkling2_counts_grid_patches_and_resolves_placeholder() {
        for (model_type, with_tc) in [("inkling2_loop", true), ("inkling2", false)] {
            let dir = tempfile::tempdir().unwrap();
            write_inkling2_checkpoint(dir.path(), model_type, with_tc);

            let counter = ImageRoutingProcessor::try_new(
                "thinkingmachines/Inkling-2-Mini",
                Some(model_type),
                dir.path(),
            )
            .unwrap();
            for ((w, h), expected) in [
                ((1, 1), 1),
                ((40, 40), 1),
                ((80, 80), 4),
                ((81, 40), 3),
                ((1920, 1080), 48 * 27),
            ] {
                assert_eq!(counter.count_tokens(w, h).unwrap(), expected);
            }
            assert_eq!(
                counter.routing_prompt_kind(),
                Some(ImagePromptKind::RepeatedPad)
            );

            let resolved = resolve_routing_tokens(
                "thinkingmachines/Inkling-2-Mini",
                dir.path(),
                Some(&counter),
            );
            assert_eq!(resolved.image_token_id, Some(200054));
            assert_eq!(resolved.chat_placeholder_token_id, Some(200054));
            assert_eq!(
                resolved.image_prompt_kind,
                Some(ImagePromptKind::RepeatedPad)
            );
            assert_eq!(
                resolve_exact_routing_image_token_id("thinkingmachines/Inkling-2-Mini", dir.path()),
                Some(200054)
            );
        }
    }

    #[test]
    fn inkling2_rejects_zero_dimensions_and_unsupported_resize() {
        let dir = tempfile::tempdir().unwrap();
        write_inkling2_checkpoint(dir.path(), "inkling2_loop", true);
        let counter = ImageRoutingProcessor::try_new(
            "thinkingmachines/Inkling-2-Mini",
            Some("inkling2_loop"),
            dir.path(),
        )
        .unwrap();
        assert!(counter.count_tokens(0, 40).is_err());
        assert!(counter.count_tokens(40, 0).is_err());

        let proc_path = dir.path().join("processor_config.json");
        let mut proc_cfg: Value =
            serde_json::from_str(&std::fs::read_to_string(&proc_path).unwrap()).unwrap();
        proc_cfg["image_processor"]["do_resize"] = Value::Bool(true);
        std::fs::write(&proc_path, proc_cfg.to_string()).unwrap();

        assert!(
            ImageRoutingProcessor::try_new(
                "thinkingmachines/Inkling-2-Mini",
                Some("inkling2_loop"),
                dir.path(),
            )
            .is_err()
        );
        assert_eq!(
            resolve_exact_routing_image_token_id("thinkingmachines/Inkling-2-Mini", dir.path()),
            None
        );
    }

    #[test]
    fn inkling2_preprocessor_routing_matches_worker_event_normalization() {
        let dir = tempfile::tempdir().unwrap();
        write_inkling2_checkpoint(dir.path(), "inkling2_loop", true);

        let mut mdc = ModelDeploymentCard::load_from_disk(
            "tests/data/sample-models/mock-llama-3.1-8b-instruct",
            None,
        )
        .unwrap();
        mdc.model_info = Some(ModelInfoType::HfConfigJson(
            CheckedFile::from_disk(dir.path().join("config.json")).unwrap(),
        ));
        mdc.kv_cache_block_size = 16;

        let preprocessor = OpenAIPreprocessor::new(mdc).unwrap();
        assert_eq!(preprocessor.routing_image_token_id, Some(200054));

        // Frontend renders `<|content_image|><|unused_200054|>` -> [200005, 200054].
        let frontend_ids = [200000, 200005, 200054, 200010];
        let mm_hash = 0xdead_beef_cafe_1234;
        let entries = [MmRoutingEntry::Image {
            mm_hash,
            width: 80,
            height: 80,
        }];
        let routing = preprocessor
            .build_mm_exact_routing_info(&entries, &frontend_ids, None)
            .unwrap();
        assert_eq!(routing.expanded_prompt_len, 7);

        // Worker strips the template seed pad and expands `<|content_image|>` to
        // `[200005] + [200054] * 4` for an 80x80 image (2x2 patches of 40x40).
        let worker_ids = [200000, 200005, 200054, 200054, 200054, 200054, 200010];
        let (normalized_worker, runs) =
            dynamo_kv_events::zmq_wire::normalize_mm_token_runs(&worker_ids, 200054, &[mm_hash])
                .unwrap();
        assert_eq!(runs, 1);
        assert_eq!(
            &routing.routing_token_ids[..routing.expanded_prompt_len],
            normalized_worker.as_slice()
        );
    }
}
