// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use serde_json::json;
use tempfile::{TempDir, tempdir};

pub fn model_dir(with_hf_config: bool, version: &str, efforts: &[&str]) -> TempDir {
    let dir = tempdir().unwrap();
    let source =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/sample-models/mock-mistral-tekken");
    std::fs::copy(source.join("params.json"), dir.path().join("params.json")).unwrap();
    let mut tekken: serde_json::Value =
        serde_json::from_slice(&std::fs::read(source.join("tekken.json")).unwrap()).unwrap();
    tekken["config"]["version"] = version.into();
    tekken["model_settings_builder"] = if version == "v13" {
        serde_json::Value::Null
    } else {
        json!({"reasoning_effort": {
            "type": "enum", "accepts_none": true, "default": "high", "values": efforts
        }})
    };
    write(dir.path(), "tekken.json", &tekken);
    write(
        dir.path(),
        "tokenizer.json",
        &json!({
            "version": "1.0", "truncation": null, "padding": null, "added_tokens": [],
            "normalizer": null, "pre_tokenizer": null, "post_processor": null, "decoder": null,
            "model": {"type": "WordLevel", "vocab": {"[UNK]": 0, "hi": 1}, "unk_token": "[UNK]"}
        }),
    );
    write(
        dir.path(),
        "tokenizer_config.json",
        &json!({
            "bos_token": "<s>", "eos_token": "</s>", "chat_template": "BUNDLED HF DEFAULT"
        }),
    );
    std::fs::write(dir.path().join("chat_template.jinja"), "BUNDLED HF DEFAULT").unwrap();
    write(
        dir.path(),
        "generation_config.json",
        &json!({"temperature": 0.4, "eos_token_id": 2}),
    );
    if with_hf_config {
        write(
            dir.path(),
            "config.json",
            &json!({
                "model_type": "mistral_large4", "architectures": ["MistralLarge4ForConditionalGeneration"], "bos_token_id": 1, "eos_token_id": 2,
                "vocab_size": 1040, "max_position_embeddings": 65536
            }),
        );
    }
    dir
}

fn write(dir: &Path, name: &str, value: &serde_json::Value) {
    std::fs::write(dir.join(name), serde_json::to_vec(value).unwrap()).unwrap();
}
