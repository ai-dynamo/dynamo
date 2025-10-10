// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::backend::Backend;
use dynamo_llm::model_card::ModelDeploymentCard;

#[test]
fn test_sequence_factory() {
    let mdc = ModelDeploymentCard::load_from_disk("tests/data/sample-models/TinyLlama_v1.1", None)
        .unwrap();

    let operator = Backend::from_mdc(&mdc);

    let mut decode_stream = operator
        .tokenizer
        .as_ref()
        .unwrap()
        .decode_stream(&[], false);
    let output = decode_stream.step(1).unwrap();
    assert_eq!(output, Some("<s>".to_string()));

    let mut decode_stream = operator
        .tokenizer
        .as_ref()
        .unwrap()
        .decode_stream(&[], true);
    let output = decode_stream.step(1).unwrap();
    assert_eq!(output, None);
}
