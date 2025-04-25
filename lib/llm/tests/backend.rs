// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use dynamo_llm::backend::Backend;
use dynamo_llm::model_card::model::ModelDeploymentCard;

#[tokio::test]
async fn test_sequence_factory() {
    let mdc = ModelDeploymentCard::from_local_path("tests/data/sample-models/TinyLlama_v1.1", None)
        .await
        .unwrap();

    let operator = Backend::from_mdc(mdc).await.unwrap();

    let mut decode_stream = operator.tokenizer.as_ref().unwrap().decode_stream(false);
    let output = decode_stream.step(1).unwrap();
    assert_eq!(output, Some("<s>".to_string()));

    let mut decode_stream = operator.tokenizer.as_ref().unwrap().decode_stream(true);
    let output = decode_stream.step(1).unwrap();
    assert_eq!(output, None);
}
