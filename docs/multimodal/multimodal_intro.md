<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
All rights reserved.
SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Multimodal Inference in Dynamo:

You can find example workflows and reference implementations for deploying a multimodal model using Dynamo in [multimodal examples](https://github.com/ai-dynamo/dynamo/tree/main/examples/multimodal).

##  EPD vs. PD Disaggregation
Dynamo supports two primary approaches for processing multimodal inputs, which differ in how the initial media encoding step is handled relative to the main LLM inference engine.

### 1. EPD (Encode-Prefill-Decode) Disaggregation
The EPD approach introduces an explicit separation of the media encoding step, maximizing the utilization of specialized hardware and increasing overall system efficiency for large multimodal models.

* **Media Input:** Image, video, audio, or an embedding URL is provided.
* **Process Flow:**
    1.  A dedicated **Encode Worker** is launched separately to handle the embedding extraction from the media input.
    2.  The extracted embeddings are transferred to the main engine via the **NVIDIA Inference Xfer Library (NIXL)**.
    3.  The main **Engine** performs the remaining **Prefill Decode Disaggregation** steps to generate the output.
* **Benefit:** This disaggregation allows for the decoupling of media encoding hardware/resources from the main LLM serving engine, making the serving of large multimodal models more efficient.

### 2. PD (Prefill-Decode) Disaggregation

The PD approach is a more traditional, aggregated method where the inference engine handles the entire process.
* **Media Input:** Image, video, or audio is loaded.
* **Process Flow:**
    1.  The main **Engine** receives the media input.
    2.  The Engine executes the full sequence: **Encode + Prefill + Decode**.
* **Note:** In this approach, the encoding step is executed within the same pipeline as the prefill and decode phases.

## Inference Framework Support Matrix

Dynamo supports multimodal capabilities across leading LLM inference backends, including **vLLM**, **TensorRT-LLM (TRT-LLM)**, and **SGLang**. The table below details the current support level for EPD/PD and various media types for each stack.

| Stack | EPD Support | PD Support | Image | Video | Audio |
| --------- | --------- | --------- | --------- |---------| --------- |
| **vLLM** | ✅  | ✅  | ✅  | ✅  | 🚧 |
| **TRT-LLM** | ✅  (Currently via precomputed Embeddings URL) | ✅  | ✅  | ❌  | ❌  |
| **SGLang** | ✅  | ❌  | ✅  | ❌  | ❌  |
