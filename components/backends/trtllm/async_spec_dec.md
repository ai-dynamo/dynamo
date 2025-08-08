<!--
SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Async Speculative Decoding

This guide demonstrates how to run Draft-Target Model (DTM) speculative decoding asynchronously in Dynamo, where the draft model and target model run as separate Dynamo workers with the TRT-LLM backend.

## Setup

Follow the [Quickstart setup](./README.md#quick-start) instructions. Then, inside the container, run the following example:

```
cd $DYNAMO_HOME/components/backends/trtllm
./launch/spec_dec.sh
```

To scale up the number of drafters:

```
cd $DYNAMO_HOME/components/backends/trtllm
export NUM_DRAFTERS=2
export DRAFTER_CUDA_VISIBLE_DEVICES:-"1,2"
./launch/spec_dec.sh
```