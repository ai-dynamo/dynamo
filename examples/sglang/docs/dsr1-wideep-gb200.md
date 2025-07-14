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

# Running DeepSeek-R1 Disaggregated with WideEP on GB200s

Dynamo has **experimental** support for SGLang's implementation of wide expert parallelilsm and large scale P/D for DeepSeek-R1 on GB200s! You can read their blog post [here](https://lmsys.org/blog/2025-06-16-gb200-part-1/) for more details and read their reproduction steps [here](https://github.com/sgl-project/sglang/issues/7227). We will clean up these instructions and unify Dockerfiles as we iterate further.

## Instructions

1. Build the SGLang GB200 container from this branch [sglang-gb200](https://github.com/sgl-project/sglang/pull/7556). Make sure you are building this on an ARM64 machine.  

2. Build the Dynamo container 

