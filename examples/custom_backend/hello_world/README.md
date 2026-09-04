<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

# Hello World

A two-stage introduction to building custom backends on Dynamo. Neither stage
needs a GPU, a model, or any downloads.

| Stage | What it teaches | Start here if |
|---|---|---|
| [`basic/`](basic/README.md) | The Dynamo runtime primitives: create a worker, expose an endpoint, call it from a client. One file, no frontend. | You are new to the Dynamo runtime, or you are building a custom service that is not a model backend. |
| [`engine/`](engine/README.md) | The unified backend contract: a complete (toy) engine with a tokenizer, streaming generation, sampling parameters, and synthetic KV events — registered with the frontend and served through the OpenAI-compatible API with KV-aware routing. | You want to bring your own engine or model server to Dynamo. |

Work through them in order: `basic/` is the runtime "hello world" the
[Runtime Development Guide](../../../docs/fern/pages/developer-guide/additional-resources/runtime-development-guide.md)
walks through; `engine/` builds on those concepts to implement everything a
real model backend provides, so the standard frontend, router, and OpenAI API
work against it unchanged.
