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


use triton_distributed_runtime::pipeline::{
    ServiceFrontend, SegmentSource, SegmentSink, SingleIn, ManyOut,
};


use crate::protocols::{
    Annotated,
    common::llm_backend::{BackendInput as BackendRequest, BackendOutput as BackendResponse},
    openai::{
        chat_completions::{
            NvCreateChatCompletionRequest , NvCreateChatCompletionStreamResponse
        },
        completions::{
            CompletionRequest, CompletionResponse,
        },
    },
};

use std::sync::Arc;

pub struct OpenAIChatService {}

impl OpenAIChatService {
    pub fn frontend() -> Arc<ServiceFrontend<SingleIn<NvCreateChatCompletionRequest>, ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>>> {
        ServiceFrontend::new()
    }

    pub fn segment_source() -> Arc<SegmentSource<SingleIn<NvCreateChatCompletionRequest>, ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>>> {
        SegmentSource::new()
    }

    pub fn segment_sink() -> Arc<SegmentSink<SingleIn<NvCreateChatCompletionRequest>, ManyOut<Annotated<NvCreateChatCompletionStreamResponse>>>> {
        SegmentSink::new()
    }
}

pub struct OpenAICompletionsService {}

impl OpenAICompletionsService {
        pub fn frontend() -> Arc<ServiceFrontend<SingleIn<CompletionRequest>, ManyOut<Annotated<CompletionResponse>>>> {
        ServiceFrontend::new()
    }

        pub fn segment_source() -> Arc<SegmentSource<SingleIn<CompletionRequest>, ManyOut<Annotated<CompletionResponse>>>> {
        SegmentSource::new()
    }

    pub fn segment_sink() -> Arc<SegmentSink<SingleIn<CompletionRequest>, ManyOut<Annotated<CompletionResponse>>>> {
        SegmentSink::new()
    }
}

pub struct BackendLLMService {}

impl BackendLLMService {
    pub fn frontend() -> Arc<ServiceFrontend<SingleIn<BackendRequest>, ManyOut<Annotated<BackendResponse>>>> {
        ServiceFrontend::new()
    }

    pub fn segment_source() -> Arc<SegmentSource<SingleIn<BackendRequest>, ManyOut<Annotated<BackendResponse>>>> {
        SegmentSource::new()
    }

    pub fn segment_sink() -> Arc<SegmentSink<SingleIn<BackendRequest>, ManyOut<Annotated<BackendResponse>>>> {
        SegmentSink::new()
    }
}
