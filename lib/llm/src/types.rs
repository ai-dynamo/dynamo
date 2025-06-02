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

use crate::protocols;

pub use protocols::{Annotated, TokenIdType};

pub mod openai {
    use super::*;
    use dynamo_runtime::pipeline::{ServerStreamingEngine, UnaryEngine};

    pub mod completions {
        use super::*;

        pub use protocols::openai::completions::{CompletionRequest, CompletionResponse};

        /// A [`UnaryEngine`] implementation for the OpenAI Completions API
        pub type OpenAICompletionsUnaryEngine = UnaryEngine<CompletionRequest, CompletionResponse>;

        /// A [`ServerStreamingEngine`] implementation for the OpenAI Completions API
        pub type OpenAICompletionsStreamingEngine =
            ServerStreamingEngine<CompletionRequest, Annotated<CompletionResponse>>;
    }

    pub mod chat_completions {
        use super::*;

        pub use protocols::openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionResponse,
            NvCreateChatCompletionStreamResponse,
        };

        /// A [`UnaryEngine`] implementation for the OpenAI Chat Completions API
        pub type OpenAIChatCompletionsUnaryEngine =
            UnaryEngine<NvCreateChatCompletionRequest, NvCreateChatCompletionResponse>;

        /// A [`ServerStreamingEngine`] implementation for the OpenAI Chat Completions API
        pub type OpenAIChatCompletionsStreamingEngine = ServerStreamingEngine<
            NvCreateChatCompletionRequest,
            Annotated<NvCreateChatCompletionStreamResponse>,
        >;
    }

    pub mod embeddings {
        use super::*;

        pub use protocols::openai::embeddings::{
            NvCreateEmbeddingRequest, NvCreateEmbeddingResponse,
        };

        /// A [`UnaryEngine`] implementation for the OpenAI Embeddings API
        pub type OpenAIEmbeddingsUnaryEngine =
            UnaryEngine<NvCreateEmbeddingRequest, NvCreateEmbeddingResponse>;

        /// A [`ServerStreamingEngine`] implementation for the OpenAI Embeddings API
        pub type OpenAIEmbeddingsStreamingEngine =
            ServerStreamingEngine<NvCreateEmbeddingRequest, Annotated<NvCreateEmbeddingResponse>>;
    }
}
