// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::protocols;

pub use protocols::{Annotated, TokenIdType};

pub mod openai {
    use super::*;
    use dynamo_runtime::pipeline::{ServerStreamingEngine, UnaryEngine};

    pub mod completions {
        use super::*;

        pub use protocols::openai::completions::{
            NvCreateCompletionRequest, NvCreateCompletionResponse,
        };

        /// A [`UnaryEngine`] implementation for the OpenAI Completions API
        pub type OpenAICompletionsUnaryEngine =
            UnaryEngine<NvCreateCompletionRequest, NvCreateCompletionResponse>;

        /// A [`ServerStreamingEngine`] implementation for the OpenAI Completions API
        pub type OpenAICompletionsStreamingEngine =
            ServerStreamingEngine<NvCreateCompletionRequest, Annotated<NvCreateCompletionResponse>>;
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

    pub mod images {
        use super::*;

        pub use protocols::openai::images::{NvCreateImageRequest, NvImagesResponse};

        /// A [`UnaryEngine`] implementation for the OpenAI Images API
        pub type OpenAIImagesUnaryEngine = UnaryEngine<NvCreateImageRequest, NvImagesResponse>;

        /// A [`ServerStreamingEngine`] implementation for the OpenAI Images API.
        ///
        /// **Note**: This "streaming" refers to the internal routing/distribution architecture,
        /// NOT client-facing Server-Sent Events (SSE) streaming. Image generation does not
        /// support progressive streaming to clients - images are generated completely and
        /// returned as finished artifacts (URLs or base64).
        ///
        /// The HTTP endpoint folds this stream into a single response before returning to clients,
        /// similar to how embeddings work. The streaming infrastructure is used for:
        /// - Consistent routing architecture across all model types
        /// - Request distribution via PushRouter
        /// - Worker fault detection and load balancing
        pub type OpenAIImagesStreamingEngine =
            ServerStreamingEngine<NvCreateImageRequest, Annotated<NvImagesResponse>>;
    }

    pub mod videos {
        use super::*;

        pub use protocols::openai::videos::{NvCreateVideoRequest, NvVideosResponse};

        /// A [`UnaryEngine`] implementation for the OpenAI Videos API
        pub type OpenAIVideosUnaryEngine = UnaryEngine<NvCreateVideoRequest, NvVideosResponse>;

        /// A [`ServerStreamingEngine`] implementation for the OpenAI Videos API
        pub type OpenAIVideosStreamingEngine =
            ServerStreamingEngine<NvCreateVideoRequest, Annotated<NvVideosResponse>>;
    }

    pub mod audios {
        use super::*;

        pub use protocols::openai::audios::{NvAudioSpeechResponse, NvCreateAudioSpeechRequest};

        /// A [`UnaryEngine`] implementation for the Audio Speech API
        pub type OpenAIAudiosUnaryEngine =
            UnaryEngine<NvCreateAudioSpeechRequest, NvAudioSpeechResponse>;

        /// A [`ServerStreamingEngine`] implementation for the Audio Speech API
        pub type OpenAIAudiosStreamingEngine =
            ServerStreamingEngine<NvCreateAudioSpeechRequest, Annotated<NvAudioSpeechResponse>>;
    }
}

pub mod generic {
    use super::*;
    use dynamo_runtime::pipeline::{
        BidirectionalStreamingEngine, ServerStreamingEngine, UnaryEngine,
    };

    pub mod tensor {
        use super::*;

        pub use protocols::tensor::{NvCreateTensorRequest, NvCreateTensorResponse};

        /// A [`UnaryEngine`] implementation for the generic Tensor API
        pub type TensorUnaryEngine = UnaryEngine<NvCreateTensorRequest, NvCreateTensorResponse>;

        /// A [`ServerStreamingEngine`] implementation for the generic Tensor API
        pub type TensorStreamingEngine =
            ServerStreamingEngine<NvCreateTensorRequest, Annotated<NvCreateTensorResponse>>;
    }

    pub mod realtime {
        use super::*;

        pub use protocols::openai::chat_completions::{
            NvCreateChatCompletionRequest, NvCreateChatCompletionStreamResponse,
        };

        /// TODO (#9175): reuses chat-completion request/response for now, as
        /// the request can carry audio data on the request side and text data
        /// on the response side. Will move to a dedicated realtime-API
        /// protocol type.
        ///
        /// A [`BidirectionalStreamingEngine`] implementation for the OpenAI Chat
        /// Completions API.
        ///
        /// Many-in / many-out: the client streams a sequence of `NvCreateChatCompletionRequest`
        /// chunks for the same logical session and receives a stream of
        /// `NvCreateChatCompletionStreamResponse` chunks back. Used by the experimental
        /// `/v1/realtime` WebSocket endpoint. The canonical concrete implementor of
        /// the input side is [`dynamo_runtime::pipeline::RequestStream`].
        pub type RealtimeBidirectionalEngine = BidirectionalStreamingEngine<
            NvCreateChatCompletionRequest,
            Annotated<NvCreateChatCompletionStreamResponse>,
        >;
    }
}
