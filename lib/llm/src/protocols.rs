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

//! # Dynamo LLM Protocols
//!
//! This module contains the protocols, i.e. messages formats, used to exchange requests and responses
//! both publicly via the HTTP API and internally between Dynamo components.
//!

use futures::{Stream, StreamExt};
use serde::{Deserialize, Serialize};

pub mod codec;
pub mod common;
pub mod openai;

/// The token ID type
pub type TokenIdType = u32;
pub use dynamo_runtime::engine::DataStream;

// TODO: This is an awkward dependency that we need to address
// Originally, all the Annotated/SSE Codec bits where in the LLM protocol module; however, [Annotated]
// has become the common response envelope for dynamo.
// We may want to move the original Annotated back here and has a Infallible conversion to the the
// ResponseEnvelop in dynamo.
pub use dynamo_runtime::protocols::annotated::Annotated;

/// The LLM responses have multiple different fields and nests of objects to get to the actual
/// text completion returned. This trait can be applied to the `choice` level objects to extract
/// the completion text.
///
/// To avoid an optional, if no completion text is found, the [`ContentProvider::content`] should
/// return an empty string.
pub trait ContentProvider {
    fn content(&self) -> String;
}

/// Converts of a stream of [codec::Message]s into a stream of [Annotated]s.
pub fn convert_sse_stream<R>(
    stream: impl Stream<Item = Result<codec::Message, codec::SseCodecError>>,
) -> impl Stream<Item = Annotated<R>>
where
    R: for<'de> Deserialize<'de> + Serialize,
{
    stream.map(|message| match message {
        Ok(message) => {
            let delta = Annotated::<R>::try_from(message);
            match delta {
                Ok(delta) => delta,
                Err(e) => Annotated::from_error(e.to_string()),
            }
        }
        Err(e) => Annotated::from_error(e.to_string()),
    })
}
