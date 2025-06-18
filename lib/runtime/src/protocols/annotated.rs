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

use super::*;
use crate::{error, Result};

pub trait AnnotationsProvider {
    fn annotations(&self) -> Option<Vec<String>>;
    fn has_annotation(&self, annotation: &str) -> bool {
        self.annotations()
            .map(|annotations| annotations.iter().any(|a| a == annotation))
            .unwrap_or(false)
    }
}

/// Our services have the option of returning an "annotated" stream, which allows use
/// to include additional information with each delta. This is useful for debugging,
/// performance benchmarking, and improved observability.
#[derive(Serialize, Deserialize, Debug)]
pub struct Annotated<R> {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<R>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub event: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub comment: Option<Vec<String>>,
}

impl<R> Annotated<R> {
    /// Create a new annotated stream from the given error
    pub fn from_error(error: String) -> Self {
        Self {
            data: None,
            id: None,
            event: Some("error".to_string()),
            comment: Some(vec![error]),
        }
    }

    /// Create a new annotated stream from the given data
    pub fn from_data(data: R) -> Self {
        Self {
            data: Some(data),
            id: None,
            event: None,
            comment: None,
        }
    }

    /// Add an annotation to the stream
    ///
    /// Annotations populate the `event` field and the `comment` field
    pub fn from_annotation<S: Serialize>(
        name: impl Into<String>,
        value: &S,
    ) -> Result<Self, serde_json::Error> {
        Ok(Self {
            data: None,
            id: None,
            event: Some(name.into()),
            comment: Some(vec![serde_json::to_string(value)?]),
        })
    }

    /// Convert to a [`Result<Self, String>`]
    /// If [`Self::event`] is "error", return an error message(s) held by [`Self::comment`]
    pub fn ok(self) -> Result<Self, String> {
        if let Some(event) = &self.event {
            if event == "error" {
                return Err(self
                    .comment
                    .unwrap_or(vec!["unknown error".to_string()])
                    .join(", "));
            }
        }
        Ok(self)
    }

    pub fn is_ok(&self) -> bool {
        self.event.as_deref() != Some("error")
    }

    pub fn is_err(&self) -> bool {
        !self.is_ok()
    }

    pub fn is_event(&self) -> bool {
        self.event.is_some()
    }

    pub fn transfer<U: Serialize>(self, data: Option<U>) -> Annotated<U> {
        Annotated::<U> {
            data,
            id: self.id,
            event: self.event,
            comment: self.comment,
        }
    }

    /// Apply a mapping/transformation to the data field
    /// If the mapping fails, the error is returned as an annotated stream
    pub fn map_data<U, F>(self, transform: F) -> Annotated<U>
    where
        F: FnOnce(R) -> Result<U, String>,
    {
        match self.data.map(transform).transpose() {
            Ok(data) => Annotated::<U> {
                data,
                id: self.id,
                event: self.event,
                comment: self.comment,
            },
            Err(e) => Annotated::from_error(e),
        }
    }

    pub fn is_error(&self) -> bool {
        self.event.as_deref() == Some("error")
    }

    pub fn into_result(self) -> Result<Option<R>> {
        match self.data {
            Some(data) => Ok(Some(data)),
            None => match self.event {
                Some(event) if event == "error" => Err(error!(self
                    .comment
                    .unwrap_or(vec!["unknown error".to_string()])
                    .join(", ")))?,
                _ => Ok(None),
            },
        }
    }
}

// impl<R> Annotated<R>
// where
//     R: for<'de> Deserialize<'de> + Serialize,
// {
//     pub fn convert_sse_stream(
//         stream: DataStream<Result<Message, SseCodecError>>,
//     ) -> DataStream<Annotated<R>> {
//         let stream = stream.map(|message| match message {
//             Ok(message) => {
//                 let delta = Annotated::<R>::try_from(message);
//                 match delta {
//                     Ok(delta) => delta,
//                     Err(e) => Annotated::from_error(e.to_string()),
//                 }
//             }
//             Err(e) => Annotated::from_error(e.to_string()),
//         });
//         Box::pin(stream)
//     }
// }
