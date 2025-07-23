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

use bitflags::bitflags;
use serde::{Deserialize, Serialize};
use strum::Display;
use std::fmt;

bitflags! {
    #[derive(Copy, Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
    pub struct ModelType: u8 {
        const Chat = 1 << 0;
        const Completion = 1 << 1;
        const Embedding = 1 << 2;
        const Backend = 1 << 3;
    }
}

impl ModelType {
    pub fn as_str(&self) -> String {
        self.as_vec().join(",")
    }

    pub fn supports_chat(&self) -> bool {
        self.contains(ModelType::Chat)
    }
    pub fn supports_completion(&self) -> bool {
        self.contains(ModelType::Completion)
    }
    pub fn supports_embedding(&self) -> bool {
        self.contains(ModelType::Embedding)
    }
    pub fn supports_backend(&self) -> bool {
        self.contains(ModelType::Backend)
    }

    pub fn as_vec(&self) -> Vec<&'static str> {
        let mut result = Vec::new();
        if self.supports_chat() { result.push("chat"); }
        if self.supports_completion() { result.push("completion"); }
        if self.supports_embedding() { result.push("embedding"); }
        if self.supports_backend() { result.push("backend"); }
        result
    }
}

impl fmt::Display for ModelType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Copy, Debug, Clone, Display, Serialize, Deserialize, Eq, PartialEq)]
pub enum ModelInput {
    /// Raw text input
    Text,
    /// Pre-processed input
    Tokens,
}

impl ModelInput {
    pub fn as_str(&self) -> &str {
        match self {
            Self::Text => "text",
            Self::Tokens => "tokens",
        }
    }
}
