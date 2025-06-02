// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

use derive_builder::Builder;
use serde::{Deserialize, Serialize};
use validator::{Validate, ValidationError};

pub trait NvExtProvider {
    fn nvext(&self) -> Option<&NvExt>;
}

/// NVIDIA LLM extensions to the OpenAI API
#[derive(Serialize, Deserialize, Builder, Validate, Debug, Clone)]
#[validate(schema(function = "validate_nv_ext"))]
pub struct NvExt {
    /// Annotations
    /// User requests triggers which result in the request issue back out-of-band information in the SSE
    /// stream using the `event:` field.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default, setter(strip_option))]
    pub annotations: Option<Vec<String>>,
}

impl Default for NvExt {
    fn default() -> Self {
        NvExt::builder().build().unwrap()
    }
}

impl NvExt {
    pub fn builder() -> NvExtBuilder {
        NvExtBuilder::default()
    }
}

fn validate_nv_ext(_nv_ext: &NvExt) -> Result<(), ValidationError> {
    Ok(())
}

impl NvExtBuilder {
    pub fn add_annotation(&mut self, annotation: impl Into<String>) -> &mut Self {
        self.annotations
            .get_or_insert_with(|| Some(vec![]))
            .as_mut()
            .expect("stop should always be Some(Vec)")
            .push(annotation.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test default builder configuration
    #[test]
    fn test_nv_ext_builder_default() {
        let nv_ext = NvExt::builder().build().unwrap();
        assert_eq!(nv_ext.annotations, None);
    }
}
