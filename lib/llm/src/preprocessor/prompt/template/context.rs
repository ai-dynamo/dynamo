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

use super::{ContextMixins, PromptContextMixin};

use chrono::{DateTime, Utc};
use minijinja::value::{Object, Value};
use std::sync::Arc;

impl Object for ContextMixins {
    fn get_value(self: &Arc<Self>, field: &Value) -> Option<Value> {
        match field.as_str()? {
            "datetime" => self.datetime(),
            _ => None,
        }
    }
}

impl ContextMixins {
    pub fn new(allowed_mixins: &[PromptContextMixin]) -> Self {
        Self {
            context_mixins: allowed_mixins.iter().cloned().collect(),
        }
    }

    /// Implements the `datetime` context mixin.
    /// Different mixins can be implemented here for the same key.
    /// We need to valiate that multiple mixins do not conflict with each other.
    fn datetime(&self) -> Option<Value> {
        if self
            .context_mixins
            .contains(&PromptContextMixin::Llama3DateTime)
        {
            let now = chrono::Utc::now();
            Some(Value::from(llama3_datetime(now)))
        } else {
            None
        }
    }
}

fn llama3_datetime(datetime: DateTime<Utc>) -> String {
    datetime.format("%d, %B, %Y").to_string()
}
