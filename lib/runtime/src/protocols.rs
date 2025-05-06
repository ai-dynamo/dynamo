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

use serde::{Deserialize, Serialize};
use std::str::FromStr;

use crate::pipeline::PipelineError;

pub mod annotated;

pub type LeaseId = i64;

/// Default namespace if user does not provide one
const DEFAULT_NAMESPACE: &str = "NS";

const DEFAULT_COMPONENT: &str = "C";

const DEFAULT_ENDPOINT: &str = "E";

/// How we identify a namespace/component/endpoint URL.
/// Technically the '://' is not part of the scheme but it eliminates several string
/// concatenations.
pub const ENDPOINT_SCHEME: &str = "dyn://";

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct Component {
    pub name: String,
    pub namespace: String,
}

/// Represents an endpoint with a namespace, component, and name.
///
/// An `Endpoint` is defined by a three-part string separated by `/` or a '.':
/// - **namespace**
/// - **component**
/// - **name**
///
/// Example format: `"namespace/component/endpoint"`
///
/// TODO: There is also an Endpoint in runtime/src/component.rs
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct Endpoint {
    pub namespace: String,
    pub component: String,
    pub name: String,
}

impl PartialEq<Vec<&str>> for Endpoint {
    fn eq(&self, other: &Vec<&str>) -> bool {
        if other.len() != 3 {
            return false;
        }

        self.namespace == other[0] && self.component == other[1] && self.name == other[2]
    }
}

impl PartialEq<Endpoint> for Vec<&str> {
    fn eq(&self, other: &Endpoint) -> bool {
        other == self
    }
}

impl Default for Endpoint {
    fn default() -> Self {
        Endpoint {
            namespace: DEFAULT_NAMESPACE.to_string(),
            component: DEFAULT_COMPONENT.to_string(),
            name: DEFAULT_ENDPOINT.to_string(),
        }
    }
}

impl From<&str> for Endpoint {
    /// Creates an `Endpoint` from a string.
    ///
    /// # Arguments
    /// - `path`: A string in the format `"namespace/component/endpoint"`.
    ///
    /// The first two parts become the first two elements of the vector.
    /// The third and subsequent parts are joined with '_' and become the third element.
    /// Default values are used for missing parts.
    ///
    /// # Examples:
    /// - "component" -> ["DEFAULT_NS", "component", "DEFAULT_E"]
    /// - "namespace.component" -> ["namespace", "component", "DEFAULT_E"]
    /// - "namespace.component.endpoint" -> ["namespace", "component", "endpoint"]
    /// - "namespace/component" -> ["namespace", "component", "DEFAULT_E"]
    /// - "namespace.component.endpoint.other.parts" -> ["namespace", "component", "endpoint_other_parts"]
    ///
    /// # Examples
    /// ```ignore
    /// use dynamo_runtime:protocols::Endpoint;
    ///
    /// let endpoint = Endpoint::from("namespace/component/endpoint");
    /// assert_eq!(endpoint.namespace, "namespace");
    /// assert_eq!(endpoint.component, "component");
    /// assert_eq!(endpoint.name, "endpoint");
    /// ```
    fn from(input: &str) -> Self {
        let mut result = Endpoint::default();

        // Split the input string on either '.' or '/'
        let elements: Vec<&str> = input
            .trim_matches([' ', '/', '.'])
            .split(['.', '/'])
            .filter(|x| !x.is_empty())
            .collect();

        match elements.len() {
            0 => {}
            1 => {
                result.component = elements[0].to_string();
            }
            2 => {
                result.namespace = elements[0].to_string();
                result.component = elements[1].to_string();
            }
            3 => {
                result.namespace = elements[0].to_string();
                result.component = elements[1].to_string();
                result.name = elements[2].to_string();
            }
            x if x > 3 => {
                result.namespace = elements[0].to_string();
                result.component = elements[1].to_string();
                result.name = elements[2..].join("_");
            }
            _ => unreachable!(),
        }
        result
    }
}

impl FromStr for Endpoint {
    type Err = PipelineError;

    /// Parses an `Endpoint` from a string using the standard Rust `.parse::<T>()` pattern.
    ///
    /// This is implemented in terms of [`From<&str>`].
    ///
    /// # Errors
    /// Does not fail
    ///
    /// # Examples
    /// ```ignore
    /// use std::str::FromStr;
    /// use dynamo_runtime:protocols::Endpoint;
    ///
    /// let endpoint: Endpoint = "namespace/component/endpoint".parse().unwrap();
    /// assert_eq!(endpoint.namespace, "namespace");
    /// assert_eq!(endpoint.component, "component");
    /// assert_eq!(endpoint.name, "endpoint");
    /// let endpoint: Endpoint = "dyn://namespace/component/endpoint".parse().unwrap();
    /// // same as above
    /// assert_eq!(endpoint.name, "endpoint");
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let cleaned = s.strip_prefix(ENDPOINT_SCHEME).unwrap_or(s);
        Ok(Endpoint::from(cleaned))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
#[serde(rename_all = "snake_case")]
pub enum RouterType {
    PushRoundRobin,
    PushRandom,
}

impl Default for RouterType {
    fn default() -> Self {
        Self::PushRandom
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq)]
pub struct ModelMetaData {
    pub name: String,
    pub component: Component,
    pub router_type: RouterType,
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryFrom;
    use std::str::FromStr;

    #[test]
    fn test_router_type_default() {
        let default_router = RouterType::default();
        assert_eq!(default_router, RouterType::PushRandom);
    }

    #[test]
    fn test_router_type_serialization() {
        let router_round_robin = RouterType::PushRoundRobin;
        let router_random = RouterType::PushRandom;

        let serialized_round_robin = serde_json::to_string(&router_round_robin).unwrap();
        let serialized_random = serde_json::to_string(&router_random).unwrap();

        assert_eq!(serialized_round_robin, "\"push_round_robin\"");
        assert_eq!(serialized_random, "\"push_random\"");
    }

    #[test]
    fn test_router_type_deserialization() {
        let round_robin: RouterType = serde_json::from_str("\"push_round_robin\"").unwrap();
        let random: RouterType = serde_json::from_str("\"push_random\"").unwrap();

        assert_eq!(round_robin, RouterType::PushRoundRobin);
        assert_eq!(random, RouterType::PushRandom);
    }

    #[test]
    fn test_valid_endpoint_from() {
        let input = "namespace1/component1/endpoint1";
        let endpoint = Endpoint::from(input);

        assert_eq!(endpoint.namespace, "namespace1");
        assert_eq!(endpoint.component, "component1");
        assert_eq!(endpoint.name, "endpoint1");
    }

    #[test]
    fn test_valid_endpoint_from_str() {
        let input = "namespace2/component2/endpoint2";
        let endpoint = Endpoint::from_str(input).unwrap();

        assert_eq!(endpoint.namespace, "namespace2");
        assert_eq!(endpoint.component, "component2");
        assert_eq!(endpoint.name, "endpoint2");
    }

    #[test]
    fn test_valid_endpoint_parse() {
        let input = "namespace3/component3/endpoint3";
        let endpoint: Endpoint = input.parse().unwrap();

        assert_eq!(endpoint.namespace, "namespace3");
        assert_eq!(endpoint.component, "component3");
        assert_eq!(endpoint.name, "endpoint3");
    }

    #[test]
    fn test_endpoint_from() {
        let result = Endpoint::from("component");
        assert_eq!(
            result,
            vec![DEFAULT_NAMESPACE, "component", DEFAULT_ENDPOINT]
        );
    }

    #[test]
    fn test_namespace_component_endpoint() {
        let result = Endpoint::from("namespace.component.endpoint");
        assert_eq!(result, vec!["namespace", "component", "endpoint"]);
    }

    #[test]
    fn test_forward_slash_separator() {
        let result = Endpoint::from("namespace/component");
        assert_eq!(result, vec!["namespace", "component", DEFAULT_ENDPOINT]);
    }

    #[test]
    fn test_multiple_parts() {
        let result = Endpoint::from("namespace.component.endpoint.other.parts");
        assert_eq!(
            result,
            vec!["namespace", "component", "endpoint_other_parts"]
        );
    }

    #[test]
    fn test_mixed_separators() {
        // Do it the .into way for variety and documentation
        let result: Endpoint = "namespace/component.endpoint".into();
        assert_eq!(result, vec!["namespace", "component", "endpoint"]);
    }

    #[test]
    fn test_empty_string() {
        let result = Endpoint::from("");
        assert_eq!(
            result,
            vec![DEFAULT_NAMESPACE, DEFAULT_COMPONENT, DEFAULT_ENDPOINT]
        );

        // White space is equivalent to an empty string
        let result = Endpoint::from("   ");
        assert_eq!(
            result,
            vec![DEFAULT_NAMESPACE, DEFAULT_COMPONENT, DEFAULT_ENDPOINT]
        );
    }
}
