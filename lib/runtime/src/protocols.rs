// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::fmt;
use std::str::FromStr;

use serde::{Deserialize, Serialize};

pub mod annotated;
pub mod maybe_error;

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
/// An [EndpointId] is defined by a three-part string separated by `/` or a '.':
/// - **namespace**
/// - **component**
/// - **name**
///
/// Example format: `"namespace/component/endpoint"`
///
#[derive(Debug, Clone, Serialize, Deserialize, Eq, PartialEq, Hash)]
pub struct EndpointId {
    pub namespace: String,
    pub component: String,
    pub name: String,
}

impl fmt::Display for EndpointId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}/{}/{}", self.namespace, self.component, self.name)
    }
}

impl PartialEq<Vec<&str>> for EndpointId {
    fn eq(&self, other: &Vec<&str>) -> bool {
        if other.len() != 3 {
            return false;
        }

        self.namespace == other[0] && self.component == other[1] && self.name == other[2]
    }
}

impl PartialEq<[&str; 3]> for EndpointId {
    fn eq(&self, other: &[&str; 3]) -> bool {
        self.namespace == other[0] && self.component == other[1] && self.name == other[2]
    }
}

impl PartialEq<EndpointId> for [&str; 3] {
    fn eq(&self, other: &EndpointId) -> bool {
        other == self
    }
}

impl PartialEq<EndpointId> for Vec<&str> {
    fn eq(&self, other: &EndpointId) -> bool {
        other == self
    }
}

impl Default for EndpointId {
    fn default() -> Self {
        EndpointId {
            namespace: DEFAULT_NAMESPACE.to_string(),
            component: DEFAULT_COMPONENT.to_string(),
            name: DEFAULT_ENDPOINT.to_string(),
        }
    }
}

impl From<&str> for EndpointId {
    /// Creates an [EndpointId] from a string.
    ///
    /// # Arguments
    /// - `path`: A string in the format `"namespace/component/endpoint"`.
    ///
    /// The first two parts become the first two elements of the vector.
    /// The third and subsequent parts are joined with '_' and become the third element.
    /// Default values are used for missing parts.
    ///
    /// # Examples:
    /// - "component" -> ["DEFAULT_NAMESPACE", "component", "DEFAULT_ENDPOINT"]
    /// - "namespace.component" -> ["namespace", "component", "DEFAULT_ENDPOINT"]
    /// - "namespace.component.endpoint" -> ["namespace", "component", "endpoint"]
    /// - "namespace/component" -> ["namespace", "component", "DEFAULT_ENDPOINT"]
    /// - "namespace.component.endpoint.other.parts" -> ["namespace", "component", "endpoint_other_parts"]
    ///
    /// # Examples
    /// ```
    /// use dynamo_runtime::protocols::EndpointId;
    ///
    /// let endpoint = EndpointId::from("namespace/component/endpoint");
    /// assert_eq!(endpoint.namespace, "namespace");
    /// assert_eq!(endpoint.component, "component");
    /// assert_eq!(endpoint.name, "endpoint");
    /// ```
    fn from(s: &str) -> Self {
        let input = s.strip_prefix(ENDPOINT_SCHEME).unwrap_or(s);

        // Split the input string on either '.' or '/'
        let mut parts = input
            .trim_matches([' ', '/', '.'])
            .split(['.', '/'])
            .filter(|x| !x.is_empty());

        // Extract the first three potential components.
        let p1 = parts.next();
        let p2 = parts.next();
        let p3 = parts.next();

        let namespace;
        let component;
        let name;

        match (p1, p2, p3) {
            (None, _, _) => {
                // 0 elements: all fields remain empty.
                // Should this be an error?
                namespace = DEFAULT_NAMESPACE.to_string();
                component = DEFAULT_COMPONENT.to_string();
                name = DEFAULT_ENDPOINT.to_string();
            }
            (Some(c), None, _) => {
                namespace = DEFAULT_NAMESPACE.to_string();
                component = c.to_string();
                name = DEFAULT_ENDPOINT.to_string();
            }
            (Some(ns), Some(c), None) => {
                // 2 elements: namespace, component
                namespace = ns.to_string();
                component = c.to_string();
                name = DEFAULT_ENDPOINT.to_string();
            }
            (Some(ns), Some(c), Some(ep)) => {
                namespace = ns.to_string();
                component = c.to_string();

                // For the 'name' field, we need to handle 'n' and any remaining parts.
                // Instead of collecting into a Vec and then joining, we can build the string directly.
                let mut endpoint_buf = String::from(ep); // Start with the third part
                for part in parts {
                    // 'parts' iterator continues from where p3 left off
                    endpoint_buf.push('_');
                    endpoint_buf.push_str(part);
                }
                name = endpoint_buf;
            }
        }

        EndpointId {
            namespace,
            component,
            name,
        }
    }
}

impl FromStr for EndpointId {
    type Err = core::convert::Infallible;

    /// Parses an `EndpointId` from a string using the standard Rust `.parse::<T>()` pattern.
    ///
    /// This is implemented in terms of [`From<&str>`].
    ///
    /// # Errors
    /// Does not fail
    ///
    /// # Examples
    /// ```
    /// use std::str::FromStr;
    /// use dynamo_runtime::protocols::EndpointId;
    ///
    /// let endpoint: EndpointId = "namespace/component/endpoint".parse().unwrap();
    /// assert_eq!(endpoint.namespace, "namespace");
    /// assert_eq!(endpoint.component, "component");
    /// assert_eq!(endpoint.name, "endpoint");
    /// let endpoint: EndpointId = "dyn://namespace/component/endpoint".parse().unwrap();
    /// // same as above
    /// assert_eq!(endpoint.name, "endpoint");
    /// ```
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(EndpointId::from(s))
    }
}

impl EndpointId {
    /// As a String like dyn://dynamo.internal.worker
    pub fn as_url(&self) -> String {
        format!(
            "{ENDPOINT_SCHEME}{}.{}.{}",
            self.namespace, self.component, self.name
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::TryFrom;
    use std::str::FromStr;

    #[test]
    fn test_valid_endpoint_from() {
        let input = "namespace1/component1/endpoint1";
        let endpoint = EndpointId::from(input);

        assert_eq!(endpoint.namespace, "namespace1");
        assert_eq!(endpoint.component, "component1");
        assert_eq!(endpoint.name, "endpoint1");
    }

    #[test]
    fn test_valid_endpoint_from_str() {
        let input = "namespace2/component2/endpoint2";
        let endpoint = EndpointId::from_str(input).unwrap();

        assert_eq!(endpoint.namespace, "namespace2");
        assert_eq!(endpoint.component, "component2");
        assert_eq!(endpoint.name, "endpoint2");
    }

    #[test]
    fn test_valid_endpoint_parse() {
        let input = "namespace3/component3/endpoint3";
        let endpoint: EndpointId = input.parse().unwrap();

        assert_eq!(endpoint.namespace, "namespace3");
        assert_eq!(endpoint.component, "component3");
        assert_eq!(endpoint.name, "endpoint3");
    }

    #[test]
    fn test_endpoint_from() {
        let result = EndpointId::from("component");
        assert_eq!(
            result,
            vec![DEFAULT_NAMESPACE, "component", DEFAULT_ENDPOINT]
        );
    }

    #[test]
    fn test_namespace_component_endpoint() {
        let result = EndpointId::from("namespace.component.endpoint");
        assert_eq!(result, vec!["namespace", "component", "endpoint"]);
    }

    #[test]
    fn test_forward_slash_separator() {
        let result = EndpointId::from("namespace/component");
        assert_eq!(result, vec!["namespace", "component", DEFAULT_ENDPOINT]);
    }

    #[test]
    fn test_multiple_parts() {
        let result = EndpointId::from("namespace.component.endpoint.other.parts");
        assert_eq!(
            result,
            vec!["namespace", "component", "endpoint_other_parts"]
        );
    }

    #[test]
    fn test_mixed_separators() {
        // Do it the .into way for variety and documentation
        let result: EndpointId = "namespace/component.endpoint".into();
        assert_eq!(result, vec!["namespace", "component", "endpoint"]);
    }

    #[test]
    fn test_empty_string() {
        let result = EndpointId::from("");
        assert_eq!(
            result,
            vec![DEFAULT_NAMESPACE, DEFAULT_COMPONENT, DEFAULT_ENDPOINT]
        );

        // White space is equivalent to an empty string
        let result = EndpointId::from("   ");
        assert_eq!(
            result,
            vec![DEFAULT_NAMESPACE, DEFAULT_COMPONENT, DEFAULT_ENDPOINT]
        );
    }

    #[test]
    fn test_parse_with_scheme_and_url_roundtrip() {
        let input = "dyn://ns/cp/ep";
        let endpoint: EndpointId = input.parse().unwrap();
        assert_eq!(endpoint, vec!["ns", "cp", "ep"]);
        assert_eq!(endpoint.as_url(), "dyn://ns.cp.ep");
    }
}
