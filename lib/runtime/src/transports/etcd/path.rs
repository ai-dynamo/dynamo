// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! EtcdPath - Parsing and validation for hierarchical etcd paths

use once_cell::sync::Lazy;
use std::str::FromStr;
use validator::ValidationError;

/// The root etcd path prefix
pub const ETCD_ROOT_PATH: &str = "v1/dynamo/";

/// Reserved keyword for component paths (with underscores to prevent user conflicts)
pub const COMPONENT_KEYWORD: &str = "_component_";

/// Reserved keyword for endpoint paths (with underscores to prevent user conflicts)
pub const ENDPOINT_KEYWORD: &str = "_endpoint_";

static ALLOWED_CHARS_REGEX: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"^[a-z0-9-_]+$").unwrap());

// TODO(ryan): this was an initial implementation that inspired the DEP; we'll keep it asis for now
// and update this impl with respect to the DEP.
//
// Notes:
// - follow up on this comment: https://github.com/ai-dynamo/dynamo/pull/1459#discussion_r2140616397
//   - we will be decoupling the "identifer" from the "extra path" bits as two separate objects
//   - this issue above is a problem, but will be solved by the DEP

/// Represents a parsed etcd path with hierarchical namespaces, components, endpoints, and extra paths
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EtcdPath {
    /// The hierarchical namespace (e.g., "ns1.ns2.ns3")
    pub namespace: String,
    /// Optional component name
    pub component: Option<String>,
    /// Optional endpoint name (requires component to be present)
    pub endpoint: Option<String>,
    /// Optional lease ID (only valid with endpoint, in hexadecimal format)
    pub lease_id: Option<i64>,
    /// Optional additional path segments beyond the standard structure
    pub extra_path: Option<Vec<String>>,
}

/// Errors that can occur during etcd path parsing
#[derive(Debug, thiserror::Error)]
pub enum EtcdPathError {
    #[error("Path must start with '{}'", ETCD_ROOT_PATH)]
    InvalidPrefix,
    #[error("Invalid namespace: {0}")]
    InvalidNamespace(String),
    #[error("Invalid component name: {0}")]
    InvalidComponent(String),
    #[error("Invalid endpoint name: {0}")]
    InvalidEndpoint(String),
    #[error("Invalid extra path segment: {0}")]
    InvalidExtraPath(String),
    #[error("Endpoint requires component to be present")]
    EndpointWithoutComponent,
    #[error("Expected '{}' keyword after namespace", COMPONENT_KEYWORD)]
    ExpectedComponentKeyword,
    #[error("Expected '{}' keyword after component", ENDPOINT_KEYWORD)]
    ExpectedEndpointKeyword,
    #[error("Reserved keyword '{0}' cannot be used in extra path")]
    ReservedKeyword(String),
    #[error("Empty namespace not allowed")]
    EmptyNamespace,
    #[error("Empty component name not allowed")]
    EmptyComponent,
    #[error("Empty endpoint name not allowed")]
    EmptyEndpoint,
}

impl EtcdPath {
    /// Create a new EtcdPath with just a namespace
    pub fn new_namespace(namespace: &str) -> Result<Self, EtcdPathError> {
        validate_namespace(namespace)?;
        Ok(Self {
            namespace: namespace.to_string(),
            component: None,
            endpoint: None,
            lease_id: None,
            extra_path: None,
        })
    }

    /// Create a new EtcdPath with namespace and component
    pub fn new_component(namespace: &str, component: &str) -> Result<Self, EtcdPathError> {
        validate_namespace(namespace)?;
        validate_component(component)?;
        Ok(Self {
            namespace: namespace.to_string(),
            component: Some(component.to_string()),
            endpoint: None,
            lease_id: None,
            extra_path: None,
        })
    }

    /// Create a new EtcdPath with namespace, component, and endpoint
    pub fn new_endpoint(
        namespace: &str,
        component: &str,
        endpoint: &str,
    ) -> Result<Self, EtcdPathError> {
        validate_namespace(namespace)?;
        validate_component(component)?;
        validate_endpoint(endpoint)?;
        Ok(Self {
            namespace: namespace.to_string(),
            component: Some(component.to_string()),
            endpoint: Some(endpoint.to_string()),
            lease_id: None,
            extra_path: None,
        })
    }

    /// Create a new EtcdPath for an endpoint with lease ID
    pub fn new_endpoint_with_lease(
        namespace: &str,
        component: &str,
        endpoint: &str,
        lease_id: i64,
    ) -> Result<Self, EtcdPathError> {
        validate_namespace(namespace)?;
        validate_component(component)?;
        validate_endpoint(endpoint)?;

        Ok(Self {
            namespace: namespace.to_string(),
            component: Some(component.to_string()),
            endpoint: Some(endpoint.to_string()),
            lease_id: Some(lease_id),
            extra_path: None,
        })
    }

    /// Add extra path segments to this EtcdPath
    pub fn with_extra_path(mut self, extra_path: Vec<String>) -> Result<Self, EtcdPathError> {
        for segment in &extra_path {
            validate_extra_path_segment(segment)?;
        }
        self.extra_path = if extra_path.is_empty() {
            None
        } else {
            Some(extra_path)
        };
        self.lease_id = None;
        Ok(self)
    }

    /// Internal method to convert the EtcdPath back to a string representation
    fn _to_string(&self) -> String {
        let mut path = format!("{}{}", ETCD_ROOT_PATH, self.namespace);

        if let Some(ref component) = self.component {
            path.push('/');
            path.push_str(COMPONENT_KEYWORD);
            path.push('/');
            path.push_str(component);

            if let Some(ref endpoint) = self.endpoint {
                path.push('/');
                path.push_str(ENDPOINT_KEYWORD);
                path.push('/');
                path.push_str(endpoint);

                // Add lease ID if present
                if let Some(lease_id) = self.lease_id {
                    path.push(':');
                    path.push_str(&format!("{:x}", lease_id));
                }
            }
        }

        if let Some(ref extra_path) = self.extra_path {
            for segment in extra_path {
                path.push('/');
                path.push_str(segment);
            }
        }

        path
    }

    /// Parse an etcd path string into its components
    pub fn parse(input: &str) -> Result<Self, EtcdPathError> {
        // Check for required prefix
        if !input.starts_with(ETCD_ROOT_PATH) {
            return Err(EtcdPathError::InvalidPrefix);
        }

        // Remove the prefix and split into segments
        let path_without_prefix = &input[ETCD_ROOT_PATH.len()..];
        let segments: Vec<&str> = path_without_prefix.split('/').collect();

        if segments.is_empty() || segments[0].is_empty() {
            return Err(EtcdPathError::EmptyNamespace);
        }

        // First segment is always the namespace
        let namespace = segments[0].to_string();
        validate_namespace(&namespace)?;

        let mut etcd_path = Self {
            namespace,
            component: None,
            endpoint: None,
            lease_id: None,
            extra_path: None,
        };

        // Parse remaining segments
        let mut i = 1;
        while i < segments.len() {
            match segments[i] {
                COMPONENT_KEYWORD => {
                    if i + 1 >= segments.len() {
                        return Err(EtcdPathError::EmptyComponent);
                    }
                    let component_name = segments[i + 1].to_string();
                    validate_component(&component_name)?;
                    etcd_path.component = Some(component_name);
                    i += 2;
                }
                ENDPOINT_KEYWORD => {
                    if etcd_path.component.is_none() {
                        return Err(EtcdPathError::EndpointWithoutComponent);
                    }
                    if i + 1 >= segments.len() {
                        return Err(EtcdPathError::EmptyEndpoint);
                    }
                    let endpoint_segment = segments[i + 1];

                    // Check if endpoint has a lease ID suffix (:lease_id)
                    if let Some(colon_pos) = endpoint_segment.find(':') {
                        let endpoint_name = endpoint_segment[..colon_pos].to_string();
                        let lease_id_str = &endpoint_segment[colon_pos + 1..];

                        validate_endpoint(&endpoint_name)?;

                        // Parse lease ID as hexadecimal
                        let lease_id = i64::from_str_radix(lease_id_str, 16).map_err(|_| {
                            EtcdPathError::InvalidEndpoint(format!(
                                "Invalid lease ID format: {}",
                                lease_id_str
                            ))
                        })?;

                        etcd_path.endpoint = Some(endpoint_name);
                        etcd_path.lease_id = Some(lease_id);
                    } else {
                        let endpoint_name = endpoint_segment.to_string();
                        validate_endpoint(&endpoint_name)?;
                        etcd_path.endpoint = Some(endpoint_name);
                    }
                    i += 2;
                }
                _ => {
                    // This is an extra path segment
                    let mut extra_path = Vec::new();
                    while i < segments.len() {
                        validate_extra_path_segment(segments[i])?;
                        extra_path.push(segments[i].to_string());
                        i += 1;
                    }
                    etcd_path.extra_path = if extra_path.is_empty() {
                        None
                    } else {
                        Some(extra_path)
                    };
                    break;
                }
            }
        }

        Ok(etcd_path)
    }
}

impl FromStr for EtcdPath {
    type Err = EtcdPathError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl EtcdPath {
    /// Try to create an EtcdPath from a String
    pub fn from_string(s: String) -> Result<Self, EtcdPathError> {
        Self::parse(&s)
    }
}

impl std::fmt::Display for EtcdPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self._to_string())
    }
}

/// Validate namespace using the existing validation function
fn validate_namespace(namespace: &str) -> Result<(), EtcdPathError> {
    if namespace.is_empty() {
        return Err(EtcdPathError::EmptyNamespace);
    }

    // Split by dots and validate each part
    for part in namespace.split('.') {
        if part.is_empty() {
            return Err(EtcdPathError::InvalidNamespace(format!(
                "Empty namespace segment in '{}'",
                namespace
            )));
        }
        validate_allowed_chars(part).map_err(|_| {
            EtcdPathError::InvalidNamespace(format!("Invalid characters in '{}'", part))
        })?;
    }
    Ok(())
}

/// Validate component name
fn validate_component(component: &str) -> Result<(), EtcdPathError> {
    if component.is_empty() {
        return Err(EtcdPathError::EmptyComponent);
    }
    validate_allowed_chars(component)
        .map_err(|_| EtcdPathError::InvalidComponent(component.to_string()))
}

/// Validate endpoint name
fn validate_endpoint(endpoint: &str) -> Result<(), EtcdPathError> {
    if endpoint.is_empty() {
        return Err(EtcdPathError::EmptyEndpoint);
    }
    validate_allowed_chars(endpoint)
        .map_err(|_| EtcdPathError::InvalidEndpoint(endpoint.to_string()))
}

/// Validate extra path segment
fn validate_extra_path_segment(segment: &str) -> Result<(), EtcdPathError> {
    if segment.is_empty() {
        return Err(EtcdPathError::InvalidExtraPath(
            "Empty path segment".to_string(),
        ));
    }

    // Check for reserved keywords
    if segment == COMPONENT_KEYWORD {
        return Err(EtcdPathError::ReservedKeyword(segment.to_string()));
    }
    if segment == ENDPOINT_KEYWORD {
        return Err(EtcdPathError::ReservedKeyword(segment.to_string()));
    }

    validate_allowed_chars(segment)
        .map_err(|_| EtcdPathError::InvalidExtraPath(segment.to_string()))
}

/// Custom validator function (same as in component.rs)
fn validate_allowed_chars(input: &str) -> Result<(), ValidationError> {
    if ALLOWED_CHARS_REGEX.is_match(input) {
        Ok(())
    } else {
        Err(ValidationError::new("invalid_characters"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_namespace_and_component() {
        let s = format!("{ETCD_ROOT_PATH}ns1.ns2/_component_/my-component");
        let path = EtcdPath::parse(&s).unwrap();
        assert_eq!(path.namespace, "ns1.ns2");
        assert_eq!(path.component, Some("my-component".to_string()));
        assert_eq!(path.endpoint, None);
        assert_eq!(path.extra_path, None);
        assert_eq!(path.to_string(), s);
    }

    #[test]
    fn test_full_path_with_endpoint() {
        let s = format!(
            "{ETCD_ROOT_PATH}ns1.ns2.ns3/_component_/component-name/_endpoint_/endpoint-name"
        );
        let path = EtcdPath::parse(&s).unwrap();
        assert_eq!(path.namespace, "ns1.ns2.ns3");
        assert_eq!(path.component, Some("component-name".to_string()));
        assert_eq!(path.endpoint, Some("endpoint-name".to_string()));
        assert_eq!(path.extra_path, None);
        assert_eq!(path.to_string(), s);
    }

    #[test]
    fn test_invalid_prefix() {
        let result = EtcdPath::parse("invalid://ns1");
        assert!(matches!(result, Err(EtcdPathError::InvalidPrefix)));
    }

    #[test]
    fn test_invalid_characters() {
        let result = EtcdPath::parse(&format!("{ETCD_ROOT_PATH}ns1!/_component_/comp1"));
        assert!(matches!(result, Err(EtcdPathError::InvalidNamespace(_))));
    }

    #[test]
    fn test_constructor_methods() {
        let path = EtcdPath::new_namespace("ns1.ns2.ns3").unwrap();
        assert_eq!(path.to_string(), format!("{ETCD_ROOT_PATH}ns1.ns2.ns3"));

        let path = EtcdPath::new_component("ns1.ns2", "comp1").unwrap();
        assert_eq!(
            path.to_string(),
            format!("{ETCD_ROOT_PATH}ns1.ns2/_component_/comp1")
        );

        let path = EtcdPath::new_endpoint("ns1", "comp1", "ep1").unwrap();
        assert_eq!(
            path.to_string(),
            format!("{ETCD_ROOT_PATH}ns1/_component_/comp1/_endpoint_/ep1")
        );
    }

    #[test]
    fn test_with_extra_path_method() {
        let path = EtcdPath::new_component("ns1", "comp1")
            .unwrap()
            .with_extra_path(vec!["path1".to_string(), "path2".to_string()])
            .unwrap();
        assert_eq!(
            path.to_string(),
            format!("{ETCD_ROOT_PATH}ns1/_component_/comp1/path1/path2")
        );
    }

    #[test]
    fn test_endpoint_with_lease_id() {
        // Test creating endpoint with lease ID
        let path = EtcdPath::new_endpoint_with_lease("ns1", "comp1", "ep1", 0xabc123).unwrap();
        assert_eq!(path.namespace, "ns1");
        assert_eq!(path.component, Some("comp1".to_string()));
        assert_eq!(path.endpoint, Some("ep1".to_string()));
        assert_eq!(path.lease_id, Some(0xabc123));
        assert_eq!(
            path.to_string(),
            format!("{ETCD_ROOT_PATH}ns1/_component_/comp1/_endpoint_/ep1:abc123")
        );
    }

    #[test]
    fn test_parse_endpoint_with_lease_id() {
        // Test parsing endpoint with lease ID
        let path = EtcdPath::parse(&format!(
            "{ETCD_ROOT_PATH}ns1/_component_/comp1/_endpoint_/ep1:abc123"
        ))
        .unwrap();
        assert_eq!(path.namespace, "ns1");
        assert_eq!(path.component, Some("comp1".to_string()));
        assert_eq!(path.endpoint, Some("ep1".to_string()));
        assert_eq!(path.lease_id, Some(0xabc123));
        assert_eq!(path.extra_path, None);
    }

    #[test]
    fn test_parse_endpoint_without_lease_id() {
        // Test that endpoints without lease ID still work
        let path = EtcdPath::parse(&format!(
            "{ETCD_ROOT_PATH}ns1/_component_/comp1/_endpoint_/ep1"
        ))
        .unwrap();
        assert_eq!(path.namespace, "ns1");
        assert_eq!(path.component, Some("comp1".to_string()));
        assert_eq!(path.endpoint, Some("ep1".to_string()));
        assert_eq!(path.lease_id, None);
        assert_eq!(path.extra_path, None);
    }

    #[test]
    fn test_invalid_lease_id_format() {
        // Test invalid lease ID format
        let result = EtcdPath::parse(&format!(
            "{ETCD_ROOT_PATH}ns1/_component_/comp1/_endpoint_/ep1:invalid"
        ));
        assert!(matches!(result, Err(EtcdPathError::InvalidEndpoint(_))));
    }

    #[test]
    fn test_lease_id_round_trip() {
        // Test round-trip: create -> to_string -> parse -> verify
        let original_path =
            EtcdPath::new_endpoint_with_lease("production", "api-gateway", "http", 0xdeadbeef)
                .unwrap();

        // Convert to string
        let path_string = original_path.to_string();
        assert_eq!(
            path_string,
            format!("{ETCD_ROOT_PATH}production/_component_/api-gateway/_endpoint_/http:deadbeef")
        );

        // Parse back from string
        let parsed_path = EtcdPath::parse(&path_string).unwrap();

        // Verify all fields match
        assert_eq!(parsed_path.namespace, "production");
        assert_eq!(parsed_path.component, Some("api-gateway".to_string()));
        assert_eq!(parsed_path.endpoint, Some("http".to_string()));
        assert_eq!(parsed_path.lease_id, Some(0xdeadbeef));
        assert_eq!(parsed_path.extra_path, None);

        // Verify the parsed path equals the original
        assert_eq!(parsed_path, original_path);
    }

    #[test]
    fn test_lease_id_edge_cases() {
        // Test with lease ID 0
        let path = EtcdPath::new_endpoint_with_lease("ns", "comp", "ep", 0).unwrap();
        assert_eq!(
            path.to_string(),
            format!("{ETCD_ROOT_PATH}ns/_component_/comp/_endpoint_/ep:0")
        );

        // Test with maximum i64 value
        let path = EtcdPath::new_endpoint_with_lease("ns", "comp", "ep", i64::MAX).unwrap();
        assert_eq!(
            path.to_string(),
            format!("{ETCD_ROOT_PATH}ns/_component_/comp/_endpoint_/ep:7fffffffffffffff")
        );

        // Test parsing maximum value
        let parsed = EtcdPath::parse(&format!(
            "{ETCD_ROOT_PATH}ns/_component_/comp/_endpoint_/ep:7fffffffffffffff"
        ))
        .unwrap();
        assert_eq!(parsed.lease_id, Some(i64::MAX));
    }
}
