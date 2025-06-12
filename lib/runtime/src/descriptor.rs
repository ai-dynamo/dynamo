// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Descriptors for pure data representation of components without runtime dependencies
//!
//! All descriptor types implement Display to produce canonical string representations
//! in the `dynamo://` path format. Use `.to_string()` to get the canonical path string.
//!
//! The Keys descriptor always inserts `_path_` before additional segments.
//! User-provided segments cannot start with underscore (reserved for internal use).
//!
//! ```ignore
//! // Always inserts _path_
//! let keys = Keys::from_identifier(id, vec!["config".to_string(), "v1".to_string()])?;
//! assert_eq!(keys.to_string(), "dynamo://ns/_component_/comp/_path_/config/v1");
//!
//! // Segments starting with underscore are rejected
//! let err = Keys::from_identifier(id, vec!["_invalid".to_string()]); // Error!
//! ```

use once_cell::sync::Lazy;
use std::str::FromStr;
use validator::ValidationError;

pub const ETCD_ROOT_PATH: &str = "dynamo://";
pub const COMPONENT_KEYWORD: &str = "_component_";
pub const ENDPOINT_KEYWORD: &str = "_endpoint_";
pub const PATH_KEYWORD: &str = "_path_";
pub const BARRIER_KEYWORD: &str = "_barrier_";
pub const STATIC_KEYWORD: &str = "_static_";

/// Errors that can occur during descriptor operations
#[derive(Debug, thiserror::Error)]
pub enum DescriptorError {
    #[error("Path must start with '{}'", ETCD_ROOT_PATH)]
    InvalidPrefix,
    #[error("Invalid namespace: {0}")]
    InvalidNamespace(String),
    #[error("Invalid component name: {0}")]
    InvalidComponent(String),
    #[error("Invalid endpoint name: {0}")]
    InvalidEndpoint(String),
    #[error("Invalid path segment: {0}")]
    InvalidPathSegment(String),
    #[error("Endpoint requires component to be present")]
    EndpointWithoutComponent,
    #[error("Reserved keyword '{0}' cannot be used in path segments")]
    ReservedKeyword(String),
    #[error("Empty namespace not allowed")]
    EmptyNamespace,
    #[error("Empty component name not allowed")]
    EmptyComponent,
    #[error("Empty endpoint name not allowed")]
    EmptyEndpoint,
    #[error("Missing instance ID in path")]
    MissingInstanceId,
    #[error("Invalid instance ID format: {0}")]
    InvalidInstanceId(String),
}

/// Intermediate representation of a parsed dynamo path
#[derive(Debug)]
struct DynamoPath {
    namespace: String,
    component: Option<String>,
    endpoint: Option<String>,
    instance_id: Option<i64>,
    is_static: bool,
    extra_segments: Vec<String>,
}

impl DynamoPath {
    /// Parse any dynamo:// path into intermediate representation
    fn parse(input: &str) -> Result<Self, DescriptorError> {
        // Check for required prefix
        if !input.starts_with(ETCD_ROOT_PATH) {
            return Err(DescriptorError::InvalidPrefix);
        }

        // Remove prefix and split into segments
        let path_without_prefix = &input[ETCD_ROOT_PATH.len()..];
        let segments: Vec<&str> = path_without_prefix.split('/').collect();

        if segments.is_empty() || segments[0].is_empty() {
            return Err(DescriptorError::EmptyNamespace);
        }

        let namespace = segments[0].to_string();
        validate_namespace(&namespace)?;

        let mut component: Option<String> = None;
        let mut endpoint: Option<String> = None;
        let mut instance_id: Option<i64> = None;
        let mut is_static = false;
        let mut extra_segments = Vec::new();

        let mut i = 1;
        while i < segments.len() {
            match segments[i] {
                COMPONENT_KEYWORD => {
                    // Check if component was already set
                    if component.is_some() {
                        return Err(DescriptorError::InvalidPathSegment(
                            "Duplicate _component_ keyword in path".to_string()
                        ));
                    }
                    if i + 1 >= segments.len() {
                        return Err(DescriptorError::EmptyComponent);
                    }
                    let component_name = segments[i + 1];
                    validate_component(component_name)?;
                    component = Some(component_name.to_string());
                    i += 2;
                }
                ENDPOINT_KEYWORD => {
                    if component.is_none() {
                        return Err(DescriptorError::EndpointWithoutComponent);
                    }
                    // Check if endpoint was already set
                    if endpoint.is_some() {
                        return Err(DescriptorError::InvalidPathSegment(
                            "Duplicate _endpoint_ keyword in path".to_string()
                        ));
                    }
                    if i + 1 >= segments.len() {
                        return Err(DescriptorError::EmptyEndpoint);
                    }

                    let endpoint_segment = segments[i + 1];

                    // Check for instance ID suffix (:hex_id)
                    if let Some(colon_pos) = endpoint_segment.find(':') {
                        let endpoint_name = &endpoint_segment[..colon_pos];
                        let id_str = &endpoint_segment[colon_pos + 1..];

                        // Parse instance ID as hexadecimal
                        instance_id = Some(i64::from_str_radix(id_str, 16).map_err(|_| {
                            DescriptorError::InvalidInstanceId(id_str.to_string())
                        })?);

                        validate_endpoint(endpoint_name)?;
                        endpoint = Some(endpoint_name.to_string());
                        i += 2;
                    } else {
                        validate_endpoint(endpoint_segment)?;
                        endpoint = Some(endpoint_segment.to_string());
                        // Check for /_static_ after endpoint
                        if i + 2 < segments.len() && segments[i + 2] == STATIC_KEYWORD {
                            is_static = true;
                            i += 3;
                        } else {
                            i += 2;
                        }
                    }
                }
                PATH_KEYWORD => {
                    // Valid _path_ extension at any level
                    i += 1;
                    while i < segments.len() {
                        validate_extra_path_segment(segments[i])?;
                        extra_segments.push(segments[i].to_string());
                        i += 1;
                    }
                }
                _ => {
                    // Any other segment is invalid - must use _path_ for extensions
                    return Err(DescriptorError::InvalidPathSegment(format!(
                        "Invalid path format: unexpected segment '{}' - use '{}' keyword for path extensions",
                        segments[i], PATH_KEYWORD
                    )));
                }
            }
        }

        Ok(DynamoPath {
            namespace,
            component,
            endpoint,
            instance_id,
            is_static,
            extra_segments,
        })
    }

    /// Convert to Identifier (drops instance_id and extra segments if present)
    fn try_into_identifier(self) -> Result<Identifier, DescriptorError> {
        // Note: We allow parsing paths with instance_id or extra segments,
        // we just drop them. This enables more flexible parsing.

        Ok(Identifier {
            namespace: self.namespace,
            component: self.component,
            endpoint: self.endpoint,
        })
    }

    /// Convert to Instance (drops extra segments if present)
    fn try_into_instance(self) -> Result<Instance, DescriptorError> {
        let component = self.component.ok_or(DescriptorError::EmptyComponent)?;
        let endpoint = self.endpoint.ok_or(DescriptorError::EmptyEndpoint)?;
        let identifier = Identifier {
            namespace: self.namespace,
            component: Some(component),
            endpoint: Some(endpoint),
        };
        if self.is_static {
            Ok(Instance {
                identifier,
                instance_id: None,
                is_static: true,
            })
        } else if let Some(instance_id) = self.instance_id {
            Ok(Instance {
                identifier,
                instance_id: Some(instance_id),
                is_static: false,
            })
        } else {
            Err(DescriptorError::MissingInstanceId)
        }
    }

    /// Convert to Keys (with validation)
    fn try_into_keys(self) -> Result<Keys, DescriptorError> {
        let base = if self.is_static || self.instance_id.is_some() {
            let component = self.component.ok_or(DescriptorError::EmptyComponent)?;
            let endpoint = self.endpoint.ok_or(DescriptorError::EmptyEndpoint)?;
            let identifier = Identifier {
                namespace: self.namespace,
                component: Some(component),
                endpoint: Some(endpoint),
            };
            KeysBase::Instance(Instance {
                identifier,
                instance_id: self.instance_id,
                is_static: self.is_static,
            })
        } else {
            let identifier = Identifier {
                namespace: self.namespace,
                component: self.component,
                endpoint: self.endpoint,
            };
            KeysBase::Identifier(identifier)
        };
        Ok(Keys {
            base,
            keys: self.extra_segments,
        })
    }
}

/// Pure data descriptor for component identification
/// Owns the canonical path format and validation logic
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Identifier {
    namespace: String,
    component: Option<String>,
    endpoint: Option<String>,
}

impl Identifier {
    /// Create namespace-only identifier
    pub fn new_namespace(namespace: &str) -> Result<Self, DescriptorError> {
        validate_namespace(namespace)?;
        Ok(Self {
            namespace: namespace.to_string(),
            component: None,
            endpoint: None,
        })
    }

    /// Create component identifier
    pub fn new_component(namespace: &str, component: &str) -> Result<Self, DescriptorError> {
        validate_namespace(namespace)?;
        validate_component(component)?;
        Ok(Self {
            namespace: namespace.to_string(),
            component: Some(component.to_string()),
            endpoint: None,
        })
    }

    /// Create endpoint identifier
    pub fn new_endpoint(namespace: &str, component: &str, endpoint: &str) -> Result<Self, DescriptorError> {
        validate_namespace(namespace)?;
        validate_component(component)?;
        validate_endpoint(endpoint)?;
        Ok(Self {
            namespace: namespace.to_string(),
            component: Some(component.to_string()),
            endpoint: Some(endpoint.to_string()),
        })
    }

    /// Parse from canonical string representation
    pub fn parse(input: &str) -> Result<Self, DescriptorError> {
        input.try_into()
    }

    /// Get the namespace
    pub fn namespace_name(&self) -> &str {
        &self.namespace
    }

    /// Get the component if present
    pub fn component_name(&self) -> Option<&str> {
        self.component.as_deref()
    }

    /// Get the endpoint if present
    pub fn endpoint_name(&self) -> Option<&str> {
        self.endpoint.as_deref()
    }

    /// Validate the identifier
    pub fn validate(&self) -> Result<(), DescriptorError> {
        validate_namespace(&self.namespace)?;

        if let Some(ref component) = self.component {
            validate_component(component)?;
        }

        if let Some(ref endpoint) = self.endpoint {
            validate_endpoint(endpoint)?;
        }

        Ok(())
    }
}

impl std::fmt::Display for Identifier {
    /// Builds the canonical string representation in the format:
    /// dynamo://namespace[/_component_/component][/_endpoint_/endpoint]
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", ETCD_ROOT_PATH, self.namespace)?;

        if let Some(ref component) = self.component {
            write!(f, "/{}/{}", COMPONENT_KEYWORD, component)?;

            if let Some(ref endpoint) = self.endpoint {
                write!(f, "/{}/{}", ENDPOINT_KEYWORD, endpoint)?;
            }
        }

        Ok(())
    }
}

impl std::str::FromStr for Identifier {
    type Err = DescriptorError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl TryFrom<&str> for Identifier {
    type Error = DescriptorError;

    fn try_from(input: &str) -> Result<Self, Self::Error> {
        DynamoPath::parse(input)?.try_into_identifier()
    }
}

/// Identifier extended with instance_id (lease_id)
/// Immutable - identifier cannot be changed after construction
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Instance {
    identifier: Identifier,  // Private to enforce immutability
    instance_id: Option<i64>,
    is_static: bool // We can have static instances which don't have an instance_id. In this case its guaranteed there will only be one instance
}

impl Instance {
    /// Create from identifier, takes ownership to enforce immutability
    pub fn new(identifier: Identifier, instance_id: i64) -> Result<Self, DescriptorError> {
        identifier.validate()?;
        if identifier.endpoint_name().is_none() {
            return Err(DescriptorError::InvalidEndpoint(
                "Instance ID can only be attached to endpoints".to_string()
            ));
        }
        Ok(Self {
            identifier,
            instance_id: Some(instance_id),
            is_static: false
        })
    }

    pub fn new_static(identifier: Identifier) -> Result<Self, DescriptorError> {
        identifier.validate()?;
        if identifier.endpoint_name().is_none() {
            return Err(DescriptorError::InvalidEndpoint(
                "Instance ID can only be attached to endpoints".to_string()
            ));
        }
        Ok(Self {
            identifier,
            instance_id: None,
            is_static: true
        })
    }

    /// Create instance from individual path components
    /// This is a convenience constructor that builds the full identifier from parts
    pub fn from_parts(
        namespace: &str,
        component: &str,
        endpoint: &str,
        instance_id: i64,
    ) -> Result<Self, DescriptorError> {
        let identifier = Identifier::new_endpoint(namespace, component, endpoint)?;
        Self::new(identifier, instance_id)
    }

    pub fn parse(input: &str) -> Result<Self, DescriptorError> {
        input.try_into()
    }

    pub fn identifier(&self) -> Identifier {
        self.identifier.clone()
    }

    pub fn instance_id(&self) -> Option<i64> {
        self.instance_id
    }
}

impl std::fmt::Display for Instance {
    /// Builds the canonical string representation in the format:
    /// dynamo://namespace/_component_/component/_endpoint_/endpoint:hex_id
    /// The instance_id is formatted as lowercase hexadecimal after the endpoint
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}{}", ETCD_ROOT_PATH, self.identifier.namespace)?;

        if let Some(ref component) = self.identifier.component {
            write!(f, "/{}/{}", COMPONENT_KEYWORD, component)?;

            if let Some(ref endpoint) = self.identifier.endpoint {
                write!(f, "/{}/{}", ENDPOINT_KEYWORD, endpoint)?;
            }
                if let Some(instance_id) = self.instance_id{
                    write!(f, ":{:x}", instance_id)?;
                } else {
                    write!(f, "/{}", STATIC_KEYWORD)?;
                }
        }

        Ok(())
    }
}

impl std::str::FromStr for Instance {
    type Err = DescriptorError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl TryFrom<&str> for Instance {
    type Error = DescriptorError;

    fn try_from(input: &str) -> Result<Self, Self::Error> {
        DynamoPath::parse(input)?.try_into_instance()
    }
}

/// Descriptor with additional path segments for extended paths
/// Always inserts _path_ before the segments
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Keys {
    base: KeysBase,  // Either Identifier or Instance
    keys: Vec<String>,
}

/// Base can be either Identifier or Instance
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum KeysBase {
    Identifier(Identifier),
    Instance(Instance),
}

impl Keys {
    /// Create from identifier with path segments
    /// All segments will be placed under _path_
    pub fn from_identifier(identifier: Identifier, keys: Vec<String>) -> Result<Self, DescriptorError> {
        identifier.validate()?;
        for key in &keys {
            validate_path_segment(key)?;
        }
        Ok(Self {
            base: KeysBase::Identifier(identifier),
            keys,
        })
    }

    /// Create from instance with path segments
    /// All segments will be placed under _path_
    pub fn from_instance(instance: Instance, keys: Vec<String>) -> Result<Self, DescriptorError> {
        for key in &keys {
            validate_path_segment(key)?;
        }
        Ok(Self {
            base: KeysBase::Instance(instance),
            keys,
        })
    }

    /// Parse from canonical string representation
    pub fn parse(input: &str) -> Result<Self, DescriptorError> {
        input.try_into()
    }

    /// Get the base descriptor
    pub fn base(&self) -> &KeysBase {
        &self.base
    }

    /// Get the additional keys
    pub fn keys(&self) -> &[String] {
        &self.keys
    }

}

impl std::fmt::Display for Keys {
    /// Builds the canonical string representation by combining the base path
    /// (either Identifier or Instance) with _path_ and additional segments.
    /// Format: base_path/_path_/segment1/segment2...
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.base {
            KeysBase::Identifier(id) => write!(f, "{}", id)?,
            KeysBase::Instance(inst) => write!(f, "{}", inst)?,
        }

        // Always insert _path_
        write!(f, "/{}", PATH_KEYWORD)?;

        // Add the segments
        for key in &self.keys {
            write!(f, "/{}", key)?;
        }

        Ok(())
    }
}

impl std::str::FromStr for Keys {
    type Err = DescriptorError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Self::parse(s)
    }
}

impl TryFrom<&str> for Keys {
    type Error = DescriptorError;

    fn try_from(input: &str) -> Result<Self, Self::Error> {
        DynamoPath::parse(input)?.try_into_keys()
    }
}

// Validation helpers

static ALLOWED_CHARS_REGEX: Lazy<regex::Regex> =
    Lazy::new(|| regex::Regex::new(r"^[a-z0-9-_]+$").unwrap());

/// Validation for namespace segments
fn validate_namespace(namespace: &str) -> Result<(), DescriptorError> {
    if namespace.is_empty() {
        return Err(DescriptorError::EmptyNamespace);
    }

    // Split by dots and validate each part
    for part in namespace.split('.') {
        if part.is_empty() {
            return Err(DescriptorError::InvalidNamespace(format!(
                "Empty namespace segment in '{}'",
                namespace
            )));
        }
        // Namespace segments cannot start with underscore (reserved for internal use)
        if part.starts_with('_') {
            return Err(DescriptorError::InvalidNamespace(
                format!("Namespace segment '{}' cannot start with underscore (reserved for internal use)", part)
            ));
        }
        validate_allowed_chars(part).map_err(|_| {
            DescriptorError::InvalidNamespace(format!("Invalid characters in '{}'", part))
        })?;
    }
    Ok(())
}

/// Validation for component names
fn validate_component(component: &str) -> Result<(), DescriptorError> {
    if component.is_empty() {
        return Err(DescriptorError::EmptyComponent);
    }
    // Component names cannot start with underscore (reserved for internal use)
    if component.starts_with('_') {
        return Err(DescriptorError::InvalidComponent(
            format!("Component name '{}' cannot start with underscore (reserved for internal use)", component)
        ));
    }
    validate_allowed_chars(component)
        .map_err(|_| DescriptorError::InvalidComponent(component.to_string()))
}

/// Validation for endpoint names
fn validate_endpoint(endpoint: &str) -> Result<(), DescriptorError> {
    if endpoint.is_empty() {
        return Err(DescriptorError::EmptyEndpoint);
    }
    // Endpoint names cannot start with underscore (reserved for internal use)
    if endpoint.starts_with('_') {
        return Err(DescriptorError::InvalidEndpoint(
            format!("Endpoint name '{}' cannot start with underscore (reserved for internal use)", endpoint)
        ));
    }
    validate_allowed_chars(endpoint)
        .map_err(|_| DescriptorError::InvalidEndpoint(endpoint.to_string()))
}

/// Validation for path segments (no segments starting with underscore)
fn validate_path_segment(segment: &str) -> Result<(), DescriptorError> {
    if segment.is_empty() {
        return Err(DescriptorError::InvalidPathSegment(
            "Empty path segment".to_string(),
        ));
    }

    // No segments starting with underscore (reserved for internal use)
    if segment.starts_with('_') {
        return Err(DescriptorError::InvalidPathSegment(
            format!("Path segment '{}' cannot start with underscore (reserved for internal use)", segment)
        ));
    }

    validate_allowed_chars(segment)
        .map_err(|_| DescriptorError::InvalidPathSegment(segment.to_string()))
}

/// Validate extra path segments (used by DynamoPath parsing)
fn validate_extra_path_segment(segment: &str) -> Result<(), DescriptorError> {
    if segment.is_empty() {
        return Err(DescriptorError::InvalidPathSegment(
            "Empty path segment".to_string(),
        ));
    }

    // No segments starting with underscore (reserved for internal use)
    if segment.starts_with('_') {
        return Err(DescriptorError::ReservedKeyword(segment.to_string()));
    }

    validate_allowed_chars(segment)
        .map_err(|_| DescriptorError::InvalidPathSegment(segment.to_string()))
}

/// Core validation for allowed characters
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
    fn test_identifier_namespace_only() {
        let id = Identifier::new_namespace("production.api.v1").unwrap();
        assert_eq!(id.namespace_name(), "production.api.v1");
        assert_eq!(id.component_name(), None);
        assert_eq!(id.endpoint_name(), None);
        assert_eq!(id.to_string(), "dynamo://production.api.v1");
    }

    #[test]
    fn test_identifier_with_component() {
        let id = Identifier::new_component("production.api.v1", "gateway").unwrap();
        assert_eq!(id.namespace_name(), "production.api.v1");
        assert_eq!(id.component_name(), Some("gateway"));
        assert_eq!(id.endpoint_name(), None);
        assert_eq!(id.to_string(), "dynamo://production.api.v1/_component_/gateway");
    }

    #[test]
    fn test_identifier_with_endpoint() {
        let id = Identifier::new_endpoint("production.api.v1", "gateway", "http").unwrap();
        assert_eq!(id.namespace_name(), "production.api.v1");
        assert_eq!(id.component_name(), Some("gateway"));
        assert_eq!(id.endpoint_name(), Some("http"));
        assert_eq!(
            id.to_string(),
            "dynamo://production.api.v1/_component_/gateway/_endpoint_/http"
        );
    }

    #[test]
    fn test_identifier_parse() {
        let id: Identifier = "dynamo://production.api.v1".parse().unwrap();
        assert_eq!(id.namespace_name(), "production.api.v1");

        let id: Identifier = "dynamo://production.api.v1/_component_/gateway".parse().unwrap();
        assert_eq!(id.component_name(), Some("gateway"));

        let id: Identifier = "dynamo://production.api.v1/_component_/gateway/_endpoint_/http"
            .parse()
            .unwrap();
        assert_eq!(id.endpoint_name(), Some("http"));
    }

    #[test]
    fn test_instance_dynamic_and_static_creation_and_parsing() {
        // Dynamic instance creation
        let identifier = Identifier::new_endpoint("ns1", "comp1", "ep1").unwrap();
        let instance = Instance::new(identifier.clone(), 0x1234).unwrap();
        assert_eq!(instance.instance_id(), Some(0x1234));
        assert!(!instance.is_static);
        assert_eq!(instance.to_string(), "dynamo://ns1/_component_/comp1/_endpoint_/ep1:1234");

        // Static instance creation
        let static_instance = Instance::new_static(identifier.clone()).unwrap();
        assert_eq!(static_instance.instance_id(), None);
        assert!(static_instance.is_static);
        assert_eq!(static_instance.to_string(), "dynamo://ns1/_component_/comp1/_endpoint_/ep1/_static_");

        // Parsing dynamic instance from string
        let parsed: Instance = "dynamo://ns1/_component_/comp1/_endpoint_/ep1:1234".parse().unwrap();
        assert_eq!(parsed.instance_id(), Some(0x1234));
        assert!(!parsed.is_static);
        assert_eq!(parsed.identifier().namespace_name(), "ns1");
        assert_eq!(parsed.identifier().component_name(), Some("comp1"));
        assert_eq!(parsed.identifier().endpoint_name(), Some("ep1"));

        // Parsing static instance from string
        let parsed_static: Instance = "dynamo://ns1/_component_/comp1/_endpoint_/ep1/_static_".parse().unwrap();
        assert_eq!(parsed_static.instance_id(), None);
        assert!(parsed_static.is_static);
        assert_eq!(parsed_static.identifier().namespace_name(), "ns1");
        assert_eq!(parsed_static.identifier().component_name(), Some("comp1"));
        assert_eq!(parsed_static.identifier().endpoint_name(), Some("ep1"));
    }

    #[test]
    fn test_keys_with_path() {
        let id = Identifier::new_component("ns1", "comp1").unwrap();
        // _path_ is always auto-inserted
        let keys = Keys::from_identifier(id.clone(), vec!["config".to_string()]).unwrap();
        assert_eq!(keys.to_string(), "dynamo://ns1/_component_/comp1/_path_/config");

        // Multiple segments
        let keys2 = Keys::from_identifier(id, vec!["config".to_string(), "v1".to_string()]).unwrap();
        assert_eq!(keys2.to_string(), "dynamo://ns1/_component_/comp1/_path_/config/v1");
    }

    #[test]
    fn test_lenient_parsing() {
        // Valid path with _path_ keyword
        let valid_path = "dynamo://ns/_component_/comp/_endpoint_/ep:1234/_path_/extra/data";

        // Can parse as Identifier (drops instance_id and extra segments)
        let as_identifier: Identifier = valid_path.try_into().unwrap();
        assert_eq!(as_identifier.to_string(), "dynamo://ns/_component_/comp/_endpoint_/ep");

        // Can parse as Instance (drops extra segments)
        let as_instance: Instance = valid_path.try_into().unwrap();
        assert_eq!(as_instance.to_string(), "dynamo://ns/_component_/comp/_endpoint_/ep:1234");

        // Can parse as Keys (preserves everything)
        let as_keys: Keys = valid_path.try_into().unwrap();
        assert_eq!(as_keys.to_string(), "dynamo://ns/_component_/comp/_endpoint_/ep:1234/_path_/extra/data");

        // Invalid paths should fail to parse entirely with InvalidPathSegment
        let invalid_paths = vec![
            "dynamo://ns/_component_/comp/_endpoint_/ep:1234/extra/data",  // Missing _path_
            "dynamo://ns/_component_/comp/config/v1",                       // Missing _path_
            "dynamo://ns/some/random/path",                                 // No structure
        ];

        for invalid_path in invalid_paths {
            // Should fail with InvalidPathSegment for all types
            assert!(matches!(
                TryInto::<Identifier>::try_into(invalid_path),
                Err(DescriptorError::InvalidPathSegment(_))
            ), "Expected InvalidPathSegment for Identifier parse of '{}'", invalid_path);

            assert!(matches!(
                TryInto::<Instance>::try_into(invalid_path),
                Err(DescriptorError::InvalidPathSegment(_))
            ), "Expected InvalidPathSegment for Instance parse of '{}'", invalid_path);

            assert!(matches!(
                TryInto::<Keys>::try_into(invalid_path),
                Err(DescriptorError::InvalidPathSegment(_))
            ), "Expected InvalidPathSegment for Keys parse of '{}'", invalid_path);
        }
    }

    #[test]
    fn test_keys_underscore_validation() {
        let id = Identifier::new_component("ns", "comp").unwrap();

        // Test that segments starting with underscore are rejected
        assert!(Keys::from_identifier(id.clone(), vec!["_invalid".to_string()]).is_err());
        assert!(Keys::from_identifier(id.clone(), vec!["_path_".to_string()]).is_err());
        assert!(Keys::from_identifier(id.clone(), vec!["_barrier_".to_string()]).is_err());

        // Test valid segments
        let keys = Keys::from_identifier(id.clone(), vec!["config".to_string(), "v1".to_string()]).unwrap();
        assert_eq!(keys.to_string(), "dynamo://ns/_component_/comp/_path_/config/v1");

        // Test parsing paths with _path_ already present
        let parsed: Keys = "dynamo://ns/_component_/comp/_path_/config/v1".parse().unwrap();
        assert_eq!(parsed.keys(), &["config", "v1"]);
        assert_eq!(parsed.to_string(), "dynamo://ns/_component_/comp/_path_/config/v1");

        let parsed = Keys::parse("dynamo://ns/_path_/config").unwrap();
        assert_eq!(parsed.keys(), &["config"]);
        assert_eq!(parsed.to_string(), "dynamo://ns/_path_/config");


        // Test parsing paths without _path_ - should fail
        let result: Result<Keys, _> = "dynamo://ns/_component_/comp/config/v1".parse();
        assert!(result.is_err());
        assert!(matches!(result, Err(DescriptorError::InvalidPathSegment(_))));

        // Underscore in middle - allowed
        assert!(Keys::from_identifier(id.clone(), vec!["config_v2".to_string()]).is_ok());
        assert!(Keys::from_identifier(id.clone(), vec!["some_file_name".to_string()]).is_ok());
    }

    #[test]
    fn test_validation_errors() {
        // Invalid prefix
        assert!(matches!(
            Identifier::parse("invalid://ns1"),
            Err(DescriptorError::InvalidPrefix)
        ));

        // Empty namespace
        assert!(matches!(
            Identifier::new_namespace(""),
            Err(DescriptorError::EmptyNamespace)
        ));

        // Invalid characters in namespace
        assert!(matches!(
            Identifier::new_namespace("ns!@#"),
            Err(DescriptorError::InvalidNamespace(_))
        ));

        // Invalid characters in component
        assert!(matches!(
            Identifier::new_component("ns1", "comp!@#"),
            Err(DescriptorError::InvalidComponent(_))
        ));

        // Invalid characters in endpoint
        assert!(matches!(
            Identifier::new_endpoint("ns1", "comp1", "ep!@#"),
            Err(DescriptorError::InvalidEndpoint(_))
        ));

        // Instance without endpoint
        let id = Identifier::new_component("ns1", "comp1").unwrap();
        assert!(matches!(
            Instance::new(id, 1234),
            Err(DescriptorError::InvalidEndpoint(_))
        ));

        // Keys with segments starting with underscore
        let id = Identifier::new_component("ns1", "comp1").unwrap();
        assert!(matches!(
            Keys::from_identifier(id.clone(), vec!["_invalid".to_string()]),
            Err(DescriptorError::InvalidPathSegment(_))
        ));
        assert!(matches!(
            Keys::from_identifier(id, vec!["valid".to_string(), "_path_".to_string()]),
            Err(DescriptorError::InvalidPathSegment(_))
        ));

        // Test colons in various positions (not allowed except for instance IDs)
        assert!(matches!(
            Identifier::parse("dynamo://ns:invalid"),
            Err(DescriptorError::InvalidNamespace(_))
        ));
        assert!(matches!(
            Identifier::parse("dynamo://ns/_component_/comp:invalid"),
            Err(DescriptorError::InvalidComponent(_))
        ));
        assert!(matches!(
            Identifier::parse("dynamo://ns/_component_/_invali:d"),
            Err(DescriptorError::InvalidComponent(_))
        ));

    }

    #[test]
    fn test_invalid_path_formats() {
        // These paths are invalid and should be rejected with InvalidPathSegment errors
        let invalid_paths = vec![
            "dynamo://ns/_component_/comp/config/v1",          // Missing _path_ keyword
            "dynamo://ns/_component_/comp/_endpoint_/ep/extra", // Extra segment after endpoint
            "dynamo://ns/random/path/segments",                 // Random segments without structure
        ];

        for path in invalid_paths {
            assert!(matches!(
                Identifier::parse(path),
                Err(DescriptorError::InvalidPathSegment(_))
            ), "Expected InvalidPathSegment for Identifier::parse({})", path);

            assert!(matches!(
                Instance::parse(path),
                Err(DescriptorError::InvalidPathSegment(_))
            ), "Expected InvalidPathSegment for Instance::parse({})", path);

            assert!(matches!(
                Keys::parse(path),
                Err(DescriptorError::InvalidPathSegment(_))
            ), "Expected InvalidPathSegment for Keys::parse({})", path);
        }

        // Valid formats should parse correctly
        let valid1 = "dynamo://ns/_component_/comp/_path_/config/v1";
        assert!(Keys::parse(valid1).is_ok());

        let valid2 = "dynamo://ns/_component_/comp/_endpoint_/ep/_path_/extra";
        assert!(Keys::parse(valid2).is_ok());
    }

    #[test]
    fn test_out_of_order_reserved_keywords() {
        assert!(matches!(
            Identifier::parse("dynamo://ns/_endpoint_/ep/_component_/comp"),
            Err(DescriptorError::EndpointWithoutComponent)
        ));

        assert!(matches!(
            Identifier::parse("dynamo://ns/_path_/config/_component_/comp"),
            Err(DescriptorError::ReservedKeyword(_))
        ));

        assert!(matches!(
            Identifier::parse("dynamo://ns/_component_/comp1/_component_/comp2"),
            Err(DescriptorError::InvalidPathSegment(_))
        ));

        assert!(matches!(
            Identifier::parse("dynamo://ns/_component_/comp/_endpoint_/ep1/_endpoint_/ep2"),
            Err(DescriptorError::InvalidPathSegment(_))
        ));

        assert!(matches!(
            Identifier::parse("dynamo://ns/_component_/comp/_path_/p1/_path_/p2"),
            Err(DescriptorError::ReservedKeyword(_))
        ));

        // Endpoint without component - specific error
        assert!(matches!(
            Identifier::parse("dynamo://ns/_endpoint_/ep"),
            Err(DescriptorError::EndpointWithoutComponent)
        ));

        // Component name starting with underscore - validation error
        assert!(matches!(
            Identifier::parse("dynamo://ns/_component_/_invalid"),
            Err(DescriptorError::InvalidComponent(_))
        ));

        // Missing component name
        assert!(matches!(
            Identifier::parse("dynamo://ns/_component_"),
            Err(DescriptorError::EmptyComponent)
        ));

        // Missing endpoint name
        assert!(matches!(
            Identifier::parse("dynamo://ns/_component_/comp/_endpoint_"),
            Err(DescriptorError::EmptyEndpoint)
        ));
    }
}
