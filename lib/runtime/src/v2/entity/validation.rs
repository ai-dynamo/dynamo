// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Centralized validation functions for entity descriptors
//!
//! This module provides the core validation logic used across the entity descriptor
//! system. These validators are private to the entity module and accessed through
//! the public EntityDescriptor validation methods.

use validator::{ValidationError, Validate};
use once_cell::sync::Lazy;
use regex::Regex;

/// Regex for identifier validation (components, endpoints, namespaces)
/// Allows: lowercase letters, digits, hyphens, underscores
static IDENTIFIER_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^[a-z0-9_-]+$").unwrap()
});

/// Regex for path segment validation (includes dots for object names)  
/// Allows: lowercase letters, digits, hyphens, underscores, dots
static PATH_SEGMENT_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(r"^[a-z0-9_.-]+$").unwrap()
});

/// Validate identifier names (components, endpoints, namespace segments)
/// Does not allow dots - stricter than path segments
pub(super) fn validate_identifier(name: &str) -> Result<(), ValidationError> {
    if name.is_empty() {
        return Err(ValidationError::new("empty_identifier"));
    }
    
    if !IDENTIFIER_REGEX.is_match(name) {
        return Err(ValidationError::new("invalid_identifier_chars"));
    }
    
    Ok(())
}

/// Validate path segment names (object names, path extensions)
/// Allows dots for file-like names (e.g., "tokenizer.json", "model.bin")
pub(super) fn validate_path_segment(segment: &str) -> Result<(), ValidationError> {
    if segment.is_empty() {
        return Err(ValidationError::new("empty_path_segment"));
    }
    
    if !PATH_SEGMENT_REGEX.is_match(segment) {
        return Err(ValidationError::new("invalid_path_segment_chars"));
    }
    
    Ok(())
}

/// Validate namespace segment with reserved prefix checking
pub(super) fn validate_namespace_segment(segment: &str, allow_internal: bool) -> Result<(), ValidationError> {
    validate_identifier(segment)?;
    
    // Check for reserved _internal prefix
    if segment.starts_with('_') && !allow_internal {
        return Err(ValidationError::new("reserved_internal_prefix"));
    }
    
    Ok(())
}

/// Validate that a collection is non-empty
pub(super) fn validate_non_empty_collection<T>(collection: &[T]) -> Result<(), ValidationError> {
    if collection.is_empty() {
        return Err(ValidationError::new("empty_collection"));
    }
    Ok(())
}

/// Validate component name with length restrictions
pub(super) fn validate_component_name(name: &str) -> Result<(), ValidationError> {
    validate_identifier(name)?;
    
    if name.len() > 63 {
        return Err(ValidationError::new("component_name_too_long"));
    }
    
    Ok(())
}

/// Validate endpoint name with length restrictions  
pub(super) fn validate_endpoint_name(name: &str) -> Result<(), ValidationError> {
    validate_identifier(name)?;
    
    if name.len() > 63 {
        return Err(ValidationError::new("endpoint_name_too_long"));
    }
    
    Ok(())
}

/// Validate object name for Oscar (allows dots, longer length limit)
pub(super) fn validate_object_name(name: &str) -> Result<(), ValidationError> {
    validate_path_segment(name)?;
    
    if name.len() > 255 {
        return Err(ValidationError::new("object_name_too_long"));
    }
    
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_identifier() {
        // Valid identifiers
        assert!(validate_identifier("valid").is_ok());
        assert!(validate_identifier("with-hyphens").is_ok());
        assert!(validate_identifier("with_underscores").is_ok());
        assert!(validate_identifier("with123numbers").is_ok());
        
        // Invalid identifiers
        assert!(validate_identifier("").is_err());
        assert!(validate_identifier("With-Capitals").is_err());
        assert!(validate_identifier("with.dots").is_err());
        assert!(validate_identifier("with spaces").is_err());
        assert!(validate_identifier("with/slash").is_err());
    }
    
    #[test]
    fn test_validate_path_segment() {
        // Valid path segments (includes dots)
        assert!(validate_path_segment("valid").is_ok());
        assert!(validate_path_segment("with-hyphens").is_ok());
        assert!(validate_path_segment("with_underscores").is_ok());
        assert!(validate_path_segment("with.dots").is_ok());
        assert!(validate_path_segment("tokenizer.json").is_ok());
        assert!(validate_path_segment("model-v1.bin").is_ok());
        
        // Invalid path segments
        assert!(validate_path_segment("").is_err());
        assert!(validate_path_segment("With-Capitals").is_err());
        assert!(validate_path_segment("with spaces").is_err());
        assert!(validate_path_segment("with/slash").is_err());
    }
    
    #[test]
    fn test_validate_namespace_segment() {
        // Valid namespace segments (no internal prefix)
        assert!(validate_namespace_segment("valid", false).is_ok());
        assert!(validate_namespace_segment("prod", false).is_ok());
        
        // Invalid - internal prefix not allowed
        assert!(validate_namespace_segment("_internal", false).is_err());
        assert!(validate_namespace_segment("_system", false).is_err());
        
        // Valid - internal prefix allowed
        assert!(validate_namespace_segment("_internal", true).is_ok());
        assert!(validate_namespace_segment("_system", true).is_ok());
    }
    
    #[test]
    fn test_validate_non_empty_collection() {
        assert!(validate_non_empty_collection(&["item"]).is_ok());
        assert!(validate_non_empty_collection(&Vec::<String>::new()).is_err());
    }
    
    #[test]
    fn test_validate_component_name() {
        assert!(validate_component_name("api").is_ok());
        assert!(validate_component_name("a".repeat(63).as_str()).is_ok());
        assert!(validate_component_name(&"a".repeat(64)).is_err()); // Too long
    }
    
    #[test]
    fn test_validate_endpoint_name() {
        assert!(validate_endpoint_name("http").is_ok());
        assert!(validate_endpoint_name(&"a".repeat(63)).is_ok());
        assert!(validate_endpoint_name(&"a".repeat(64)).is_err()); // Too long
    }
    
    #[test]
    fn test_validate_object_name() {
        assert!(validate_object_name("tokenizer.json").is_ok());
        assert!(validate_object_name(&"a".repeat(255)).is_ok());
        assert!(validate_object_name(&"a".repeat(256)).is_err()); // Too long
        assert!(validate_object_name("model.bin").is_ok());
        assert!(validate_object_name("config-v1_final.yaml").is_ok());
    }
}