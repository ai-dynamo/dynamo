// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub const GLOBAL_NAMESPACE: &str = "dynamo";
const DYN_NAMESPACE_PREFIX_STRICT_ENV: &str = "DYN_NAMESPACE_PREFIX_STRICT";

/// Determines how namespaces are filtered during model discovery.
///
/// This supports the hierarchical model architecture where multiple WorkerSets
/// with different namespaces (e.g., during rolling updates) should be discovered
/// together under the same Model.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NamespaceFilter {
    /// Discover models from all namespaces (no filtering)
    Global,
    /// Discover models only from an exact namespace match
    Exact(String),
    /// Discover models from namespaces starting with the given prefix.
    Prefix(String),
    /// Discover models from the base namespace and its worker-generation namespaces.
    WorkerGenerationPrefix(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamespacePrefixMode {
    Literal,
    WorkerGeneration,
}

impl NamespaceFilter {
    /// Create a NamespaceFilter from optional namespace and namespace_prefix.
    /// If prefix is provided, it takes precedence over exact namespace.
    pub fn from_namespace_and_prefix(
        namespace: Option<&str>,
        namespace_prefix: Option<&str>,
    ) -> Self {
        // Prefix takes precedence if both are specified
        if let Some(prefix) = namespace_prefix {
            if prefix.is_empty() || is_global_namespace(prefix) {
                return NamespaceFilter::Global;
            }
            return NamespaceFilter::Prefix(prefix.to_string());
        }

        Self::from_namespace(namespace)
    }

    /// Create a NamespaceFilter using explicit namespace prefix matching semantics.
    /// If prefix is provided, it takes precedence over exact namespace.
    pub fn from_namespace_and_prefix_with_mode(
        namespace: Option<&str>,
        namespace_prefix: Option<&str>,
        mode: NamespacePrefixMode,
    ) -> Self {
        if let Some(prefix) = namespace_prefix {
            if prefix.is_empty() || is_global_namespace(prefix) {
                return NamespaceFilter::Global;
            }
            if matches!(mode, NamespacePrefixMode::WorkerGeneration) {
                return NamespaceFilter::WorkerGenerationPrefix(prefix.to_string());
            }
            return NamespaceFilter::Prefix(prefix.to_string());
        }

        Self::from_namespace(namespace)
    }

    fn from_namespace(namespace: Option<&str>) -> Self {
        if let Some(ns) = namespace {
            if ns.is_empty() || is_global_namespace(ns) {
                return NamespaceFilter::Global;
            }
            return NamespaceFilter::Exact(ns.to_string());
        }

        NamespaceFilter::Global
    }

    /// Check if a given namespace matches this filter.
    pub fn matches(&self, namespace: &str) -> bool {
        match self {
            NamespaceFilter::Global => true,
            NamespaceFilter::Exact(target) => namespace == target,
            NamespaceFilter::Prefix(prefix) => namespace.starts_with(prefix),
            NamespaceFilter::WorkerGenerationPrefix(prefix) => {
                namespace_matches_prefix(namespace, prefix, NamespacePrefixMode::WorkerGeneration)
            }
        }
    }

    /// Returns true if this is global namespace filtering (no filtering).
    pub fn is_global(&self) -> bool {
        matches!(self, NamespaceFilter::Global)
    }
}

pub fn is_global_namespace(namespace: &str) -> bool {
    namespace == GLOBAL_NAMESPACE || namespace.is_empty()
}

pub fn namespace_matches_prefix(namespace: &str, prefix: &str, mode: NamespacePrefixMode) -> bool {
    match mode {
        NamespacePrefixMode::Literal => namespace.starts_with(prefix),
        NamespacePrefixMode::WorkerGeneration => {
            namespace_matches_worker_generation_prefix(namespace, prefix)
        }
    }
}

fn namespace_matches_worker_generation_prefix(namespace: &str, prefix: &str) -> bool {
    if namespace == prefix {
        return true;
    }

    let Some(suffix) = namespace.strip_prefix(prefix) else {
        return false;
    };
    let Some(worker_hash) = suffix.strip_prefix('-') else {
        return false;
    };

    worker_hash.len() == 8
        && worker_hash
            .chars()
            .all(|c| matches!(c, '0'..='9' | 'a'..='f'))
}

pub fn namespace_prefix_mode_from_env() -> NamespacePrefixMode {
    std::env::var(DYN_NAMESPACE_PREFIX_STRICT_ENV)
        .ok()
        .filter(|value| {
            matches!(
                value.trim().to_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            )
        })
        .map(|_| NamespacePrefixMode::WorkerGeneration)
        .unwrap_or(NamespacePrefixMode::Literal)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_from_namespace_and_prefix_global() {
        assert_eq!(
            NamespaceFilter::from_namespace_and_prefix(None, None),
            NamespaceFilter::Global
        );
        assert_eq!(
            NamespaceFilter::from_namespace_and_prefix(Some(""), None),
            NamespaceFilter::Global
        );
        assert_eq!(
            NamespaceFilter::from_namespace_and_prefix(Some(GLOBAL_NAMESPACE), None),
            NamespaceFilter::Global
        );
    }

    #[test]
    fn test_from_namespace_and_prefix_exact() {
        assert_eq!(
            NamespaceFilter::from_namespace_and_prefix(Some("my-namespace"), None),
            NamespaceFilter::Exact("my-namespace".to_string())
        );
    }

    #[test]
    fn test_from_namespace_and_prefix_prefix_takes_precedence() {
        assert_eq!(
            NamespaceFilter::from_namespace_and_prefix(Some("exact"), Some("prefix")),
            NamespaceFilter::Prefix("prefix".to_string())
        );
    }

    #[test]
    fn test_from_namespace_and_prefix_with_worker_generation_mode() {
        assert_eq!(
            NamespaceFilter::from_namespace_and_prefix_with_mode(
                Some("exact"),
                Some("prefix"),
                NamespacePrefixMode::WorkerGeneration,
            ),
            NamespaceFilter::WorkerGenerationPrefix("prefix".to_string())
        );
        assert_eq!(
            NamespaceFilter::from_namespace_and_prefix_with_mode(
                Some("exact"),
                Some("prefix"),
                NamespacePrefixMode::Literal,
            ),
            NamespaceFilter::Prefix("prefix".to_string())
        );
    }

    #[test]
    fn test_matches_global() {
        let filter = NamespaceFilter::Global;
        assert!(filter.matches("anything"));
        assert!(filter.matches(""));
        assert!(filter.matches("default"));
        assert!(filter.matches("ns-abc123"));
    }

    #[test]
    fn test_matches_exact() {
        let filter = NamespaceFilter::Exact("my-namespace".to_string());
        assert!(filter.matches("my-namespace"));
        assert!(!filter.matches("my-namespace-abc123"));
        assert!(!filter.matches("other"));
        assert!(!filter.matches(""));
    }

    #[test]
    fn test_matches_prefix() {
        let filter = NamespaceFilter::Prefix("default-foo".to_string());
        assert!(filter.matches("default-foo"));
        assert!(filter.matches("default-foo-1a2b3c4d"));
        assert!(filter.matches("default-foo-DEADBEEF"));
        assert!(filter.matches("default-foo-bar"));
        assert!(filter.matches("default-foo-1a2b3c4"));
        assert!(filter.matches("default-foo-1a2b3c4g"));
        assert!(filter.matches("default-foobar"));
        assert!(!filter.matches("other-default-foo"));
        assert!(!filter.matches(""));
    }

    #[test]
    fn test_matches_worker_generation_prefix() {
        let filter = NamespaceFilter::WorkerGenerationPrefix("default-foo".to_string());
        assert!(filter.matches("default-foo"));
        assert!(filter.matches("default-foo-1a2b3c4d"));
        assert!(!filter.matches("default-foo-DEADBEEF"));
        assert!(!filter.matches("default-foo-bar"));
        assert!(!filter.matches("default-foo-1a2b3c4"));
        assert!(!filter.matches("default-foo-1a2b3c4g"));
        assert!(!filter.matches("default-foobar"));
        assert!(!filter.matches("other-default-foo"));
        assert!(!filter.matches(""));
    }

    #[test]
    fn test_namespace_matches_prefix_modes() {
        assert!(namespace_matches_prefix(
            "default-foo-bar",
            "default-foo",
            NamespacePrefixMode::Literal,
        ));
        assert!(!namespace_matches_prefix(
            "default-foo-bar",
            "default-foo",
            NamespacePrefixMode::WorkerGeneration,
        ));
        assert!(namespace_matches_prefix(
            "default-foo-1a2b3c4d",
            "default-foo",
            NamespacePrefixMode::WorkerGeneration,
        ));
    }

    #[test]
    fn test_is_global() {
        assert!(NamespaceFilter::Global.is_global());
        assert!(!NamespaceFilter::Exact("ns".to_string()).is_global());
        assert!(!NamespaceFilter::Prefix("ns".to_string()).is_global());
        assert!(!NamespaceFilter::WorkerGenerationPrefix("ns".to_string()).is_global());
    }
}
