// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub const GLOBAL_NAMESPACE: &str = "dynamo";

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
    /// Discover models from the prefix namespace and its hyphen-delimited
    /// worker generations (e.g., prefix "ns" matches "ns", "ns-abc123",
    /// "ns-def456", but not "ns2")
    Prefix(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NamespacePrefixMode {
    Literal,
    WorkerGeneration,
}

impl NamespacePrefixMode {
    pub fn from_env() -> Self {
        if crate::config::env_is_truthy("DYN_NAMESPACE_PREFIX_STRICT") {
            Self::WorkerGeneration
        } else {
            Self::Literal
        }
    }
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

        if let Some(ns) = namespace {
            if ns.is_empty() || is_global_namespace(ns) {
                return NamespaceFilter::Global;
            }
            return NamespaceFilter::Exact(ns.to_string());
        }

        NamespaceFilter::Global
    }

    /// Check if a given namespace matches this filter.
    ///
    /// A prefix scope stops at a namespace boundary. A bare `starts_with` also
    /// admits a sibling deployment whose name merely begins with the prefix:
    /// `ComputeDynamoNamespace` builds `<k8s namespace>-<deployment name>`, so
    /// under `DYN_NAMESPACE_PREFIX=myns-dgd` a bare match would take in
    /// `myns-dgd2`, a different deployment in the same Kubernetes namespace.
    /// The scope is the prefix itself plus the hyphen-delimited worker
    /// generations beneath it — the shape `DYN_NAMESPACE_WORKER_SUFFIX`
    /// produces. A prefix ending in `-` already includes that boundary.
    pub fn matches(&self, namespace: &str) -> bool {
        match self {
            NamespaceFilter::Global => true,
            NamespaceFilter::Exact(target) => namespace == target,
            NamespaceFilter::Prefix(prefix) => {
                namespace.strip_prefix(prefix.as_str()).is_some_and(|rest| {
                    prefix.ends_with('-') || rest.is_empty() || rest.starts_with('-')
                })
            }
        }
    }

    /// Match operator worker generations when requested, preserving exact and global scopes.
    pub fn matches_with_prefix_mode(&self, namespace: &str, mode: NamespacePrefixMode) -> bool {
        match (self, mode) {
            (NamespaceFilter::Prefix(prefix), NamespacePrefixMode::WorkerGeneration) => {
                if namespace == prefix {
                    return true;
                }
                let Some(suffix) = namespace
                    .strip_prefix(prefix.as_str())
                    .and_then(|rest| rest.strip_prefix('-'))
                else {
                    return false;
                };
                // The operator uses `legacy` while migrating pre-generation workers.
                suffix == "legacy"
                    || (suffix.len() == 8
                        && suffix
                            .bytes()
                            .all(|c| matches!(c, b'0'..=b'9' | b'a'..=b'f')))
            }
            _ => self.matches(namespace),
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
        let filter = NamespaceFilter::Prefix("ns".to_string());
        assert!(filter.matches("ns"));
        assert!(filter.matches("ns-abc123"));
        assert!(filter.matches("ns-def456"));
        assert!(!filter.matches("other-ns"));
        assert!(!filter.matches(""));
        assert!(!filter.matches("ns2"));
        assert!(!filter.matches("nsother-abc123"));

        let filter = NamespaceFilter::Prefix("ns-".to_string());
        assert!(filter.matches("ns-abc"));
        assert!(!filter.matches("ns2-abc"));

        let filter = NamespaceFilter::Prefix("myns-dgd".to_string());
        assert!(filter.matches("myns-dgd"));
        assert!(filter.matches("myns-dgd-abc123"));
        assert!(!filter.matches("myns-dgd2"));
        assert!(!filter.matches("myns"));
    }

    #[test]
    fn test_is_global() {
        assert!(NamespaceFilter::Global.is_global());
        assert!(!NamespaceFilter::Exact("ns".to_string()).is_global());
        assert!(!NamespaceFilter::Prefix("ns".to_string()).is_global());
    }

    #[test]
    fn operator_prefix_excludes_sibling_deployments() {
        let literal = NamespaceFilter::from_namespace_and_prefix(None, Some("default-foo"));
        for namespace in ["default-foo", "default-foo-1a2b3c4d", "default-foo-legacy"] {
            assert!(
                literal.matches_with_prefix_mode(namespace, NamespacePrefixMode::WorkerGeneration),
                "{namespace}"
            );
        }
        assert!(!literal.matches("default-foobar"));
        assert!(
            !literal
                .matches_with_prefix_mode("default-foobar", NamespacePrefixMode::WorkerGeneration)
        );
        for namespace in [
            "default-foo-bar",
            "default-foo-bar-1a2b3c4d",
            "default-foo-DEADBEEF",
            "default-foo-1a2b3c4g",
            "default-foo-1a2b3c4",
        ] {
            assert!(literal.matches(namespace), "manual prefix: {namespace}");
            assert!(
                !literal.matches_with_prefix_mode(namespace, NamespacePrefixMode::WorkerGeneration),
                "operator prefix: {namespace}"
            );
        }
    }

    #[test]
    fn strict_prefix_preserves_exact_and_global_scopes() {
        for (namespace, prefix) in [
            (None, None),
            (Some("default-foo"), None),
            (None, Some("dynamo")),
            (None, Some("")),
        ] {
            let filter = NamespaceFilter::from_namespace_and_prefix(namespace, prefix);
            for candidate in ["default-foo", "default-foo-1a2b3c4d", "default-foo-bar"] {
                assert_eq!(
                    filter
                        .matches_with_prefix_mode(candidate, NamespacePrefixMode::WorkerGeneration),
                    filter.matches(candidate)
                );
            }
        }
    }

    #[test]
    fn prefix_mode_uses_canonical_environment_flag() {
        for (value, expected) in [
            (None, NamespacePrefixMode::Literal),
            (Some("false"), NamespacePrefixMode::Literal),
            (Some(" TRUE "), NamespacePrefixMode::WorkerGeneration),
            (Some("on"), NamespacePrefixMode::WorkerGeneration),
        ] {
            temp_env::with_var("DYN_NAMESPACE_PREFIX_STRICT", value, || {
                assert_eq!(NamespacePrefixMode::from_env(), expected);
            });
        }
    }
}
