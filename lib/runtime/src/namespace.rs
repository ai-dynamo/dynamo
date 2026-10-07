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
    /// Discover models from namespaces starting with the given prefix
    /// (e.g., prefix "ns" matches "ns", "ns-abc123", "ns-def456")
    Prefix(String),
}

impl NamespaceFilter {
    /// Create a NamespaceFilter from optional namespace and namespace_prefix.
    /// If prefix is provided, it takes precedence over exact namespace.
    pub fn from_namespace_and_prefix(
        namespace: Option<&str>,
        namespace_prefix: Option<&str>,
    ) -> Self {
        Self::from_namespace_prefix_and_suffix(namespace, namespace_prefix, None)
    }

    /// Resolve the discovery scope from the three namespace inputs: the namespace
    /// (`--namespace` / `DYN_NAMESPACE`), the prefix (`--namespace-prefix` /
    /// `DYN_NAMESPACE_PREFIX`) and the worker suffix (`DYN_NAMESPACE_WORKER_SUFFIX`).
    ///
    /// Frontend model discovery and the RL worker listing both use this, so they see
    /// the same set of namespaces. Precedence, highest first:
    ///
    /// 1. A prefix, when given: empty or [`GLOBAL_NAMESPACE`] means every namespace,
    ///    anything else is a literal [`NamespaceFilter::Prefix`]. The suffix is ignored.
    /// 2. A namespace that is absent, empty or [`GLOBAL_NAMESPACE`]: every namespace.
    ///    The suffix is ignored because `Global` already contains `dynamo-<suffix>`.
    /// 3. A namespace with a non-empty suffix: append `-{suffix}` unless already
    ///    present, matching the apply-once rule in Rust backend `CommonArgs`.
    /// 4. Otherwise exactly the namespace.
    pub fn from_namespace_prefix_and_suffix(
        namespace: Option<&str>,
        namespace_prefix: Option<&str>,
        worker_suffix: Option<&str>,
    ) -> Self {
        // Prefix takes precedence if both are specified
        if let Some(prefix) = namespace_prefix {
            if is_global_namespace(prefix) {
                return NamespaceFilter::Global;
            }
            return NamespaceFilter::Prefix(prefix.to_string());
        }

        let Some(ns) = namespace.filter(|ns| !is_global_namespace(ns)) else {
            return NamespaceFilter::Global;
        };
        match worker_suffix.filter(|suffix| !suffix.is_empty()) {
            Some(suffix) if !ns.ends_with(&format!("-{suffix}")) => {
                NamespaceFilter::Exact(format!("{ns}-{suffix}"))
            }
            _ => NamespaceFilter::Exact(ns.to_string()),
        }
    }

    /// Check if a given namespace matches this filter.
    pub fn matches(&self, namespace: &str) -> bool {
        match self {
            NamespaceFilter::Global => true,
            NamespaceFilter::Exact(target) => namespace == target,
            NamespaceFilter::Prefix(prefix) => namespace.starts_with(prefix),
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
    fn test_from_namespace_prefix_and_suffix_precedence() {
        let cases = [
            (
                "prefix wins over a suffix that is also set",
                Some("ns"),
                Some("ns"),
                Some("abc123"),
                NamespaceFilter::Prefix("ns".to_string()),
            ),
            (
                "an empty prefix means every namespace",
                Some("ns"),
                Some(""),
                Some("abc123"),
                NamespaceFilter::Global,
            ),
            (
                "a global prefix wins over a suffix that is also set",
                Some("ns"),
                Some(GLOBAL_NAMESPACE),
                Some("abc123"),
                NamespaceFilter::Global,
            ),
            (
                "suffix composes the worker namespace",
                Some("ns"),
                None,
                Some("abc123"),
                NamespaceFilter::Exact("ns-abc123".to_string()),
            ),
            (
                "an already-applied suffix is not repeated",
                Some("team-blue"),
                None,
                Some("blue"),
                NamespaceFilter::Exact("team-blue".to_string()),
            ),
            (
                "the suffix must include its hyphen to count as applied",
                Some("teamblue"),
                None,
                Some("blue"),
                NamespaceFilter::Exact("teamblue-blue".to_string()),
            ),
            (
                "an empty suffix counts as absent",
                Some("ns"),
                None,
                Some(""),
                NamespaceFilter::Exact("ns".to_string()),
            ),
            (
                "no namespace means every namespace, suffix or not",
                None,
                None,
                Some("abc123"),
                NamespaceFilter::Global,
            ),
            (
                "an empty namespace means every namespace, suffix or not",
                Some(""),
                None,
                Some("abc123"),
                NamespaceFilter::Global,
            ),
            (
                "the global namespace already contains its suffixed workers",
                Some(GLOBAL_NAMESPACE),
                None,
                Some("abc123"),
                NamespaceFilter::Global,
            ),
        ];

        for (description, namespace, prefix, suffix, expected) in cases {
            assert_eq!(
                NamespaceFilter::from_namespace_prefix_and_suffix(namespace, prefix, suffix),
                expected,
                "{description}"
            );
        }
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
    }

    #[test]
    fn test_is_global() {
        assert!(NamespaceFilter::Global.is_global());
        assert!(!NamespaceFilter::Exact("ns".to_string()).is_global());
        assert!(!NamespaceFilter::Prefix("ns".to_string()).is_global());
    }
}
