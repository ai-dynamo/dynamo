// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// The global namespace for all models
pub const GLOBAL_NAMESPACE: &str = "dynamo";

pub fn is_global_namespace(namespace: &str) -> bool {
    namespace == GLOBAL_NAMESPACE || namespace.is_empty()
}

/// Specifies how to filter workers by namespace during discovery.
#[derive(Debug, Clone)]
pub enum NamespaceFilter {
    /// Accept workers from all namespaces (global discovery)
    Global,
    /// Accept only workers whose namespace exactly matches the given string
    Exact(String),
    /// Accept workers whose namespace starts with the given prefix (multi-set mode)
    Prefix(String),
}

impl NamespaceFilter {
    /// Create a NamespaceFilter from optional namespace and namespace_prefix.
    /// - If namespace_prefix is Some, use Prefix mode (multi-set discovery)
    /// - If namespace is Some and not global, use Exact mode (single namespace)
    /// - Otherwise, use Global mode (discover all)
    pub fn from_options(namespace: Option<&str>, namespace_prefix: Option<&str>) -> Self {
        if let Some(prefix) = namespace_prefix {
            NamespaceFilter::Prefix(prefix.to_string())
        } else if let Some(ns) = namespace {
            if is_global_namespace(ns) {
                NamespaceFilter::Global
            } else {
                NamespaceFilter::Exact(ns.to_string())
            }
        } else {
            NamespaceFilter::Global
        }
    }

    /// Check if a namespace matches this filter.
    pub fn matches(&self, namespace: &str) -> bool {
        match self {
            NamespaceFilter::Global => true,
            NamespaceFilter::Exact(target) => namespace == target,
            NamespaceFilter::Prefix(prefix) => namespace.starts_with(prefix),
        }
    }

    /// Returns true if this is global mode (no filtering).
    pub fn is_global(&self) -> bool {
        matches!(self, NamespaceFilter::Global)
    }

    /// Returns true if this is prefix mode (multi-set).
    pub fn is_prefix(&self) -> bool {
        matches!(self, NamespaceFilter::Prefix(_))
    }

    /// Get the exact namespace if in Exact mode, None otherwise.
    pub fn exact_namespace(&self) -> Option<&str> {
        match self {
            NamespaceFilter::Exact(ns) => Some(ns),
            _ => None,
        }
    }

    /// Get the prefix string if in Prefix mode, None otherwise.
    pub fn prefix(&self) -> Option<&str> {
        match self {
            NamespaceFilter::Prefix(prefix) => Some(prefix),
            _ => None,
        }
    }
}
