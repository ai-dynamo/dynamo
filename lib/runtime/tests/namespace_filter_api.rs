// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_runtime::namespace::NamespaceFilter;

#[test]
fn namespace_filter_supports_existing_downstream_exhaustive_matches() {
    fn classify(filter: &NamespaceFilter) -> &str {
        match filter {
            NamespaceFilter::Global => "global",
            NamespaceFilter::Exact(_) => "exact",
            NamespaceFilter::Prefix(_) => "prefix",
        }
    }

    for (filter, expected, matches_sibling) in [
        (NamespaceFilter::Global, "global", true),
        (NamespaceFilter::Exact("default-foo".into()), "exact", false),
        (
            NamespaceFilter::Prefix("default-foo".into()),
            "prefix",
            true,
        ),
    ] {
        assert_eq!(classify(&filter), expected);
        assert_eq!(filter.matches("default-foo-bar"), matches_sibling);
    }
}
