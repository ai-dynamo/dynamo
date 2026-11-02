// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub(super) fn assert_bounded_labels(
    registry: &prometheus::Registry,
    prefix: &str,
    label: &str,
    expected: &[&str],
) {
    let families: Vec<_> = registry
        .gather()
        .into_iter()
        .filter(|family| family.name().starts_with(prefix))
        .collect();
    for suffix in expected {
        assert!(
            families
                .iter()
                .any(|family| family.name() == format!("{prefix}{suffix}"))
        );
    }
    for family in families {
        for metric in family.get_metric() {
            let model = metric
                .get_label()
                .iter()
                .find(|item| item.name() == "model")
                .expect("decision metrics have a bounded model label");
            assert_eq!(model.value(), label);
        }
    }
}
