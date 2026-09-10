// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub mod lora;
pub mod zmq;

pub use lora::lora_name_to_id;

/// Sort every JSON object in `value` by key, recursively.
///
/// `serde_json` runs with `preserve_order` in this workspace, so an object
/// keeps the key order it was parsed in. Anything that hashes JSON as an
/// identity has to canonicalize first, or two logically identical values
/// hash apart.
pub(crate) fn canonicalize_json(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Object(object) => {
            let mut entries = std::mem::take(object).into_iter().collect::<Vec<_>>();
            entries.sort_by(|left, right| left.0.cmp(&right.0));
            for (key, mut value) in entries {
                canonicalize_json(&mut value);
                object.insert(key, value);
            }
        }
        serde_json::Value::Array(values) => {
            for value in values {
                canonicalize_json(value);
            }
        }
        _ => {}
    }
}
