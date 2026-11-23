// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// Sorts object keys at every depth before serialization for checksums and fingerprints.
/// Array order and scalar values, including number representations, stay unchanged.
pub(crate) fn canonicalize_json_object_keys(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Array(values) => {
            values.iter_mut().for_each(canonicalize_json_object_keys);
        }
        serde_json::Value::Object(map) => {
            map.values_mut().for_each(canonicalize_json_object_keys);
            map.sort_keys();
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::canonicalize_json_object_keys;
    use serde_json::json;

    /// Serializes a value after sorting its object keys.
    fn canonical_string(mut value: serde_json::Value) -> String {
        canonicalize_json_object_keys(&mut value);
        value.to_string()
    }

    /// Checks key order through several levels of objects.
    #[test]
    fn sorts_nested_object_keys() {
        assert_eq!(
            canonical_string(json!({"z": {"y": {"b": 1, "a": 2}, "x": 3}, "a": 0})),
            r#"{"a":0,"z":{"x":3,"y":{"a":2,"b":1}}}"#
        );
    }

    /// Checks nested keys while preserving array order and scalar values.
    #[test]
    fn sorts_objects_inside_arrays() {
        assert_eq!(
            canonical_string(
                json!({"list": [{"b": null, "a": 2}, [{"d": 3, "c": 4}], [3, 1, 2], "text", 42, -1.5, true, null], "k": 1})
            ),
            r#"{"k":1,"list":[{"a":2,"b":null},[{"c":4,"d":3}],[3,1,2],"text",42,-1.5,true,null]}"#
        );
    }
}
