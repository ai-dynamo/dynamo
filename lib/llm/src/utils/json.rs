// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

/// Sorts object keys at every depth and keeps array order, so that equal JSON
/// values serialize to the same bytes for checksum and fingerprint inputs.
pub(crate) fn canonicalize_json(value: &mut serde_json::Value) {
    match value {
        serde_json::Value::Array(values) => {
            values.iter_mut().for_each(canonicalize_json);
        }
        serde_json::Value::Object(map) => {
            map.values_mut().for_each(canonicalize_json);
            map.sort_keys();
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests {
    use super::canonicalize_json;
    use serde_json::json;

    fn canonical_string(mut value: serde_json::Value) -> String {
        canonicalize_json(&mut value);
        value.to_string()
    }

    #[test]
    fn sorts_nested_object_keys() {
        assert_eq!(
            canonical_string(json!({"z": {"y": {"b": 1, "a": 2}, "x": 3}, "a": 0})),
            r#"{"a":0,"z":{"x":3,"y":{"a":2,"b":1}}}"#
        );
    }

    #[test]
    fn sorts_objects_inside_arrays() {
        assert_eq!(
            canonical_string(json!({"list": [{"b": 1, "a": 2}, [{"d": 3, "c": 4}]], "k": 1})),
            r#"{"k":1,"list":[{"a":2,"b":1},[{"c":4,"d":3}]]}"#
        );
    }

    #[test]
    fn keeps_array_order() {
        assert_eq!(canonical_string(json!([3, 1, 2])), "[3,1,2]");
        assert_eq!(
            canonical_string(json!([{"b": 1}, {"a": 2}])),
            r#"[{"b":1},{"a":2}]"#
        );
    }

    #[test]
    fn leaves_scalars_unchanged() {
        assert_eq!(canonical_string(json!("text")), r#""text""#);
        assert_eq!(canonical_string(json!(42)), "42");
        assert_eq!(canonical_string(json!(-1.5)), "-1.5");
        assert_eq!(canonical_string(json!(true)), "true");
    }

    #[test]
    fn leaves_null_unchanged() {
        assert_eq!(canonical_string(json!(null)), "null");
        assert_eq!(
            canonical_string(json!({"b": null, "a": [null]})),
            r#"{"a":[null],"b":null}"#
        );
    }
}
