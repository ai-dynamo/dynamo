// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::{HashMap, HashSet};

use anyhow::{Result, anyhow};
use dynamo_protocols::types::FunctionObject;
use serde_json::{Map, Value};

// OpenAI's documented composition exclusions, plus `oneOf` (the API answers
// "'oneOf' is not permitted") and the array keywords absent from its supported list.
const UNSUPPORTED_KEYWORDS: &[&str] = &[
    "allOf",
    "oneOf",
    "not",
    "dependentRequired",
    "dependentSchemas",
    "if",
    "then",
    "else",
    "uniqueItems",
    "contains",
    "minContains",
    "maxContains",
    "unevaluatedItems",
];
const MAX_OBJECT_DEPTH: usize = 10;
const TYPES: &[&str] = &[
    "null", "boolean", "object", "array", "number", "integer", "string",
];

pub(super) fn validate_strict_function(function: &FunctionObject) -> Result<()> {
    if function.strict != Some(true) {
        return Ok(());
    }
    let Some(schema) = &function.parameters else {
        return Ok(());
    };
    validate_schema(schema)
        .map_err(|error| anyhow!("Invalid schema for function '{}': {error}", function.name))
}

fn invalid(path: &str, detail: impl std::fmt::Display) -> anyhow::Error {
    anyhow!("In context=#{path}, {detail}")
}

fn child_path(path: &str, key: &str) -> String {
    format!("{path}/{}", key.replace('~', "~0").replace('/', "~1"))
}

fn sorted_entries(map: &Map<String, Value>) -> Vec<(&String, &Value)> {
    let mut entries: Vec<_> = map.iter().collect();
    entries.sort_unstable_by(|a, b| a.0.cmp(b.0));
    entries
}

fn unique_strings<'a>(value: &'a Value, path: &str) -> Result<HashSet<&'a str>> {
    let array = value
        .as_array()
        .ok_or_else(|| invalid(path, "must be an array of unique strings"))?;
    let mut strings = HashSet::new();
    for value in array {
        let string = value
            .as_str()
            .ok_or_else(|| invalid(path, "must be an array of unique strings"))?;
        if !strings.insert(string) {
            return Err(invalid(path, "must be an array of unique strings"));
        }
    }
    Ok(strings)
}

#[derive(Default)]
struct Budgets {
    properties: usize,
    enums: usize,
    characters: usize,
}

impl Budgets {
    fn check(&self, path: &str) -> Result<()> {
        for (count, limit, name) in [
            (self.properties, 5_000, "declared properties"),
            (self.enums, 1_000, "enum entries"),
            (self.characters, 120_000, "name and value characters"),
        ] {
            if count > limit {
                return Err(invalid(path, format!("exceeds {limit} {name}")));
            }
        }
        Ok(())
    }
}

fn validate_schema(root: &Value) -> Result<()> {
    let mut nodes = vec![(String::new(), root, 0)];
    let mut budgets = Budgets::default();
    let mut nested_id = None;
    let mut references = Vec::new();
    let mut index = 0;
    // Breadth-first traversal, sorted map keys, and fixed keyword order make diagnostics stable
    // even when serde_json's preserve_order feature is enabled by another workspace dependency.
    while index < nodes.len() {
        let (path, value, depth) = nodes[index].clone();
        index += 1;
        if value.is_boolean() {
            continue;
        }
        let schema = value
            .as_object()
            .ok_or_else(|| invalid(&path, "schema must be an object or boolean"))?;
        for keyword in UNSUPPORTED_KEYWORDS {
            if schema.contains_key(*keyword) {
                return Err(invalid(
                    &child_path(&path, keyword),
                    "keyword is not supported in strict schemas",
                ));
            }
        }
        for keyword in ["$dynamicRef", "$recursiveRef"] {
            if schema.contains_key(keyword) {
                return Err(invalid(
                    &child_path(&path, keyword),
                    "Dynamo does not support this reference mechanism",
                ));
            }
        }
        if !path.is_empty()
            && (schema.contains_key("$id") || schema.contains_key("id"))
            && nested_id.is_none()
        {
            nested_id = Some(path.clone());
        }
        validate_shapes(schema, &path)?;
        let is_object = is_object_schema(schema);
        if is_object && depth > MAX_OBJECT_DEPTH {
            return Err(invalid(
                &path,
                format!("exceeds {MAX_OBJECT_DEPTH} levels of object nesting"),
            ));
        }
        let child_depth = depth + usize::from(is_object);
        validate_object(schema, &path)?;
        if let Some(reference) = schema.get("$ref") {
            references.push((index - 1, reference.as_str().unwrap()));
        }
        if let Some(values) = schema.get("enum") {
            let values = values.as_array().unwrap();
            budgets.enums += values.len();
            let characters: usize = values
                .iter()
                .filter_map(Value::as_str)
                .map(|s| s.chars().count())
                .sum();
            budgets.characters += characters;
            if values.len() > 250 && values.iter().all(Value::is_string) && characters > 15_000 {
                return Err(invalid(
                    &child_path(&path, "enum"),
                    "exceeds 15000 characters in an all-string enum with more than 250 entries",
                ));
            }
        }
        if let Some(value) = schema.get("const").and_then(Value::as_str) {
            budgets.characters += value.chars().count();
        }
        for keyword in ["properties", "patternProperties", "$defs", "definitions"] {
            if let Some(value) = schema.get(keyword) {
                let location = child_path(&path, keyword);
                let map = value
                    .as_object()
                    .ok_or_else(|| invalid(&location, "must be a map of schemas"))?;
                if keyword == "properties" {
                    budgets.properties += map.len();
                }
                for (name, value) in sorted_entries(map) {
                    if keyword != "patternProperties" {
                        budgets.characters += name.chars().count();
                    }
                    nodes.push((child_path(&location, name), value, child_depth));
                }
            }
        }
        for keyword in ["anyOf", "prefixItems", "items"] {
            if let Some(value) = schema.get(keyword) {
                let location = child_path(&path, keyword);
                if keyword == "items" && !value.is_array() {
                    nodes.push((location, value, child_depth));
                } else {
                    let values = value
                        .as_array()
                        .filter(|a| !a.is_empty())
                        .ok_or_else(|| invalid(&location, "must be a nonempty array of schemas"))?;
                    for (i, value) in values.iter().enumerate() {
                        nodes.push((child_path(&location, &i.to_string()), value, child_depth));
                    }
                }
            }
        }
        for keyword in [
            "additionalProperties",
            "additionalItems",
            "propertyNames",
            "unevaluatedProperties",
            "contentSchema",
        ] {
            if let Some(value) = schema.get(keyword) {
                nodes.push((child_path(&path, keyword), value, child_depth));
            }
        }
        if let Some(value) = schema.get("dependencies") {
            let location = child_path(&path, "dependencies");
            let map = value.as_object().ok_or_else(|| {
                invalid(
                    &location,
                    "must be a map of schemas or property dependencies",
                )
            })?;
            for (name, value) in sorted_entries(map) {
                let location = child_path(&location, name);
                if value.is_array() {
                    unique_strings(value, &location)?;
                } else {
                    nodes.push((location, value, child_depth));
                }
            }
        }
        budgets.check(&path)?;
    }
    if !references.is_empty()
        && let Some(path) = nested_id
    {
        return Err(invalid(
            &path,
            "Dynamo does not support references with nested identifier scopes",
        ));
    }
    let locations: HashMap<_, _> = nodes
        .iter()
        .enumerate()
        .map(|(i, (path, _, _))| (path.as_str(), i))
        .collect();
    let mut targets = vec![None; nodes.len()];
    for (index, reference) in references {
        let path = child_path(&nodes[index].0, "$ref");
        let pointer = reference_pointer(reference, &path)?;
        if root.pointer(&pointer).is_none() {
            return Err(invalid(&path, "reference target does not exist"));
        }
        let target = locations.get(pointer.as_str()).ok_or_else(|| {
            invalid(
                &path,
                "reference target is not a recognized schema location",
            )
        })?;
        targets[index] = Some(*target);
    }
    validate_reference_cycles(&nodes, &targets)?;
    validate_reference_closure(&nodes, &targets)?;
    validate_root(&nodes, &targets)
}

fn validate_shapes(schema: &Map<String, Value>, path: &str) -> Result<()> {
    if let Some(value) = schema.get("type") {
        let location = child_path(path, "type");
        let valid = if let Some(name) = value.as_str() {
            TYPES.contains(&name)
        } else {
            let names = unique_strings(value, &location)?;
            !names.is_empty() && names.iter().all(|name| TYPES.contains(name))
        };
        if !valid {
            return Err(invalid(
                &location,
                "must contain recognized, unique JSON Schema type names",
            ));
        }
    }
    if let Some(value) = schema.get("required") {
        unique_strings(value, &child_path(path, "required"))?;
    }
    if let Some(value) = schema.get("enum")
        && !value.is_array()
    {
        return Err(invalid(&child_path(path, "enum"), "must be an array"));
    }
    for keyword in ["$ref", "pattern", "format"] {
        if let Some(value) = schema.get(keyword)
            && !value.is_string()
        {
            return Err(invalid(&child_path(path, keyword), "must be a string"));
        }
    }
    for keyword in [
        "minimum",
        "maximum",
        "exclusiveMinimum",
        "exclusiveMaximum",
        "multipleOf",
    ] {
        if let Some(value) = schema.get(keyword) {
            if !value.is_number() {
                return Err(invalid(&child_path(path, keyword), "must be numeric"));
            }
            if keyword == "multipleOf" && !value.as_f64().is_some_and(|n| n > 0.0) {
                return Err(invalid(&child_path(path, keyword), "must be positive"));
            }
        }
    }
    for keyword in [
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
    ] {
        if let Some(value) = schema.get(keyword)
            && !value.as_f64().is_some_and(|n| n >= 0.0 && n.fract() == 0.0)
        {
            return Err(invalid(
                &child_path(path, keyword),
                "must be a nonnegative integer",
            ));
        }
    }
    Ok(())
}

fn declares_properties(schema: &Map<String, Value>) -> bool {
    schema.contains_key("properties") || schema.contains_key("patternProperties")
}

fn is_object_schema(schema: &Map<String, Value>) -> bool {
    let object_type = schema.get("type").is_some_and(|value| {
        value == "object"
            || value
                .as_array()
                .is_some_and(|types| types.iter().any(|t| t == "object"))
    });
    object_type || declares_properties(schema)
}

fn has_local_constraints(schema: &Map<String, Value>) -> bool {
    declares_properties(schema)
        || schema.contains_key("required")
        || schema.contains_key("additionalProperties")
}

fn validate_object(schema: &Map<String, Value>, path: &str) -> Result<()> {
    // A type-only $ref sibling constrains the target, not a second local property set;
    // validate_reference_closure checks that the target closes the object.
    let needs_object_checks = if schema.contains_key("$ref") {
        has_local_constraints(schema)
    } else {
        is_object_schema(schema)
    };
    if !needs_object_checks {
        return Ok(());
    }
    if schema.get("additionalProperties") != Some(&Value::Bool(false)) {
        return Err(invalid(
            path,
            "object schemas require additionalProperties: false",
        ));
    }
    let properties = schema
        .get("properties")
        .map(|v| {
            v.as_object()
                .ok_or_else(|| invalid(&child_path(path, "properties"), "must be a map of schemas"))
        })
        .transpose()?;
    let patterns = schema
        .get("patternProperties")
        .map(|v| {
            v.as_object().ok_or_else(|| {
                invalid(
                    &child_path(path, "patternProperties"),
                    "must be a map of schemas",
                )
            })
        })
        .transpose()?;
    let required = schema
        .get("required")
        .map(|v| unique_strings(v, &child_path(path, "required")))
        .transpose()?
        .unwrap_or_default();
    if let Some(properties) = properties {
        for (name, _) in sorted_entries(properties) {
            if !required.contains(name.as_str()) {
                return Err(invalid(
                    &child_path(path, "required"),
                    format!("must include property '{name}'"),
                ));
            }
        }
    }
    if patterns.is_none_or(|p| p.is_empty()) {
        let mut extra: Vec<_> = required
            .iter()
            .filter(|name| !properties.is_some_and(|p| p.contains_key(**name)))
            .collect();
        extra.sort_unstable();
        if let Some(name) = extra.first() {
            return Err(invalid(
                &child_path(path, "required"),
                format!("unknown property '{name}' in a closed object"),
            ));
        }
    }
    Ok(())
}

fn reference_pointer(reference: &str, path: &str) -> Result<String> {
    let fragment = reference
        .strip_prefix('#')
        .ok_or_else(|| invalid(path, "Dynamo supports only document-local references"))?;
    let mut bytes = fragment.bytes();
    while let Some(byte) = bytes.next() {
        if byte == b'%'
            && !(bytes.next().is_some_and(|b| b.is_ascii_hexdigit())
                && bytes.next().is_some_and(|b| b.is_ascii_hexdigit()))
        {
            return Err(invalid(path, "invalid URI-fragment escape"));
        }
    }
    let pointer = percent_encoding::percent_decode_str(fragment)
        .decode_utf8()
        .map_err(|_| invalid(path, "reference fragment must decode to UTF-8"))?;
    if !pointer.is_empty() && !pointer.starts_with('/') {
        return Err(invalid(
            path,
            "Dynamo does not support named-anchor references",
        ));
    }
    let mut bytes = pointer.bytes();
    while let Some(byte) = bytes.next() {
        if byte == b'~' && !matches!(bytes.next(), Some(b'0' | b'1')) {
            return Err(invalid(path, "invalid JSON Pointer escape"));
        }
    }
    Ok(pointer.into_owned())
}

// Only direct `$ref` edges are followed. Recursion through a child schema is not a cycle
// here because the child is a separate node with no edge back to its parent.
fn validate_reference_cycles(
    nodes: &[(String, &Value, usize)],
    targets: &[Option<usize>],
) -> Result<()> {
    let mut finished = HashSet::new();
    for start in 0..nodes.len() {
        let mut chain = HashSet::new();
        let mut current = start;
        while !finished.contains(&current) {
            if !chain.insert(current) {
                return Err(invalid(
                    &nodes[current].0,
                    "Dynamo does not support direct reference cycles",
                ));
            }
            let Some(target) = targets[current] else {
                break;
            };
            current = target;
        }
        finished.extend(chain);
        finished.insert(current);
    }
    Ok(())
}

// A `$ref` sibling that asserts an object type without closing it locally must reach a
// target that does, stopping at the first target with its own object constraints because
// validate_object already checked those.
fn validate_reference_closure(
    nodes: &[(String, &Value, usize)],
    targets: &[Option<usize>],
) -> Result<()> {
    for (index, (path, value, _)) in nodes.iter().enumerate() {
        let Some(schema) = value.as_object() else {
            continue;
        };
        if targets[index].is_none() || !is_object_schema(schema) || has_local_constraints(schema) {
            continue;
        }
        let mut current = index;
        while let Some(target) = targets[current] {
            current = target;
            if nodes[current]
                .1
                .as_object()
                .is_some_and(has_local_constraints)
            {
                break;
            }
        }
        if nodes[current].1.get("additionalProperties") != Some(&Value::Bool(false)) {
            return Err(invalid(
                path,
                "reference target must close the object with additionalProperties: false",
            ));
        }
    }
    Ok(())
}

fn validate_root(nodes: &[(String, &Value, usize)], targets: &[Option<usize>]) -> Result<()> {
    let mut current = 0;
    let mut seen = HashSet::new();
    let mut object = false;
    while seen.insert(current) {
        let (path, schema, _) = &nodes[current];
        if schema.get("anyOf").is_some() {
            return Err(invalid(path, "root must not use anyOf"));
        }
        if let Some(value) = schema.get("type") {
            if value == "object"
                || value
                    .as_array()
                    .is_some_and(|types| types.len() == 1 && types[0] == "object")
            {
                object = true;
            } else {
                return Err(invalid(path, "root must have object-only type"));
            }
        }
        let Some(target) = targets[current] else {
            break;
        };
        current = target;
    }
    if !object {
        return Err(invalid(
            "",
            "unsupported root representation: Dynamo requires an object-only type or local-reference chain",
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn check(schema: Value) -> Result<()> {
        validate_strict_function(&FunctionObject {
            name: "search".into(),
            description: None,
            parameters: Some(schema),
            strict: Some(true),
        })
    }

    fn object() -> Value {
        json!({"type": "object", "additionalProperties": false})
    }

    fn nested(schema: Value) -> Value {
        json!({"type": "object", "properties": {"value": schema}, "required": ["value"], "additionalProperties": false})
    }

    fn rejects(schema: Value, detail: &str) {
        let error = check(schema).unwrap_err().to_string();
        assert!(error.contains(detail), "expected {detail:?}, got {error}");
    }

    #[test]
    fn activation_and_absent_parameters() {
        for strict in [None, Some(false), Some(true)] {
            let mut function = FunctionObject {
                name: "search".into(),
                description: None,
                parameters: None,
                strict,
            };
            validate_strict_function(&function).unwrap();
            function.parameters = Some(json!({}));
            assert_eq!(
                validate_strict_function(&function).is_err(),
                strict == Some(true)
            );
        }
    }

    #[test]
    fn roots_and_object_constraints() {
        for schema in [
            object(),
            json!({"type": ["object"], "additionalProperties": false}),
            nested(json!({"type": ["object", "null"], "additionalProperties": false})),
            nested(json!({"properties": {}, "additionalProperties": false})),
            nested(
                json!({"patternProperties": {".*": true}, "required": ["anything"], "additionalProperties": false}),
            ),
        ] {
            check(schema).unwrap();
        }
        for (schema, detail) in [
            (json!({}), "unsupported root representation"),
            (json!(true), "unsupported root representation"),
            (json!({"type": "array"}), "object-only type"),
            (
                json!({"type": ["object", "null"], "additionalProperties": false}),
                "object-only type",
            ),
            (
                json!({"type": "object", "additionalProperties": false, "anyOf": [true]}),
                "root must not use anyOf",
            ),
            (nested(json!({"type": "object"})), "additionalProperties"),
            (nested(json!({"properties": {}})), "additionalProperties"),
            (
                nested(json!({"patternProperties": {}})),
                "additionalProperties",
            ),
            (
                json!({"type": "object", "additionalProperties": true}),
                "additionalProperties",
            ),
            (
                json!({"type": "object", "additionalProperties": {}}),
                "additionalProperties",
            ),
            (
                json!({"type": "object", "additionalProperties": false, "properties": {"x": true}}),
                "must include property 'x'",
            ),
            (
                json!({"type": "object", "additionalProperties": false, "required": ["extra"]}),
                "unknown property 'extra'",
            ),
            (
                json!({"type": "object", "additionalProperties": false, "patternProperties": {}, "required": ["extra"]}),
                "unknown property 'extra'",
            ),
        ] {
            rejects(schema, detail);
        }
    }

    #[test]
    fn stable_diagnostics_and_sorted_properties() {
        let mut properties = Map::new();
        properties.insert("z".into(), json!({"type": "object"}));
        properties.insert("a/b~c".into(), json!({"type": "object"}));
        let schema = json!({"type": "object", "properties": properties, "required": ["z", "a/b~c"], "additionalProperties": false});
        assert_eq!(
            check(schema).unwrap_err().to_string(),
            "Invalid schema for function 'search': In context=#/properties/a~1b~0c, object schemas require additionalProperties: false"
        );
        assert_eq!(
            check(
                json!({"type": "object", "additionalProperties": false, "properties": {"x": true}})
            )
            .unwrap_err()
            .to_string(),
            "Invalid schema for function 'search': In context=#/required, must include property 'x'"
        );
    }

    #[test]
    fn recognized_schema_locations() {
        for keyword in [
            "properties",
            "patternProperties",
            "$defs",
            "definitions",
            "dependencies",
        ] {
            let mut schema = object();
            schema[keyword] = json!({"x": {"type": "object"}});
            if keyword == "properties" {
                schema["required"] = json!(["x"]);
            }
            rejects(schema.clone(), &format!("context=#/{keyword}/x"));
            schema[keyword]["x"] = object();
            check(schema).unwrap();
        }
        for keyword in ["anyOf", "prefixItems", "items"] {
            let mut schema = json!({});
            schema[keyword] = json!([{"type": "object"}]);
            rejects(nested(schema.clone()), &format!("/{keyword}/0"));
            schema[keyword] = json!([object(), true, false, {}, {"description": "annotation"}]);
            check(nested(schema)).unwrap();
        }
        for keyword in [
            "items",
            "additionalProperties",
            "additionalItems",
            "propertyNames",
            "unevaluatedProperties",
            "contentSchema",
        ] {
            let mut schema = json!({});
            schema[keyword] = json!({"type": "object"});
            rejects(nested(schema.clone()), &format!("/{keyword}"));
            for value in [
                object(),
                json!(true),
                json!(false),
                json!({}),
                json!({"title": "annotation"}),
            ] {
                schema[keyword] = value;
                check(nested(schema.clone())).unwrap();
            }
        }
    }

    #[test]
    fn malformed_shapes() {
        let cases = [
            ("type", json!("unknown")),
            ("type", json!([])),
            ("type", json!(["string", "string"])),
            ("type", json!([1])),
            ("type", json!(null)),
            ("required", json!(["x", "x"])),
            ("required", json!([1])),
            ("required", json!("x")),
            ("enum", json!({})),
            ("$ref", json!(1)),
            ("pattern", json!([])),
            ("format", json!(false)),
            ("minimum", json!("0")),
            ("maximum", json!(true)),
            ("exclusiveMinimum", json!(null)),
            ("exclusiveMaximum", json!([])),
            ("multipleOf", json!(0)),
            ("multipleOf", json!(-1)),
            ("minLength", json!(-1)),
            ("maxLength", json!(0.5)),
            ("minItems", json!("1")),
            ("maxItems", json!(null)),
            ("minProperties", json!(0.5)),
            ("maxProperties", json!(-1)),
            ("properties", json!([])),
            ("patternProperties", json!(false)),
            ("$defs", json!([])),
            ("definitions", json!(null)),
            ("anyOf", json!([])),
            ("prefixItems", json!([null])),
            ("items", json!([])),
            ("items", json!(null)),
            ("dependencies", json!([])),
            ("dependencies", json!({"x": ["a", "a"]})),
            ("dependencies", json!({"x": [1]})),
        ];
        for (keyword, value) in cases {
            let mut schema = object();
            schema[keyword] = value;
            rejects(nested(schema), keyword);
        }
        for value in [json!(null), json!(1), json!("schema"), json!([])] {
            rejects(nested(value), "schema must be an object or boolean");
        }
    }

    #[test]
    fn keyword_policy_and_data_boundaries() {
        for keyword in UNSUPPORTED_KEYWORDS {
            let mut schema = json!({});
            schema[*keyword] = json!(null);
            rejects(nested(schema), "keyword is not supported");
        }
        check(nested(json!({
            "type": ["string", "null"], "minLength": 1, "maxLength": 100,
            "pattern": "^[a-z]+$", "format": "email", "default": {"allOf": []},
            "examples": [{"not": true}], "enum": [{"type": "object"}, "string", null],
            "const": {"$ref": "https://example.com"}, "extension": {"type": "object"}
        })))
        .unwrap();
        check(nested(json!({"type": "number", "minimum": 0, "maximum": 10, "exclusiveMinimum": -1, "exclusiveMaximum": 11, "multipleOf": 0.5}))).unwrap();
        check(nested(
            json!({"type": "array", "items": {"type": "string"}, "minItems": 0, "maxItems": 2}),
        ))
        .unwrap();
        check(json!({"type": "object", "properties": {"not": true}, "required": ["not"], "additionalProperties": false, "dependencies": {"not": ["not"]}, "$schema": "https://example.com/schema"})).unwrap();
    }

    #[test]
    fn supported_references_and_recursion() {
        for reference in ["#/$defs/a~1b~0c", "#/%24defs/a~1b~0c"] {
            check(json!({"$ref": reference, "$defs": {"a/b~c": object()}})).unwrap();
        }
        check(json!({"$ref": "#/$defs/%E9%9B%AA", "$defs": {"雪": object()}})).unwrap();
        check(json!({"$ref": "#/$defs/%252F", "$defs": {"%2F": object()}})).unwrap();
        check(nested(json!({"$ref": "#"}))).unwrap();
        check(json!({"$ref": "#/$defs/node", "$defs": {"node": {
            "type": "object", "additionalProperties": false, "properties": {
                "next": {"anyOf": [{"$ref": "#/$defs/node"}, {"type": "null"}]}
            }, "required": ["next"]
        }}}))
        .unwrap();
        for sibling in [
            json!({}),
            json!({"description": "target"}),
            json!({"type": "object"}),
        ] {
            let mut schema = sibling;
            schema["$ref"] = json!("#/$defs/alias");
            schema["$defs"] = json!({"alias": {"$ref": "#/$defs/target"}, "target": object()});
            check(schema).unwrap();
        }
        check(json!({"$id": "https://example.com/root", "$anchor": "root", "type": "object", "additionalProperties": false, "$defs": {"unused": {"$id": "child", "$anchor": "child"}}})).unwrap();
        check(json!({
            "type": "object", "additionalProperties": false, "required": ["value"],
            "properties": {"value": {"type": ["object", "null"], "$ref": "#/$defs/base"}},
            "$defs": {"base": object()}
        }))
        .unwrap();
    }

    #[test]
    fn unsupported_and_invalid_references() {
        rejects(
            json!({"$ref": "#/$defs/target", "required": ["x"], "$defs": {"target": object()}}),
            "additionalProperties",
        );
        rejects(
            json!({"$ref": "#/$defs/target", "required": ["x"], "additionalProperties": false, "$defs": {"target": nested(json!(true))}}),
            "unknown property 'x'",
        );
        for (reference, detail) in [
            ("https://example.com/schema", "document-local"),
            ("#named", "named-anchor"),
            ("#/%", "URI-fragment escape"),
            ("#/%0", "URI-fragment escape"),
            ("#/%GG", "URI-fragment escape"),
            ("#/%FF", "UTF-8"),
            ("#/~2", "JSON Pointer escape"),
            ("#/~", "JSON Pointer escape"),
            ("#/$defs/missing", "does not exist"),
            ("#/default", "not a recognized schema location"),
            ("#/properties", "not a recognized schema location"),
        ] {
            let mut schema = object();
            schema["$ref"] = json!(reference);
            schema["default"] = json!({});
            schema["properties"] = json!({});
            rejects(schema, detail);
        }
        for keyword in ["$dynamicRef", "$recursiveRef"] {
            let mut schema = object();
            schema[keyword] = json!("#");
            rejects(schema, "reference mechanism");
        }
        rejects(json!({"$ref": "#"}), "reference cycles");
        rejects(
            json!({"$ref": "#/$defs/a", "$defs": {"a": {"$ref": "#/$defs/b"}, "b": {"$ref": "#/$defs/a"}}}),
            "reference cycles",
        );
        let mut schema = object();
        schema["$defs"] = json!({"unused": {"$ref": "#/$defs/unused"}});
        rejects(schema, "reference cycles");
        let mut schema = object();
        schema["$ref"] = json!("#");
        rejects(schema.clone(), "reference cycles");
        rejects(
            json!({"type": "object", "additionalProperties": false, "$ref": "#/$defs/base", "$defs": {"base": schema}}),
            "reference cycles",
        );
        for target in [json!(true), json!({}), json!({"description": "open"})] {
            rejects(
                json!({"type": "object", "$ref": "#/$defs/base", "$defs": {"base": target}}),
                "must close the object",
            );
        }
        rejects(
            json!({
                "type": "object", "additionalProperties": false, "required": ["value"],
                "properties": {"value": {"type": ["object", "null"], "$ref": "#/$defs/base"}},
                "$defs": {"base": true}
            }),
            "must close the object",
        );
        rejects(
            json!({"$ref": "#/$defs/target", "$defs": {"target": {"type": "object"}}}),
            "additionalProperties",
        );
        rejects(
            json!({"$ref": "#/$defs/target", "type": "object", "$defs": {"target": {"type": "object", "additionalProperties": false, "anyOf": [true]}}}),
            "root must not use anyOf",
        );
        rejects(
            json!({"$ref": "#/$defs/target", "$defs": {"target": {"type": ["object", "null"], "additionalProperties": false}}}),
            "object-only type",
        );
        rejects(
            json!({"$ref": "#/$defs/target", "type": "object", "properties": {"local": true}, "$defs": {"target": object()}}),
            "additionalProperties",
        );
        rejects(
            json!({"$ref": "#/$defs/target", "$defs": {"target": object(), "scoped": {"$id": "child"}}}),
            "nested identifier scopes",
        );
    }

    #[test]
    fn property_and_enum_budgets() {
        for count in [5_000, 5_001] {
            let properties: Map<_, _> =
                (0..count).map(|i| (format!("p{i}"), json!(true))).collect();
            let required: Vec<_> = properties.keys().cloned().collect();
            let schema = json!({"type": "object", "properties": properties, "required": required, "additionalProperties": false});
            assert_eq!(check(schema).is_ok(), count == 5_000);
        }
        for count in [1_000, 1_001] {
            let values: Vec<_> = (0..count).collect();
            assert_eq!(
                check(nested(json!({"enum": values}))).is_ok(),
                count == 1_000
            );
        }
        let values: Vec<_> = (0..501).collect();
        let mut schema = object();
        schema["$defs"] = json!({"a": {"enum": values}, "b": {"enum": values}});
        rejects(schema, "1000 enum entries");
    }

    #[test]
    fn character_budgets_and_unicode() {
        for count in [120_000, 120_001] {
            let mut schema = object();
            schema["const"] = json!("雪".repeat(count));
            assert_eq!(check(schema).is_ok(), count == 120_000);
        }
        for keyword in ["properties", "$defs", "definitions"] {
            for count in [120_000, 120_001] {
                let name = "雪".repeat(count);
                let mut schema = object();
                schema[keyword] = json!({name.clone(): true});
                if keyword == "properties" {
                    schema["required"] = json!([name]);
                }
                assert_eq!(check(schema).is_ok(), count == 120_000);
            }
        }
        for count in [15_000, 15_001] {
            let mut values: Vec<Value> = (0..250).map(|_| json!("")).collect();
            values.push(json!("雪".repeat(count)));
            assert_eq!(
                check(nested(json!({"enum": values}))).is_ok(),
                count == 15_000
            );
        }
        for count in [250, 251] {
            let values: Vec<_> = (0..count).map(|_| "x".repeat(61)).collect();
            assert_eq!(check(nested(json!({"enum": values}))).is_ok(), count == 250);
        }
        let mut values = vec![json!(null); 251];
        values[0] = json!("雪".repeat(120_000));
        let mut schema = object();
        schema["enum"] = json!(values);
        check(schema.clone()).unwrap();
        schema["const"] = json!("x");
        rejects(schema, "120000 name and value characters");
    }

    #[test]
    fn annotations_and_repeated_references_do_not_inflate_budgets() {
        let mut schema = object();
        schema["description"] = json!("x".repeat(120_001));
        schema["pattern"] = json!("x".repeat(120_001));
        schema["$defs"] = json!({"x": {"enum": (0..1_000).collect::<Vec<_>>()}});
        schema["properties"] = json!({"a": {"$ref": "#/$defs/x"}, "b": {"$ref": "#/$defs/x"}});
        schema["required"] = json!(["a", "b"]);
        check(schema).unwrap();
    }

    #[test]
    fn object_nesting_depth() {
        let mut schema = json!(true);
        for _ in 0..512 {
            schema = json!({"items": schema});
        }
        check(nested(schema)).unwrap();
        let mut schema = object();
        for _ in 0..10 {
            schema = nested(schema);
        }
        check(schema.clone()).unwrap();
        rejects(nested(schema), "10 levels of object nesting");
    }
}
