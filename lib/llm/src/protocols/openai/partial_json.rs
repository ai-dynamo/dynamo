// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Partial JSON Parser for streaming tool calls.
//!
//! This implementation is heavily inspired by the `partial-json-parser` library:
//! https://github.com/promplate/partial-json-parser
//!
//! The original Python library is licensed under MIT License.
//! We've adapted the core logic to Rust for use in Dynamo's streaming tool calls functionality.

use std::collections::VecDeque;

/// Options for what types of partial JSON are allowed
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AllowPartial {
    pub strings: bool,
    pub objects: bool,
    pub arrays: bool,
}

impl Default for AllowPartial {
    fn default() -> Self {
        Self {
            strings: true,
            objects: true,
            arrays: true,
        }
    }
}

impl AllowPartial {
    pub fn all() -> Self {
        Self::default()
    }

    pub fn none() -> Self {
        Self {
            strings: false,
            objects: false,
            arrays: false,
        }
    }
}

/// Represents a token found during JSON scanning
#[derive(Debug, Clone, PartialEq, Eq)]
struct Token {
    index: usize,
    char: char,
}

/// Scans the JSON string for structural characters
fn scan_tokens(json_string: &str) -> Vec<Token> {
    json_string
        .char_indices()
        .filter_map(|(i, c)| {
            if matches!(c, '"' | '[' | ']' | '{' | '}') {
                Some(Token { index: i, char: c })
            } else {
                None
            }
        })
        .collect()
}

/// Checks if a quote at the given position is escaped
fn is_escaped(json_string: &str, index: usize) -> bool {
    let text_before = &json_string[..index];
    let count = index - text_before.trim_end_matches('\\').len();
    count % 2 == 1
}

/// Joins closing tokens for unclosed containers
fn join_closing_tokens(stack: &VecDeque<Token>) -> String {
    stack
        .iter()
        .rev()
        .map(|token| if token.char == '{' { '}' } else { ']' })
        .collect()
}

/// Completes a partial JSON string by adding necessary closing tokens
///
/// Returns a tuple of (head, tail) where head is the potentially truncated
/// input and tail is the completion string
pub fn fix_json(json_string: &str, allow: AllowPartial) -> (String, String) {
    let tokens = scan_tokens(json_string);

    // Empty or starts with quote - use simple fix
    if tokens.is_empty() || tokens[0].char == '"' {
        return simple_fix(json_string, allow);
    }

    let mut stack: VecDeque<Token> = VecDeque::new();
    let mut in_string = false;
    let mut last_string_start = None;
    let mut last_string_end = None;

    for token in &tokens {
        if token.char == '"' {
            if !in_string {
                in_string = true;
                last_string_start = Some(token.index);
            } else if !is_escaped(json_string, token.index) {
                in_string = false;
                last_string_end = Some(token.index);
            }
        } else if !in_string {
            match token.char {
                '}' => {
                    if let Some(open) = stack.pop_back() {
                        assert_eq!(open.char, '{', "Mismatched braces");
                    }
                }
                ']' => {
                    if let Some(open) = stack.pop_back() {
                        assert_eq!(open.char, '[', "Mismatched brackets");
                    }
                }
                _ => {
                    stack.push_back(token.clone());
                }
            }
        }
    }

    // If stack is empty, JSON is complete
    if stack.is_empty() {
        return (json_string.to_string(), String::new());
    }

    // Remove trailing comma if present
    let mut head = json_string.trim_end();
    if head.ends_with(',') {
        head = head[..head.len() - 1].trim_end();
    }

    // Handle unclosed strings
    if !allow.strings && in_string {
        if let Some(last_container) = stack.back()
            && last_container.char == '{' {
                // Truncate before the unclosed string key
                return (
                    head[..=last_container.index].to_string(),
                    join_closing_tokens(&stack),
                );
            }

        // Find last comma before the unclosed string
        if let Some(string_start) = last_string_start {
            let last_container_pos = stack.back().map(|t| t.index).unwrap_or(0);
            let search_start = last_container_pos.max(last_string_end.unwrap_or(0)) + 1;

            if let Some(comma_pos) = head[search_start..string_start].rfind(',') {
                let absolute_comma = search_start + comma_pos;
                return (
                    head[..absolute_comma].to_string(),
                    join_closing_tokens(&stack),
                );
            }
        }
    }

    // Simple case: just close all open containers
    if in_string && allow.strings
        && let Some(string_start) = last_string_start {
            // Fix the partial string
            let partial_str = &head[string_start..];
            let (fixed_head, fixed_tail) = simple_fix(partial_str, allow);
            return (
                format!("{}{}", &head[..string_start], fixed_head),
                format!("{}{}", fixed_tail, join_closing_tokens(&stack)),
            );
        }

    (head.to_string(), join_closing_tokens(&stack))
}

/// Simple fix for basic cases (strings, atoms)
fn simple_fix(json_string: &str, allow: AllowPartial) -> (String, String) {
    let trimmed = json_string.trim_end();

    // Handle unclosed strings
    if trimmed.starts_with('"')
        && allow.strings {
            // Count how many unescaped quotes we have
            let mut escaped = false;
            let mut quote_count = 0;
            for ch in trimmed.chars() {
                if ch == '\\' && !escaped {
                    escaped = true;
                } else {
                    if ch == '"' && !escaped {
                        quote_count += 1;
                    }
                    escaped = false;
                }
            }

            if quote_count % 2 == 1 {
                // Unclosed string
                return (trimmed.to_string(), "\"".to_string());
            }
        }

    // Already complete or can't fix
    (trimmed.to_string(), String::new())
}

/// Ensures the JSON string is complete by adding necessary tokens
pub fn ensure_json(json_string: &str, allow: AllowPartial) -> String {
    let (head, tail) = fix_json(json_string, allow);
    format!("{}{}", head, tail)
}

/// Parses partial JSON string into a serde_json::Value
///
/// This is the main function inspired by partial-json-parser's `loads()`.
/// It completes the partial JSON and then parses it.
pub fn loads(json_string: &str, allow: AllowPartial) -> Result<serde_json::Value, serde_json::Error> {
    let completed = ensure_json(json_string, allow);
    serde_json::from_str(&completed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_complete_json() {
        let result = ensure_json(r#"{"key":"value"}"#, AllowPartial::all());
        assert_eq!(result, r#"{"key":"value"}"#);
    }

    #[test]
    fn test_unclosed_object() {
        let result = ensure_json(r#"{"key":"value""#, AllowPartial::all());
        assert_eq!(result, r#"{"key":"value"}"#);
    }

    #[test]
    fn test_unclosed_string() {
        let result = ensure_json(r#"{"key":"val"#, AllowPartial::all());
        assert_eq!(result, r#"{"key":"val"}"#);
    }

    #[test]
    fn test_nested_objects() {
        let result = ensure_json(r#"{"outer":{"inner":"val"#, AllowPartial::all());
        assert_eq!(result, r#"{"outer":{"inner":"val"}}"#);
    }

    #[test]
    fn test_array() {
        let result = ensure_json(r#"[{"name":"test","args":{"val":"a""#, AllowPartial::all());
        assert_eq!(result, r#"[{"name":"test","args":{"val":"a"}}]"#);
    }

    #[test]
    fn test_parallel_tool_calls() {
        let result = ensure_json(
            r#"[{"name":"search","parameters":{"query":"rust"}},{"name":"summ"#,
            AllowPartial::all(),
        );
        // Should complete both the string and close all containers
        assert!(result.contains("search"));
        assert!(result.ends_with("}]"));
    }

    #[test]
    fn test_loads_incremental() {
        // Test 1: Unclosed string value
        let result1 = loads(r#"{"location":""#, AllowPartial::all()).unwrap();
        assert_eq!(result1["location"], "");

        // Test 2: Complete first field, starting second
        let result2 = loads(r#"{"location":"Paris","#, AllowPartial::all()).unwrap();
        assert_eq!(result2["location"], "Paris");

        // Test 3: Complete object
        let result3 = loads(r#"{"location":"Paris","unit":"celsius"}"#, AllowPartial::all()).unwrap();
        assert_eq!(result3["location"], "Paris");
        assert_eq!(result3["unit"], "celsius");
    }

    #[test]
    fn test_loads_array_incremental() {
        // Test 4: Array with unclosed parameter value
        let result4 = loads(r#"[{"name":"search","parameters":{"query":""#, AllowPartial::all()).unwrap();
        assert!(result4.is_array());
        let arr = result4.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["name"], "search");
        assert_eq!(arr[0]["parameters"]["query"], "");

        // Test 5: Complete first tool, starting second
        let result5 = loads(r#"[{"name":"search","parameters":{"query":"rust"}},"#, AllowPartial::all()).unwrap();
        assert!(result5.is_array());
        let arr = result5.as_array().unwrap();
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["name"], "search");
        assert_eq!(arr[0]["parameters"]["query"], "rust");
    }
}

