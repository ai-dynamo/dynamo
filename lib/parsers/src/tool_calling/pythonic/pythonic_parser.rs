// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::response::{CalledFunction, ToolCallResponse, ToolCallType};
use regex::Regex;
use rustpython_parser::{
    Mode,
    ast::{Constant, Expr, Mod},
    parse,
};
use serde_json::{Number, Value, json};
use std::sync::OnceLock;

static PYTHONIC_REGEX: OnceLock<Regex> = OnceLock::new();

/// Get the compiled regex pattern for pythonic tool calls
/// Initialize the regex pattern once, no need to compile it everytime
fn get_pythonic_regex() -> &'static Regex {
    PYTHONIC_REGEX.get_or_init(|| {
        // Format Structure: [tool1(arg1=val1, arg2=val2), tool2(arg1=val3)]
        let pattern = r"\[([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*?,\s*)*([a-zA-Z]+\w*=.*?\s?)?\),\s*)*([a-zA-Z]+\w*\(([a-zA-Z]+\w*=.*?,\s*)*([a-zA-Z]+\w*=.*?\s*)?\)\s*)+\]";
        Regex::new(pattern).expect("Failed to compile pythonic regex pattern")
    })
}

fn strip_text(message: &str) -> String {
    // Remove unexpected python tags if any
    message
        .replace("<|python_start|>", "")
        .replace("<|python_end|>", "")
}

fn get_regex_matches(message: &str) -> Vec<String> {
    let re = get_pythonic_regex();
    let mut matches = Vec::new();
    for cap in re.find_iter(message) {
        matches.push(cap.as_str().to_string());
    }
    matches
}

pub fn parse_tool_calls(src: &str) -> anyhow::Result<Vec<ToolCallResponse>> {
    let ast = parse(src, Mode::Expression, "<input>")?;

    /*
    AST: Expression(ModExpression {
        range: (),
        body: List(ExprList {
            range: 0..25,
            elts: [Call(...), Call(...)]
            ctx: Load
        })
    })
    */
    let body = match ast {
        Mod::Expression(mod_expr) => mod_expr.body,
        _ => return Ok(vec![]),
    };

    let elts = match *body {
        Expr::List(expr_list) => expr_list.elts,
        _ => return Ok(vec![]),
    };

    let mut res = Vec::with_capacity(elts.len());
    for (idx, elt) in elts.iter().enumerate() {
        let (func, keywords) = match elt {
            Expr::Call(call) => (&call.func, &call.keywords),
            _ => continue,
        };

        let name = match func.as_ref() {
            Expr::Name(name) => name.id.clone(),
            _ => continue,
        };

        let mut obj = serde_json::Map::new();
        for keyword in keywords.iter() {
            let Some(arg_ident) = keyword.arg.as_ref() else {
                tracing::debug!(
                    "Skipping **kwargs in pythonic tool call for function {}",
                    name
                );
                continue;
            };

            match const_expr(&keyword.value) {
                Ok(value) => {
                    obj.insert(arg_ident.to_string(), value);
                }
                Err(e) => {
                    tracing::debug!("Skipping non-constant argument {}: {}", arg_ident, e);
                }
            }
        }

        res.push(ToolCallResponse {
            id: format!("call-{}", idx + 1),
            tp: ToolCallType::Function,
            function: CalledFunction {
                name: name.to_string(),
                // Safety: `Value::Object` is always valid JSON, so serialization cannot fail
                arguments: serde_json::to_string(&Value::Object(obj))?,
            },
        });
    }
    Ok(res)
}

fn const_expr(e: &Expr) -> Result<Value, Box<dyn std::error::Error>> {
    match e {
        Expr::Constant(constant) => Ok(match &constant.value {
            Constant::Bool(b) => json!(b),
            Constant::None => Value::Null,
            Constant::Int(i) => {
                // Try to downcast to i64/u64; fallback to string if out of range
                use num_traits::ToPrimitive;
                if let Some(v) = i.to_i64() {
                    Value::Number(Number::from(v))
                } else if let Some(v) = i.to_u64() {
                    Value::Number(Number::from(v))
                } else {
                    Value::String(i.to_string())
                }
            }
            Constant::Float(f) => json!(f),
            Constant::Str(s) => json!(s),
            _ => return Err("unsupported constant type".into()),
        }),
        // Handle Python lists as expressions, not constants
        Expr::List(expr_list) => {
            let list_values: Result<Vec<Value>, Box<dyn std::error::Error>> =
                expr_list.elts.iter().map(|e| const_expr(e)).collect();
            Ok(json!(list_values?))
        }
        // Handle Python dictionaries as expressions, not constants
        Expr::Dict(expr_dict) => {
            let mut dict_map = std::collections::HashMap::new();
            for (key_expr, value_expr) in expr_dict.keys.iter().zip(expr_dict.values.iter()) {
                // Keys should be strings for JSON compatibility
                // Handle the case where key_expr is Option<Expr>
                let key = match key_expr {
                    Some(k) => match const_expr(k)? {
                        Value::String(s) => s,
                        other => other.to_string(),
                    },
                    None => {
                        return Err(
                            "dictionary unpacking (**kwargs) not supported in constants".into()
                        );
                    }
                };
                let value = const_expr(value_expr)?;
                dict_map.insert(key, value);
            }
            Ok(json!(dict_map))
        }
        _ => Err("only constant values, lists, and dicts are allowed".into()),
    }
}

pub fn try_tool_call_parse_pythonic(
    message: &str,
) -> anyhow::Result<(Vec<ToolCallResponse>, Option<String>)> {
    let stripped = strip_text(message).trim().to_string();

    // Early exit if no content
    if stripped.is_empty() {
        return Ok((vec![], Some(String::new())));
    }

    let matches = get_regex_matches(&stripped);
    if matches.is_empty() {
        return Ok((vec![], Some(stripped)));
    }

    let tool_response = parse_tool_calls(&matches[0]);

    // normal text is everything before the first match
    let normal_text = stripped
        .split(&matches[0])
        .next()
        .unwrap() // Safety: `split()` always returns at least one element (the string before the first delimiter, or the entire string if delimiter not found)
        .trim()
        .to_string();

    Ok((tool_response?, Some(normal_text)))
}

pub fn detect_tool_call_start_pythonic(chunk: &str) -> bool {
    let trimmed = chunk.trim();
    // Early return for empty input
    if trimmed.is_empty() {
        return false;
    }
    // Heuristic: Pythonic tool calls always start with a '[' somewhere in the chunk
    trimmed.contains('[')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn extract_name_and_args(call: ToolCallResponse) -> (String, serde_json::Value) {
        let args: serde_json::Value = serde_json::from_str(&call.function.arguments).unwrap();
        (call.function.name, args)
    }

    #[test]
    fn test_strip_text() {
        let message = "Hello, world!";
        let stripped = strip_text(message);
        assert_eq!(stripped, "Hello, world!");

        let message = "<|python_start|>foo(a=1, b=2)<|python_end|>";
        let stripped = strip_text(message);
        assert_eq!(stripped, "foo(a=1, b=2)");

        let message = "<|python_start|>foo(a=1, b=2)";
        let stripped = strip_text(message);
        assert_eq!(stripped, "foo(a=1, b=2)");

        let message = "foo(a=1, b=2)<|python_end|>";
        let stripped = strip_text(message);
        assert_eq!(stripped, "foo(a=1, b=2)");
    }

    #[test]
    fn test_get_regex_matches_simple_case() {
        // Simple Case
        let message = "[foo(a=1, b=2), bar(x=3)]";
        let matches = get_regex_matches(message);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0], "[foo(a=1, b=2), bar(x=3)]");
    }

    #[test]
    fn test_get_regex_matches_text_before_and_after() {
        // Spacing in arg and value and text before and after
        let message = "Hey yo ! [foo(a=1, b=2), bar(x= 3)] Hey yo";
        let matches = get_regex_matches(message);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0], "[foo(a=1, b=2), bar(x= 3)]");
    }

    #[test]
    fn test_get_regex_matches_new_line_in_arg_and_value() {
        // New Line in Arg and value
        let message = "Hey \n yo ! [foo(a=1,b=2), \n bar(x=3)] Hey yo";
        let matches = get_regex_matches(message);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0], "[foo(a=1,b=2), \n bar(x=3)]");
    }

    #[test]
    fn test_get_regex_matches_no_call() {
        // No Call
        let message = "Hey yo !";
        let matches = get_regex_matches(message);
        assert_eq!(matches.len(), 0);
    }

    #[test]
    fn test_parse_tool_call_parse_pythonic_basic() {
        let message = "[foo(a=1, b=2), bar(x=3)]";
        let (result, content) = try_tool_call_parse_pythonic(message).unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 2);
        let (name, args) = extract_name_and_args(result[0].clone()); // TODO: Add support for normal text
        assert_eq!(name, "foo");
        assert_eq!(args["a"], 1);
        assert_eq!(args["b"], 2);
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "bar");
        assert_eq!(args["x"], 3);
    }

    #[test]
    fn test_parse_tool_call_parse_pythonic_with_text() {
        let message = "Hey yo ! [foo(a=1, b=2), bar(x=3)] Hey yo";
        let (result, content) = try_tool_call_parse_pythonic(message).unwrap();
        assert_eq!(content, Some("Hey yo !".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 2);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "foo");
        assert_eq!(args["a"], 1);
        assert_eq!(args["b"], 2);
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "bar");
        assert_eq!(args["x"], 3);
    }

    #[test]
    fn test_parse_tool_call_parse_pythonic_with_text_and_new_line() {
        let message = "Hey \n yo ! [foo(a=1, b=2), bar(x=3)] Hey yo";
        let (result, content) = try_tool_call_parse_pythonic(message).unwrap();
        assert_eq!(content, Some("Hey \n yo !".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 2);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "foo");
        assert_eq!(args["a"], 1);
        assert_eq!(args["b"], 2);
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "bar");
        assert_eq!(args["x"], 3);
    }

    #[test]
    fn test_parse_tool_call_parse_pythonic_with_no_calls() {
        let message = "Hey \n yo !";
        let (result, content) = try_tool_call_parse_pythonic(message).unwrap();
        assert_eq!(content, Some("Hey \n yo !".to_string()));
        assert!(result.is_empty());
        assert_eq!(result.len(), 0)
    }

    #[test]
    fn test_parse_tool_call_parse_pythonic_with_python_tags() {
        let message = "<|python_start|>[foo(a=1, b=2), bar(x=3)]<|python_end|>";
        let (result, content) = try_tool_call_parse_pythonic(message).unwrap();
        assert_eq!(content, Some("".to_string()));
        assert!(!result.is_empty());
        assert_eq!(result.len(), 2);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "foo");
        assert_eq!(args["a"], 1);
        assert_eq!(args["b"], 2);
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "bar");
        assert_eq!(args["x"], 3);
    }

    #[test]
    fn test_parse_tool_call_parse_pythonic_with_list_arg_values() {
        let message = "[foo(a=[1, 2, 3], b=2), bar(x=[3, 4, 5])]";
        let (result, _) = try_tool_call_parse_pythonic(message).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 2);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "foo");
        assert_eq!(args["a"], json!([1, 2, 3]));
        assert_eq!(args["b"], 2);
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "bar");
        assert_eq!(args["x"], json!([3, 4, 5]));
    }

    #[test]
    fn test_parse_tool_call_parse_pythonic_with_dict_arg_values() {
        let message = "[foo(a={'a': 1, 'b': 2}, b=2), bar(x={'x': 3, 'y': {'e': 'f'}})]";
        let (result, _) = try_tool_call_parse_pythonic(message).unwrap();
        assert!(!result.is_empty());
        assert_eq!(result.len(), 2);
        let (name, args) = extract_name_and_args(result[0].clone());
        assert_eq!(name, "foo");
        assert_eq!(args["a"], json!({"a": 1, "b": 2}));
        assert_eq!(args["b"], 2);
        let (name, args) = extract_name_and_args(result[1].clone());
        assert_eq!(name, "bar");
        assert_eq!(args["x"], json!({"x": 3, "y": {"e": "f"}}));
    }
}

#[cfg(test)]
mod detect_parser_tests {
    use super::*;

    #[test]
    fn test_detect_tool_call_start_pythonic_chunk_with_tool_call_start_token() {
        let text = r#"[foo(a=1, b=2), bar(x=3)]"#;
        let result = detect_tool_call_start_pythonic(text);
        assert!(result);
    }

    #[test]
    fn test_detect_tool_call_start_pythonic_chunk_without_tool_call_start_token() {
        let text = r#"foo(a=1, b=2)"#;
        let result = detect_tool_call_start_pythonic(text);
        assert!(!result);
    }

    #[test]
    fn test_detect_tool_call_start_pythonic_chunk_with_tool_call_start_token_in_middle() {
        let text = r#"information: [foo(a=1, b=2), bar(x=3)]"#;
        let result = detect_tool_call_start_pythonic(text);
        assert!(result);
    }

    #[test]
    fn test_detect_tool_call_start_pythonic_false_positive() {
        // Since we detect just "[" as tool call start token, this will be a false positive
        let text = r#"Hey [ There is one tool call here . foo(a=1, b=2)"#;
        let result = detect_tool_call_start_pythonic(text);
        assert!(result);
    }
}
