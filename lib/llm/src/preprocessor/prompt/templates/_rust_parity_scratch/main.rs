//! Rust minijinja parity test for DIS-1850.
//! Renders fixtures through V4 inline template via Rust minijinja, byte-diffs
//! against the saved test_output_*.txt files. Pre-processed fixture-1 messages
//! are loaded from fixture1_pre.json (Python-preprocessed) so this binary
//! doesn't need to re-implement merge_tool_messages / sort_tool_results.

use minijinja::{Environment, Value};
use serde_json::Value as JsonValue;
use std::fs;
use std::path::Path;

fn render(env: &Environment, messages: JsonValue, thinking_mode: &str, drop_thinking: bool) -> Result<String, minijinja::Error> {
    env.get_template("deepseek_v4_inline.jinja")?.render(minijinja::context! {
        messages => messages,
        thinking_mode => thinking_mode,
        drop_thinking => drop_thinking,
        reasoning_effort => Value::from(()),
        add_bos_token => true,
    })
}

fn check(label: &str, expected: &str, actual: &str) -> bool {
    let exp = expected.trim_end_matches('\n');
    let act = actual.trim_end_matches('\n');
    if exp == act {
        println!("PASS  {} ({} bytes)", label, act.len());
        true
    } else {
        let n = exp.len().min(act.len());
        let diff = (0..n).find(|&i| exp.as_bytes()[i] != act.as_bytes()[i]).unwrap_or(n);
        let lo = diff.saturating_sub(40);
        let hi_e = (diff + 40).min(exp.len());
        let hi_a = (diff + 40).min(act.len());
        eprintln!("FAIL  {} byte_pos={} exp_len={} act_len={}", label, diff, exp.len(), act.len());
        eprintln!("  expected: ...{:?}...", &exp[lo..hi_e]);
        eprintln!("  actual:   ...{:?}...", &act[lo..hi_a]);
        false
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repo = Path::new("/home/keivenc/dynamo/dynamo3");
    let tpl_path = repo.join("lib/llm/src/preprocessor/prompt/templates/deepseek_v4_inline.jinja");
    let tpl_src = fs::read_to_string(&tpl_path)?;

    let mut env = Environment::new();
    env.set_lstrip_blocks(true);
    env.set_trim_blocks(true);
    env.add_template_owned("deepseek_v4_inline.jinja", tpl_src)?;
    env.add_filter("tojson", |v: Value| -> Result<String, minijinja::Error> {
        let json_v: JsonValue = serde_json::to_value(v).map_err(|e| minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, format!("tojson: {}", e)))?;
        serde_json::to_string(&json_v).map_err(|e| minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, format!("tojson: {}", e)))
    });
    env.add_filter("fromjson", |s: String| -> Result<Value, minijinja::Error> {
        serde_json::from_str::<JsonValue>(&s)
            .map(Value::from_serialize)
            .map_err(|e| minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, format!("fromjson: {}", e)))
    });

    let mut all_pass = true;

    // Fixture 1 — pre-processed (uses merge_tool_messages + sort_tool_results).
    {
        let pre: JsonValue = serde_json::from_str(&fs::read_to_string("fixture1_pre.json")?)?;
        let expected = fs::read_to_string(repo.join("lib/llm/tests/data/deepseek-v4/test_output_1.txt"))?;
        let actual = render(&env, pre, "thinking", true)?;
        all_pass &= check("v4 fixture 1 (pre-processed)", &expected, &actual);
    }

    // Fixture 2 — bare array, no preprocessing required.
    {
        let raw = fs::read_to_string(repo.join("lib/llm/tests/data/deepseek-v4/test_input_2.json"))?;
        let messages: JsonValue = serde_json::from_str(&raw)?;
        let expected = fs::read_to_string(repo.join("lib/llm/tests/data/deepseek-v4/test_output_2.txt"))?;
        let actual = render(&env, messages, "thinking", true)?;
        all_pass &= check("v4 fixture 2", &expected, &actual);
    }

    // Fixture 4 — chat mode, no preprocessing required.
    {
        let raw = fs::read_to_string(repo.join("lib/llm/tests/data/deepseek-v4/test_input_4.json"))?;
        let messages: JsonValue = serde_json::from_str(&raw)?;
        let expected = fs::read_to_string(repo.join("lib/llm/tests/data/deepseek-v4/test_output_4.txt"))?;
        let actual = render(&env, messages, "chat", true)?;
        all_pass &= check("v4 fixture 4", &expected, &actual);
    }

    if !all_pass { std::process::exit(1) }
    println!("ALL Rust minijinja parity checks PASS");
    Ok(())
}
