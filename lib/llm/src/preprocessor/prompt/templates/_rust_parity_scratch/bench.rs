//! DIS-1850 — head-to-head bench: Rust minijinja vs raw Rust port.
//!
//! Runs N renders per fixture through:
//!   (A) `deepseek_v4::encode_messages` — the existing Rust port
//!   (B) minijinja + `deepseek_v4_inline.jinja` — the spike's Jinja path
//!         + the same Rust pre-pass (merge_tool_messages + sort_tool_results_by_call_order)
//!
//! Reports p50, p99, mean, throughput.

mod deepseek_v4;

use deepseek_v4::{encode_messages, ThinkingMode, merge_tool_messages, sort_tool_results_by_call_order};
use minijinja::{Environment, Value};
use serde_json::Value as JsonValue;
use std::fs;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

const ITERS: usize = 5000;
const WARMUP: usize = 100;

/// Python json.dumps-compatible formatter for the tojson filter.
struct PythonStyleFormatter;
impl serde_json::ser::Formatter for PythonStyleFormatter {
    fn begin_array_value<W: ?Sized + Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_key<W: ?Sized + Write>(&mut self, w: &mut W, first: bool) -> std::io::Result<()> {
        if first { Ok(()) } else { w.write_all(b", ") }
    }
    fn begin_object_value<W: ?Sized + Write>(&mut self, w: &mut W) -> std::io::Result<()> {
        w.write_all(b": ")
    }
}

fn python_style_to_string(v: &JsonValue) -> Result<String, serde_json::Error> {
    let mut buf = Vec::new();
    let mut ser = serde_json::Serializer::with_formatter(&mut buf, PythonStyleFormatter);
    serde::Serialize::serialize(v, &mut ser)?;
    Ok(String::from_utf8(buf).expect("utf-8"))
}

fn make_env(tpl_src: String) -> Environment<'static> {
    let mut env = Environment::new();
    env.set_lstrip_blocks(true);
    env.set_trim_blocks(true);
    env.add_template_owned("v4_inline", tpl_src).unwrap();
    env.add_filter("fromjson", |s: String| -> Result<Value, minijinja::Error> {
        serde_json::from_str::<JsonValue>(&s).map(Value::from_serialize)
            .map_err(|e| minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, format!("fromjson: {}", e)))
    });
    env.add_filter("tojson", |v: Value| -> Result<String, minijinja::Error> {
        let json_v: JsonValue = serde_json::to_value(v).map_err(|e|
            minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, format!("tojson: {}", e)))?;
        python_style_to_string(&json_v).map_err(|e|
            minijinja::Error::new(minijinja::ErrorKind::InvalidOperation, format!("tojson: {}", e)))
    });
    env
}

fn time_ns<F: FnMut()>(mut f: F, iters: usize) -> Vec<u64> {
    for _ in 0..WARMUP { f(); }
    let mut samples = Vec::with_capacity(iters);
    for _ in 0..iters {
        let t = Instant::now();
        f();
        samples.push(t.elapsed().as_nanos() as u64);
    }
    samples.sort_unstable();
    samples
}

fn report(label: &str, samples: &[u64], bytes: usize) {
    let p50 = samples[samples.len() / 2];
    let p99 = samples[samples.len() * 99 / 100];
    let mean = samples.iter().sum::<u64>() / samples.len() as u64;
    let throughput_per_sec = 1_000_000_000.0 / mean as f64;
    println!(
        "  {:34}  p50={:7.2}us  p99={:8.2}us  mean={:7.2}us  ~{:.0}/sec  ({} bytes)",
        label, p50 as f64 / 1000.0, p99 as f64 / 1000.0, mean as f64 / 1000.0,
        throughput_per_sec, bytes,
    );
}

fn load_fixture(path: &Path) -> Vec<JsonValue> {
    let raw: JsonValue = serde_json::from_str(&fs::read_to_string(path).unwrap()).unwrap();
    if let Some(messages) = raw.get("messages").and_then(|m| m.as_array()) {
        let mut messages = messages.clone();
        if let Some(tools) = raw.get("tools") {
            if let Some(first) = messages.get_mut(0) {
                if let Some(obj) = first.as_object_mut() {
                    obj.insert("tools".to_string(), tools.clone());
                }
            }
        }
        messages
    } else if let Some(arr) = raw.as_array() {
        arr.clone()
    } else {
        panic!("unexpected fixture shape")
    }
}

fn main() {
    let repo = Path::new("/workspace");
    let tpl_src = fs::read_to_string(
        repo.join("lib/llm/src/preprocessor/prompt/templates/deepseek_v4_inline.jinja"),
    ).unwrap();
    let env = make_env(tpl_src);

    let fixtures: &[(&str, ThinkingMode, &str)] = &[
        ("test_input_1.json", ThinkingMode::Thinking, "fixture 1 (tools)"),
        ("test_input_2.json", ThinkingMode::Thinking, "fixture 2 (no tools)"),
        ("test_input_3.json", ThinkingMode::Thinking, "fixture 3 (developer+tools)"),
        ("test_input_4.json", ThinkingMode::Chat,     "fixture 4 (chat+task)"),
    ];

    println!("Iterations: {} per measurement (+ {} warmup)", ITERS, WARMUP);
    println!();

    for (file, mode, label) in fixtures {
        let messages = load_fixture(&repo.join("lib/llm/tests/data/deepseek-v4").join(file));

        // Bench A: raw Rust port
        let port_samples = time_ns(|| {
            let _ = encode_messages(&messages, *mode, true).unwrap();
        }, ITERS);
        let port_bytes = encode_messages(&messages, *mode, true).unwrap().len();

        // Bench B: minijinja with same pre-pass
        let mode_str: &str = match mode { ThinkingMode::Thinking => "thinking", ThinkingMode::Chat => "chat" };
        let jinja_samples = time_ns(|| {
            let merged = merge_tool_messages(&messages);
            let sorted = sort_tool_results_by_call_order(merged);
            let json_v: JsonValue = JsonValue::Array(sorted);
            let _ = env.get_template("v4_inline").unwrap().render(minijinja::context! {
                messages => Value::from_serialize(&json_v),
                thinking_mode => mode_str,
                drop_thinking => true,
                reasoning_effort => Value::from(()),
                add_bos_token => true,
            }).unwrap();
        }, ITERS);
        let jinja_first = {
            let merged = merge_tool_messages(&messages);
            let sorted = sort_tool_results_by_call_order(merged);
            let json_v: JsonValue = JsonValue::Array(sorted);
            env.get_template("v4_inline").unwrap().render(minijinja::context! {
                messages => Value::from_serialize(&json_v),
                thinking_mode => mode_str,
                drop_thinking => true,
                reasoning_effort => Value::from(()),
                add_bos_token => true,
            }).unwrap()
        };

        // Bench B': minijinja WITHOUT the pre-pass (just the render cost)
        let merged_pre = merge_tool_messages(&messages);
        let sorted_pre = sort_tool_results_by_call_order(merged_pre);
        let json_v_pre: JsonValue = JsonValue::Array(sorted_pre);
        let render_only_samples = time_ns(|| {
            let _ = env.get_template("v4_inline").unwrap().render(minijinja::context! {
                messages => Value::from_serialize(&json_v_pre),
                thinking_mode => mode_str,
                drop_thinking => true,
                reasoning_effort => Value::from(()),
                add_bos_token => true,
            }).unwrap();
        }, ITERS);

        // Bench A': pre-pass alone (so we can subtract)
        let prepass_samples = time_ns(|| {
            let merged = merge_tool_messages(&messages);
            let _ = sort_tool_results_by_call_order(merged);
        }, ITERS);

        println!("{} ({} bytes raw / {} bytes jinja):", label, port_bytes, jinja_first.len());
        report("raw Rust port (encode_messages)", &port_samples, port_bytes);
        report("Rust minijinja + pre-pass",       &jinja_samples, jinja_first.len());
        report("  └ render only (pre-cached)",    &render_only_samples, jinja_first.len());
        report("  └ pre-pass alone",              &prepass_samples, 0);
        let port_p50 = port_samples[port_samples.len() / 2] as f64;
        let jinja_p50 = jinja_samples[jinja_samples.len() / 2] as f64;
        println!("  ratio (jinja+pre / port) p50: {:.2}x", jinja_p50 / port_p50);
        println!();
    }
}
