// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Binary to generate Python prometheus_names from Rust source

use anyhow::{Context, Result};
use dynamo_codegen::prometheus_parser::{ModuleDef, PrometheusParser};
use std::collections::HashMap;
use std::path::PathBuf;

/// Generates Python module code from parsed Rust prometheus_names modules.
/// Converts Rust const declarations into Python class attributes with deterministic ordering.
struct PythonGenerator<'a> {
    modules: &'a HashMap<String, ModuleDef>,
}

impl<'a> PythonGenerator<'a> {
    fn new(parser: &'a PrometheusParser) -> Self {
        Self {
            modules: &parser.modules,
        }
    }

    fn load_template(template_name: &str) -> String {
        let template_path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("templates")
            .join(template_name);

        std::fs::read_to_string(&template_path)
            .unwrap_or_else(|_| panic!("Failed to read template: {}", template_path.display()))
    }

    fn generate_python_file(&self) -> String {
        let mut output = Self::load_template("prometheus_names.py.template");

        // Append generated classes
        output.push_str(&self.generate_classes());

        output
    }

    fn generate_classes(&self) -> String {
        let mut lines = Vec::new();

        // Sort module names to ensure deterministic output
        let mut module_names: Vec<&String> = self.modules.keys().collect();
        module_names.sort();

        let total = module_names.len();

        // Generate simple classes with constants as class attributes
        for (idx, module_name) in module_names.iter().enumerate() {
            let module = &self.modules[module_name.as_str()];
            render_class(module, 0, &mut lines);

            // PEP 8 / black requires two blank lines between top-level class definitions,
            // but no trailing blank lines at end of file.
            if idx + 1 < total {
                lines.push("".to_string());
                lines.push("".to_string());
            }
        }

        // End file with a single trailing newline (no blank lines after last class)
        lines.push("".to_string());

        lines.join("\n")
    }
}

/// Black's line limit, from `pyproject.toml` (`line-length = 88`). The generator matches
/// it so its output is already canonical and `cargo run` alone reproduces the checked-in
/// file; keep the two in sync if the repo ever changes the limit.
const MAX_LINE_LEN: usize = 88;

/// Emit one `NAME = "value"` assignment, wrapping in parentheses exactly the way black
/// does when the single-line form would exceed `MAX_LINE_LEN`. Comments are left alone —
/// black does not reflow them, and flake8 has `E501` in `extend-ignore`.
fn push_assignment(body_pad: &str, name: &str, value: &str, lines: &mut Vec<String>) {
    let single = format!("{}{} = \"{}\"", body_pad, name, value);
    if single.chars().count() <= MAX_LINE_LEN {
        lines.push(single);
        return;
    }
    lines.push(format!("{}{} = (", body_pad, name));
    lines.push(format!("{}    \"{}\"", body_pad, value));
    lines.push(format!("{})", body_pad));
}

/// Render one Rust module as a Python class, recursing into nested `pub mod` so
/// `transport::tcp::ERRORS_TOTAL` becomes `transport.tcp.ERRORS_TOTAL`. `depth` is the
/// nesting level; each level adds four spaces of indentation.
fn render_class(module: &ModuleDef, depth: usize, lines: &mut Vec<String>) {
    let pad = "    ".repeat(depth);
    let body_pad = "    ".repeat(depth + 1);

    lines.push(format!("{}class {}:", pad, module.name));

    // Use doc comment from module if available
    let mut wrote_body = false;
    if !module.doc_comment.is_empty() {
        let first_line = module.doc_comment.lines().next().unwrap_or("").trim();
        if !first_line.is_empty() {
            lines.push(format!("{}\"\"\"{}\"\"\"", body_pad, first_line));
            wrote_body = true;
        }
    }

    // The blank separator is emitted only when something already sits in the body.
    // Emitting it unconditionally puts a blank line directly under a docstring-free
    // `class X:` header, which black then deletes — so the generator alone could not
    // reproduce its own checked-in artifact and every regeneration needed a second
    // formatting pass.
    if !module.constants.is_empty() {
        if wrote_body {
            lines.push(String::new());
        }
        wrote_body = true;
        for constant in &module.constants {
            if !constant.doc_comment.is_empty() {
                for comment_line in constant.doc_comment.lines() {
                    lines.push(format!("{}# {}", body_pad, comment_line));
                }
            }
            push_assignment(&body_pad, &constant.name, &constant.value, lines);
        }
    }

    // Nested classes go after the constants, separated by one blank line each — PEP 8
    // uses a single blank line between nested definitions, not two.
    for child in &module.submodules {
        if wrote_body {
            lines.push(String::new());
        }
        render_class(child, depth + 1, lines);
        wrote_body = true;
    }

    // A module with no doc comment, no constants and no submodules would otherwise emit
    // `class X:` with an empty body, which is a syntax error rather than the
    // silently-empty class this generator used to produce.
    if !wrote_body {
        lines.push(format!("{}pass", body_pad));
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();

    let mut source_path: Option<PathBuf> = None;
    let mut output_path: Option<PathBuf> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--source" => {
                i += 1;
                if i < args.len() {
                    source_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--output" => {
                i += 1;
                if i < args.len() {
                    output_path = Some(PathBuf::from(&args[i]));
                }
            }
            "--help" | "-h" => {
                print_usage();
                return Ok(());
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                print_usage();
                std::process::exit(1);
            }
        }
        i += 1;
    }

    // Determine paths relative to codegen directory
    let codegen_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));

    let source = source_path.unwrap_or_else(|| {
        // From: lib/bindings/python/codegen
        // To:   lib/runtime/src/metrics/prometheus_names.rs
        codegen_dir
            .join("../../../runtime/src/metrics/prometheus_names.rs")
            .canonicalize()
            .expect("Failed to resolve source path")
    });

    let output = output_path.unwrap_or_else(|| {
        // From: lib/bindings/python/codegen
        // To:   lib/bindings/python/src/dynamo/prometheus_names.py
        codegen_dir
            .join("../src/dynamo/prometheus_names.py")
            .canonicalize()
            .unwrap_or_else(|_| {
                // If file doesn't exist yet, resolve the parent directory
                let dir = codegen_dir
                    .join("../src/dynamo")
                    .canonicalize()
                    .expect("Failed to resolve output directory");
                dir.join("prometheus_names.py")
            })
    });

    println!("Generating Python prometheus_names from Rust source");
    println!("Source: {}", source.display());
    println!("Output: {}", output.display());
    println!();

    let content = std::fs::read_to_string(&source)
        .with_context(|| format!("Failed to read source file: {}", source.display()))?;

    println!("Parsing Rust AST...");
    let parser = PrometheusParser::parse_file(&content)?;

    println!("Found {} modules:", parser.modules.len());
    let mut module_names: Vec<&String> = parser.modules.keys().collect();
    module_names.sort();
    for name in module_names.iter() {
        let module = &parser.modules[name.as_str()];
        print_module_summary(module, 0);
    }

    println!("\nGenerating Python prometheus_names module...");
    let generator = PythonGenerator::new(&parser);
    let python_code = generator.generate_python_file();

    // Ensure output directory exists
    if let Some(parent) = output.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create output directory: {}", parent.display()))?;
    }

    std::fs::write(&output, python_code)
        .with_context(|| format!("Failed to write output file: {}", output.display()))?;

    println!("✓ Generated Python prometheus_names: {}", output.display());
    println!("\nSuccess! Python module ready for import.");

    Ok(())
}

/// Print one module and its nested modules, indented by nesting level. Nested modules are
/// listed explicitly so a run that emits an empty parent class is visible in the log —
/// the flat listing reported `transport: 0 constants` while silently dropping five names.
fn print_module_summary(module: &ModuleDef, depth: usize) {
    println!(
        "  {}- {}: {} constants{}",
        "  ".repeat(depth),
        module.name,
        module.constants.len(),
        if module.is_macro_generated {
            " (macro-generated)"
        } else {
            ""
        }
    );
    for child in &module.submodules {
        print_module_summary(child, depth + 1);
    }
}

fn print_usage() {
    println!(
        r#"
gen-python-prometheus-names - Generate Python prometheus_names from Rust source

Usage: gen-python-prometheus-names [OPTIONS]

Parses lib/runtime/src/metrics/prometheus_names.rs and generates a pure Python
module with 1:1 constant mappings at lib/bindings/python/src/dynamo/prometheus_names.py

This allows Python code to import Prometheus metric constants without Rust bindings:
    from dynamo.prometheus_names import frontend_service

OPTIONS:
    --source PATH    Path to Rust source file
                     (default: lib/runtime/src/metrics/prometheus_names.rs)

    --output PATH    Path to Python output file
                     (default: lib/bindings/python/src/dynamo/prometheus_names.py)

    --help, -h       Print this help message

EXAMPLES:
    # Generate with default paths
    cargo run -p dynamo-codegen --bin gen-python-prometheus-names

    # Generate with custom output
    cargo run -p dynamo-codegen --bin gen-python-prometheus-names -- --output /tmp/test.py
"#
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_codegen::prometheus_parser::PrometheusParser;
    /// End-to-end: parse Rust source, generate the Python module, then actually import it
    /// with `python3` and read the attributes back. This is the only test that proves the
    /// whole chain — recursion, nested-class indentation, and black-canonical formatting —
    /// produces a module where `transport.tcp.ERRORS_TOTAL` resolves. Asserting on the
    /// rendered text instead would pass even if the indentation made `tcp` a sibling of
    /// `transport` rather than a member.
    #[test]
    fn generated_module_exposes_nested_metric_names() {
        let parser = PrometheusParser::parse_file(
            r#"
/// Transport-specific metrics (TCP / NATS)
pub mod transport {
    pub mod tcp {
        pub const ERRORS_TOTAL: &str = "tcp_errors_total";
    }
}

pub mod frontend_service {
    pub const REQUESTS_TOTAL: &str = "requests_total";
    pub mod operation {
        pub const TOKENIZE: &str = "tokenize";
    }
}

pub mod empty_module {}
"#,
        )
        .expect("source should parse");

        let code = PythonGenerator::new(&parser).generate_python_file();

        // Unique per process so concurrent test binaries cannot collide on the path.
        let path =
            std::env::temp_dir().join(format!("dynamo_prometheus_names_{}.py", std::process::id()));
        std::fs::write(&path, &code).expect("write generated module");

        let out = std::process::Command::new("python3")
            .arg("-c")
            .arg(format!(
                "import importlib.util as u;                  s = u.spec_from_file_location('gen', r'{}');                  m = u.module_from_spec(s); s.loader.exec_module(m);                  print(m.transport.tcp.ERRORS_TOTAL);                  print(m.frontend_service.REQUESTS_TOTAL);                  print(m.frontend_service.operation.TOKENIZE)",
                path.display()
            ))
            .output()
            .expect("python3 should be available to import the generated module");

        let _ = std::fs::remove_file(&path);

        assert!(
            out.status.success(),
            "generated module failed to import:\n{}\n--- generated ---\n{}",
            String::from_utf8_lossy(&out.stderr),
            code
        );
        let values: Vec<&str> = std::str::from_utf8(&out.stdout)
            .expect("utf8 stdout")
            .lines()
            .collect();
        assert_eq!(
            values,
            vec!["tcp_errors_total", "requests_total", "tokenize"],
            "nested and flat names must both resolve as attributes"
        );

        // The generator must emit canonical black output on its own; a stray blank line
        // under a docstring-free header, or an over-long assignment left unwrapped, would
        // mean `cargo run` alone could not reproduce the checked-in file.
        assert!(
            !code.contains("class tcp:\n\n"),
            "no blank line belongs directly under a docstring-free class header:\n{code}"
        );
        assert!(
            code.contains("class empty_module:\n    pass"),
            "a module with no body must emit `pass`, not a syntax error:\n{code}"
        );
    }

    /// The committed Python module is a build artifact of this generator. If it
    /// drifts, importers such as components/src/dynamo/common/utils/prometheus.py
    /// fail with AttributeError at worker startup rather than here. Regenerate with:
    ///     cargo run -p dynamo-codegen --bin gen-python-prometheus-names
    #[test]
    fn committed_python_matches_generated() {
        let codegen_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
        let rust_source = codegen_dir.join("../../../runtime/src/metrics/prometheus_names.rs");
        let python_module = codegen_dir.join("../src/dynamo/prometheus_names.py");

        let rust_src = std::fs::read_to_string(&rust_source)
            .unwrap_or_else(|e| panic!("read {}: {e}", rust_source.display()));
        let parser = PrometheusParser::parse_file(&rust_src).expect("parse Rust source");
        let generated = PythonGenerator::new(&parser).generate_python_file();

        let committed = std::fs::read_to_string(&python_module)
            .unwrap_or_else(|e| panic!("read {}: {e}", python_module.display()));

        if generated == committed {
            return;
        }

        // assert_eq! on two ~500-line strings prints both in full, which buries
        // the one line that actually drifted. Report the first divergence instead.
        let (mut gen_lines, mut got_lines) = (generated.lines(), committed.lines());
        let mut lineno = 0;
        loop {
            lineno += 1;
            match (gen_lines.next(), got_lines.next()) {
                (None, None) => break,
                (expected, actual) if expected == actual => continue,
                (expected, actual) => panic!(
                    "prometheus_names.py is out of sync with prometheus_names.rs at line {lineno}.\n  \
                     expected (from generator): {:?}\n  \
                     found    (committed file): {:?}\n\
                     Regenerate: cargo run -p dynamo-codegen --bin gen-python-prometheus-names",
                    expected.unwrap_or("<end of file>"),
                    actual.unwrap_or("<end of file>"),
                ),
            }
        }
    }

    /// Unit-test the line-wrapping helper so long `NAME = "value"` assignments are
    /// emitted exactly the way black would format them.
    #[test]
    fn push_assignment_emits_black_compatible_lines() {
        let mut lines = Vec::new();
        push_assignment("    ", "SHORT", "short_value", &mut lines);
        assert_eq!(lines, vec![r#"    SHORT = "short_value""#]);

        lines.clear();
        let long_name = "MODEL_MIGRATION_MAX_SEQ_LEN_EXCEEDED_TOTAL";
        push_assignment(
            "    ",
            long_name,
            "model_migration_max_seq_len_exceeded_total",
            &mut lines,
        );
        assert_eq!(
            lines,
            vec![
                format!("    {long_name} = ("),
                r#"        "model_migration_max_seq_len_exceeded_total""#.to_string(),
                "    )".to_string(),
            ]
        );

        for line in lines {
            assert!(line.chars().count() <= MAX_LINE_LEN);
        }
    }
}
