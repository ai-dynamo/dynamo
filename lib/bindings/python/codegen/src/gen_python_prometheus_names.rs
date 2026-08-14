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

    if !module.constants.is_empty() {
        lines.push("".to_string());
        wrote_body = true;
        for constant in &module.constants {
            if !constant.doc_comment.is_empty() {
                for comment_line in constant.doc_comment.lines() {
                    lines.push(format!("{}# {}", body_pad, comment_line));
                }
            }
            lines.push(format!(
                "{}{} = \"{}\"",
                body_pad, constant.name, constant.value
            ));
        }
    }

    // Nested classes go after the constants, separated by one blank line each — PEP 8
    // uses a single blank line between nested definitions, not two.
    for child in &module.submodules {
        lines.push("".to_string());
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
    use dynamo_codegen::prometheus_parser::ConstantDef;

    fn constant(name: &str, value: &str) -> ConstantDef {
        ConstantDef {
            name: name.to_string(),
            value: value.to_string(),
            doc_comment: String::new(),
        }
    }

    fn module(name: &str, constants: Vec<ConstantDef>, submodules: Vec<ModuleDef>) -> ModuleDef {
        ModuleDef {
            name: name.to_string(),
            constants,
            doc_comment: String::new(),
            is_macro_generated: false,
            macro_prefix: None,
            submodules,
        }
    }

    fn render(module: &ModuleDef) -> String {
        let mut lines = Vec::new();
        render_class(module, 0, &mut lines);
        lines.join("\n")
    }

    /// A nested module must be emitted indented inside its parent's body. Emitting it at
    /// column zero would make `transport.tcp` a sibling of `transport` instead of a
    /// member, so the attribute access the whole change exists for would fail.
    #[test]
    fn nested_class_is_indented_inside_its_parent() {
        let rendered = render(&module(
            "transport",
            vec![],
            vec![module(
                "tcp",
                vec![constant("ERRORS_TOTAL", "tcp_errors_total")],
                vec![],
            )],
        ));

        assert_eq!(
            rendered,
            "class transport:\n\
             \n\
             \x20   class tcp:\n\
             \n\
             \x20       ERRORS_TOTAL = \"tcp_errors_total\""
        );
    }

    /// Three levels deep the indentation must keep compounding; a fixed four-space pad
    /// would silently flatten `outer.middle.inner` into `outer.middle`.
    #[test]
    fn indentation_compounds_with_depth() {
        let rendered = render(&module(
            "outer",
            vec![],
            vec![module(
                "middle",
                vec![],
                vec![module(
                    "inner",
                    vec![constant("LEAF", "leaf_value")],
                    vec![],
                )],
            )],
        ));

        assert!(
            rendered.contains("\n        class inner:"),
            "inner class should sit at 8 spaces, got:\n{rendered}"
        );
        assert!(
            rendered.contains("\n            LEAF = \"leaf_value\""),
            "leaf constant should sit at 12 spaces, got:\n{rendered}"
        );
    }

    /// Parent constants come first, then nested classes. Interleaving them would still be
    /// valid Python but churns the diff whenever a constant is added.
    #[test]
    fn parent_constants_precede_nested_classes() {
        let rendered = render(&module(
            "frontend_service",
            vec![constant("REQUESTS_TOTAL", "requests_total")],
            vec![module(
                "status",
                vec![constant("SUCCESS", "success")],
                vec![],
            )],
        ));

        let const_at = rendered.find("REQUESTS_TOTAL").expect("parent constant");
        let class_at = rendered.find("class status:").expect("nested class");
        assert!(
            const_at < class_at,
            "parent constants must precede nested classes, got:\n{rendered}"
        );
    }

    /// An otherwise-empty module must emit `pass`. Without it the generator writes
    /// `class X:` with nothing under it, which is a SyntaxError — the one failure mode
    /// that is worse than the silent empty class this change replaced.
    #[test]
    fn empty_module_emits_pass() {
        assert_eq!(
            render(&module("empty", vec![], vec![])),
            "class empty:\n    pass"
        );
    }

    /// The `pass` guard must not fire when a docstring alone forms the body, or every
    /// documented-but-constant-free module grows a redundant statement.
    #[test]
    fn docstring_only_module_does_not_emit_pass() {
        let mut m = module("documented", vec![], vec![]);
        m.doc_comment = "Only a doc comment".to_string();
        let rendered = render(&m);
        assert_eq!(
            rendered,
            "class documented:\n    \"\"\"Only a doc comment\"\"\""
        );
        assert!(!rendered.contains("pass"));
    }
}
