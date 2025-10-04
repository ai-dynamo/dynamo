// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Binary to generate Python prometheus_names from Rust source

use anyhow::{Context, Result};
use dynamo_codegen::prometheus_parser::PrometheusParser;
use dynamo_codegen::python_generator::PythonGenerator;
use std::path::PathBuf;

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
    for (name, module) in &parser.modules {
        println!(
            "  - {}: {} constants{}",
            name,
            module.constants.len(),
            if module.is_macro_generated {
                " (macro-generated)"
            } else {
                ""
            }
        );
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

fn print_usage() {
    println!(
        r#"
gen-python-prometheus-names - Generate Python prometheus_names from Rust source

Usage: gen-python-prometheus-names [OPTIONS]

Parses lib/runtime/src/metrics/prometheus_names.rs and generates a pure Python
module with 1:1 constant mappings at lib/bindings/python/src/dynamo/prometheus_names.py

This allows Python code to import Prometheus metric constants without Rust bindings:
    from dynamo.prometheus_names import frontend_service, kvstats

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
