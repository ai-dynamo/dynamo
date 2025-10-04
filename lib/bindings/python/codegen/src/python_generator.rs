// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python code generator for prometheus_names

use crate::prometheus_parser::{ModuleDef, PrometheusParser};
use std::collections::HashMap;
use std::path::PathBuf;

pub struct PythonGenerator<'a> {
    modules: &'a HashMap<String, ModuleDef>,
}

impl<'a> PythonGenerator<'a> {
    pub fn new(parser: &'a PrometheusParser) -> Self {
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

    pub fn generate_python_file(&self) -> String {
        let mut output = Self::load_template("prometheus_names.py.template");

        // Append generated classes
        output.push_str(&self.generate_classes());

        output
    }

    fn generate_classes(&self) -> String {
        let mut lines = Vec::new();

        // Generate simple classes with constants as class attributes
        for (module_name, module) in self.modules.iter() {
            lines.push(format!("class {}:", module_name));

            // Use doc comment from module if available
            if !module.doc_comment.is_empty() {
                let first_line = module.doc_comment.lines().next().unwrap_or("").trim();
                if !first_line.is_empty() {
                    lines.push(format!("    \"\"\"{}\"\"\"", first_line));
                }
            }
            lines.push("".to_string());

            for constant in &module.constants {
                if !constant.doc_comment.is_empty() {
                    for comment_line in constant.doc_comment.lines() {
                        lines.push(format!("    # {}", comment_line));
                    }
                }
                lines.push(format!("    {} = \"{}\"", constant.name, constant.value));
            }

            lines.push("".to_string());
        }

        lines.join("\n")
    }
}
