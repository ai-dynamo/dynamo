// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::coding::common::{dedupe_paths, expand_user_path, home_dir};
use anyhow::{Result, bail};
use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};

const IGNORED_FILENAMES: &[&str] = &["history.jsonl"];

pub fn iter_ancestor_roots(start: &Path) -> Vec<PathBuf> {
    let mut roots = Vec::new();
    let mut current = start.to_path_buf();
    loop {
        roots.push(current.clone());
        let Some(parent) = current.parent() else {
            break;
        };
        if parent == current {
            break;
        }
        current = parent.to_path_buf();
    }
    dedupe_paths(roots)
}

pub fn claude_project_dir_for_root(root: &Path, home_dir: &Path) -> PathBuf {
    let encoded = root.to_string_lossy().replace('/', "-");
    home_dir.join(".claude").join("projects").join(encoded)
}

pub fn discover_trace_files(explicit_inputs: &[String], start_dir: &Path) -> Result<Vec<PathBuf>> {
    let Some(home_dir) = home_dir() else {
        bail!("could not resolve HOME for Claude trace discovery");
    };
    let claude_projects_root = home_dir.join(".claude").join("projects");
    let mut discovered = Vec::new();

    if !explicit_inputs.is_empty() {
        for raw_path in explicit_inputs {
            let input_path = expand_user_path(raw_path);
            let input_path = input_path.canonicalize().unwrap_or(input_path);
            if input_path.is_file() {
                if !is_trace_path(&input_path) || is_ignored_path(&input_path) {
                    bail!("not a Claude session trace file: {}", input_path.display());
                }
                discovered.push(input_path);
                continue;
            }

            if !input_path.exists() {
                bail!("input path does not exist: {}", input_path.display());
            }

            if !input_path.is_dir() {
                bail!("unsupported input path: {}", input_path.display());
            }

            let in_claude_tree =
                input_path == claude_projects_root || input_path.starts_with(&claude_projects_root);
            if in_claude_tree {
                let directory_hits = scan_trace_dir(&input_path)?;
                if !directory_hits.is_empty() {
                    discovered.extend(directory_hits);
                    continue;
                }
            } else {
                let repo_hits =
                    scan_trace_dir(&claude_project_dir_for_root(&input_path, &home_dir))?;
                if !repo_hits.is_empty() {
                    discovered.extend(repo_hits);
                    continue;
                }

                let directory_hits = scan_trace_dir(&input_path)?;
                if !directory_hits.is_empty() {
                    discovered.extend(directory_hits);
                    continue;
                }
            }

            bail!(
                "no Claude session traces found under input path or its encoded Claude project directory: {}",
                input_path.display()
            );
        }

        return Ok(dedupe_paths(discovered));
    }

    for candidate_root in iter_ancestor_roots(start_dir) {
        discovered.extend(scan_trace_dir(&claude_project_dir_for_root(
            &candidate_root,
            &home_dir,
        ))?);
    }
    discovered.extend(scan_trace_dir(&claude_projects_root)?);

    Ok(dedupe_paths(discovered))
}

fn scan_trace_dir(root: &Path) -> Result<Vec<PathBuf>> {
    if !root.exists() {
        return Ok(Vec::new());
    }

    let mut queue = VecDeque::from([root.to_path_buf()]);
    let mut discovered = Vec::new();
    while let Some(directory) = queue.pop_front() {
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            let path = entry.path();
            let file_type = entry.file_type()?;
            if file_type.is_dir() {
                if path.file_name().and_then(|value| value.to_str()) == Some("subagents") {
                    continue;
                }
                queue.push_back(path);
                continue;
            }
            if file_type.is_file() && is_trace_path(&path) && !is_ignored_path(&path) {
                discovered.push(path);
            }
        }
    }

    discovered.sort();
    Ok(discovered)
}

fn is_trace_path(path: &Path) -> bool {
    path.extension().and_then(|value| value.to_str()) == Some("jsonl")
        && !IGNORED_FILENAMES
            .iter()
            .any(|ignored| path.file_name().and_then(|value| value.to_str()) == Some(ignored))
}

fn is_ignored_path(path: &Path) -> bool {
    path.components()
        .any(|component| component.as_os_str() == "subagents")
        || IGNORED_FILENAMES
            .iter()
            .any(|ignored| path.file_name().and_then(|value| value.to_str()) == Some(ignored))
}

#[cfg(test)]
mod tests {
    use super::iter_ancestor_roots;
    use tempfile::TempDir;

    #[test]
    fn ancestors_walk_to_root() {
        let temp = TempDir::new().unwrap();
        let nested = temp.path().join("a").join("b").join("c");
        std::fs::create_dir_all(&nested).unwrap();

        let roots = iter_ancestor_roots(&nested);
        assert_eq!(roots.first().unwrap(), &nested.canonicalize().unwrap());
        let last = roots.last().unwrap();
        assert!(
            last.parent().is_none(),
            "expected filesystem root, got {}",
            last.display()
        );
    }
}
