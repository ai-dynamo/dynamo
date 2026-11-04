// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use crate::coding::common::{dedupe_paths, expand_user_path, home_dir};
use anyhow::{Result, bail};
use std::fs;
use std::path::{Path, PathBuf};

/// Returns `$CODEX_HOME`, or `~/.codex` when it is unset.
pub fn codex_home() -> Option<PathBuf> {
    match std::env::var_os("CODEX_HOME") {
        Some(home) if !home.is_empty() => Some(PathBuf::from(home)),
        _ => home_dir().map(|home| home.join(".codex")),
    }
}

/// Finds Codex rollout files under the explicit inputs, or under the Codex home.
///
/// Codex moves rollouts from `sessions` into `archived_sessions` while it runs, so default
/// discovery scans both to keep agent trees complete.
pub fn discover_rollout_files(explicit_inputs: &[String]) -> Result<Vec<PathBuf>> {
    if explicit_inputs.is_empty() {
        let Some(codex_home) = codex_home() else {
            bail!("could not resolve CODEX_HOME or HOME for Codex rollout discovery");
        };
        let mut discovered = scan_rollout_dir(&codex_home.join("sessions"))?;
        discovered.extend(scan_rollout_dir(&codex_home.join("archived_sessions"))?);
        return Ok(discovered);
    }

    let mut discovered = Vec::new();
    for raw_path in explicit_inputs {
        let input_path = expand_user_path(raw_path);
        if input_path.is_file() {
            if !is_rollout_path(&input_path) {
                bail!("not a Codex rollout file: {}", input_path.display());
            }
            discovered.push(input_path);
            continue;
        }
        if !input_path.is_dir() {
            bail!("input path does not exist: {}", input_path.display());
        }
        let hits = scan_rollout_dir(&input_path)?;
        if hits.is_empty() {
            bail!(
                "no Codex rollout files found under {}",
                input_path.display()
            );
        }
        discovered.extend(hits);
    }
    Ok(dedupe_paths(discovered))
}

fn scan_rollout_dir(root: &Path) -> Result<Vec<PathBuf>> {
    if !root.exists() {
        return Ok(Vec::new());
    }
    let mut pending = vec![root.to_path_buf()];
    let mut discovered = Vec::new();
    while let Some(directory) = pending.pop() {
        for entry in fs::read_dir(&directory)? {
            let entry = entry?;
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                pending.push(path);
            } else if is_rollout_path(&path) {
                discovered.push(path);
            }
        }
    }
    discovered.sort();
    Ok(discovered)
}

fn is_rollout_path(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
        return false;
    };
    name.starts_with("rollout-") && name.ends_with(".jsonl")
}

#[cfg(test)]
mod tests {
    use super::discover_rollout_files;
    use std::fs;
    use tempfile::TempDir;

    #[test]
    fn discovers_nested_rollouts_and_rejects_other_files() {
        let temp = TempDir::new().unwrap();
        let day = temp.path().join("2026").join("09").join("30");
        fs::create_dir_all(&day).unwrap();
        fs::write(day.join("rollout-a.jsonl"), "").unwrap();
        fs::write(day.join("notes.jsonl"), "").unwrap();

        let found = discover_rollout_files(&[temp.path().display().to_string()]).unwrap();
        assert_eq!(found, [day.join("rollout-a.jsonl").canonicalize().unwrap()]);
        assert!(discover_rollout_files(&[day.join("notes.jsonl").display().to_string()]).is_err());
    }
}
