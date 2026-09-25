// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::path::Path;

use crate::Result;

const MANIFEST_NAME: &str = "model.protection.json";
const SIGNATURE_NAME: &str = "model.protection.sig";

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelProtectionKind {
    Plain,
    ProtectedCandidate,
}

/// Detect only reserved package markers. Package verification decides whether a candidate is valid.
pub fn detect_model_protection(model: &Path) -> Result<ModelProtectionKind> {
    match std::fs::metadata(model) {
        Ok(metadata) if metadata.is_dir() => {}
        Ok(_) => return Ok(ModelProtectionKind::Plain),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Ok(ModelProtectionKind::Plain);
        }
        Err(error) => return Err(error.into()),
    }
    for marker in [MANIFEST_NAME, SIGNATURE_NAME] {
        match model.join(marker).symlink_metadata() {
            Ok(_) => {
                return Ok(ModelProtectionKind::ProtectedCandidate);
            }
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => {}
            Err(error) => return Err(error.into()),
        }
    }
    Ok(ModelProtectionKind::Plain)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_reserved_markers_select_the_protected_path() {
        let model = tempfile::tempdir().unwrap();
        std::fs::write(model.path().join("pytorch_model.bin"), b"plain").unwrap();
        std::fs::write(model.path().join("manifest.json"), b"generic").unwrap();
        assert_eq!(
            detect_model_protection(model.path()).unwrap(),
            ModelProtectionKind::Plain
        );

        std::fs::write(model.path().join(MANIFEST_NAME), b"protected").unwrap();
        assert_eq!(
            detect_model_protection(model.path()).unwrap(),
            ModelProtectionKind::ProtectedCandidate
        );

        let oversized = tempfile::tempdir().unwrap();
        for index in 0..=4096 {
            std::fs::write(oversized.path().join(index.to_string()), []).unwrap();
        }
        assert_eq!(
            detect_model_protection(oversized.path()).unwrap(),
            ModelProtectionKind::Plain
        );
    }
}
