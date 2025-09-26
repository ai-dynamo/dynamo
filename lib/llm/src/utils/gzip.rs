// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Gzip decompression utilities for handling compressed files in temporary directories.
//!
//! This module provides a clean API for decompressing gzip files to temporary directories
//! with automatic cleanup through RAII. The primary use case is for compressed files stored
//! in git-lfs that need to be decompressed for processing by code expecting file paths.
//!
//! Uses pure Rust implementation with no system dependencies.

use anyhow::{Context, Result};
use flate2::read::GzDecoder;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};
use tempfile::TempDir;

/// Builder for configuring gzip extraction operations.
///
/// This struct uses the Builder pattern to configure extraction options.
/// The `extract()` method performs the actual decompression operation.
#[derive(Debug, Clone, Default)]
pub struct GzipExtractorBuilder {
    /// Path to the source gzip compressed file
    source_path: Option<PathBuf>,
    /// Optional custom filename for the extracted file in temp directory
    target_filename: Option<String>,
}

/// Configuration for gzip extraction operations.
#[derive(Debug, Clone)]
pub struct GzipExtractor {
    /// Path to the source gzip compressed file
    source_path: PathBuf,
    /// Optional custom filename for the extracted file in temp directory
    /// If None, derives name from source_path by removing .gz extension
    target_filename: Option<String>,
}

/// Result of a successful gzip extraction operation.
///
/// This struct provides RAII cleanup of the temporary directory and
/// access to the extracted file path. The temporary directory and its
/// contents are automatically removed when this struct is dropped.
#[derive(Debug)]
pub struct GzipExtraction {
    /// RAII temporary directory - kept private to prevent misuse
    _temp_dir: TempDir,
    /// Path to the extracted file within the temporary directory
    file_path: PathBuf,
}

impl GzipExtractorBuilder {
    /// Set the source path for the gzip file to extract.
    pub fn source_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.source_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set a custom target filename for the extracted file.
    pub fn target_filename<S: Into<String>>(mut self, filename: S) -> Self {
        self.target_filename = Some(filename.into());
        self
    }

    /// Build the extractor configuration and immediately perform the extraction.
    ///
    /// This method validates the configuration, performs the gzip decompression,
    /// and returns a `GzipExtraction` handle for accessing the extracted file.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - Source file doesn't exist or can't be read
    /// - Source file is not valid gzip format
    /// - Target filename is invalid
    /// - Temporary directory creation fails
    /// - Decompression operation fails
    pub fn extract(self) -> Result<GzipExtraction> {
        let source_path = self
            .source_path
            .ok_or_else(|| anyhow::anyhow!("source_path is required"))?;

        let extractor = GzipExtractor {
            source_path,
            target_filename: self.target_filename,
        };

        extractor.perform_extraction()
    }
}

impl GzipExtractor {
    /// Create a new builder for configuring gzip extraction.
    pub fn builder() -> GzipExtractorBuilder {
        GzipExtractorBuilder::default()
    }

    /// Perform the actual gzip extraction operation.
    ///
    /// This method creates a temporary directory, decompresses the source file,
    /// and returns a handle to the extracted file with RAII cleanup.
    fn perform_extraction(self) -> Result<GzipExtraction> {
        // Validate source file exists
        if !self.source_path.exists() {
            anyhow::bail!("Source file does not exist: {}", self.source_path.display());
        }

        // Create temporary directory
        let temp_dir = tempfile::tempdir().context("Failed to create temporary directory")?;

        // Determine target filename
        let target_filename = match &self.target_filename {
            Some(filename) => filename.clone(),
            None => {
                // Derive filename from source by removing .gz extension
                let source_filename = self
                    .source_path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .ok_or_else(|| anyhow::anyhow!("Invalid source filename"))?;

                if source_filename.ends_with(".gz") {
                    source_filename.strip_suffix(".gz").unwrap().to_string()
                } else {
                    format!("{}.extracted", source_filename)
                }
            }
        };

        let target_path = temp_dir.path().join(&target_filename);

        // Open source file and create decoder
        let source_file = File::open(&self.source_path).with_context(|| {
            format!("Failed to open source file: {}", self.source_path.display())
        })?;

        let mut decoder = GzDecoder::new(source_file);

        // Create target file and decompress
        let mut target_file = File::create(&target_path)
            .with_context(|| format!("Failed to create target file: {}", target_path.display()))?;

        std::io::copy(&mut decoder, &mut target_file).with_context(|| {
            format!(
                "Failed to decompress {} to {}",
                self.source_path.display(),
                target_path.display()
            )
        })?;

        target_file.flush().context("Failed to flush target file")?;

        Ok(GzipExtraction {
            _temp_dir: temp_dir,
            file_path: target_path,
        })
    }
}

impl GzipExtraction {
    /// Get the path to the extracted file.
    ///
    /// The file exists within a temporary directory that will be cleaned up
    /// when this `GzipExtraction` instance is dropped.
    pub fn file_path(&self) -> &Path {
        &self.file_path
    }

    /// Get the path to the temporary directory containing the extracted file.
    ///
    /// The directory will be cleaned up when this `GzipExtraction` instance is dropped.
    pub fn temp_dir_path(&self) -> &Path {
        self._temp_dir.path()
    }

    /// Read the entire extracted file contents as a UTF-8 string.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be read
    /// - The file contents are not valid UTF-8
    pub fn read_to_string(&self) -> Result<String> {
        std::fs::read_to_string(&self.file_path).with_context(|| {
            format!(
                "Failed to read file as string: {}",
                self.file_path.display()
            )
        })
    }

    /// Read the entire extracted file contents as raw bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read.
    pub fn read_to_bytes(&self) -> Result<Vec<u8>> {
        std::fs::read(&self.file_path)
            .with_context(|| format!("Failed to read file as bytes: {}", self.file_path.display()))
    }

    /// Validate the extracted file against a BLAKE3 hash.
    ///
    /// # Arguments
    ///
    /// * `expected_hash` - The expected BLAKE3 hash as a hexadecimal string
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The file cannot be read
    /// - The computed hash doesn't match the expected hash
    pub fn validate_blake3_hash(&self, expected_hash: &str) -> Result<()> {
        let file_contents = self.read_to_bytes()?;
        let computed_hash = blake3::hash(&file_contents);
        let computed_hash_hex = computed_hash.to_hex();

        if computed_hash_hex.as_str() == expected_hash {
            Ok(())
        } else {
            anyhow::bail!(
                "BLAKE3 hash mismatch for file {}: expected {}, got {}",
                self.file_path.display(),
                expected_hash,
                computed_hash_hex
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use flate2::Compression;
    use flate2::write::GzEncoder;
    use std::io::Write;

    /// Create a test gzip file with known content for testing
    fn create_test_gzip_file(content: &str) -> Result<tempfile::NamedTempFile> {
        let mut temp_file = tempfile::NamedTempFile::with_suffix(".gz")?;

        // Compress the test data
        let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(content.as_bytes())?;
        let compressed_data = encoder.finish()?;

        // Write compressed data to temp file
        temp_file.write_all(&compressed_data)?;
        temp_file.flush()?;

        Ok(temp_file)
    }

    #[test]
    fn test_basic_extraction() -> Result<()> {
        let test_content = "Hello, World!\nThis is a test file for gzip compression.\n";
        let temp_file = create_test_gzip_file(test_content)?;

        let extraction = GzipExtractor::builder()
            .source_path(temp_file.path())
            .extract()?;

        // Verify the extracted file exists and has correct content
        assert!(extraction.file_path().exists());
        let content = extraction.read_to_string()?;
        assert_eq!(content, test_content);

        Ok(())
    }

    #[test]
    fn test_custom_target_filename() -> Result<()> {
        let test_content = "Custom filename test content";
        let temp_file = create_test_gzip_file(test_content)?;

        let extraction = GzipExtractor::builder()
            .source_path(temp_file.path())
            .target_filename("custom.txt")
            .extract()?;

        // Verify the file has the custom name
        assert_eq!(extraction.file_path().file_name().unwrap(), "custom.txt");
        let content = extraction.read_to_string()?;
        assert_eq!(content, test_content);

        Ok(())
    }

    #[test]
    fn test_automatic_filename_derivation() -> Result<()> {
        let test_content = "Automatic filename test";

        // Create a temp file with a .gz extension
        let mut temp_file = tempfile::NamedTempFile::with_suffix(".json.gz")?;
        let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
        encoder.write_all(test_content.as_bytes())?;
        let compressed_data = encoder.finish()?;
        temp_file.write_all(&compressed_data)?;
        temp_file.flush()?;

        let extraction = GzipExtractor::builder()
            .source_path(temp_file.path())
            .extract()?;

        // Should derive filename by removing .gz extension
        let expected_filename = temp_file
            .path()
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .strip_suffix(".gz")
            .unwrap();

        assert_eq!(
            extraction
                .file_path()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap(),
            expected_filename
        );
        let content = extraction.read_to_string()?;
        assert_eq!(content, test_content);

        Ok(())
    }

    #[test]
    fn test_missing_source_path_error() {
        let result = GzipExtractor::builder().extract();
        assert!(result.is_err());
        assert!(
            result
                .unwrap_err()
                .to_string()
                .contains("source_path is required")
        );
    }

    #[test]
    fn test_nonexistent_file_error() {
        let result = GzipExtractor::builder()
            .source_path("/nonexistent/file.gz")
            .extract();
        assert!(result.is_err());
        assert!(result.unwrap_err().to_string().contains("does not exist"));
    }

    #[test]
    fn test_raii_cleanup() -> Result<()> {
        let test_content = "RAII cleanup test";
        let temp_file = create_test_gzip_file(test_content)?;
        let temp_dir_path;

        {
            let extraction = GzipExtractor::builder()
                .source_path(temp_file.path())
                .extract()?;

            temp_dir_path = extraction.temp_dir_path().to_path_buf();
            assert!(temp_dir_path.exists());
            assert!(extraction.file_path().exists());
        } // extraction goes out of scope here

        // Temporary directory should be cleaned up
        assert!(!temp_dir_path.exists());

        Ok(())
    }

    #[test]
    fn test_builder_validation() {
        // Test that builder validates required fields
        let result = GzipExtractor::builder().extract();
        assert!(result.is_err());

        // Test that builder accepts valid configuration
        let temp_file = tempfile::NamedTempFile::with_suffix(".gz").unwrap();
        let result = GzipExtractor::builder()
            .source_path(temp_file.path())
            .target_filename("test.txt")
            .extract();

        // This should fail because the file is empty/invalid gzip, but not due to validation
        assert!(result.is_err());
    }

    #[test]
    fn test_complex_filename_patterns() -> Result<()> {
        let test_content = "Complex filename test";

        // Test various filename patterns
        let test_cases = vec![
            ("data.json.gz", "data.json"),
            ("file.txt.gz", "file.txt"),
            ("archive.tar.gz", "archive.tar"),
            ("simple.gz", "simple"),
            ("no-extension", "no-extension.extracted"),
        ];

        for (input_filename, expected_output) in test_cases {
            // Create temp directory and file with exact name we want to test
            let temp_dir = tempfile::tempdir()?;
            let temp_file_path = temp_dir.path().join(input_filename);

            // Create compressed file with the exact name
            let mut encoder = GzEncoder::new(Vec::new(), Compression::best());
            encoder.write_all(test_content.as_bytes())?;
            let compressed_data = encoder.finish()?;

            std::fs::write(&temp_file_path, &compressed_data)?;

            let extraction = GzipExtractor::builder()
                .source_path(&temp_file_path)
                .extract()?;

            assert_eq!(
                extraction
                    .file_path()
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap(),
                expected_output,
                "Failed for input: {}",
                input_filename
            );
        }

        Ok(())
    }

    #[test]
    fn test_extract_and_validate_with_mock_tokenizer() -> Result<()> {
        // Use the existing mock tokenizer instead of the deepseek one
        let _tokenizer_file_path =
            "tests/data/sample-models/mock-llama-3.1-8b-instruct/tokenizer.json";

        // For this test, we'll just read the file and validate it exists and is readable
        // Since it's not compressed, we'll test a different aspect of the gzip functionality
        use flate2::Compression;
        use flate2::write::GzEncoder;
        use std::io::Write;
        use tempfile::NamedTempFile;

        // Create a temporary gzipped file for testing
        let test_content = r#"{"test": "tokenizer", "vocab_size": 1000}"#;
        let mut temp_file = NamedTempFile::new()?;
        {
            let mut encoder = GzEncoder::new(&mut temp_file, Compression::default());
            encoder.write_all(test_content.as_bytes())?;
            encoder.finish()?;
        }

        let extraction = GzipExtractor::builder()
            .source_path(temp_file.path())
            .target_filename("test-tokenizer.json")
            .extract()?;

        // Verify the file was extracted with the correct name
        assert_eq!(
            extraction
                .file_path()
                .file_name()
                .unwrap()
                .to_str()
                .unwrap(),
            "test-tokenizer.json"
        );

        // Verify the extracted content matches what we compressed
        let extracted_content = std::fs::read_to_string(extraction.file_path())?;
        assert_eq!(extracted_content, test_content);

        println!("Successfully created, compressed, and extracted test tokenizer file");

        Ok(())
    }
}
