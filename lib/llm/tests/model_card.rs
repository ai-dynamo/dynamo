// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::model_card::{
    ArtifactRef, ArtifactRole, ModelDeploymentCard, PromptFormatterArtifact, TokenizerKind,
};
use tempfile::tempdir;

const HF_PATH: &str = "tests/data/sample-models/TinyLlama_v1.1";

#[tokio::test]
async fn test_model_info_from_hf_like_local_repo() {
    let mdc = ModelDeploymentCard::load_from_disk(HF_PATH, None).unwrap();
    let info = mdc.model_info.unwrap().get_model_info().unwrap();
    assert_eq!(info.model_type(), "llama");
    assert_eq!(info.bos_token_id(), Some(1));
    assert_eq!(info.eos_token_ids(), vec![2]);
    assert_eq!(info.max_position_embeddings(), Some(2048));
    assert_eq!(info.vocab_size(), Some(32000));
}

#[tokio::test]
async fn test_model_info_from_non_existent_local_repo() {
    let path = "tests/data/sample-models/this-model-does-not-exist";
    let result = ModelDeploymentCard::load_from_disk(path, None);
    assert!(result.is_err());
}

#[tokio::test]
async fn test_tokenizer_from_hf_like_local_repo() {
    let mdc = ModelDeploymentCard::load_from_disk(HF_PATH, None).unwrap();
    // Verify tokenizer file was found
    match mdc.tokenizer.unwrap() {
        TokenizerKind::HfTokenizerJson(_) => (),
        TokenizerKind::TikTokenModel(_) => panic!("Expected HfTokenizerJson, got TikTokenModel"),
    }
}

#[tokio::test]
async fn test_prompt_formatter_from_hf_like_local_repo() {
    let mdc = ModelDeploymentCard::load_from_disk(HF_PATH, None).unwrap();
    // Verify prompt formatter was found
    match mdc.prompt_formatter {
        Some(PromptFormatterArtifact::HfTokenizerConfigJson(_)) => (),
        _ => panic!("Expected HfTokenizerConfigJson prompt formatter"),
    }
}

#[tokio::test]
async fn test_missing_required_files() {
    // Create empty temp directory
    let temp_dir = tempdir().unwrap();
    let result = ModelDeploymentCard::load_from_disk(temp_dir.path(), None);
    assert!(result.is_err());
    let err = result.unwrap_err().to_string();
    // Should fail because config.json is missing
    assert!(err.contains("unable to extract"));
}

/// Models without tokenizer.json (e.g. Qwen3-Omni which ships vocab.json + merges.txt)
/// should load successfully with tokenizer set to None. The frontend must use a
/// non-Rust chat processor for these models (e.g. --dyn-chat-processor vllm).
#[tokio::test]
async fn test_model_loads_without_tokenizer_json() {
    let path = "tests/data/sample-models/mock-no-tokenizer-json";
    let mdc = ModelDeploymentCard::load_from_disk(path, None).unwrap();
    assert!(
        mdc.tokenizer.is_none(),
        "Expected tokenizer to be None for model without tokenizer.json"
    );
    assert!(!mdc.has_tokenizer(), "has_tokenizer() should be false");
    // Model info should still be loaded
    assert!(mdc.model_info.is_some());
}

/// chat_template.json should be picked up as a fallback when chat_template.jinja
/// does not exist (e.g. Qwen3-Omni). The fixture's tokenizer_config.json has no
/// inline chat_template, so this is the only template source.
#[tokio::test]
async fn test_chat_template_json_fallback() {
    let path = "tests/data/sample-models/mock-no-tokenizer-json";
    let mdc = ModelDeploymentCard::load_from_disk(path, None).unwrap();
    match &mdc.chat_template_file {
        Some(PromptFormatterArtifact::HfChatTemplateJson { file, is_custom }) => {
            assert!(!is_custom, "Should not be marked as custom template");
            let p = file.path().expect("Should be a local path");
            assert!(
                p.ends_with("chat_template.json"),
                "Expected chat_template.json, got {:?}",
                p
            );
        }
        other => panic!("Expected HfChatTemplateJson, got {:?}", other),
    }
}

// =============================================================
// `files` vector + download tests (gh-8749)
//
// These tests exercise the per-file artifact list and the
// content-addressed cache populated by `ModelDeploymentCard::download_config`.
//
// To force `download_config` to actually run the new download path
// (rather than short-circuiting via `has_local_files()`), each test
// URL-backs the typed CheckedFiles via `move_to_url`, then restores
// the `files` vector to its original `file://` URIs. That sets up:
//   - typed enums: URL-backed → has_local_files() returns false
//   - files vector: file:// URIs → resolved via the new path
//
// Cache lives under $HOME/.cache/dynamo/mdc/. Tests use `serial_test`
// + a per-test tempdir-overridden HOME so they don't pollute the
// user's real cache or interfere with each other.
// =============================================================

/// Override `HOME` for the duration of a test scope. Returns the
/// `TempDir` so it stays alive (otherwise it'd be cleaned up before
/// the test runs). Pair with `#[serial_test::serial]` so the env
/// override doesn't race with other tests.
fn isolated_home() -> tempfile::TempDir {
    let dir = tempdir().expect("tempdir for isolated $HOME");
    // SAFETY: tests on this file are serialized via serial_test.
    unsafe { std::env::set_var("HOME", dir.path()) };
    dir
}

/// Fresh MDC loaded from TinyLlama, with the typed CheckedFiles
/// URL-backed (so `has_local_files()` returns false) but the
/// `files` vector still carrying the original `file://` URIs that
/// `from_repo_checkout` populated.
fn mdc_with_file_scheme_files() -> ModelDeploymentCard {
    let mut mdc = ModelDeploymentCard::load_from_disk(HF_PATH, None).unwrap();
    let original_files = mdc.files.clone();
    // Forces typed enums to URL-backed (and rewrites files; we'll
    // immediately undo that part).
    mdc.move_to_url("hf://test/").unwrap();
    mdc.files = original_files;
    mdc
}

#[tokio::test]
async fn test_files_vector_populated_from_local_repo() {
    let mdc = ModelDeploymentCard::load_from_disk(HF_PATH, None).unwrap();

    // TinyLlama has tokenizer.json, config.json, tokenizer_config.json,
    // chat_template.jinja, generation_config.json — all five roles.
    assert!(
        !mdc.files.is_empty(),
        "files vector must be populated by from_repo_checkout"
    );

    // Every entry has a file:// URI pointing into the test fixture
    // dir, a blake3 checksum, and matching role.
    for artifact in &mdc.files {
        assert!(
            artifact.uri.starts_with("file://"),
            "expected file:// uri, got {}",
            artifact.uri
        );
        assert!(
            artifact.checksum.starts_with("blake3:"),
            "expected blake3 checksum, got {}",
            artifact.checksum
        );
        assert!(artifact.size > 0, "expected non-zero size for {artifact:?}");
    }

    // Roles must be unique within the closed five-role set.
    let roles: std::collections::HashSet<_> = mdc.files.iter().map(|a| a.role).collect();
    for role in &roles {
        assert!(matches!(
            role,
            ArtifactRole::Config
                | ArtifactRole::Tokenizer
                | ArtifactRole::TokenizerConfig
                | ArtifactRole::ChatTemplate
                | ArtifactRole::GenerationConfig
        ));
    }
}

#[serial_test::serial]
#[tokio::test]
async fn test_download_files_resolves_local_file_scheme() {
    let _home = isolated_home();
    let mut mdc = mdc_with_file_scheme_files();
    let expected_files = mdc.files.clone();
    let slug = mdc.slug().to_string();

    mdc.download_config().await.expect("download_config");

    // Each artifact's blob must exist under blobs/<blake3-hex> and
    // its bytes must blake3-match the MDC entry.
    let blobs_dir =
        std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/dynamo/mdc/blobs");
    let slug_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/dynamo/mdc/by-slug")
        .join(&slug);
    for artifact in &expected_files {
        let hex = artifact.checksum.strip_prefix("blake3:").unwrap();
        let blob = blobs_dir.join(hex);
        assert!(blob.exists(), "expected blob at {}", blob.display());

        let actual = format!("blake3:{}", blake3::hash(&std::fs::read(&blob).unwrap()));
        assert_eq!(actual, artifact.checksum, "blob bytes blake3-match");

        let link = slug_dir.join(&artifact.filename);
        assert!(link.exists(), "expected by-slug link at {}", link.display());
    }

    // The tokenizer should now load via the symlinked path — proves
    // update_dir routed downstream consumers to the cached files.
    let _tokenizer = mdc.tokenizer().expect("tokenizer should load");
}

#[serial_test::serial]
#[tokio::test]
async fn test_download_files_dedupes_by_blake3() {
    let _home = isolated_home();

    // First registration: populate the cache.
    let mut mdc1 = mdc_with_file_scheme_files();
    let expected_files = mdc1.files.clone();
    mdc1.download_config().await.expect("first download_config");

    let blobs_dir =
        std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/dynamo/mdc/blobs");
    let mtimes_before: Vec<_> = expected_files
        .iter()
        .map(|a| {
            let hex = a.checksum.strip_prefix("blake3:").unwrap();
            std::fs::metadata(blobs_dir.join(hex))
                .unwrap()
                .modified()
                .unwrap()
        })
        .collect();

    // Second registration with a fresh MDC (simulates a new worker
    // replica advertising the same blake3s, or a frontend restart
    // reading the same MDC again).
    let mut mdc2 = mdc_with_file_scheme_files();
    mdc2.download_config()
        .await
        .expect("second download_config");

    let mtimes_after: Vec<_> = expected_files
        .iter()
        .map(|a| {
            let hex = a.checksum.strip_prefix("blake3:").unwrap();
            std::fs::metadata(blobs_dir.join(hex))
                .unwrap()
                .modified()
                .unwrap()
        })
        .collect();

    assert_eq!(
        mtimes_before, mtimes_after,
        "second download must not re-write blobs (mtimes should be unchanged)"
    );
}

#[serial_test::serial]
#[tokio::test]
async fn test_download_files_rejects_blake3_mismatch() {
    let _home = isolated_home();
    let mut mdc = mdc_with_file_scheme_files();

    // Tamper with one artifact's expected checksum so the verifier
    // sees a mismatch.
    assert!(!mdc.files.is_empty());
    mdc.files[0].checksum =
        "blake3:0000000000000000000000000000000000000000000000000000000000000000".to_string();

    let err = mdc
        .download_config()
        .await
        .expect_err("checksum mismatch must error")
        .to_string();
    assert!(
        err.contains("checksum mismatch"),
        "expected checksum-mismatch error, got: {err}"
    );

    // The tampered artifact's blob (under the bad blake3) must NOT
    // exist — atomic write guarantees no partial file.
    let blobs_dir =
        std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/dynamo/mdc/blobs");
    let bad_blob =
        blobs_dir.join("0000000000000000000000000000000000000000000000000000000000000000");
    assert!(
        !bad_blob.exists(),
        "no blob should exist for the mismatched checksum"
    );
}

/// End-to-end of the unified download path against a real HuggingFace
/// repo. Skipped at runtime when `HF_TOKEN` is unset, matching the
/// pattern in `lib/llm/tests/preprocessor.rs`. Uses the same gated
/// model the preprocessor tests use so the HF cache is shared.
#[serial_test::serial]
#[tokio::test]
async fn test_download_files_resolves_hf_scheme_with_token() {
    if std::env::var("HF_TOKEN")
        .ok()
        .filter(|t| !t.trim().is_empty())
        .is_none()
    {
        println!("HF_TOKEN is not set, skipping test");
        return;
    }

    let _home = isolated_home();

    // Construct an MDC carrying hf:// URIs for the metadata files.
    // We fabricate a minimal MDC with just the `files` vector
    // populated; the test exercises `download_files` end-to-end via
    // `download_config`.
    let repo = "meta-llama/Llama-3.1-70B-Instruct";

    // First, prime the HF cache by calling hub::from_hf — this is
    // the same path the resolver uses internally, but we need it
    // here too so we can compute the expected blake3 of each file.
    let snapshot = dynamo_llm::hub::from_hf(repo, /* ignore_weights = */ true)
        .await
        .expect("priming HF cache");

    // Build the files vector from the snapshot's metadata files.
    // URIs use the same shape `move_to_url` produces today
    // (`hf://repo/filename`, no `@revision` segment).
    let candidates = [
        ("config.json", ArtifactRole::Config),
        ("tokenizer.json", ArtifactRole::Tokenizer),
        ("tokenizer_config.json", ArtifactRole::TokenizerConfig),
    ];
    let mut files = Vec::new();
    for (filename, role) in candidates {
        let path = snapshot.join(filename);
        if !path.exists() {
            continue;
        }
        let bytes = std::fs::read(&path).unwrap();
        let checksum = format!("blake3:{}", blake3::hash(&bytes));
        files.push(ArtifactRef {
            filename: filename.to_string(),
            uri: format!("hf://{repo}/{filename}"),
            checksum,
            size: bytes.len() as u64,
            role,
        });
    }
    assert!(
        !files.is_empty(),
        "expected to find at least config.json + tokenizer.json in the HF snapshot"
    );

    // Manufacture an MDC with just `display_name` + `files` set.
    // No typed-enum CheckedFiles, so `has_local_files()` returns
    // true via `unwrap_or(true)` on each None — meaning
    // download_config short-circuits without entering download_files.
    // Call download_files directly via download_config by giving
    // it at least one URL-backed typed enum so has_local_files()
    // returns false. Construct one minimal placeholder via
    // `move_to_url` of a freshly loaded TinyLlama MDC, then swap in
    // our HF-sourced files vector.
    let mut mdc = ModelDeploymentCard::load_from_disk(HF_PATH, None).unwrap();
    mdc.move_to_url("hf://test/").unwrap();
    mdc.files = files.clone();

    mdc.download_config()
        .await
        .expect("download_config (hf://)");

    // Assert each artifact landed in the content-addressed cache
    // and the bytes blake3-match.
    let blobs_dir =
        std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/dynamo/mdc/blobs");
    for artifact in &files {
        let hex = artifact.checksum.strip_prefix("blake3:").unwrap();
        let blob = blobs_dir.join(hex);
        assert!(
            blob.exists(),
            "expected blob for {} at {}",
            artifact.filename,
            blob.display()
        );

        let actual = format!("blake3:{}", blake3::hash(&std::fs::read(&blob).unwrap()));
        assert_eq!(actual, artifact.checksum);
    }
}
