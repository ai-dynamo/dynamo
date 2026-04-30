// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use dynamo_llm::model_card::{ModelDeploymentCard, PromptFormatterArtifact, TokenizerKind};
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

/// RAII guard: overrides `HOME` to a fresh tempdir for the test, restores the
/// previous value (or unsets) on drop. The tempdir is cleaned up with the
/// guard, so the MDC cache leaves no residue for the next test.
struct IsolatedHome {
    _tempdir: tempfile::TempDir,
    prev_home: Option<String>,
}

impl IsolatedHome {
    // Only called from the integration_tests module, gated behind the
    // `integration` feature flag.
    #[allow(dead_code)]
    fn path(&self) -> &std::path::Path {
        self._tempdir.path()
    }
}

impl Drop for IsolatedHome {
    fn drop(&mut self) {
        // SAFETY: tests using this guard are serialized via serial_test.
        unsafe {
            match self.prev_home.take() {
                Some(prev) => std::env::set_var("HOME", prev),
                None => std::env::remove_var("HOME"),
            }
        }
    }
}

fn isolated_home() -> IsolatedHome {
    let tempdir = tempdir().expect("tempdir for isolated $HOME");
    let prev_home = std::env::var("HOME").ok();
    // SAFETY: tests using this guard are serialized via serial_test.
    unsafe { std::env::set_var("HOME", tempdir.path()) };
    IsolatedHome {
        _tempdir: tempdir,
        prev_home,
    }
}

/// RAII guard for a single env var. Restores the previous value (or unsets)
/// on drop, so a panicking test doesn't leak state into the next one.
#[allow(dead_code)]
struct EnvVarGuard {
    key: &'static str,
    prev: Option<String>,
}

#[allow(dead_code)]
impl EnvVarGuard {
    fn set(key: &'static str, value: &str) -> Self {
        let prev = std::env::var(key).ok();
        // SAFETY: tests using this guard are serialized via serial_test.
        unsafe { std::env::set_var(key, value) };
        Self { key, prev }
    }

    fn unset(key: &'static str) -> Self {
        let prev = std::env::var(key).ok();
        // SAFETY: tests using this guard are serialized via serial_test.
        unsafe { std::env::remove_var(key) };
        Self { key, prev }
    }
}

impl Drop for EnvVarGuard {
    fn drop(&mut self) {
        // SAFETY: tests using this guard are serialized via serial_test.
        unsafe {
            match self.prev.take() {
                Some(v) => std::env::set_var(self.key, v),
                None => std::env::remove_var(self.key),
            }
        }
    }
}

/// Fresh MDC loaded from TinyLlama, with each typed-enum CheckedFile
/// rewritten from its local path to a `file://<absolute>` URL that
/// points back at the same on-disk bytes. After this transformation:
///   - `has_local_files()` returns false (URLs aren't "local"), so
///     `download_config` dispatches into `resolve_metadata_files`.
///   - Each `CheckedFile.size` stays populated from `from_disk`, so
///     `is_new_format()` returns true.
fn mdc_with_url_backed_metadata() -> ModelDeploymentCard {
    let mut mdc = ModelDeploymentCard::load_from_disk(HF_PATH, None).unwrap();
    for cf in mdc.iter_metadata_files_mut() {
        let path = cf
            .path()
            .expect("from_repo_checkout should produce local paths")
            .to_path_buf();
        let absolute = std::fs::canonicalize(&path).expect("canonicalize");
        let url = url::Url::from_file_path(&absolute).expect("url from path");
        cf.move_to_url(url);
    }
    mdc
}

#[tokio::test]
async fn test_metadata_iter_populated_from_local_repo() {
    let mdc = ModelDeploymentCard::load_from_disk(HF_PATH, None).unwrap();

    let mut count = 0usize;
    for cf in mdc.iter_metadata_files() {
        assert!(
            cf.size().is_some(),
            "from_disk must populate size on every CheckedFile"
        );
        assert!(cf.size().unwrap() > 0, "non-zero size expected");
        assert!(
            cf.path().is_some() && cf.path().unwrap().exists(),
            "expected local file at {:?}",
            cf.path()
        );
        count += 1;
    }
    // TinyLlama_v1.1 ships config + tokenizer + tokenizer_config +
    // generation_config (no separate chat_template). At least 4 slots.
    assert!(
        count >= 4,
        "expected at least 4 metadata files for TinyLlama, got {count}"
    );

    // Every populated slot carries size → MDC is new-format.
    assert!(mdc.is_new_format());
}

/// Snapshot of (filename, expected blake3 hex) pairs from an MDC's
/// typed-enum metadata slots. Used by tests to verify the
/// content-addressed cache landed with the right blobs and per-slug
/// symlinks.
fn metadata_fingerprint(mdc: &ModelDeploymentCard) -> Vec<(String, String)> {
    mdc.iter_metadata_files()
        .map(|cf| {
            let filename = cf
                .path()
                .and_then(|p| p.file_name())
                .and_then(|f| f.to_str())
                .map(String::from)
                .or_else(|| {
                    cf.url().and_then(|u| {
                        u.path()
                            .rsplit('/')
                            .find(|s| !s.is_empty())
                            .map(String::from)
                    })
                })
                .expect("derivable filename");
            let hex = cf
                .checksum()
                .to_string()
                .strip_prefix("blake3:")
                .expect("blake3:<hex> on CheckedFile")
                .to_string();
            (filename, hex)
        })
        .collect()
}

#[serial_test::serial]
#[tokio::test]
async fn test_download_files_resolves_local_file_scheme() {
    let _home = isolated_home();
    let mut mdc = mdc_with_url_backed_metadata();
    let expected = metadata_fingerprint(&mdc);
    let slug = mdc.slug().to_string();
    let mdcsum = mdc.mdcsum().to_string();

    mdc.download_config().await.expect("download_config");

    let blobs_dir =
        std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/dynamo/mdc/blobs");
    let slug_dir = std::path::PathBuf::from(std::env::var("HOME").unwrap())
        .join(".cache/dynamo/mdc/by-slug")
        .join(&slug)
        .join(&mdcsum);
    for (filename, hex) in &expected {
        let blob = blobs_dir.join(hex);
        assert!(blob.exists(), "expected blob at {}", blob.display());

        let actual = format!("blake3:{}", blake3::hash(&std::fs::read(&blob).unwrap()));
        assert_eq!(actual, format!("blake3:{hex}"), "blob bytes blake3-match");

        let link = slug_dir.join(filename);
        assert!(link.exists(), "expected by-slug link at {}", link.display());
    }

    // The tokenizer should now load via the symlinked path — proves
    // update_dir routed downstream consumers to the cached files.
    let _tokenizer = mdc.tokenizer().expect("tokenizer should load");
}

/// Single-process simulation of multiple replicas registering the same
/// model concurrently. Each replica's MDC carries identical CheckedFile
/// checksums, so they all converge on the same content-addressed blobs.
/// Without per-task tmp suffixes, two concurrent fetches of the same
/// blake3 would race on `<blob>.tmp` and produce spurious blake3-mismatch
/// errors. This test exercises that exact path: N concurrent
/// `download_config` calls against the same MDC must all succeed.
#[serial_test::serial]
#[tokio::test]
async fn test_concurrent_registrations_do_not_race() {
    let _home = isolated_home();

    // Fan out 8 concurrent registrations of fresh-but-identical MDCs.
    // 8 is enough to provoke real interleaving on multi-core hosts;
    // the test still completes in well under a second.
    let mut handles = Vec::new();
    for _ in 0..8 {
        handles.push(tokio::spawn(async move {
            let mut mdc = mdc_with_url_backed_metadata();
            mdc.download_config().await
        }));
    }
    for handle in handles {
        handle
            .await
            .expect("task panicked")
            .expect("download_config must succeed under concurrent fetches of same blake3");
    }

    // After all 8 finished, the cache is fully populated and consistent.
    let blobs_dir =
        std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/dynamo/mdc/blobs");
    let blobs: Vec<_> = std::fs::read_dir(&blobs_dir)
        .unwrap()
        .filter_map(|e| {
            let entry = e.ok()?;
            let name = entry.file_name().into_string().ok()?;
            // Filter out leftover `tmp.*` files — they're tolerated but
            // shouldn't be the only thing left.
            if name.contains(".tmp.") {
                None
            } else {
                Some(entry)
            }
        })
        .collect();
    assert!(
        !blobs.is_empty(),
        "at least one blob should land in the cache"
    );

    // No leftover tmp files — successful rename cleans up.
    let leftover_tmps: Vec<_> = std::fs::read_dir(&blobs_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.file_name().to_string_lossy().contains(".tmp."))
        .collect();
    assert!(
        leftover_tmps.is_empty(),
        "no `*.tmp.*` files should be left after successful concurrent registrations, got {} leftovers",
        leftover_tmps.len(),
    );
}

#[serial_test::serial]
#[tokio::test]
async fn test_download_files_dedupes_by_blake3() {
    let _home = isolated_home();

    // First registration: populate the cache.
    let mut mdc1 = mdc_with_url_backed_metadata();
    let expected = metadata_fingerprint(&mdc1);
    mdc1.download_config().await.expect("first download_config");

    let blobs_dir =
        std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/dynamo/mdc/blobs");
    let mtimes_before: Vec<_> = expected
        .iter()
        .map(|(_, hex)| {
            std::fs::metadata(blobs_dir.join(hex))
                .unwrap()
                .modified()
                .unwrap()
        })
        .collect();

    // Second registration with a fresh MDC (simulates a new worker
    // replica advertising the same blake3s, or a frontend restart
    // reading the same MDC again).
    let mut mdc2 = mdc_with_url_backed_metadata();
    mdc2.download_config()
        .await
        .expect("second download_config");

    let mtimes_after: Vec<_> = expected
        .iter()
        .map(|(_, hex)| {
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
    let mut mdc = mdc_with_url_backed_metadata();

    // Tamper with one slot's URL to point at a different file on disk.
    // The CheckedFile's checksum still references the original bytes,
    // so resolve_uri's content fetched-vs-claimed check will reject.
    let other = std::fs::canonicalize(std::path::Path::new(HF_PATH).join("generation_config.json"))
        .unwrap();
    let other_url = url::Url::from_file_path(&other).unwrap();
    if let Some(t) = mdc.tokenizer.as_mut() {
        match t {
            TokenizerKind::HfTokenizerJson(cf) | TokenizerKind::TikTokenModel(cf) => {
                cf.move_to_url(other_url);
            }
        }
    }

    let err = mdc
        .download_config()
        .await
        .expect_err("checksum mismatch must error")
        .to_string();
    assert!(
        err.contains("checksum mismatch"),
        "expected checksum-mismatch error, got: {err}"
    );
}

/// End-to-end of the unified download path against a real HuggingFace
/// repo. Skipped at runtime when `HF_TOKEN` is unset.
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
    let repo = "meta-llama/Llama-3.1-70B-Instruct";

    // Prime the HF cache so we can construct CheckedFiles with the
    // right blake3 + size, then point the typed-enum slots at
    // hf://repo/filename URIs that resolve_uri will dispatch.
    let snapshot = dynamo_llm::hub::from_hf(repo, /* ignore_weights = */ true)
        .await
        .expect("priming HF cache");

    // Build a minimal MDC carrying just config + tokenizer slots,
    // both backed by HF snapshot files (so blake3 + size reflect the
    // HF bytes). Other slots stay None so they don't enter the
    // download path with stale TinyLlama checksums.
    let hf_config = snapshot.join("config.json");
    let hf_tokenizer = snapshot.join("tokenizer.json");
    assert!(hf_config.exists(), "config.json missing in HF snapshot");
    assert!(
        hf_tokenizer.exists(),
        "tokenizer.json missing in HF snapshot"
    );

    let mut mdc = ModelDeploymentCard::load_from_disk(HF_PATH, None).unwrap();
    mdc.prompt_formatter = None;
    mdc.chat_template_file = None;
    mdc.gen_config = None;
    mdc.model_info = Some(dynamo_llm::model_card::ModelInfoType::HfConfigJson(
        dynamo_llm::common::checked_file::CheckedFile::from_disk(&hf_config)
            .expect("from_disk(config.json)"),
    ));
    mdc.tokenizer = Some(TokenizerKind::HfTokenizerJson(
        dynamo_llm::common::checked_file::CheckedFile::from_disk(&hf_tokenizer)
            .expect("from_disk(tokenizer.json)"),
    ));

    // Switch every populated CheckedFile to hf://<repo>/<filename>.
    for cf in mdc.iter_metadata_files_mut() {
        let filename = cf
            .path()
            .and_then(|p| p.file_name())
            .and_then(|f| f.to_str())
            .map(String::from)
            .expect("derivable filename");
        let url = url::Url::parse(&format!("hf://{repo}/{filename}")).expect("hf url");
        cf.move_to_url(url);
    }

    let expected = metadata_fingerprint(&mdc);
    assert_eq!(
        expected.len(),
        2,
        "expected exactly config + tokenizer slots"
    );
    mdc.download_config()
        .await
        .expect("download_config (hf://)");

    let blobs_dir =
        std::path::PathBuf::from(std::env::var("HOME").unwrap()).join(".cache/dynamo/mdc/blobs");
    for (_, hex) in &expected {
        let blob = blobs_dir.join(hex);
        assert!(blob.exists(), "expected blob at {}", blob.display());
        let actual = format!("blake3:{}", blake3::hash(&std::fs::read(&blob).unwrap()));
        assert_eq!(actual, format!("blake3:{hex}"));
    }
}

// =============================================================
// DRT-level integration test (gh-8749)
//
// Mirrors lib/llm/tests/http_metrics.rs `integration_tests`. Builds
// a real DistributedRuntime (requires etcd and a process-spawned
// system_status_server). Single-process worker + frontend: the
// worker's `LocalModel::attach` publishes the MDC into discovery,
// the frontend's `ModelWatcher` observes it and runs
// `download_config`, which fetches via `http://` from the worker's
// own `/v1/metadata/...` route.
//
// Gated by the `integration` feature and `#[ignore]`d by default
// because it requires etcd. Run with:
//   cargo test -p dynamo-llm --test model_card --features integration -- --ignored
// =============================================================

#[cfg(feature = "integration")]
mod integration_tests {
    use super::*;
    use dynamo_llm::discovery::{ModelManager, ModelWatcher};
    use dynamo_llm::http::service::metrics::Metrics;
    use dynamo_llm::local_model::LocalModelBuilder;
    use dynamo_llm::namespace::NamespaceFilter;
    use dynamo_runtime::DistributedRuntime;
    use dynamo_runtime::discovery::DiscoveryQuery;
    use std::sync::Arc;
    use std::time::Duration;

    /// Serialized because `HOME` and `DYN_SYSTEM_PORT` are
    /// process-global. Across separate `cargo test` processes the
    /// test is parallelizable: each gets its own tempdir-rooted
    /// HOME, its own random port, and a unique namespace so etcd
    /// keys don't collide.
    #[serial_test::serial]
    #[tokio::test]
    #[ignore = "Requires etcd"]
    async fn worker_self_host_round_trip_via_drt() {
        // SAFETY: serialized within-binary; tempdir HOME + random
        // port + unique namespace handle cross-binary parallelism.
        // RAII guards so a panic mid-test doesn't leak state.
        let _system_port = EnvVarGuard::set("DYN_SYSTEM_PORT", "0");
        let _self_host = EnvVarGuard::unset("DYN_SELF_HOST_METADATA");
        let _home = isolated_home();
        let test_namespace = format!(
            "self-host-it-{}-{}",
            std::process::id(),
            uuid::Uuid::new_v4().simple()
        );

        let runtime = dynamo_runtime::Runtime::from_settings().expect("runtime");
        let drt = DistributedRuntime::from_settings(runtime.clone())
            .await
            .expect("DRT");
        assert!(
            drt.system_status_server_info().is_some(),
            "system_status_server must be running for self-host registration"
        );

        let abs_path = std::fs::canonicalize(HF_PATH).expect("canonicalize HF_PATH");
        let mut local_model = LocalModelBuilder::default()
            .model_path(abs_path)
            .self_host_metadata(true)
            .build()
            .await
            .expect("build LocalModel");

        let manager = Arc::new(ModelManager::new());
        let metrics = Arc::new(Metrics::new());
        let watcher = ModelWatcher::new(
            drt.clone(),
            manager.clone(),
            dynamo_llm::entrypoint::RouterConfig::default(),
            0,
            None,
            None,
            None,
            metrics,
        );
        let discovery_stream = drt
            .discovery()
            .list_and_watch(DiscoveryQuery::AllModels, Some(drt.primary_token()))
            .await
            .expect("list_and_watch");
        let watcher_filter = NamespaceFilter::Exact(test_namespace.clone());
        let _watcher_task = tokio::spawn(async move {
            Arc::new(watcher)
                .watch(discovery_stream, watcher_filter)
                .await;
        });

        let namespace = drt.namespace(&test_namespace).unwrap();
        let component = namespace.component("worker").unwrap();
        let endpoint = component.endpoint("generate");
        local_model
            .attach(
                &endpoint,
                dynamo_llm::model_type::ModelType::Chat,
                dynamo_llm::model_type::ModelInput::Text,
                None,
            )
            .await
            .expect("attach");

        let want_slug = local_model.card().slug().to_string();
        let card = poll_for_card(&manager, &want_slug, 30).await;
        let want_mdcsum = card.mdcsum().to_string();

        // Cache must be populated under the isolated HOME — the
        // only writer is `download_files`, so its presence proves
        // the http path ran end-to-end. Count by-slug links rather
        // than blobs: blobs are content-addressed and could collapse
        // for byte-identical artifacts.
        let blobs_dir = _home.path().join(".cache/dynamo/mdc/blobs");
        let slug_dir = _home
            .path()
            .join(".cache/dynamo/mdc/by-slug")
            .join(&want_slug)
            .join(&want_mdcsum);
        assert!(
            blobs_dir.exists(),
            "expected blobs dir at {}",
            blobs_dir.display()
        );
        let blobs: Vec<_> = std::fs::read_dir(&blobs_dir).unwrap().collect();
        let links: Vec<_> = std::fs::read_dir(&slug_dir).unwrap().collect();
        assert!(!blobs.is_empty(), "expected at least one blob");
        let metadata_count = card.iter_metadata_files().count();
        assert_eq!(
            metadata_count,
            links.len(),
            "by-slug entries must match the count of populated metadata files"
        );

        let _tokenizer = card.tokenizer().expect("tokenizer should load");
    }

    async fn poll_for_card(
        manager: &ModelManager,
        slug: &str,
        timeout_secs: u64,
    ) -> ModelDeploymentCard {
        let deadline = std::time::Instant::now() + Duration::from_secs(timeout_secs);
        loop {
            for card in manager.get_model_cards() {
                if card.slug().as_ref() == slug {
                    return card;
                }
            }
            if std::time::Instant::now() >= deadline {
                panic!("watcher never discovered card for slug {slug}");
            }
            tokio::time::sleep(Duration::from_millis(200)).await;
        }
    }
}
