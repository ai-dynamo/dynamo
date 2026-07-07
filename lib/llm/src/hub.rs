// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::path::{Path, PathBuf};

use anyhow::Context;
use hf_hub::{Cache, Repo, RepoType, api::tokio::ApiBuilder};
use modelexpress_client::{
    Client as MxClient, ClientConfig as MxClientConfig, ModelProvider as MxModelProvider,
};
use modelexpress_common::download as mx;

use dynamo_runtime::config::environment_names::model as env_model;

mod huggingface;

pub(crate) use huggingface::{
    HfRepoSpec, cached_hf_snapshot, download_hf_snapshot, finalize_hf_snapshot, huggingface_cache,
};

/// Check if a model is already cached in the HuggingFace hub cache directory.
/// Returns the path to the cached model directory if found, None otherwise.
///
/// Uses hf-hub's Cache API to check for cached files. For tokenizer-only downloads
/// (ignore_weights=true), we check for config.json and tokenizer files.
/// For full downloads, we also require weight files to be present.
fn get_cached_model_path(model_name: &str, ignore_weights: bool) -> Option<PathBuf> {
    get_cached_model_path_in(model_name, ignore_weights, get_model_express_cache_dir())
}

fn get_cached_model_path_in(
    model_name: &str,
    ignore_weights: bool,
    cache_dir: PathBuf,
) -> Option<PathBuf> {
    let cache = Cache::new(cache_dir);
    let repo = cache.model(model_name.to_string());

    // Check for required config file
    let config_path = repo.get("config.json")?;

    // Check for tokenizer files (at least one must exist). Only count
    // artifacts that ``ModelDeploymentCard::TokenizerKind::from_disk`` can
    // actually load -- ``tokenizer_config.json`` is metadata describing the
    // tokenizer and cannot be used on its own, so a snapshot with only
    // ``config.json`` + ``tokenizer_config.json`` would fall through to a
    // download even though the cache appears "populated".
    let has_tokenizer = repo.get("tokenizer.json").is_some()
        || repo.get("tiktoken.model").is_some()
        || has_tiktoken_file(config_path.parent()?);

    if !has_tokenizer {
        return None;
    }

    // For full downloads, check for weight files. When an index file is present,
    // verify the shard files it references are also cached — an index without its
    // shards is an incomplete cache that should fall through to download.
    if !ignore_weights {
        let has_weights = repo.get("model.safetensors").is_some()
            || repo.get("pytorch_model.bin").is_some()
            || repo
                .get("model.safetensors.index.json")
                .is_some_and(|p| shard_files_present(&p))
            || repo
                .get("pytorch_model.bin.index.json")
                .is_some_and(|p| shard_files_present(&p));

        if !has_weights {
            return None;
        }
    }

    // Return the parent directory (snapshot dir) containing the model files
    let snapshot_path = config_path.parent()?.to_path_buf();
    tracing::info!("Found cached model '{model_name}' at {snapshot_path:?}, skipping download");
    Some(snapshot_path)
}

/// Returns the snapshot path if that exact revision is already on disk.
fn get_cached_model_path_at_revision(
    model_name: &str,
    revision: &str,
    ignore_weights: bool,
    required_files: Option<&[String]>,
    cache_dir: PathBuf,
) -> Option<PathBuf> {
    // HF cache layout: models--{org}--{model}/snapshots/{sha}/
    let model_key = model_name.replace('/', "--");
    let snapshot_dir = cache_dir
        .join(format!("models--{model_key}"))
        .join("snapshots")
        .join(revision);

    if !snapshot_dir.exists() {
        return None;
    }

    if let Some(files) = required_files {
        // Caller knows the exact MDC file set (frontend resolving an
        // already-published card) — require every one of them, so a
        // sparse/interrupted snapshot is correctly treated as incomplete
        // instead of passing a config.json+tokenizer-only heuristic.
        if !files.iter().all(|f| snapshot_dir.join(f).exists()) {
            return None;
        }
    } else {
        // No MDC yet (worker discovering its own file set for the first
        // time) — fall back to the config.json + tokenizer heuristic.
        if !snapshot_dir.join("config.json").exists() {
            return None;
        }

        let has_tokenizer = snapshot_dir.join("tokenizer.json").exists()
            || snapshot_dir.join("tiktoken.model").exists()
            || has_tiktoken_file(&snapshot_dir);

        if !has_tokenizer {
            return None;
        }
    }

    if !ignore_weights {
        let index = snapshot_dir.join("model.safetensors.index.json");
        let pt_index = snapshot_dir.join("pytorch_model.bin.index.json");
        let has_weights = snapshot_dir.join("model.safetensors").exists()
            || snapshot_dir.join("pytorch_model.bin").exists()
            || (index.exists() && shard_files_present(&index))
            || (pt_index.exists() && shard_files_present(&pt_index));
        if !has_weights {
            return None;
        }
    }

    tracing::info!(
        "Found cached model '{model_name}' at revision {revision}({snapshot_dir:?}), skipping download"
    );
    Some(snapshot_dir)
}

/// If `path` sits inside an HF-cache-style snapshot dir
/// (".../models--{org}--{name}/snapshots/{sha}"), return the repo id "org/name".
/// Returns `None` for any path that isn't shaped like an HF cache snapshot
/// (e.g. a plain local checkpoint directory).
pub(crate) fn hf_repo_from_snapshot_path(path: &Path) -> Option<String> {
    let snapshots_dir = path.parent()?;
    if snapshots_dir.file_name()?.to_str()? != "snapshots" {
        return None;
    }
    let models_dir = snapshots_dir.parent()?;
    let repo_key = models_dir.file_name()?.to_str()?.strip_prefix("models--")?;
    match repo_key.split_once("--") {
        Some((org, name)) => Some(format!("{org}/{name}")),
        None => Some(repo_key.to_string()),
    }
}

/// Check if the snapshot directory contains any `*.tiktoken` file (e.g. `qwen.tiktoken`).
fn has_tiktoken_file(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .into_iter()
        .flatten()
        .flatten()
        .any(|e| e.path().extension().is_some_and(|ext| ext == "tiktoken"))
}

/// For a sharded-weights index file (e.g. `model.safetensors.index.json`), verify
/// that every shard file it references is present in the same snapshot directory.
/// Returns false on parse error, missing weight_map, empty weight_map, or any
/// missing shard file.
fn shard_files_present(index_path: &Path) -> bool {
    let Some(snapshot_dir) = index_path.parent() else {
        return false;
    };
    let Ok(contents) = std::fs::read_to_string(index_path) else {
        return false;
    };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&contents) else {
        return false;
    };
    let Some(weight_map) = value.get("weight_map").and_then(|v| v.as_object()) else {
        return false;
    };
    let shards: std::collections::HashSet<&str> =
        weight_map.values().filter_map(|v| v.as_str()).collect();
    if shards.is_empty() {
        return false;
    }
    shards.iter().all(|s| snapshot_dir.join(s).exists())
}

/// Check if offline mode is enabled via HF_HUB_OFFLINE environment variable.
fn is_offline_mode() -> bool {
    dynamo_runtime::config::env_is_truthy(env_model::huggingface::HF_HUB_OFFLINE)
}

/// Check if shared-storage mode is disabled via MODEL_EXPRESS_NO_SHARED_STORAGE.
/// When true, the Model Express client streams files from the server over gRPC
/// instead of relying on a shared filesystem path. This is required when the
/// server and worker pods do not share a filesystem (e.g. RWO PVCs, cross-namespace
/// deployments).
fn is_no_shared_storage() -> bool {
    dynamo_runtime::config::env_is_truthy(env_model::model_express::MODEL_EXPRESS_NO_SHARED_STORAGE)
}

/// Download a model using ModelExpress client. The client first requests for the model
/// from the server and fallbacks to direct download in case of server failure.
/// If ignore_weights is true, model weight files will be skipped
/// Returns the path to the model files
///
/// If the model is already cached locally with the required files, returns the cached
/// path without making any API calls to HuggingFace, regardless of HF_HUB_OFFLINE.
pub async fn from_hf(name: impl AsRef<Path>, ignore_weights: bool) -> anyhow::Result<PathBuf> {
    let name = name.as_ref();
    let model_name = name.display().to_string();

    // Cache-first in all modes: if the snapshot is already on disk with the files we
    // need, return it without touching the network.
    if let Some(cached_path) = get_cached_model_path(&model_name, ignore_weights) {
        return Ok(cached_path);
    }

    if is_offline_mode() {
        tracing::warn!(
            "Offline mode enabled but model '{model_name}' not found in cache, attempting download anyway"
        );
    }

    let mut config: MxClientConfig = MxClientConfig::default();
    if let Ok(endpoint) = env::var(env_model::model_express::MODEL_EXPRESS_URL) {
        config = config.with_endpoint(endpoint);
    }
    if is_no_shared_storage() {
        config.cache.shared_storage = false;
    }

    let result = match MxClient::new(config).await {
        Ok(mut client) => {
            tracing::info!("Successfully connected to ModelExpress server");
            match client
                .request_model_with_provider_and_fallback(
                    &model_name,
                    MxModelProvider::HuggingFace,
                    ignore_weights,
                )
                .await
            {
                Ok(()) => {
                    tracing::info!("Server download succeeded for model: {model_name}");
                    match client
                        .get_model_path(&model_name, MxModelProvider::HuggingFace)
                        .await
                    {
                        Ok(path) => Ok(path),
                        Err(e) => {
                            tracing::warn!(
                                "Failed to resolve local model path after server download for '{model_name}': {e}. \
                                Falling back to direct download."
                            );
                            mx_download_direct(&model_name, ignore_weights).await
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Server download failed for model '{model_name}': {e}. Falling back to direct download."
                    );
                    mx_download_direct(&model_name, ignore_weights).await
                }
            }
        }
        Err(e) => {
            tracing::warn!("Cannot connect to ModelExpress server: {e}. Using direct download.");
            mx_download_direct(&model_name, ignore_weights).await
        }
    };

    match result {
        Ok(path) => {
            tracing::info!("ModelExpress download completed successfully for model: {model_name}");
            Ok(path)
        }
        Err(e) => {
            tracing::warn!("ModelExpress download failed for model '{model_name}': {e}");
            Err(e)
        }
    }
}

/// Like `from_hf`, but resolves a specific commit SHA instead of latest.
/// If the snapshot is already on disk at that revision, returns it immediately.
/// Otherwise downloads just that revision directly from HuggingFace (bypassing
/// ModelExpress, which has no concept of pinned revisions) into the shared
/// cache directory.
pub async fn from_hf_at_revision(
    name: impl AsRef<Path>,
    revision: &str,
    required_files: Option<&[String]>,
    ignore_weights: bool,
) -> anyhow::Result<PathBuf> {
    let name = name.as_ref();
    let model_name = name.display().to_string();

    if let Some(cached) = get_cached_model_path_at_revision(
        &model_name,
        revision,
        ignore_weights,
        required_files,
        get_model_express_cache_dir(),
    ) {
        return Ok(cached);
    }

    if !ignore_weights {
        anyhow::bail!(
            "from_hf_at_revision does not support downloading weights for a pinned \
            revision yet (ignore_weights=false); only tokenizer/config metadata \
            download is implemented."
        );
    }

    let token = env::var(env_model::huggingface::HF_TOKEN).ok();
    let api = ApiBuilder::from_env()
        .with_cache_dir(get_model_express_cache_dir())
        .with_token(token)
        .build()
        .context("building HuggingFace API client for pinned-revision download")?;

    let repo = api.repo(Repo::with_revision(
        model_name.clone(),
        RepoType::Model,
        revision.to_string(),
    ));

    let config_path = if let Some(files) = required_files {
        // Caller knows exactly which files this MDC needs - fetch precisely
        // those instead of guessing from a hardcoded list, so arbitrary
        // worker-harvested siblings (added_tokens.json, merges.txt, ...)
        // are actually downloaded instead of silently missing later.
        let mut config_path = None;
        for f in files {
            let path = repo
                .get(f)
                .await
                .with_context(|| format!("downloading {f} for {model_name}@{revision}"))?;
            if f == "config.json" {
                config_path = Some(path);
            }
        }
        config_path.context("MDC required_files did not include config.json")?
    } else {
        let config_path = repo
            .get("config.json")
            .await
            .with_context(|| format!("downloading config.json for {model_name}@{revision}"))?;

        if repo.get("tokenizer.json").await.is_err() && repo.get("tiktoken.model").await.is_err() {
            let info = repo
                .info()
                .await
                .with_context(|| format!("listing files for {model_name}@{revision}"))?;
            let tiktoken_file = info
                .siblings
                .iter()
                .map(|s| s.rfilename.as_str())
                .find(|f| f.ends_with(".tiktoken"));
            match tiktoken_file {
                Some(f) => {
                    repo.get(f)
                        .await
                        .with_context(|| format!("downloading {f}"))?;
                }
                None => anyhow::bail!("no tokenizer file found for {model_name}@{revision}"),
            }
        }

        for extra in [
            "tokenizer_config.json",
            "generation_config.json",
            "chat_template.jinja",
            "chat_template.json",
            "preprocessor_config.json",
            "special_tokens_map.json",
        ] {
            if let Err(e) = repo.get(extra).await {
                tracing::debug!(
                    "optional file {extra} not available for {model_name}@{revision}: {e}"
                );
            }
        }
        config_path
    };

    Ok(config_path
        .parent()
        .context("config.json path has no parent directory")?
        .to_path_buf())
}

// Direct download using the ModelExpress client.
async fn mx_download_direct(model_name: &str, ignore_weights: bool) -> anyhow::Result<PathBuf> {
    let cache_dir = get_model_express_cache_dir();
    mx::download_model(
        model_name,
        MxModelProvider::HuggingFace,
        Some(cache_dir),
        ignore_weights,
    )
    .await
}

// TODO: remove in the future. This is a temporary workaround to find common
// cache directory between client and server.
fn get_model_express_cache_dir() -> PathBuf {
    cache_dir_from_values(
        env::var(env_model::huggingface::HF_HUB_CACHE).ok(),
        env::var(env_model::huggingface::HF_HOME).ok(),
        env::var(env_model::model_express::MODEL_EXPRESS_CACHE_PATH).ok(),
        env::var("HOME").ok(),
        env::var("USERPROFILE").ok(),
    )
}

fn cache_dir_from_values(
    hf_hub_cache: Option<String>,
    hf_home: Option<String>,
    model_express_cache: Option<String>,
    home: Option<String>,
    userprofile: Option<String>,
) -> PathBuf {
    if let Some(cache_path) = hf_hub_cache {
        return PathBuf::from(cache_path);
    }
    if let Some(hf_home) = hf_home {
        return PathBuf::from(hf_home).join("hub");
    }
    if let Some(cache_path) = model_express_cache {
        return PathBuf::from(cache_path);
    }

    PathBuf::from(home.or(userprofile).unwrap_or_else(|| ".".to_string()))
        .join(".cache/huggingface/hub")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use tempfile::TempDir;

    #[serial_test::serial]
    #[test]
    fn hf_offline_mode_accepts_huggingface_truthy_values() {
        for value in ["1", "true", "TRUE", "on", "ON", "yes", "YES"] {
            temp_env::with_var(env_model::huggingface::HF_HUB_OFFLINE, Some(value), || {
                assert!(is_offline_mode(), "rejected {value}")
            });
        }
    }

    #[test]
    fn cache_dir_precedence_and_fallback() {
        assert_eq!(
            cache_dir_from_values(
                Some("/hub-cache".to_string()),
                Some("/hf-home".to_string()),
                Some("/model-express".to_string()),
                Some("/home".to_string()),
                None,
            ),
            PathBuf::from("/hub-cache")
        );
        assert_eq!(
            cache_dir_from_values(
                None,
                Some("/hf-home".to_string()),
                Some("/model-express".to_string()),
                Some("/home".to_string()),
                None,
            ),
            PathBuf::from("/hf-home/hub")
        );
        assert_eq!(
            cache_dir_from_values(
                None,
                None,
                Some("/model-express".to_string()),
                Some("/home".to_string()),
                None,
            ),
            PathBuf::from("/model-express")
        );
        assert_eq!(
            cache_dir_from_values(None, None, None, None, Some("/profile".to_string())),
            PathBuf::from("/profile/.cache/huggingface/hub")
        );
    }

    /// Build an hf-hub-format cache layout for `model_name` in `cache_root`,
    /// populated with the given filenames at a fake snapshot revision. Returns
    /// the snapshot directory path that `Cache::model().get()` should resolve to.
    fn build_hf_cache(cache_root: &Path, model_name: &str, files: &[&str]) -> PathBuf {
        let repo_dir = cache_root.join(format!("models--{}", model_name.replace('/', "--")));
        let snapshot_hash = "0000000000000000000000000000000000000000";
        let snapshot_dir = repo_dir.join("snapshots").join(snapshot_hash);
        let refs_dir = repo_dir.join("refs");
        fs::create_dir_all(&snapshot_dir).unwrap();
        fs::create_dir_all(&refs_dir).unwrap();
        fs::write(refs_dir.join("main"), snapshot_hash).unwrap();
        for f in files {
            fs::write(snapshot_dir.join(f), "{}").unwrap();
        }
        snapshot_dir
    }

    #[test]
    fn test_cached_path_metadata_only_satisfies_ignore_weights_true() {
        // A cache with only metadata files should satisfy ignore_weights=true
        // but NOT ignore_weights=false (no weight files present).
        let temp = TempDir::new().unwrap();
        let model = "test-org/metadata-only";
        let snapshot = build_hf_cache(temp.path(), model, &["config.json", "tokenizer.json"]);

        let with_weights = get_cached_model_path_in(model, false, temp.path().to_path_buf());
        let no_weights = get_cached_model_path_in(model, true, temp.path().to_path_buf());

        assert!(
            with_weights.is_none(),
            "metadata-only cache must NOT satisfy ignore_weights=false"
        );
        assert_eq!(
            no_weights.as_deref(),
            Some(snapshot.as_path()),
            "metadata-only cache must satisfy ignore_weights=true"
        );
    }

    #[test]
    fn test_cached_path_full_cache_satisfies_both_modes() {
        let temp = TempDir::new().unwrap();
        let model = "test-org/full-cache";
        let snapshot = build_hf_cache(
            temp.path(),
            model,
            &["config.json", "tokenizer.json", "model.safetensors"],
        );

        let with_weights = get_cached_model_path_in(model, false, temp.path().to_path_buf());
        let no_weights = get_cached_model_path_in(model, true, temp.path().to_path_buf());

        assert_eq!(with_weights.as_deref(), Some(snapshot.as_path()));
        assert_eq!(no_weights.as_deref(), Some(snapshot.as_path()));
    }

    #[test]
    fn test_cached_path_sharded_requires_all_shard_files() {
        // A cache containing only `model.safetensors.index.json` (without the
        // shard files it points to) is incomplete and must NOT satisfy
        // ignore_weights=false. Once all shards are written, it should.
        let temp = TempDir::new().unwrap();
        let model = "test-org/sharded";
        let snapshot = build_hf_cache(temp.path(), model, &["config.json", "tokenizer.json"]);
        fs::write(
            snapshot.join("model.safetensors.index.json"),
            r#"{"weight_map": {"a.weight": "model-00001-of-00002.safetensors", "b.weight": "model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();

        let incomplete = get_cached_model_path_in(model, false, temp.path().to_path_buf());
        assert!(
            incomplete.is_none(),
            "sharded cache without shard files must NOT satisfy ignore_weights=false"
        );

        fs::write(snapshot.join("model-00001-of-00002.safetensors"), "").unwrap();
        fs::write(snapshot.join("model-00002-of-00002.safetensors"), "").unwrap();
        let complete = get_cached_model_path_in(model, false, temp.path().to_path_buf());
        assert_eq!(complete.as_deref(), Some(snapshot.as_path()));
    }

    #[test]
    fn test_cached_path_rejects_tokenizer_config_without_real_tokenizer() {
        // A snapshot with only ``config.json`` and ``tokenizer_config.json``
        // (no ``tokenizer.json`` / ``tiktoken.model`` / ``*.tiktoken``) cannot
        // actually load a tokenizer at runtime via
        // ``TokenizerKind::from_disk``. The cache-hit probe must reject this
        // partial state in BOTH modes so ``from_hf`` falls through to a
        // download that populates the real tokenizer artifact.
        let temp = TempDir::new().unwrap();
        let model = "test-org/tokenizer-config-only";
        build_hf_cache(
            temp.path(),
            model,
            &["config.json", "tokenizer_config.json"],
        );

        assert!(
            get_cached_model_path_in(model, true, temp.path().to_path_buf()).is_none(),
            "tokenizer_config.json alone must NOT satisfy ignore_weights=true",
        );
        assert!(
            get_cached_model_path_in(model, false, temp.path().to_path_buf()).is_none(),
            "tokenizer_config.json alone must NOT satisfy ignore_weights=false",
        );
    }

    #[test]
    fn test_get_cached_model_path_at_revision_finds_pinned_snapshot() {
        let temp = TempDir::new().unwrap();
        let model = "test-org/my-model";
        // build_hf_cache always uses "0000000000000000000000000000000000000000" as the SHA
        let snapshot = build_hf_cache(temp.path(), model, &["config.json", "tokenizer.json"]);

        let result = get_cached_model_path_at_revision(
            model,
            "0000000000000000000000000000000000000000",
            true,
            None,
            temp.path().to_path_buf(),
        );

        assert_eq!(result.as_deref(), Some(snapshot.as_path()));
    }

    #[test]
    fn test_get_cached_model_path_at_revision_wrong_sha_returns_none() {
        let temp = TempDir::new().unwrap();
        let model = "test-org/my-model";
        build_hf_cache(temp.path(), model, &["config.json", "tokenizer.json"]);

        let result = get_cached_model_path_at_revision(
            model,
            "different_sha",
            true,
            None,
            temp.path().to_path_buf(),
        );

        assert!(result.is_none(), "wrong SHA should return None");
    }

    #[test]
    fn test_hf_repo_from_snapshot_path_recognizes_hf_cache_layout() {
        let temp = TempDir::new().unwrap();
        let model = "test-org/my-model";
        let snapshot = build_hf_cache(temp.path(), model, &["config.json", "tokenizer.json"]);

        assert_eq!(
            hf_repo_from_snapshot_path(&snapshot),
            Some(model.to_string())
        );
    }

    #[test]
    fn test_hf_repo_from_snapshot_path_recognizes_single_segment_repo() {
        let temp = TempDir::new().unwrap();
        let model = "gpt2";
        let snapshot = build_hf_cache(temp.path(), model, &["config.json", "tokenizer.json"]);

        assert_eq!(
            hf_repo_from_snapshot_path(&snapshot),
            Some(model.to_string())
        );
    }

    #[test]
    fn test_hf_repo_from_snapshot_path_rejects_plain_local_dir() {
        let temp = TempDir::new().unwrap();
        let local_checkpoint = temp.path().join("my-finetuned-model");
        fs::create_dir_all(&local_checkpoint).unwrap();

        assert_eq!(hf_repo_from_snapshot_path(&local_checkpoint), None);
    }

    #[tokio::test]
    async fn test_from_hf_at_revision_rejects_weights_on_cache_miss() {
        let temp = TempDir::new().unwrap();
        // Empty cache dir — guarantees a cache miss, so we hit the new guard
        // before any network call would happen.
        let result = temp_env::async_with_vars(
            [(
                env_model::huggingface::HF_HUB_CACHE,
                Some(temp.path().to_str().unwrap()),
            )],
            async {
                from_hf_at_revision(
                    PathBuf::from("test-org/some-model"),
                    "deadbeef",
                    None,
                    /* ignore_weights = */ false,
                )
                .await
            },
        )
        .await;

        let err = result.expect_err("ignore_weights=false must be rejected on cache miss");
        assert!(
            err.to_string()
                .contains("does not support downloading weights"),
            "unexpected error: {err}"
        );
    }

    #[serial_test::serial]
    #[tokio::test]
    async fn test_from_hf_cache_first_in_online_mode() {
        // The cache-first short-circuit must fire even when HF_HUB_OFFLINE is
        // not set. If it does, from_hf returns the cached path without touching
        // MxClient or the HF network.
        let temp = TempDir::new().unwrap();
        let model = "test-org/cache-first-online";
        let snapshot = build_hf_cache(
            temp.path(),
            model,
            &["config.json", "tokenizer.json", "model.safetensors"],
        );

        temp_env::async_with_vars(
            [
                (
                    env_model::huggingface::HF_HUB_CACHE,
                    Some(temp.path().to_str().unwrap()),
                ),
                (env_model::huggingface::HF_HUB_OFFLINE, None),
                (env_model::huggingface::HF_HOME, None),
                (env_model::model_express::MODEL_EXPRESS_CACHE_PATH, None),
            ],
            async {
                let result = from_hf(PathBuf::from(model), false).await;

                assert_eq!(
                    result.ok().as_deref(),
                    Some(snapshot.as_path()),
                    "from_hf must return cached path in online mode without network"
                );
            },
        )
        .await;
    }
}
