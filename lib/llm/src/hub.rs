// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::path::{Path, PathBuf};

use anyhow::Context;
use hf_hub::Cache;
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
    let config_path = repo.get("config.json")?;
    let snapshot_path = config_path.parent()?;

    if !is_snapshot_complete(snapshot_path, ignore_weights, None) {
        return None;
    }

    let snapshot_path = snapshot_path.to_path_buf();
    tracing::info!("Found cached model '{model_name}' at {snapshot_path:?}, skipping download");
    Some(snapshot_path)
}

fn is_snapshot_complete(
    dir: &Path,
    ignore_weights: bool,
    required_files: Option<&[String]>,
) -> bool {
    if let Some(files) = required_files {
        if !files.iter().all(|file| dir.join(file).exists()) {
            return false;
        }
    } else {
        if !dir.join("config.json").exists() {
            return false;
        }

        let has_tokenizer = dir.join("tokenizer.json").exists()
            || dir.join("tiktoken.model").exists()
            || has_tiktoken_file(dir);
        if !has_tokenizer {
            return false;
        }
    }

    if ignore_weights {
        return true;
    }

    let safetensors_index = dir.join("model.safetensors.index.json");
    let pytorch_index = dir.join("pytorch_model.bin.index.json");
    dir.join("model.safetensors").exists()
        || dir.join("pytorch_model.bin").exists()
        || (safetensors_index.exists() && shard_files_present(&safetensors_index))
        || (pytorch_index.exists() && shard_files_present(&pytorch_index))
}

/// Returns the snapshot path if that exact revision is already on disk.
fn get_cached_model_path_at_revision(
    model_name: &str,
    revision: &str,
    ignore_weights: bool,
    required_files: Option<&[String]>,
    cache_dir: PathBuf,
) -> Option<PathBuf> {
    // This revision reaches the join below from another worker's card over etcd, and a
    // `..` segment in it would walk out of the cache directory. The HF cache only ever
    // names a snapshot directory after a commit SHA, so anything else is a miss.
    if !is_hf_commit_sha(revision) {
        return None;
    }

    // HF cache layout: models--{org}--{model}/snapshots/{sha}/
    let model_key = model_name.replace('/', "--");
    let snapshot_dir = cache_dir
        .join(format!("models--{model_key}"))
        .join("snapshots")
        .join(revision);

    if !snapshot_dir.exists()
        || !is_snapshot_complete(&snapshot_dir, ignore_weights, required_files)
    {
        return None;
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

/// Build the ModelExpress client config shared by `from_hf` and `from_hf_at_revision`
/// from the same environment variables.
fn mx_client_config() -> MxClientConfig {
    let mut config: MxClientConfig = MxClientConfig::default();
    if let Ok(endpoint) = env::var(env_model::model_express::MODEL_EXPRESS_URL) {
        config = config.with_endpoint(endpoint);
    }
    if is_no_shared_storage() {
        config.cache.shared_storage = false;
    }
    config
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

    match download_from_model_express(&model_name, None, ignore_weights).await {
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
/// Otherwise downloads that revision through ModelExpress, which resolves and
/// fetches pinned branches/tags/commit SHAs natively (falling back to a direct
/// HuggingFace download of the same pinned revision if the server is
/// unreachable).
pub async fn from_hf_at_revision(
    name: impl AsRef<Path>,
    revision: &str,
    required_files: Option<&[String]>,
    ignore_weights: bool,
) -> anyhow::Result<PathBuf> {
    let name = name.as_ref();
    let model_name = name.display().to_string();

    // Both branches below join this revision onto a cache directory — the lookup here,
    // and hf-hub's ref writer under the download — and it arrives from another worker's
    // card over etcd. Reject anything that could climb out before either one runs. A
    // branch or tag is still allowed through; only the cache lookup needs a full SHA.
    if !is_joinable_revision(revision) {
        anyhow::bail!("invalid revision for {model_name}: {revision:?}");
    }

    if let Some(cached) = get_cached_model_path_at_revision(
        &model_name,
        revision,
        ignore_weights,
        required_files,
        get_model_express_cache_dir(),
    ) {
        return Ok(cached);
    }

    let snapshot = download_from_model_express(&model_name, Some(revision), ignore_weights)
        .await
        .with_context(|| {
            format!("downloading {model_name} at revision {revision} via ModelExpress")
        })?;

    // The direct-download fallback inside `download_from_model_express` reports no
    // resolved revision of its own, so when a full SHA was requested the snapshot's own
    // directory name is the only confirmation available that the pin was honored.
    if is_hf_commit_sha(revision) && snapshot.file_name().and_then(|n| n.to_str()) != Some(revision)
    {
        anyhow::bail!(
            "{model_name} resolved to {snapshot:?}, which is not the pinned revision {revision}"
        );
    }

    Ok(snapshot)
}

/// The first of `required_files` that is not present in `dir`, if any.
pub(crate) fn first_missing_file<'a>(
    dir: &Path,
    required_files: Option<&'a [String]>,
) -> Option<&'a String> {
    required_files?
        .iter()
        .find(|filename| !dir.join(filename).exists())
}

async fn download_from_model_express(
    model_name: &str,
    revision: Option<&str>,
    ignore_weights: bool,
) -> anyhow::Result<PathBuf> {
    let model_ref = revision
        .map(|revision| format!("{model_name} at revision {revision}"))
        .unwrap_or_else(|| model_name.to_string());

    match MxClient::new(mx_client_config()).await {
        Ok(mut client) => {
            tracing::info!("Successfully connected to ModelExpress server");
            match client
                .request_model_revision(
                    model_name,
                    MxModelProvider::HuggingFace,
                    ignore_weights,
                    revision,
                )
                .await
            {
                Ok(result) => {
                    tracing::info!("Server download succeeded for model: {model_ref}");
                    // Compatibility with ModelExpress servers older than 0.6.0, which
                    // predate the pinned-revision protocol: they parse the request
                    // without its revision field and answer with the default snapshot,
                    // reporting no resolved revision. Accepting that would hand back a
                    // different commit under the name of the pinned one — the exact skew
                    // this path exists to prevent — so a revision the server does not
                    // confirm is a miss.
                    // TODO: remove once 0.6.0 is the oldest ModelExpress server in the
                    // supported window; the client crate and both runtime images are
                    // already pinned to it, so only an operator-run external server can
                    // still be older.
                    if let Some(revision) = revision
                        && !revision_honored(revision, result.resolved_revision.as_deref())
                    {
                        tracing::warn!(
                            resolved_revision = ?result.resolved_revision,
                            "Server did not confirm the pinned revision for '{model_ref}'. \
                            Falling back to direct download."
                        );
                        return mx_download_direct(model_name, Some(revision), ignore_weights)
                            .await;
                    }
                    // `get_model_path` takes no revision, so it cannot stand in for a
                    // pinned request even though the server did confirm one.
                    if revision.is_some() && result.path.is_none() {
                        tracing::warn!(
                            "Server confirmed the pinned revision for '{model_ref}' but reported \
                            no local snapshot path. Falling back to direct download."
                        );
                        return mx_download_direct(model_name, revision, ignore_weights).await;
                    }
                    let resolved = match result.path {
                        Some(path) => Ok(path),
                        None => {
                            client
                                .get_model_path(model_name, MxModelProvider::HuggingFace)
                                .await
                        }
                    };
                    match resolved {
                        Ok(path) => Ok(path),
                        Err(e) => {
                            tracing::warn!(
                                "Failed to resolve local model path after server download for '{model_ref}': {e}. \
                                Falling back to direct download."
                            );
                            mx_download_direct(model_name, revision, ignore_weights).await
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Server download failed for model '{model_ref}': {e}. \
                        Falling back to direct download."
                    );
                    mx_download_direct(model_name, revision, ignore_weights).await
                }
            }
        }
        Err(e) => {
            tracing::warn!("Cannot connect to ModelExpress server: {e}. Using direct download.");
            mx_download_direct(model_name, revision, ignore_weights).await
        }
    }
}

async fn mx_download_direct(
    model_name: &str,
    revision: Option<&str>,
    ignore_weights: bool,
) -> anyhow::Result<PathBuf> {
    mx::download_model_revision(
        model_name,
        MxModelProvider::HuggingFace,
        Some(get_model_express_cache_dir()),
        ignore_weights,
        revision,
    )
    .await
    .map(|result| result.path)
}

/// Whether the revision a ModelExpress server reports resolving confirms that it
/// honored `requested`.
///
/// `None` means the server never resolved a revision at all, which is what a server
/// older than the pinned-revision protocol reports. A branch or tag legitimately
/// resolves to an unrelated immutable commit SHA, so only a requested SHA can be
/// compared against the answer, and an abbreviated one is a prefix of it.
fn revision_honored(requested: &str, resolved: Option<&str>) -> bool {
    let Some(resolved) = resolved else {
        return false;
    };
    if !is_commit_sha_prefix(requested) {
        return true;
    }
    resolved.len() >= requested.len()
        && resolved.as_bytes()[..requested.len()].eq_ignore_ascii_case(requested.as_bytes())
}

/// Whether `revision` names a commit directly — full or abbreviated — rather than a
/// branch or tag. Looser than [`is_hf_commit_sha`] because a caller may abbreviate a
/// SHA on the command line; use that one for anything joined into a path or published
/// on a card.
fn is_commit_sha_prefix(revision: &str) -> bool {
    (7..=40).contains(&revision.len()) && revision.bytes().all(|b| b.is_ascii_hexdigit())
}

/// Whether `revision` is a full Hugging Face commit SHA. This is the only shape that
/// names a snapshot directory, so it is the bar both for joining a revision into a
/// cache path and for publishing one on a model deployment card.
pub(crate) fn is_hf_commit_sha(revision: &str) -> bool {
    huggingface::validate_hf_commit_sha(revision).is_ok()
}

/// Whether `filename` is a safe relative path within a repository snapshot. Rejects
/// absolute paths and `.`/`..` components, which would otherwise resolve outside the
/// snapshot directory they are joined onto.
pub(crate) fn is_hf_repo_file(filename: &str) -> bool {
    huggingface::validate_hf_repo_file(filename).is_ok()
}

/// Whether `repo` is a safe repository id: no absolute or `..` segments, since it is
/// both a cache key and an outbound Hub request path.
pub(crate) fn is_hf_repo_path(repo: &str) -> bool {
    huggingface::validate_hf_relative_path(repo, "repository id").is_ok()
}

/// Whether `revision` is safe to join onto a cache directory. Looser than
/// [`is_hf_commit_sha`], so a branch or tag still reaches the download, but it still
/// rejects anything that could climb out of the cache root.
fn is_joinable_revision(revision: &str) -> bool {
    huggingface::validate_hf_relative_path(revision, "revision").is_ok()
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
pub(crate) mod tests {
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
    pub(crate) fn build_hf_cache(cache_root: &Path, model_name: &str, files: &[&str]) -> PathBuf {
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
    fn test_get_cached_model_path_at_revision_rejects_traversal_out_of_the_cache() {
        let temp = TempDir::new().unwrap();
        let model = "test-org/my-model";
        // The escape target is a complete snapshot, so only the revision check can
        // reject it: reaching the join at all would return a path outside the cache.
        let outside = temp.path().join("outside");
        std::fs::create_dir_all(&outside).unwrap();
        std::fs::write(outside.join("config.json"), "{}").unwrap();
        std::fs::write(outside.join("tokenizer.json"), "{}").unwrap();
        build_hf_cache(temp.path(), model, &["config.json", "tokenizer.json"]);

        for revision in ["../../outside", "..", ""] {
            assert!(
                get_cached_model_path_at_revision(
                    model,
                    revision,
                    true,
                    None,
                    temp.path().to_path_buf(),
                )
                .is_none(),
                "revision {revision:?} must not resolve a path",
            );
        }
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

    #[test]
    fn test_get_cached_model_path_at_revision_finds_pinned_snapshot_with_weights() {
        // ignore_weights=false is satisfied once weight files are on disk at the
        // pinned revision.
        let temp = TempDir::new().unwrap();
        let model = "test-org/my-model";
        let snapshot = build_hf_cache(
            temp.path(),
            model,
            &["config.json", "tokenizer.json", "model.safetensors"],
        );

        let result = get_cached_model_path_at_revision(
            model,
            "0000000000000000000000000000000000000000",
            false,
            None,
            temp.path().to_path_buf(),
        );

        assert_eq!(result.as_deref(), Some(snapshot.as_path()));
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

    const SHA: &str = "1a2b3c4d5e6f78901a2b3c4d5e6f78901a2b3c4d";

    #[test]
    fn test_revision_unconfirmed_by_the_server_is_not_honored() {
        // A server older than the pinned-revision protocol ignores the request field
        // and resolves nothing, so its default snapshot must not pass as the pin.
        assert!(!revision_honored(SHA, None));
    }

    #[test]
    fn test_revision_honored_when_the_server_resolves_the_requested_sha() {
        assert!(revision_honored(SHA, Some(SHA)));
        assert!(revision_honored(SHA, Some(&SHA.to_uppercase())));
        assert!(revision_honored(&SHA[..12], Some(SHA)));
    }

    #[test]
    fn test_revision_not_honored_when_the_server_resolves_a_different_sha() {
        let other = "9f8e7d6c5b4a39209f8e7d6c5b4a39209f8e7d6c";

        assert!(!revision_honored(SHA, Some(other)));
        assert!(!revision_honored(SHA, Some(&SHA[..12])));
    }

    #[test]
    fn test_branch_or_tag_accepts_the_sha_it_resolves_to() {
        // A branch or tag has no SHA to compare against; resolving it at all is the
        // confirmation that the server understood the request.
        assert!(revision_honored("main", Some(SHA)));
        assert!(revision_honored("v1.0", Some(SHA)));
        assert!(!revision_honored("main", None));
    }

    #[test]
    fn test_first_missing_file_holds_a_download_to_the_required_set() {
        let temp = TempDir::new().unwrap();
        std::fs::write(temp.path().join("config.json"), "{}").unwrap();

        let present = ["config.json".to_string()];
        let short = ["config.json".to_string(), "added_tokens.json".to_string()];

        assert_eq!(first_missing_file(temp.path(), None), None);
        assert_eq!(first_missing_file(temp.path(), Some(&present)), None);
        assert_eq!(
            first_missing_file(temp.path(), Some(&short)).map(String::as_str),
            Some("added_tokens.json"),
        );
    }

    #[test]
    fn test_is_hf_commit_sha_accepts_only_a_full_sha() {
        assert!(is_hf_commit_sha(SHA));
        assert!(is_hf_commit_sha(&SHA.to_uppercase()));
        assert!(
            !is_hf_commit_sha(&SHA[..12]),
            "abbreviated is not a snapshot"
        );
        assert!(!is_hf_commit_sha("main"));
        assert!(!is_hf_commit_sha(".."));
        assert!(!is_hf_commit_sha(""));
    }

    #[test]
    fn test_is_hf_repo_file_rejects_paths_that_leave_the_snapshot() {
        assert!(is_hf_repo_file("config.json"));
        assert!(is_hf_repo_file("subdir/config.json"));
        assert!(!is_hf_repo_file(".."));
        assert!(!is_hf_repo_file("."));
        assert!(!is_hf_repo_file("../escape.json"));
        assert!(!is_hf_repo_file("/etc/passwd"));
        assert!(!is_hf_repo_file(""));
    }

    #[test]
    fn test_is_commit_sha_prefix_separates_shas_from_ref_names() {
        assert!(is_commit_sha_prefix(SHA));
        assert!(is_commit_sha_prefix("1a2b3c4"));
        assert!(!is_commit_sha_prefix("1a2b3c")); // too short to be unambiguous
        assert!(!is_commit_sha_prefix("main"));
        assert!(!is_commit_sha_prefix("refs/pr/1"));
        assert!(!is_commit_sha_prefix(&format!("{SHA}0"))); // longer than a SHA
    }
}
