// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::env;
use std::path::{Component, Path, PathBuf};

use hf_hub::{Cache, Repo};
use modelexpress_client::{
    Client as MxClient, ClientConfig as MxClientConfig, ModelProvider as MxModelProvider,
};
use modelexpress_common::{cache as mx_cache, download as mx, envs as mx_envs};

use dynamo_runtime::config::environment_names::model as env_model;

const NGC_URI_PREFIX: &str = "ngc://";

fn provider_for(model_name: &str) -> MxModelProvider {
    if model_name.starts_with(NGC_URI_PREFIX) {
        MxModelProvider::Ngc
    } else {
        MxModelProvider::HuggingFace
    }
}

/// Check if a model is already cached locally with the files needed to serve it.
/// Returns the path to the cached model directory if found, None otherwise.
///
/// For tokenizer-only downloads (ignore_weights=true), we check for config.json and
/// tokenizer files. For full downloads, we also require weight files to be present.
fn get_cached_model_path(
    model_name: &str,
    provider: MxModelProvider,
    ignore_weights: bool,
) -> Option<PathBuf> {
    let cache_dir = get_model_express_cache_dir(provider);
    match provider {
        MxModelProvider::HuggingFace => {
            get_cached_model_path_in(model_name, ignore_weights, cache_dir)
        }
        _ => get_cached_provider_model_path_in(model_name, provider, ignore_weights, &cache_dir),
    }
}

fn get_cached_model_path_in(
    model_name: &str,
    ignore_weights: bool,
    cache_dir: PathBuf,
) -> Option<PathBuf> {
    let cache = Cache::new(cache_dir);
    let snapshot_path = cache
        .model(model_name.to_string())
        .get("config.json")?
        .parent()?
        .to_path_buf();
    cached_model_dir(model_name, snapshot_path, ignore_weights)
}

/// Look up a model in a non-HuggingFace provider's cache. ModelExpress owns the layout.
fn get_cached_provider_model_path_in(
    model_name: &str,
    provider: MxModelProvider,
    ignore_weights: bool,
    cache_dir: &Path,
) -> Option<PathBuf> {
    let model_dir = mx_cache::resolve_model_path(cache_dir, provider, model_name, None).ok()?;
    cached_model_dir(model_name, model_dir, ignore_weights)
}

fn cached_model_dir(model_name: &str, dir: PathBuf, ignore_weights: bool) -> Option<PathBuf> {
    if !has_required_model_files(&dir, ignore_weights) {
        return None;
    }
    tracing::info!("Found cached model '{model_name}' at {dir:?}, skipping download");
    Some(dir)
}

/// Require model config and tokenizer files, plus weights unless explicitly skipped.
fn has_required_model_files(dir: &Path, ignore_weights: bool) -> bool {
    let has = |filename: &str| dir.join(filename).exists();

    if !has("config.json") {
        return false;
    }

    // Check for tokenizer files (at least one must exist). Only count
    // artifacts that ``ModelDeploymentCard::TokenizerKind::from_disk`` can
    // actually load -- ``tokenizer_config.json`` is metadata describing the
    // tokenizer and cannot be used on its own, so a snapshot with only
    // ``config.json`` + ``tokenizer_config.json`` would fall through to a
    // download even though the cache appears "populated".
    let has_tokenizer = has("tokenizer.json") || has("tiktoken.model") || has_tiktoken_file(dir);
    if !has_tokenizer {
        return false;
    }

    // For full downloads, check for weight files. When an index file is present,
    // verify the shard files it references are also cached — an index without its
    // shards is an incomplete cache that should fall through to download.
    ignore_weights
        || has("model.safetensors")
        || has("pytorch_model.bin")
        || shard_files_present(&dir.join("model.safetensors.index.json"))
        || shard_files_present(&dir.join("pytorch_model.bin.index.json"))
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
/// `ngc://` names are downloaded from NGC; all other names are Hugging Face repos.
/// If ignore_weights is true, model weight files will be skipped
/// Returns the path to the model files
///
/// If the model is already cached locally with the required files, returns the cached
/// path without making any network calls, regardless of HF_HUB_OFFLINE.
pub async fn from_hf(name: impl AsRef<Path>, ignore_weights: bool) -> anyhow::Result<PathBuf> {
    let name = name.as_ref();
    let model_name = name.display().to_string();
    let provider = provider_for(&model_name);

    // Cache-first in all modes: if the snapshot is already on disk with the files we
    // need, return it without touching the network.
    if let Some(cached_path) = get_cached_model_path(&model_name, provider, ignore_weights) {
        return Ok(cached_path);
    }

    if is_offline_mode() {
        tracing::warn!(
            "Offline mode enabled but model '{model_name}' not found in cache, attempting download anyway"
        );
    }

    let result = match MxClient::new(mx_client_config(provider)).await {
        Ok(mut client) => {
            tracing::info!("Successfully connected to ModelExpress server");
            match client
                .request_model(&model_name, provider, ignore_weights)
                .await
            {
                Ok(()) => {
                    tracing::info!("Server download succeeded for model: {model_name}");
                    match client.get_model_path(&model_name, provider).await {
                        Ok(path) => Ok(path),
                        Err(e) => {
                            tracing::warn!(
                                "Failed to resolve local model path after server download for '{model_name}': {e}. \
                                Falling back to direct download."
                            );
                            mx_download_direct(&model_name, provider, ignore_weights).await
                        }
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        "Server download failed for model '{model_name}': {e}. Falling back to direct download."
                    );
                    mx_download_direct(&model_name, provider, ignore_weights).await
                }
            }
        }
        Err(e) => {
            tracing::warn!("Cannot connect to ModelExpress server: {e}. Using direct download.");
            mx_download_direct(&model_name, provider, ignore_weights).await
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

fn mx_client_config(provider: MxModelProvider) -> MxClientConfig {
    let mut config = MxClientConfig::default();
    if provider == MxModelProvider::Ngc {
        config.cache.local_path = get_model_express_cache_dir(provider);
    }
    if let Ok(endpoint) = env::var(env_model::model_express::MODEL_EXPRESS_URL) {
        config = config.with_endpoint(endpoint);
    }
    if is_no_shared_storage() {
        config.cache.shared_storage = false;
    }
    config
}

async fn mx_download_direct(
    model_name: &str,
    provider: MxModelProvider,
    ignore_weights: bool,
) -> anyhow::Result<PathBuf> {
    let cache_dir = get_model_express_cache_dir(provider);
    mx::download_model(model_name, provider, Some(cache_dir), ignore_weights).await
}

/// A full Hugging Face commit SHA: the name of a snapshot directory in the hub cache.
fn is_commit_hash(revision: &str) -> bool {
    revision.len() == 40
        && revision
            .bytes()
            .all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

/// The commit a Hugging Face hub-cache snapshot directory (`.../snapshots/<commit>`) holds, or
/// `None` for any other directory.
pub(crate) fn snapshot_commit(dir: &Path) -> Option<String> {
    let commit = dir.file_name()?.to_str()?;
    let parent = dir.parent()?.file_name()?;
    (parent == "snapshots" && is_commit_hash(commit)).then(|| commit.to_string())
}

/// Revisions are Hugging Face branches, tags, or commits. One becomes a path below the
/// repository's `refs/` cache directory, so it must stay inside it.
fn check_revision_request(model_name: &str, revision: &str) -> anyhow::Result<()> {
    if provider_for(model_name) != MxModelProvider::HuggingFace {
        anyhow::bail!(
            "revision '{revision}' applies only to Hugging Face models, not '{model_name}'"
        );
    }
    let mut components = Path::new(revision).components().peekable();
    if components.peek().is_none() || !components.all(|c| matches!(c, Component::Normal(_))) {
        anyhow::bail!("invalid Hugging Face revision '{revision}'");
    }
    Ok(())
}

/// The cached snapshot of `model_name` at `revision`, if it holds the files needed to serve.
/// A branch or tag resolves through the cache's `refs/`; a commit names its snapshot directly.
/// Never falls back to another revision.
fn get_cached_model_path_at_revision_in(
    model_name: &str,
    revision: &str,
    ignore_weights: bool,
    cache_dir: &Path,
) -> Option<PathBuf> {
    let repo_dir = cache_dir.join(Repo::model(model_name.to_string()).folder_name());
    let commit = if is_commit_hash(revision) {
        revision.to_string()
    } else {
        std::fs::read_to_string(repo_dir.join("refs").join(revision))
            .ok()?
            .trim()
            .to_string()
    };
    if !is_commit_hash(&commit) {
        return None;
    }
    cached_model_dir(
        model_name,
        repo_dir.join("snapshots").join(commit),
        ignore_weights,
    )
}

/// [`from_hf`] at a Hugging Face `revision` (branch, tag, or commit SHA) instead of the
/// repository's default branch; `None` is [`from_hf`].
///
/// A cached snapshot of the revision is returned without network calls. Otherwise the
/// revision is downloaded, through the ModelExpress server when one is reachable. The
/// result is always the requested revision, never another cached one.
pub async fn from_hf_at_revision(
    name: impl AsRef<Path>,
    revision: Option<&str>,
    ignore_weights: bool,
) -> anyhow::Result<PathBuf> {
    let Some(revision) = revision else {
        return from_hf(name, ignore_weights).await;
    };
    // The hub names commits in lower case; accept a full SHA in either case.
    let lower_case = revision.to_ascii_lowercase();
    let revision = if is_commit_hash(&lower_case) {
        lower_case.as_str()
    } else {
        revision
    };
    let model_name = name.as_ref().display().to_string();
    check_revision_request(&model_name, revision)?;
    let provider = MxModelProvider::HuggingFace;
    let cache_dir = get_model_express_cache_dir(provider);

    if let Some(cached_path) =
        get_cached_model_path_at_revision_in(&model_name, revision, ignore_weights, &cache_dir)
    {
        return Ok(cached_path);
    }

    if is_offline_mode() {
        tracing::warn!(
            "Offline mode enabled but model '{model_name}' at revision '{revision}' not found in cache, attempting download anyway"
        );
    }

    let server_path = match MxClient::new(mx_client_config(provider)).await {
        Ok(mut client) => match client
            .request_model_revision(&model_name, provider, ignore_weights, Some(revision))
            .await
        {
            Ok(download) => {
                let path = download.path.or_else(|| {
                    let commit = download.resolved_revision.as_deref()?;
                    get_cached_model_path_at_revision_in(
                        &model_name,
                        commit,
                        ignore_weights,
                        &cache_dir,
                    )
                });
                if path.is_none() {
                    tracing::warn!(
                        "Server downloaded model '{model_name}' at revision '{revision}' but its local path is unknown. Falling back to direct download."
                    );
                }
                path
            }
            Err(e) => {
                tracing::warn!(
                    "Server download failed for model '{model_name}' at revision '{revision}': {e}. Falling back to direct download."
                );
                None
            }
        },
        Err(e) => {
            tracing::warn!("Cannot connect to ModelExpress server: {e}. Using direct download.");
            None
        }
    };
    if let Some(path) = server_path {
        return Ok(path);
    }

    let download = mx::download_model_revision(
        &model_name,
        provider,
        Some(cache_dir),
        ignore_weights,
        Some(revision),
    )
    .await
    .inspect_err(|e| {
        tracing::warn!(
            "ModelExpress download failed for model '{model_name}' at revision '{revision}': {e}"
        )
    })?;
    tracing::info!(
        "ModelExpress download completed for model '{model_name}' at revision '{revision}' ({})",
        download
            .resolved_revision
            .as_deref()
            .unwrap_or("unresolved")
    );
    Ok(download.path)
}

// TODO: remove in the future. This is a temporary workaround to find common
// cache directory between client and server.
fn get_model_express_cache_dir(provider: MxModelProvider) -> PathBuf {
    if provider == MxModelProvider::Ngc {
        // ModelExpress 0.6 adds the provider's `ngc/` layout beneath this root.
        return mx_envs::cache_directory()
            .unwrap_or_else(|| mx_envs::home_dir_or_cwd().join(".cache"));
    }
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

    #[test]
    fn provider_for_routes_ngc_uris_to_ngc() {
        assert_eq!(
            provider_for("ngc://nvstaging/nim/nemotron-3-ultra:hf-rl-052726-nvfp4-e9744cc"),
            MxModelProvider::Ngc
        );
        assert_eq!(
            provider_for("Qwen/Qwen3-0.6B"),
            MxModelProvider::HuggingFace
        );
        assert_eq!(
            provider_for("/data/llms/Qwen3-0.6B"),
            MxModelProvider::HuggingFace
        );
    }

    const NGC_TEST_MODEL: &str = "ngc://test-org/test-team/test-model:v1";

    fn build_ngc_cache(cache_root: &Path, files: &[&str]) -> PathBuf {
        let model_dir = cache_root.join("ngc/test-org/test-team/models/test-model/v1");
        fs::create_dir_all(&model_dir).unwrap();
        for f in files {
            fs::write(model_dir.join(f), "{}").unwrap();
        }
        model_dir
    }

    #[test]
    fn test_ngc_cached_path_requires_weights_for_full_download() {
        let temp = TempDir::new().unwrap();
        let model_dir = build_ngc_cache(temp.path(), &["config.json", "tokenizer.json"]);
        let lookup = |ignore_weights: bool| {
            get_cached_provider_model_path_in(
                NGC_TEST_MODEL,
                MxModelProvider::Ngc,
                ignore_weights,
                temp.path(),
            )
        };

        assert!(
            lookup(false).is_none(),
            "NGC cache without weights must NOT satisfy ignore_weights=false"
        );
        assert_eq!(lookup(true).as_deref(), Some(model_dir.as_path()));

        fs::write(model_dir.join("model.safetensors"), "").unwrap();
        assert_eq!(lookup(false).as_deref(), Some(model_dir.as_path()));
    }

    #[serial_test::serial]
    #[tokio::test]
    async fn test_from_hf_ngc_cache_first() {
        for explicit_cache in [true, false] {
            let temp = TempDir::new().unwrap();
            let home = temp.path().join("home");
            let hf_cache = temp.path().join("hf");
            let cache_root = if explicit_cache {
                temp.path().join("mx")
            } else {
                home.join(".cache")
            };
            let model_dir = build_ngc_cache(
                &cache_root,
                &["config.json", "tokenizer.json", "model.safetensors"],
            );

            temp_env::async_with_vars(
                [
                    (
                        mx_envs::MODEL_EXPRESS_CACHE_DIRECTORY,
                        explicit_cache.then(|| cache_root.to_str().unwrap()),
                    ),
                    (mx_envs::HOME, home.to_str()),
                    (env_model::huggingface::HF_HUB_CACHE, hf_cache.to_str()),
                    (env_model::huggingface::HF_HOME, hf_cache.to_str()),
                    (
                        env_model::model_express::MODEL_EXPRESS_CACHE_PATH,
                        hf_cache.to_str(),
                    ),
                ],
                async {
                    assert_eq!(
                        get_cached_model_path(NGC_TEST_MODEL, MxModelProvider::Ngc, false)
                            .as_deref(),
                        Some(model_dir.as_path())
                    );
                    assert_eq!(
                        get_model_express_cache_dir(MxModelProvider::HuggingFace),
                        hf_cache
                    );
                    let result = from_hf(PathBuf::from(NGC_TEST_MODEL), false).await;
                    assert_eq!(result.ok().as_deref(), Some(model_dir.as_path()));
                },
            )
            .await;
        }
    }

    const MAIN_COMMIT: &str = "1111111111111111111111111111111111111111";
    const PINNED_COMMIT: &str = "2222222222222222222222222222222222222222";

    /// Add a snapshot of `model_name` at `commit` to an hf-hub cache, with `refs` pointing at
    /// it, and return the snapshot directory.
    fn build_hf_snapshot(
        cache_root: &Path,
        model_name: &str,
        commit: &str,
        refs: &[&str],
        files: &[&str],
    ) -> PathBuf {
        let repo_dir = cache_root.join(format!("models--{}", model_name.replace('/', "--")));
        let snapshot_dir = repo_dir.join("snapshots").join(commit);
        fs::create_dir_all(&snapshot_dir).unwrap();
        for reference in refs {
            let ref_path = repo_dir.join("refs").join(reference);
            fs::create_dir_all(ref_path.parent().unwrap()).unwrap();
            fs::write(ref_path, commit).unwrap();
        }
        for f in files {
            fs::write(snapshot_dir.join(f), "{}").unwrap();
        }
        snapshot_dir
    }

    const FULL_MODEL: &[&str] = &["config.json", "tokenizer.json", "model.safetensors"];

    #[test]
    fn cached_revision_reads_the_commit_snapshot_without_a_ref() {
        let temp = TempDir::new().unwrap();
        let model = "test-org/pinned";
        let main = build_hf_snapshot(temp.path(), model, MAIN_COMMIT, &["main"], FULL_MODEL);
        let pinned = build_hf_snapshot(temp.path(), model, PINNED_COMMIT, &[], FULL_MODEL);

        let lookup =
            |revision| get_cached_model_path_at_revision_in(model, revision, false, temp.path());
        assert_eq!(lookup(PINNED_COMMIT).as_deref(), Some(pinned.as_path()));
        assert_eq!(lookup(MAIN_COMMIT).as_deref(), Some(main.as_path()));
        assert_eq!(lookup("main").as_deref(), Some(main.as_path()));
    }

    #[test]
    fn cached_revision_follows_a_branch_or_tag_ref() {
        let temp = TempDir::new().unwrap();
        let model = "test-org/pinned";
        build_hf_snapshot(temp.path(), model, MAIN_COMMIT, &["main"], FULL_MODEL);
        let pinned = build_hf_snapshot(
            temp.path(),
            model,
            PINNED_COMMIT,
            &["release/v1"],
            FULL_MODEL,
        );

        assert_eq!(
            get_cached_model_path_at_revision_in(model, "release/v1", false, temp.path())
                .as_deref(),
            Some(pinned.as_path())
        );
    }

    #[test]
    fn cached_revision_never_falls_back_to_the_default_branch() {
        let temp = TempDir::new().unwrap();
        let model = "test-org/pinned";
        build_hf_snapshot(temp.path(), model, MAIN_COMMIT, &["main"], FULL_MODEL);

        for revision in [PINNED_COMMIT, "release/v1"] {
            assert!(
                get_cached_model_path_at_revision_in(model, revision, true, temp.path()).is_none(),
                "{revision} is not cached, so the lookup must miss rather than return main"
            );
        }
    }

    #[test]
    fn cached_revision_still_requires_the_files_to_serve() {
        let temp = TempDir::new().unwrap();
        let model = "test-org/pinned";
        build_hf_snapshot(
            temp.path(),
            model,
            PINNED_COMMIT,
            &[],
            &["config.json", "tokenizer.json"],
        );

        assert!(
            get_cached_model_path_at_revision_in(model, PINNED_COMMIT, false, temp.path())
                .is_none()
        );
        assert!(
            get_cached_model_path_at_revision_in(model, PINNED_COMMIT, true, temp.path()).is_some()
        );
    }

    #[test]
    fn snapshot_commit_reads_only_a_snapshot_directory_named_by_a_commit() {
        let snapshots = Path::new("/cache/models--org--name/snapshots");
        assert_eq!(
            snapshot_commit(&snapshots.join(PINNED_COMMIT)).as_deref(),
            Some(PINNED_COMMIT)
        );
        assert_eq!(snapshot_commit(&snapshots.join("main")), None);
        let hex_commit = "0123456789abcdef0123456789abcdef01234567";
        assert_eq!(
            snapshot_commit(&snapshots.join(hex_commit)).as_deref(),
            Some(hex_commit)
        );
        assert_eq!(
            snapshot_commit(&snapshots.join(hex_commit.to_uppercase())),
            None
        );
        assert_eq!(
            snapshot_commit(&Path::new("/models/org--name").join(PINNED_COMMIT)),
            None
        );
        assert_eq!(snapshot_commit(Path::new("/models/Qwen3-0.6B")), None);
    }

    #[test]
    fn a_revision_must_be_a_hugging_face_ref_inside_the_cache() {
        for revision in ["main", "release/v1", "refs/pr/1", PINNED_COMMIT] {
            assert!(
                check_revision_request("org/name", revision).is_ok(),
                "{revision}"
            );
        }
        for revision in ["", "../escape", "refs/../../escape", "/abs", "./main"] {
            let err = check_revision_request("org/name", revision).unwrap_err();
            assert!(
                err.to_string().contains("invalid Hugging Face revision"),
                "{revision}: {err}"
            );
        }
        let err = check_revision_request(NGC_TEST_MODEL, "v1").unwrap_err();
        assert!(err.to_string().contains("only to Hugging Face"), "{err}");
    }

    #[serial_test::serial]
    #[tokio::test]
    async fn from_hf_at_revision_accepts_a_commit_in_upper_case() {
        let temp = TempDir::new().unwrap();
        let model = "test-org/pinned-upper-case";
        let commit = "0123456789abcdef0123456789abcdef01234567";
        let pinned = build_hf_snapshot(temp.path(), model, commit, &[], FULL_MODEL);

        temp_env::async_with_vars(
            [
                (env_model::huggingface::HF_HUB_CACHE, temp.path().to_str()),
                (env_model::huggingface::HF_HUB_OFFLINE, Some("1")),
                (env_model::huggingface::HF_HOME, None),
                (env_model::model_express::MODEL_EXPRESS_CACHE_PATH, None),
                (
                    env_model::model_express::MODEL_EXPRESS_URL,
                    Some("http://127.0.0.1:9"),
                ),
            ],
            async {
                let path = from_hf_at_revision(model, Some(&commit.to_uppercase()), false).await;
                assert_eq!(path.ok().as_deref(), Some(pinned.as_path()));
            },
        )
        .await;
    }

    #[serial_test::serial]
    #[tokio::test]
    async fn from_hf_at_revision_serves_the_cached_commit_not_main() {
        let temp = TempDir::new().unwrap();
        let model = "test-org/pinned-cache-first";
        let main = build_hf_snapshot(temp.path(), model, MAIN_COMMIT, &["main"], FULL_MODEL);
        let pinned = build_hf_snapshot(temp.path(), model, PINNED_COMMIT, &[], FULL_MODEL);

        temp_env::async_with_vars(
            [
                (env_model::huggingface::HF_HUB_CACHE, temp.path().to_str()),
                (env_model::huggingface::HF_HUB_OFFLINE, None),
                (env_model::huggingface::HF_HOME, None),
                (env_model::model_express::MODEL_EXPRESS_CACHE_PATH, None),
            ],
            async {
                let at_commit = from_hf_at_revision(model, Some(PINNED_COMMIT), false).await;
                assert_eq!(at_commit.ok().as_deref(), Some(pinned.as_path()));
                let default = from_hf_at_revision(model, None, false).await;
                assert_eq!(default.ok().as_deref(), Some(main.as_path()));
            },
        )
        .await;
    }
}
