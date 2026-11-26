// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::io::{BufReader, ErrorKind, Read};
use std::path::{Path, PathBuf};
use std::{env, fs};

use anyhow::Context as _;

use gguf_rs_lib::{GGUFError, reader::stream_reader::GGUFStreamReader};
use hf_hub::Cache;
use modelexpress_client::{
    Client as MxClient, ClientConfig as MxClientConfig, ModelProvider as MxModelProvider,
};
use modelexpress_common::{cache as mx_cache, download as mx, envs as mx_envs};

use dynamo_runtime::config::environment_names::model as env_model;

mod onnx;

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
    if !ignore_weights && !hf_safetensors_ready(&snapshot_path).unwrap_or(false) {
        return None;
    }
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

    if !hf_tokenizer_file(dir).is_ok_and(|file| file.is_some()) {
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

/// Prefer tokenizer.json, then tiktoken.model, then a single named .tiktoken file.
pub(crate) fn hf_tokenizer_file(dir: &Path) -> anyhow::Result<Option<PathBuf>> {
    let mut first_error = None;
    let mut readable = |path: &Path| match readable_metadata_file(path) {
        Err(error) if error.kind() != ErrorKind::InvalidData => {
            first_error.get_or_insert(error);
            Ok(false)
        }
        result => result,
    };
    for name in ["tokenizer.json", "tiktoken.model"] {
        let path = dir.join(name);
        if readable(&path)? {
            return Ok(Some(path));
        }
    }
    let mut candidate: Option<PathBuf> = None;
    for entry in fs::read_dir(dir).with_context(|| format!("Reading {}", dir.display()))? {
        let path = entry?.path();
        if path.extension().is_some_and(|ext| ext == "tiktoken") && readable(&path)? {
            if let Some(first) = candidate {
                anyhow::bail!(
                    "Multiple .tiktoken files found in {}: {} and {}",
                    dir.display(),
                    first.display(),
                    path.display()
                );
            }
            candidate = Some(path);
        }
    }
    if candidate.is_none()
        && let Some(error) = first_error
    {
        return Err(error.into());
    }
    Ok(candidate)
}

pub(crate) fn mistral_sentencepiece_version(path: &Path) -> Option<u32> {
    let name = path.file_name()?.to_str()?;
    let version = name
        .strip_prefix("tokenizer.model.v")
        .or_else(|| name.strip_prefix("tokenizer.mm.model.v"))?
        .parse::<u32>()
        .ok()?;
    (version < 11).then_some(version)
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
    from_hf_filtered(name, ignore_weights, None).await
}

type SnapshotFilter = dyn Fn(&Path) -> bool + Send + Sync;

pub(crate) async fn from_hf_filtered(
    name: impl AsRef<Path>,
    ignore_weights: bool,
    filter: Option<std::sync::Arc<SnapshotFilter>>,
) -> anyhow::Result<PathBuf> {
    let model_name = name.as_ref().display().to_string();
    let provider = provider_for(&model_name);

    let mut config: MxClientConfig = MxClientConfig::default();
    if provider == MxModelProvider::Ngc {
        config.cache.local_path = get_model_express_cache_dir(provider);
    }
    if let Ok(endpoint) = env::var(env_model::model_express::MODEL_EXPRESS_URL) {
        config = config.with_endpoint(endpoint);
    }
    if is_no_shared_storage() {
        config.cache.shared_storage = false;
    }

    let huggingface = MxModelProvider::resolve_provider_for_model_name(&model_name, provider)
        == MxModelProvider::HuggingFace;
    if !huggingface || (config.cache.shared_storage && filter.is_none()) {
        let name = model_name.clone();
        let cached = tokio::task::spawn_blocking(move || {
            get_cached_model_path(&name, provider, ignore_weights)
        })
        .await?;
        if let Some(path) = cached {
            return Ok(path);
        }
    }

    let cache_dir = get_model_express_cache_dir(provider);
    let repo_dir = || cache_dir.join(hf_hub::Repo::model(model_name.clone()).folder_name());
    // The HF cache can include streamed snapshots even in shared-storage mode.
    if huggingface {
        let repo = repo_dir();
        let matches = filter.clone();
        let cached = tokio::task::spawn_blocking(move || {
            matching_stream_snapshot(&repo, ignore_weights, matches.as_deref()).unwrap_or_else(
                |error| {
                    tracing::warn!(?repo, %error, "Failed to inspect cached Hugging Face snapshot");
                    None
                },
            )
        })
        .await?;
        if let Some(path) = cached {
            tracing::info!("Found cached model '{model_name}' at {path:?}, skipping download");
            return Ok(path);
        }
    }

    let filter = filter.filter(|_| huggingface);
    let path = download_model(
        &model_name,
        config,
        provider,
        cache_dir.clone(),
        ignore_weights,
        huggingface,
        filter.clone(),
    )
    .await?;
    if let Some(matches) = filter {
        let repo = repo_dir();
        return tokio::task::spawn_blocking(move || {
            if matches(&path) {
                return Ok(path);
            }
            // A shared fetch may have populated the HF cache while we waited for ModelExpress.
            matching_stream_snapshot(&repo, ignore_weights, Some(matches.as_ref()))?.with_context(
                || format!("No HF snapshot matches the model card checksums for {model_name}"),
            )
        })
        .await?;
    }
    if huggingface && !ignore_weights {
        return tokio::task::spawn_blocking(move || {
            anyhow::ensure!(
                hf_safetensors_ready(&path)?,
                "Hugging Face safetensors checkpoint in {} is incomplete",
                path.display()
            );
            Ok(path)
        })
        .await?;
    }
    Ok(path)
}

async fn download_model(
    model_name: &str,
    mut config: MxClientConfig,
    provider: MxModelProvider,
    cache_dir: PathBuf,
    ignore_weights: bool,
    huggingface: bool,
    filter: Option<std::sync::Arc<SnapshotFilter>>,
) -> anyhow::Result<PathBuf> {
    if is_offline_mode() {
        tracing::warn!(
            "Offline mode enabled but model '{model_name}' not found in cache, attempting download anyway"
        );
    }

    let stream_hf = huggingface && !config.cache.shared_storage;
    let server = async {
        let staging = if stream_hf {
            tokio::fs::create_dir_all(&cache_dir).await?;
            // The client writes final filenames as chunks arrive. Keep those writes
            // outside the HF snapshot tree until it verifies the end of the stream.
            let dir = tempfile::Builder::new()
                .prefix(".mx-")
                .tempdir_in(&cache_dir)?;
            config.cache.local_path = dir.path().to_owned();
            Some(dir)
        } else {
            None
        };
        let mut client = MxClient::new(config).await?;
        let download = client
            .request_model_revision(model_name, provider, ignore_weights, None)
            .await?;
        let path = if staging.is_some() {
            download
                .path
                .context("ModelExpress returned no streamed snapshot")?
        } else {
            let shared = if huggingface
                && let Some(revision) = download.resolved_revision.as_deref()
            {
                match download
                    .path
                    .filter(|path| path.file_name() == Some(revision.as_ref()) && path.is_dir())
                {
                    Some(path) => Some(path),
                    None => {
                        // MX and HF_HUB_CACHE can point at different shared cache roots.
                        let repo = hf_hub::Repo::model(model_name.to_owned()).folder_name();
                        let snapshots = cache_dir.join(repo).join("snapshots");
                        let revision = revision.to_owned();
                        let matches = filter.clone();
                        tokio::task::spawn_blocking(move || {
                            revision_snapshot(&snapshots, &revision).filter(|path| {
                                snapshot_matches_request(path, ignore_weights, matches.as_deref())
                                    .unwrap_or(false)
                            })
                        })
                        .await?
                    }
                }
            } else {
                None
            };
            let path = match shared {
                Some(path) => path,
                None => client.get_model_path(model_name, provider).await?,
            };
            if huggingface && (is_upgrade_snapshot(&path) || filter.is_some()) {
                let repo = cache_dir.join(hf_hub::Repo::model(model_name.to_owned()).folder_name());
                recover_published_snapshot(path, Some(repo), ignore_weights, filter.clone(), false)
                    .await?
            } else {
                path
            }
        };
        Ok::<_, anyhow::Error>((staging, path, download.resolved_revision))
    };
    let error = match server.await {
        Err(error) => error,
        Ok((None, path, _)) => return Ok(path),
        Ok((Some(staging), snapshot, revision)) => {
            let repo = hf_hub::Repo::model(model_name.to_owned()).folder_name();
            let cache = cache_dir.clone();
            let lock = lock_hf_repo(&cache, &repo).await?;
            let matches = filter.clone();
            let result = tokio::task::spawn_blocking(move || {
                let _lock = lock;
                let result = publish_hf_snapshot(
                    &staging,
                    &cache,
                    &repo,
                    &snapshot,
                    revision.as_deref(),
                    ignore_weights,
                    matches.as_deref(),
                );
                close_staging(staging);
                result
            })
            .await?;
            match result {
                Err(error)
                    if error.is::<SnapshotInstallFailed>()
                        || error.is::<SnapshotChecksumMismatch>() =>
                {
                    error
                }
                result => return result,
            }
        }
    };
    tracing::warn!(%model_name, %error, "ModelExpress fetch failed, falling back to direct download");
    mx_download_direct(model_name, provider, cache_dir, ignore_weights, stream_hf, filter).await
        .map_err(|direct| anyhow::anyhow!(
            "ModelExpress fetch failed for '{model_name}': {error:#}. Direct fallback failed: {direct:#}"
        ))
}

#[derive(Debug, thiserror::Error)]
#[error("Failed to install the staged Hugging Face snapshot")]
struct SnapshotInstallFailed;

#[derive(Debug, thiserror::Error)]
#[error("Streamed Hugging Face snapshot does not match the model card checksums")]
struct SnapshotChecksumMismatch;

fn require_stream_weights(snapshot: &Path, ignore_weights: bool) -> anyhow::Result<()> {
    anyhow::ensure!(
        ignore_weights || stream_weights_present(snapshot, false)?,
        "Hugging Face snapshot {} is incomplete for the requested download",
        snapshot.display()
    );
    Ok(())
}

// The direct provider writes live files. Serialize those writes with the stream's copy and swap.
async fn lock_hf_repo(cache: &Path, repo: &str) -> anyhow::Result<fs::File> {
    let repo_dir = cache.join(repo);
    // Eviction removes the repository directory, so keep the lock outside it.
    let locks = cache.join(".locks");
    let lock_path = locks.join(format!("{repo}.dynamo-stream.lock"));
    let cache = cache.to_path_buf();
    let path_for_error = lock_path.clone();
    let lock = tokio::task::spawn_blocking(move || {
        ensure_cache_directory(&locks)?;
        let lock = open_hf_lock(&lock_path);
        #[cfg(target_os = "linux")]
        let lock = lock.or_else(|error| {
            use rustix::fs::{Access, AtFlags, CWD, accessat};
            if error.kind() == ErrorKind::PermissionDenied
                && accessat(CWD, &cache, Access::WRITE_OK, AtFlags::EACCESS).is_ok()
            {
                // Local flock can lock the same inode through a read-only descriptor.
                fs::File::open(&lock_path).or(Err(error))
            } else {
                Err(error)
            }
        });
        lock.with_context(|| format!("Failed to open Hugging Face lock {}", lock_path.display()))
    })
    .await??;
    // A direct download needs the Tokio blocking pool for file I/O while it owns the lock.
    loop {
        match lock.try_lock() {
            Ok(()) => {
                return tokio::task::spawn_blocking(move || {
                    ensure_cache_directory(&repo_dir)?;
                    Ok(lock)
                })
                .await?;
            }
            Err(fs::TryLockError::WouldBlock) => {
                tokio::time::sleep(std::time::Duration::from_millis(50)).await
            }
            Err(fs::TryLockError::Error(error)) => {
                #[cfg(target_os = "linux")]
                if error.raw_os_error() == Some(rustix::io::Errno::BADF.raw_os_error()) {
                    return Err(error).with_context(|| {
                        format!(
                            "Hugging Face lock {} requires write access for each cache writer on this filesystem",
                            path_for_error.display()
                        )
                    });
                }
                return Err(error).with_context(|| {
                    format!(
                        "Failed to lock Hugging Face repository {repo} at {}",
                        path_for_error.display()
                    )
                });
            }
        }
    }
}

fn open_hf_lock(path: &Path) -> std::io::Result<fs::File> {
    let mut options = fs::File::options();
    options.write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt as _;

        let locks = path.parent().expect("cache lock has a parent");
        match options.open(path) {
            // A group-writable setgid directory explicitly designates the lock's writer group.
            Err(error)
                if error.kind() == ErrorKind::NotFound
                    && locks.metadata()?.permissions().mode() & 0o2020 == 0o2020 =>
            {
                // Set permissions before linking the file so all writers see the same inode
                // with a writable descriptor available as soon as the lock name exists.
                let staged = tempfile::NamedTempFile::new_in(locks)?;
                staged
                    .as_file()
                    .set_permissions(fs::Permissions::from_mode(0o660))?;
                match fs::hard_link(staged.path(), path) {
                    Ok(()) => return Ok(staged.into_file()),
                    Err(error) if error.kind() == ErrorKind::AlreadyExists => {
                        return options.open(path);
                    }
                    Err(error) if error.kind() == ErrorKind::Unsupported => {}
                    Err(error) => return Err(error),
                }
            }
            Err(error) if error.kind() == ErrorKind::NotFound => {}
            result => return result,
        }
    }
    options.create(true).truncate(false).open(path)
}

const UPGRADE_PREFIX: &str = ".dynamo-upgrade-";

fn is_upgrade_snapshot(path: &Path) -> bool {
    path.file_name().is_some_and(|name| {
        name.as_encoded_bytes()
            .starts_with(UPGRADE_PREFIX.as_bytes())
    })
}

fn revision_snapshot(snapshots: &Path, revision: &str) -> Option<PathBuf> {
    if Path::new(revision).file_name()?.to_str()? != revision
        || revision.starts_with(UPGRADE_PREFIX)
    {
        return None;
    }
    let path = snapshots.join(revision);
    path.is_dir().then_some(path)
}

fn cached_stream_snapshot(repo: &Path, ignore_weights: bool) -> anyhow::Result<Option<PathBuf>> {
    let revision = match fs::read_to_string(repo.join("refs/main")) {
        Ok(revision) => revision,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error).context("Failed to read Hugging Face refs/main"),
    };
    let Some(path) = revision_snapshot(&repo.join("snapshots"), revision.trim()) else {
        return Ok(None);
    };
    Ok(stream_model_ready(&path, ignore_weights)?.then_some(path))
}

// A metadata-only update can install a new revision while refs/main still serves full weights.
fn matching_stream_snapshot(
    repo: &Path,
    ignore_weights: bool,
    filter: Option<&SnapshotFilter>,
) -> anyhow::Result<Option<PathBuf>> {
    let preferred = cached_stream_snapshot(repo, ignore_weights);
    let Some(matches) = filter else {
        return preferred;
    };
    let preferred = preferred.ok().flatten();
    if preferred.as_deref().is_some_and(matches) {
        return Ok(preferred);
    }
    let snapshots = match fs::read_dir(repo.join("snapshots")) {
        Ok(entries) => entries,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error.into()),
    };
    let mut candidates = Vec::new();
    for entry in snapshots {
        let entry = entry?;
        let path = entry.path();
        if !is_upgrade_snapshot(&path)
            && entry.file_type()?.is_dir()
            && Some(&path) != preferred.as_ref()
        {
            candidates.push(path);
        }
    }
    candidates.sort_unstable();
    Ok(candidates.into_iter().find(|path| matches(path)))
}

async fn recover_published_snapshot(
    snapshot: PathBuf,
    hf_repo: Option<PathBuf>,
    ignore_weights: bool,
    filter: Option<std::sync::Arc<SnapshotFilter>>,
    shared_offline: bool,
) -> anyhow::Result<PathBuf> {
    tokio::task::spawn_blocking(move || {
        let filtered = filter.is_some() || shared_offline;
        let ready = move |path: &Path| {
            if shared_offline {
                filter.as_deref().is_none_or(|matches| matches(path))
                    && shared_offline_snapshot_ready(path, filter.is_none()).unwrap_or(false)
            } else {
                snapshot_matches_request(path, ignore_weights, filter.as_deref()).unwrap_or(false)
            }
        };
        if !is_upgrade_snapshot(&snapshot) && ready(&snapshot) {
            return Ok(snapshot);
        }
        if let Some(repo) = hf_repo {
            // Prefer ModelExpress's revision across shared cache roots before other revisions.
            if let Some(revision) = snapshot.file_name().and_then(|name| name.to_str())
                && let Some(path) =
                    revision_snapshot(&repo.join("snapshots"), revision).filter(|path| ready(path))
            {
                return Ok(path);
            }
            if let Some(path) = matching_stream_snapshot(
                &repo,
                ignore_weights,
                filtered.then_some(&ready as &SnapshotFilter),
            )
            .ok()
            .flatten()
            {
                return Ok(path);
            }
        }

        // The SDK can pick a private upgrade when it falls back to the newest directory.
        let repo = snapshot
            .parent()
            .and_then(Path::parent)
            .context("Missing HF repository")?;
        if shared_offline
            && let Ok(revision) = fs::read_to_string(repo.join("refs/main"))
            && let Some(path) = revision_snapshot(&repo.join("snapshots"), revision.trim())
                .filter(|path| ready(path))
        {
            return Ok(path);
        }
        matching_stream_snapshot(repo, ignore_weights, Some(&ready))?.with_context(|| {
            format!(
                "No published HF snapshot available beside {}",
                snapshot.display()
            )
        })
    })
    .await?
}

fn publish_hf_snapshot(
    staging: &tempfile::TempDir,
    cache: &Path,
    repo: &str,
    snapshot: &Path,
    resolved_revision: Option<&str>,
    ignore_weights: bool,
    filter: Option<&SnapshotFilter>,
) -> anyhow::Result<PathBuf> {
    let revision = snapshot
        .file_name()
        .and_then(|name| name.to_str())
        .context("Missing revision in streamed Hugging Face snapshot")?;
    anyhow::ensure!(
        snapshot == staging.path().join(repo).join("snapshots").join(revision)
            && !is_upgrade_snapshot(snapshot)
            && resolved_revision.is_none_or(|resolved| resolved == revision),
        "Unexpected streamed Hugging Face snapshot {}",
        snapshot.display()
    );
    let repo = cache.join(repo);
    let snapshots = repo.join("snapshots");
    let destination = snapshots.join(revision);
    if !ignore_weights {
        discard_corrupt_safetensors(snapshot)?;
    }
    let ready = |path: &Path| snapshot_matches_request(path, ignore_weights, filter);
    let staged_ready = ready(snapshot)?;
    if !staged_ready
        && filter.is_none()
        && (ignore_weights
            || resolved_revision != Some(revision)
            || !destination
                .symlink_metadata()
                .is_ok_and(|meta| meta.is_dir()))
    {
        // Only retry through the direct provider when the requested weights are accounted for.
        require_stream_weights(snapshot, ignore_weights)?;
        return Err(anyhow::anyhow!("Unsupported model metadata")).context(SnapshotInstallFailed);
    }
    ensure_cache_directory(&repo)?;
    ensure_cache_directory(&snapshots)?;
    let refs = repo.join("refs");
    ensure_cache_directory(&refs)?;
    let main_ref = refs.join("main");
    let previous = match fs::read_to_string(&main_ref) {
        Ok(revision) => Some(revision),
        Err(error) if matches!(error.kind(), ErrorKind::NotFound | ErrorKind::InvalidData) => None,
        Err(error) => return Err(error).context("Failed to read Hugging Face refs/main"),
    };
    let keep_ref = match previous.as_deref() {
        Some(current) if current.trim() == revision => true,
        // Artifact-only requests may be cached without becoming the default model snapshot.
        _ if filter.is_some() && !stream_metadata_present(snapshot).unwrap_or(false) => true,
        Some(current) if ignore_weights => match revision_snapshot(&snapshots, current.trim()) {
            Some(path) if filter.is_some() => {
                ignore_invalid_metadata(stream_model_ready(&path, false))?
            }
            // A standalone GGUF may satisfy a full request without providing a tokenizer file.
            Some(path) if ignore_invalid_metadata(stream_metadata_present(&path))? => {
                stream_weights_present(&path, false)?
            }
            _ => false,
        },
        _ => false,
    };
    anyhow::ensure!(
        staged_ready || destination.is_dir(),
        SnapshotChecksumMismatch
    );
    let pending_ref = staging.path().join("main-ref");
    if !keep_ref {
        fs::write(&pending_ref, revision)?;
    }
    let merge = !staged_ready
        || match fs::rename(snapshot, &destination) {
            Ok(()) => false,
            Err(error)
                if matches!(
                    error.kind(),
                    ErrorKind::AlreadyExists | ErrorKind::DirectoryNotEmpty
                ) =>
            {
                true
            }
            Err(error) => {
                let failure = Err(error)
                    .with_context(|| format!("Failed to install {}", destination.display()));
                return match destination.symlink_metadata() {
                    Ok(meta) if meta.is_dir() => failure.context(SnapshotInstallFailed),
                    Err(error) if error.kind() == ErrorKind::NotFound => {
                        failure.context(SnapshotInstallFailed)
                    }
                    _ => failure,
                };
            }
        };
    if merge {
        anyhow::ensure!(
            destination.symlink_metadata()?.is_dir(),
            "Expected a cache directory at {}",
            destination.display()
        );
        // A same-parent exchange works even for a read-only checkout and keeps
        // native HF blob links at the correct relative depth during validation.
        let sibling = tempfile::Builder::new()
            .prefix(UPGRADE_PREFIX)
            .tempdir_in(&snapshots)
            .context(SnapshotInstallFailed)?;
        let staged = sibling.path();
        let result = (|| {
            fs::rename(snapshot, staged).context(SnapshotInstallFailed)?;
            let keep_full =
                ignore_weights && ignore_invalid_metadata(stream_model_ready(&destination, false))?;
            let replace_corrupt_safetensors =
                is_corrupt_safetensors(&destination.join("model.safetensors"))
                    && stream_index_complete(staged, "model.safetensors.index.json")
                        .unwrap_or(false);
            preserve_cached_files(
                &destination,
                staged,
                &repo.join("blobs"),
                replace_corrupt_safetensors,
            )
            .context(SnapshotInstallFailed)?;
            if !ignore_weights {
                discard_corrupt_safetensors(staged)?;
            }
            preserve_directory_permissions(&destination, staged).context(SnapshotInstallFailed)?;
            let staged_ready = ready(staged)?;
            anyhow::ensure!(staged_ready || filter.is_none(), SnapshotChecksumMismatch);
            anyhow::ensure!(
                staged_ready && (!keep_full || stream_model_ready(staged, false)?),
                "Staged Hugging Face snapshot is incomplete after preserving cached files"
            );
            exchange_snapshots(staged, &destination)
                .with_context(|| {
                    format!("Failed to exchange HF snapshot {}", destination.display())
                })
                .context(SnapshotInstallFailed)
        })();
        close_staging(sibling);
        result?;
    }
    if !keep_ref {
        fs::rename(pending_ref, main_ref).context("Failed to publish Hugging Face refs/main")?;
    }
    Ok(destination)
}

// Keep local-only files and native Hub blob links at the same relative paths.
// Hard links let a metadata request retain existing weights without copying them.
fn preserve_cached_files(
    old: &Path,
    staged: &Path,
    blobs: &Path,
    replace_corrupt_safetensors: bool,
) -> anyhow::Result<()> {
    for entry in fs::read_dir(old)? {
        let entry = entry?;
        let dest = staged.join(entry.file_name());
        let source = entry.path();
        let kind = entry.file_type()?;
        if kind.is_dir() {
            ensure_cache_directory(&dest)?;
            preserve_cached_files(&source, &dest, blobs, false)?;
            preserve_directory_permissions(&source, &dest)?;
        } else if dest.try_exists()? {
            anyhow::ensure!(
                dest.symlink_metadata()?.is_file(),
                "Conflicting staged file {}",
                dest.display()
            );
            // Keep access restrictions on a file replaced by the complete stream.
            match source.metadata() {
                Ok(meta) if meta.is_file() => {
                    let permissions = meta.permissions();
                    #[cfg(unix)]
                    let permissions = {
                        use std::os::unix::fs::PermissionsExt as _;
                        fs::Permissions::from_mode(permissions.mode() | 0o400)
                    };
                    fs::set_permissions(&dest, permissions)?;
                }
                Err(error) if error.kind() == ErrorKind::NotFound => {}
                Ok(_) => anyhow::bail!("Conflicting cache entry {}", source.display()),
                Err(error) => return Err(error.into()),
            }
        } else if replace_corrupt_safetensors && entry.file_name() == "model.safetensors" {
            // Loaders can prefer the corrupt file over the complete staged index.
            continue;
        } else if let Err(error) = fs::hard_link(&source, &dest) {
            if error.kind() != ErrorKind::PermissionDenied || !(kind.is_file() || kind.is_symlink())
            {
                return Err(error)
                    .with_context(|| format!("Failed to preserve {}", source.display()));
            }
            let source = if kind.is_symlink() {
                hf_blob_target(&source, &blobs.canonicalize()?)?
            } else {
                source
            };
            fs::copy(&source, &dest)
                .with_context(|| format!("Failed to copy cached file {}", source.display()))?;
            #[cfg(unix)]
            {
                use std::os::unix::fs::PermissionsExt as _;
                let mut permissions = dest.metadata()?.permissions();
                if permissions.mode() & 0o400 == 0 {
                    permissions.set_mode(permissions.mode() | 0o400);
                    fs::set_permissions(&dest, permissions).with_context(|| {
                        format!(
                            "Failed to add owner-read to copied cache file {}",
                            dest.display()
                        )
                    })?;
                }
            }
        }
    }
    Ok(())
}

fn preserve_directory_permissions(old: &Path, staged: &Path) -> std::io::Result<()> {
    let permissions = old.metadata()?.permissions();
    #[cfg(unix)]
    let permissions = {
        use std::os::unix::fs::PermissionsExt as _;
        fs::Permissions::from_mode(permissions.mode() | 0o500)
    };
    fs::set_permissions(staged, permissions)
}

fn close_staging(staging: tempfile::TempDir) {
    let path = staging.path().to_owned();
    let cleanup = staging.close();
    #[cfg(unix)]
    let cleanup = cleanup.or_else(|error| {
        if error.kind() != ErrorKind::PermissionDenied {
            return Err(error);
        }
        make_staging_dirs_writable(&path)?;
        fs::remove_dir_all(&path)
    });
    if let Err(error) = cleanup {
        tracing::warn!(?path, %error, "Failed to remove ModelExpress staging directory");
    }
}

// An exchanged checkout can retain read-only directories inside private staging.
#[cfg(unix)]
fn make_staging_dirs_writable(root: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::{MetadataExt, PermissionsExt};
    let owner = root.symlink_metadata()?.uid();
    let mut pending = vec![root.to_owned()];
    while let Some(dir) = pending.pop() {
        let meta = dir.symlink_metadata()?;
        let mode = meta.permissions().mode();
        if meta.uid() == owner && mode & 0o700 != 0o700 {
            fs::set_permissions(&dir, fs::Permissions::from_mode(mode | 0o700))?;
        }
        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            if entry.file_type()?.is_dir() {
                pending.push(entry.path());
            }
        }
    }
    Ok(())
}

#[cfg(any(target_os = "linux", target_os = "android", target_vendor = "apple"))]
fn exchange_snapshots(staged: &Path, live: &Path) -> std::io::Result<()> {
    use rustix::fs::{CWD, RenameFlags, renameat_with};
    Ok(renameat_with(
        CWD,
        staged,
        CWD,
        live,
        RenameFlags::EXCHANGE,
    )?)
}

#[cfg(not(any(target_os = "linux", target_os = "android", target_vendor = "apple")))]
fn exchange_snapshots(_: &Path, _: &Path) -> std::io::Result<()> {
    Err(std::io::Error::from(ErrorKind::Unsupported))
}

// The SDK can resolve a directory without checking which files it contains.
fn snapshot_matches_request(
    dir: &Path,
    ignore_weights: bool,
    filter: Option<&SnapshotFilter>,
) -> anyhow::Result<bool> {
    if let Some(matches) = filter {
        return Ok(matches(dir) && (ignore_weights || stream_model_ready(dir, false)?));
    }
    stream_model_ready(dir, ignore_weights)
}

fn stream_model_ready(dir: &Path, ignore_weights: bool) -> anyhow::Result<bool> {
    let has_metadata = stream_metadata_present(dir);
    if ignore_weights {
        return has_metadata;
    }
    // A standalone GGUF carries its own metadata, even if separate metadata is unreadable.
    if stream_weights_present(dir, !matches!(has_metadata, Ok(true)))? {
        return Ok(true);
    }
    has_metadata.map(|_| false)
}

fn readable_file_len(path: &Path) -> std::io::Result<Option<u64>> {
    match fs::metadata(path) {
        Ok(meta) if meta.is_file() => {
            let opened = fs::File::open(path)?.metadata()?;
            Ok(opened.is_file().then_some(opened.len()))
        }
        Ok(_) => Ok(None),
        Err(error) if error.kind() == ErrorKind::NotFound => Ok(None),
        Err(error) => Err(error),
    }
}

fn readable_metadata_file(path: &Path) -> std::io::Result<bool> {
    match readable_file_len(path)? {
        Some(0) => Err(std::io::Error::new(
            ErrorKind::InvalidData,
            format!("Empty model metadata file {}", path.display()),
        )),
        size => Ok(size.is_some()),
    }
}

fn ignore_invalid_metadata(result: anyhow::Result<bool>) -> anyhow::Result<bool> {
    match result {
        Err(error)
            if error
                .downcast_ref::<std::io::Error>()
                .is_some_and(|error| error.kind() == ErrorKind::InvalidData) =>
        {
            Ok(false)
        }
        result => result,
    }
}

fn stream_metadata_present(dir: &Path) -> anyhow::Result<bool> {
    let mut first_error = None;
    let mut readable = |path: &Path| match readable_metadata_file(path) {
        Ok(readable) => readable,
        Err(error) => {
            first_error.get_or_insert(error);
            false
        }
    };
    let mut has = |name: &str| readable(&dir.join(name));
    let hf_config = has("config.json");
    if hf_config && hf_tokenizer_file(dir)?.is_some() {
        return Ok(true);
    }
    // Native loaders may use these sidecars when there is no Rust tokenizer.
    let mistral = has("params.json");
    let native = (hf_config
        && (has("tokenizer.model")
            || has("spiece.model")
            || has("spm.model")
            || has("sentencepiece.bpe.model")
            || has("vocab.txt")
            || (has("vocab.json") && has("merges.txt"))))
        || (mistral && (has("tekken.json") || has("tokenizer.model")));
    if native {
        return Ok(true);
    }
    if mistral {
        for entry in fs::read_dir(dir)? {
            let path = entry?.path();
            if mistral_sentencepiece_version(&path).is_some() && readable(&path) {
                return Ok(true);
            }
        }
    }
    first_error.map_or(Ok(false), |error| Err(error.into()))
}

const SHARDED_WEIGHT_NAMES: [&str; 6] = [
    "model.safetensors",
    "consolidated.safetensors",
    "pytorch_model.bin",
    "model.bin",
    "consolidated.pth",
    "model.pth",
];

fn stream_weights_present(dir: &Path, gguf_only: bool) -> anyhow::Result<bool> {
    // The standard standalone file takes precedence over the sharded index.
    if !gguf_only
        && let Some(complete) = safetensors_complete(&dir.join("model.safetensors"), true)?
    {
        return Ok(complete);
    }
    let mut first_error = None;
    let mut nonempty = |name: &str| {
        let file = dir.join(name);
        let result = if name.ends_with(".safetensors") {
            safetensors_complete(&file, true)
        } else {
            readable_file_len(&file).map(|size| size.map(|len| len > 0))
        };
        match result {
            Ok(ready) => ready.unwrap_or(false),
            Err(error) => {
                first_error.get_or_insert_with(|| error.into());
                false
            }
        }
    };
    if !gguf_only
        && SHARDED_WEIGHT_NAMES
            .into_iter()
            .filter(|&name| !matches!(name, "model.bin" | "model.safetensors"))
            .chain([
                "model.tflite",
                "model.pt",
                "model.h5",
                "tf_model.h5",
                "model.msgpack",
                "flax_model.msgpack",
            ])
            .any(&mut nonempty)
    {
        return Ok(true);
    }
    // model.bin can be either standalone weights or an ONNX external data file.
    let mut standalone_bin = !gguf_only && nonempty("model.bin");

    let onnx_dir = if gguf_only {
        None
    } else {
        match fs::read_dir(dir.join("onnx")) {
            Ok(entries) => Some(entries),
            Err(error)
                if matches!(error.kind(), ErrorKind::NotFound | ErrorKind::NotADirectory) =>
            {
                None
            }
            Err(error) => {
                first_error.get_or_insert_with(|| error.into());
                None
            }
        }
    };
    // A bad index for one format can coexist with a complete export in another.
    for entry in fs::read_dir(dir)?.chain(onnx_dir.into_iter().flatten()) {
        let entry = entry?;
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|s| s.to_str()) else {
            continue;
        };
        let at_root = path.parent() == Some(dir);
        let result = match name {
            _ if !gguf_only && name.ends_with(".onnx") && !entry.file_type()?.is_dir() => {
                standalone_bin &= !at_root;
                if !path.is_file() {
                    continue;
                }
                onnx::external_data_complete(&path).map_err(Into::into)
            }
            _ if !at_root => continue,
            _ if !gguf_only
                && name
                    .strip_suffix(".index.json")
                    .is_some_and(|base| SHARDED_WEIGHT_NAMES.contains(&base))
                && path.is_file() =>
            {
                stream_index_complete(dir, name)
            }
            _ if name.ends_with(".gguf") => gguf_weights_complete(dir, name).map_err(Into::into),
            _ if !gguf_only => numbered_weights_complete(dir, name),
            _ => continue,
        };
        match result {
            Ok(true) => return Ok(true),
            Ok(false) => {}
            Err(error) => {
                first_error.get_or_insert(error);
            }
        }
    }
    if standalone_bin {
        return Ok(true);
    }
    first_error.map_or(Ok(false), Err)
}

fn numbered_weights_complete(dir: &Path, name: &str) -> anyhow::Result<bool> {
    let Some((prefix, suffix)) = name.split_once("-00001-of-") else {
        return Ok(false);
    };
    let Some((count, extension)) = suffix.split_once('.') else {
        return Ok(false);
    };
    if count.len() != 5
        || !count.bytes().all(|byte| byte.is_ascii_digit())
        || !SHARDED_WEIGHT_NAMES
            .iter()
            .any(|base| base.rsplit_once('.') == Some((prefix, extension)))
    {
        return Ok(false);
    }

    // The canonical index can name shards with a different prefix (e.g. consolidated).
    for base in SHARDED_WEIGHT_NAMES {
        if base
            .rsplit_once('.')
            .is_some_and(|(_, ext)| ext == extension)
        {
            match fs::metadata(dir.join(format!("{base}.index.json"))) {
                Ok(_) => return Ok(false),
                Err(error) if error.kind() == ErrorKind::NotFound => {}
                Err(error) => return Err(error.into()),
            }
        }
    }
    let total: u32 = count.parse()?;
    let mut has_tensors = extension != "safetensors";
    for part in 1..=total {
        let shard = dir.join(format!("{prefix}-{part:05}-of-{count}.{extension}"));
        let complete = if extension == "safetensors" {
            safetensors_metadata(&shard)?.is_some_and(|(metadata, actual)| {
                has_tensors |= metadata.data_len() != 0 || !metadata.tensors().is_empty();
                metadata.data_len() as u64 == actual
            })
        } else {
            readable_file_len(&shard)?.is_some_and(|len| len != 0)
        };
        if !complete {
            return Ok(false);
        }
    }
    Ok(total > 0 && has_tensors)
}

// A referenced file can also be an independent export. Keep ambiguous legacy checkpoints.
fn shared_sidecar_may_be_checkpoint(path: &Path) -> std::io::Result<bool> {
    let metadata = match fs::metadata(path) {
        Ok(metadata) => metadata,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error),
    };
    if !metadata.is_file()
        || (crate::model_card::is_weight_file(path)
            && path.extension().is_none_or(|extension| extension != "bin"))
        || (path.file_name() == Some("model.bin".as_ref()) && metadata.len() != 0)
    {
        return Ok(true);
    }
    let mut prefix = [0; 4];
    match fs::File::open(path)?.read_exact(&mut prefix) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::UnexpectedEof => {}
        Err(error) => return Err(error),
    }
    // ZIP and binary or legacy-text pickle may be independently loadable checkpoints.
    Ok(prefix.starts_with(b"PK") || matches!(prefix[0], 0x80 | b'L'))
}

fn shared_offline_snapshot_ready(dir: &Path, check_safetensors: bool) -> anyhow::Result<bool> {
    // Unfiltered full requests apply this same check before returning to the caller.
    Ok(shared_offline_weights_ready(dir)? != Some(false)
        && (!check_safetensors || hf_safetensors_ready(dir)?))
}

// Shared SDK caches can contain exports whose layout the stream validator does not support.
fn shared_offline_weights_ready(dir: &Path) -> anyhow::Result<Option<bool>> {
    let strict = stream_weights_present(dir, false);
    if matches!(strict, Ok(true)) {
        return Ok(Some(true));
    }
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(error) if error.kind() == ErrorKind::PermissionDenied => return Ok(None),
        Err(error) if error.kind() == ErrorKind::NotADirectory => return Ok(Some(false)),
        Err(error) => return Err(error.into()),
    };
    let mut files = std::collections::HashSet::new();
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if entry.file_type()?.is_dir() {
            if entry.file_name() != "onnx" {
                return Ok(None);
            }
            let children = match fs::read_dir(&path) {
                Ok(children) => children,
                Err(error) if error.kind() == ErrorKind::PermissionDenied => return Ok(None),
                Err(error) => return Err(error.into()),
            };
            for child in children {
                let child = child?;
                if child.file_type()?.is_dir() {
                    return Ok(None);
                }
                files.insert(child.path());
            }
        } else {
            files.insert(path);
        }
    }
    let onnx_files: Vec<_> = files
        .iter()
        .filter(|path| path.extension().is_some_and(|ext| ext == "onnx"))
        .cloned()
        .collect();
    let mut error = strict.err().filter(|error| {
        error
            .downcast_ref::<std::io::Error>()
            .is_none_or(|error| error.kind() != ErrorKind::UnexpectedEof)
    });
    let mut sidecars = std::collections::HashSet::new();
    for path in onnx_files {
        match onnx::shared_external_data_complete(&path, &mut sidecars) {
            Ok(Some(true)) => return Ok(Some(true)),
            Ok(None) => continue,
            Ok(Some(false)) => {}
            Err(failure) => {
                error.get_or_insert_with(|| failure.into());
            }
        }
        files.remove(&path);
    }
    for path in sidecars {
        if !files.contains(&path) {
            continue;
        }
        match shared_sidecar_may_be_checkpoint(&path) {
            Ok(true) => continue,
            Ok(false) => {}
            Err(failure) => {
                error.get_or_insert_with(|| failure.into());
            }
        }
        files.remove(&path);
    }

    // A shard's filename is not evidence of an independent alternative export.
    let unsupported_shards = |paths: &[PathBuf]| -> anyhow::Result<bool> {
        let mut unsupported = false;
        for path in paths {
            let length = match fs::metadata(path) {
                Ok(metadata) if !metadata.is_file() => return Ok(true),
                Ok(metadata) if metadata.len() < 8 => return Ok(false),
                Ok(metadata) => metadata.len(),
                Err(error) if error.kind() == ErrorKind::NotFound => return Ok(false),
                Err(error) => return Err(error.into()),
            };
            let mut prefix = [0; 8];
            match fs::File::open(path)?.read_exact(&mut prefix) {
                Ok(()) => {}
                Err(error) if error.kind() == ErrorKind::UnexpectedEof => return Ok(false),
                Err(error) => return Err(error.into()),
            }
            if u64::from_le_bytes(prefix) > length - 8 {
                return Ok(false);
            }
            match safetensors_metadata(path) {
                Ok(metadata) => unsupported |= metadata.is_none(),
                Err(error) if error.kind() == ErrorKind::UnexpectedEof => return Ok(false),
                Err(error) => return Err(error.into()),
            }
        }
        Ok(unsupported)
    };
    let mut has_index = false;
    for name in [
        "model.safetensors.index.json",
        "consolidated.safetensors.index.json",
    ] {
        let path = dir.join(name);
        match fs::metadata(&path) {
            Ok(metadata) if !metadata.is_file() => continue,
            Err(failure) if failure.kind() == ErrorKind::NotFound => {
                files.remove(&path);
                continue;
            }
            Err(failure) => {
                error.get_or_insert_with(|| failure.into());
                files.remove(&path);
                continue;
            }
            Ok(_) => {}
        }
        let index = match fs::File::open(&path) {
            Ok(file) => serde_json::from_reader::<_, serde_json::Value>(BufReader::new(file)),
            Err(failure) if failure.kind() == ErrorKind::NotFound => {
                files.remove(&path);
                continue;
            }
            Err(failure) => {
                error.get_or_insert_with(|| failure.into());
                files.remove(&path);
                continue;
            }
        };
        has_index = true;
        let index = match index {
            Ok(index) => index,
            Err(failure) if failure.is_io() => {
                error.get_or_insert_with(|| failure.into());
                files.remove(&path);
                continue;
            }
            Err(_) => continue,
        };
        let Some(shards) = index["weight_map"].as_object() else {
            continue;
        };
        let paths: Option<Vec<_>> = shards
            .values()
            .map(|value| {
                let path = Path::new(value.as_str()?);
                (path.extension().is_some_and(|ext| ext == "safetensors")
                    && path.components().all(|part| {
                        matches!(
                            part,
                            std::path::Component::Normal(_) | std::path::Component::CurDir
                        )
                    }))
                .then(|| dir.join(path))
            })
            .collect();
        let Some(mut paths) = paths else {
            continue;
        };
        let mut seen = std::collections::HashSet::new();
        paths.retain(|path| seen.insert(path.clone()));
        for path in &paths {
            files.remove(path);
        }
        files.remove(&path);
        match unsupported_shards(&paths) {
            Ok(true) => {
                files.insert(path);
            }
            Ok(false) => {}
            Err(failure) => {
                error.get_or_insert(failure);
            }
        }
    }
    let mut numbered = std::collections::HashMap::<_, Vec<PathBuf>>::new();
    for path in files.clone() {
        if path.parent() != Some(dir) {
            continue;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if matches!(name, "model.safetensors" | "consolidated.safetensors") {
            match unsupported_shards(std::slice::from_ref(&path)) {
                Ok(true) => continue,
                Ok(false) => {}
                Err(failure) => {
                    error.get_or_insert(failure);
                }
            }
            files.remove(&path);
            continue;
        }
        let Some((head, total)) = name
            .strip_suffix(".safetensors")
            .and_then(|name| name.rsplit_once("-of-"))
        else {
            continue;
        };
        let Some((prefix, part)) = head.rsplit_once('-') else {
            continue;
        };
        if !matches!(prefix, "model" | "consolidated")
            || part.len() != 5
            || total.len() != 5
            || !part
                .bytes()
                .chain(total.bytes())
                .all(|byte| byte.is_ascii_digit())
            || part.parse::<u32>()? == 0
            || part.parse::<u32>()? > total.parse::<u32>()?
        {
            continue;
        }
        numbered
            .entry((prefix.to_owned(), total.parse::<u32>()?))
            .or_default()
            .push(path.clone());
        files.remove(&path);
    }
    if !has_index {
        for ((_, total), paths) in numbered {
            if paths.len() != total as usize {
                continue;
            }
            match unsupported_shards(&paths) {
                Ok(true) => {
                    files.extend(paths);
                }
                Ok(false) => {}
                Err(failure) => {
                    error.get_or_insert(failure);
                }
            }
        }
    }
    files.retain(|path| {
        !matches!(
            path.file_name().and_then(|name| name.to_str()),
            Some(
                "config.json"
                    | "tokenizer.json"
                    | "tokenizer_config.json"
                    | "generation_config.json"
                    | "special_tokens_map.json"
                    | "README.md"
                    | "LICENSE"
                    | ".gitattributes"
                    | "params.json"
                    | "tekken.json"
                    | "merges.txt"
                    | "vocab.json"
                    | "vocab.txt"
                    | "added_tokens.json"
                    | "tokenizer.model"
                    | "spiece.model"
                    | "spm.model"
                    | "sentencepiece.bpe.model"
                    | "tiktoken.model"
            )
        ) && mistral_sentencepiece_version(path).is_none()
            && path.extension().is_none_or(|ext| ext != "tiktoken")
    });
    if !files.is_empty() {
        return Ok(None);
    }
    error.map_or(Ok(Some(false)), Err)
}

fn hf_safetensors_ready(dir: &Path) -> anyhow::Result<bool> {
    if let Some(complete) = safetensors_complete(&dir.join("model.safetensors"), true)? {
        return Ok(complete
            || (!matches!(stream_metadata_present(dir), Ok(true))
                && stream_weights_present(dir, true)?));
    }
    if dir.join("model.safetensors.index.json").is_file() {
        return stream_weights_present(dir, false);
    }
    Ok(true)
}

// Omit an unusable standalone checkpoint only after validating its replacement index.
// An I/O failure leaves it untouched.
fn is_corrupt_safetensors(path: &Path) -> bool {
    matches!(safetensors_complete(path, true), Ok(Some(false)))
}

fn discard_corrupt_safetensors(dir: &Path) -> anyhow::Result<()> {
    let checkpoint = dir.join("model.safetensors");
    if dir.join("model.safetensors.index.json").is_file()
        && is_corrupt_safetensors(&checkpoint)
        && stream_index_complete(dir, "model.safetensors.index.json")?
    {
        fs::remove_file(checkpoint)?;
    }
    Ok(())
}

// Validate the safetensors header and payload length without reading the tensors.
fn safetensors_complete(path: &Path, require_tensors: bool) -> std::io::Result<Option<bool>> {
    match fs::metadata(path) {
        Ok(meta) if !meta.is_file() => return Ok(None),
        Ok(meta) if meta.len() < 8 => return Ok(Some(false)),
        Ok(_) => {}
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    }
    Ok(Some(safetensors_metadata(path)?.is_some_and(
        |(metadata, actual)| {
            metadata.data_len() as u64 == actual
                && (!require_tensors || metadata.data_len() != 0 || !metadata.tensors().is_empty())
        },
    )))
}

fn safetensors_metadata(
    path: &Path,
) -> std::io::Result<Option<(safetensors::tensor::Metadata, u64)>> {
    match fs::metadata(path) {
        Ok(meta) if !meta.is_file() || meta.len() < 8 => return Ok(None),
        Ok(_) => {}
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
    }
    let mut file = fs::File::open(path)?;
    let mut prefix = [0; 8];
    file.read_exact(&mut prefix)?;
    let header_len = u64::from_le_bytes(prefix);
    // The safetensors format limits the JSON header to 100 MB.
    if header_len > 100_000_000 {
        return Ok(None);
    }
    let Some(data_len) = file.metadata()?.len().checked_sub(8 + header_len) else {
        return Ok(None);
    };
    let mut header = vec![0; header_len as usize];
    file.read_exact(&mut header)?;
    if !header.starts_with(b"{") {
        return Ok(None);
    }
    Ok(
        serde_json::from_slice::<safetensors::tensor::Metadata>(&header)
            .ok()
            .map(|metadata| (metadata, data_len)),
    )
}

fn stream_index_complete(dir: &Path, name: &str) -> anyhow::Result<bool> {
    let file = match fs::File::open(dir.join(name)) {
        Ok(file) => file,
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error.into()),
    };
    let index = match serde_json::from_reader::<_, serde_json::Value>(BufReader::new(file)) {
        Ok(index) => index,
        Err(error) if error.is_io() => return Err(error.into()),
        Err(_) => return Ok(false),
    };
    let Some(shards) = index["weight_map"].as_object() else {
        return Ok(false);
    };
    let extension = name
        .strip_suffix(".index.json")
        .and_then(|file| Path::new(file).extension())
        .context("Missing weight format in index filename")?;
    let mut checked = std::collections::HashMap::new();
    for (tensor, shard) in shards {
        let Some(shard) = shard.as_str() else {
            return Ok(false);
        };
        let path = Path::new(shard);
        if path.extension() != Some(extension)
            || !path.components().all(|c| {
                matches!(
                    c,
                    std::path::Component::CurDir | std::path::Component::Normal(_)
                )
            })
        {
            return Ok(false);
        }
        let metadata = match checked.entry(path) {
            std::collections::hash_map::Entry::Occupied(entry) => entry.into_mut(),
            std::collections::hash_map::Entry::Vacant(entry) => {
                let file = dir.join(path);
                let metadata = if extension == "safetensors" {
                    let Some((metadata, actual)) = safetensors_metadata(&file)? else {
                        return Ok(false);
                    };
                    if metadata.data_len() as u64 != actual {
                        return Ok(false);
                    }
                    Some(metadata)
                } else {
                    if readable_file_len(&file)?.is_none_or(|len| len == 0) {
                        return Ok(false);
                    }
                    None
                };
                entry.insert(metadata)
            }
        };
        if metadata
            .as_ref()
            .is_some_and(|metadata| metadata.info(tensor).is_none())
        {
            return Ok(false);
        }
    }
    Ok(!checked.is_empty())
}

fn gguf_weights_complete(dir: &Path, name: &str) -> std::io::Result<bool> {
    let Some(stem) = name.strip_suffix(".gguf") else {
        return Ok(false);
    };
    let digits = |s: &str| !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit());
    let split = stem.rsplit_once("-of-").and_then(|(head, count)| {
        let (base, part) = head.rsplit_once('-')?;
        (digits(part) && digits(count)).then_some((base, part, count))
    });
    let base = split.map_or(stem, |(base, _, _)| base);
    if base.is_empty()
        || matches!(base.split('-').next(), Some("mmproj" | "mtp"))
        || base
            .rsplit('-')
            .next()
            .is_some_and(|suffix| suffix.eq_ignore_ascii_case("lora"))
    {
        return Ok(false);
    }
    if let Some((_, part, count)) = split {
        if part != "00001" || count.len() != 5 {
            return Ok(false);
        }
        let Ok(parts) = count.parse::<u16>() else {
            return Ok(false);
        };
        if parts != 1 {
            return split_gguf_complete(dir, base, parts);
        }
    }
    let path = dir.join(name);
    let Some(reader) = open_model_gguf(&path)? else {
        return Ok(false);
    };
    let split_count = reader.metadata().get("split.count");
    if split.is_some() && split_count.is_some_and(|count| count.as_u64() == Some(1)) {
        return split_gguf_complete(dir, base, 1);
    }
    Ok(!reader.tensor_infos().is_empty()
        && split_count.is_none_or(|count| matches!(count.as_u64(), Some(0 | 1))))
}

fn split_gguf_complete(dir: &Path, base: &str, parts: u16) -> std::io::Result<bool> {
    use gguf_rs_lib::format::metadata::MetadataValue::{I32, U16};
    let mut expected_tensors = 0;
    let mut names = std::collections::HashSet::new();
    for index in 0..parts {
        let path = dir.join(format!("{base}-{:05}-of-{parts:05}.gguf", index + 1));
        let Some(reader) = open_model_gguf(&path)? else {
            return Ok(false);
        };
        let metadata = reader.metadata();
        let Some(I32(total)) = metadata.get("split.tensors.count") else {
            return Ok(false);
        };
        if index == 0 {
            expected_tensors = *total;
        }
        if *total <= 0
            || *total != expected_tensors
            || metadata.get("split.no") != Some(&U16(index))
            || metadata.get("split.count") != Some(&U16(parts))
        {
            return Ok(false);
        }
        if reader
            .tensor_infos()
            .iter()
            .any(|tensor| !names.insert(tensor.name().to_owned()))
        {
            return Ok(false);
        }
    }
    Ok(expected_tensors > 0 && usize::try_from(expected_tensors).ok() == Some(names.len()))
}

fn open_model_gguf(path: &Path) -> std::io::Result<Option<GGUFStreamReader<impl Read>>> {
    match fs::metadata(path) {
        Ok(meta) if !meta.is_file() => return Ok(None),
        Err(error) if error.kind() == ErrorKind::NotFound => return Ok(None),
        Err(error) => return Err(error),
        Ok(_) => {}
    }
    let file = fs::File::open(path)?;
    let len = file.metadata()?.len();
    let mut input = BufReader::new(file);
    let mut prefix = [0; 8];
    match input.read_exact(&mut prefix) {
        Ok(()) => {}
        Err(error) if error.kind() == ErrorKind::UnexpectedEof => return Ok(None),
        Err(error) => return Err(error),
    }
    // GGUF v3 added big-endian support. The little-endian v2 body is identical.
    if prefix == *b"GGUF\x02\0\0\0" {
        prefix[4] = 3;
    }
    let reader = match GGUFStreamReader::new(std::io::Cursor::new(prefix).chain(input)) {
        Ok(reader) => reader,
        Err(GGUFError::Io(error)) if error.kind() != ErrorKind::UnexpectedEof => return Err(error),
        Err(_) => return Ok(None),
    };
    let layout = reader.summary();
    if reader
        .metadata()
        .get("general.type")
        .is_some_and(|kind| kind.as_str() != Some("model"))
        || layout
            .tensor_data_offset
            .checked_add(layout.tensor_data_span)
            .is_none_or(|end| end > len)
    {
        return Ok(None);
    }
    Ok(Some(reader))
}

fn ensure_cache_directory(path: &Path) -> anyhow::Result<()> {
    match fs::create_dir(path) {
        Ok(()) => Ok(()),
        Err(error)
            if error.kind() == ErrorKind::AlreadyExists && path.symlink_metadata()?.is_dir() =>
        {
            Ok(())
        }
        Err(error) => Err(error)
            .with_context(|| format!("Failed to create cache directory {}", path.display())),
    }
}

// A reader without write access to the lock can reuse a complete snapshot but cannot download into it.
fn cached_snapshot_after_lock_error(
    repo: &Path,
    ignore_weights: bool,
    error: anyhow::Error,
) -> anyhow::Result<PathBuf> {
    match error
        .downcast_ref::<std::io::Error>()
        .map(std::io::Error::kind)
    {
        Some(ErrorKind::PermissionDenied | ErrorKind::ReadOnlyFilesystem) => {
            cached_stream_snapshot(repo, ignore_weights)?.ok_or(error)
        }
        _ => Err(error),
    }
}

fn hf_blob_target(path: &Path, blobs: &Path) -> anyhow::Result<PathBuf> {
    let target = path.canonicalize()?;
    anyhow::ensure!(
        target.starts_with(blobs) && target.is_file(),
        "Unexpected Hugging Face blob link {}",
        path.display()
    );
    Ok(target)
}

// Hub snapshots may point back to blobs in the private download cache.
fn materialize_hf_blobs(dir: &Path, blobs: &Path) -> anyhow::Result<()> {
    for entry in fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        let kind = entry.file_type()?;
        if kind.is_dir() {
            materialize_hf_blobs(&path, blobs)?;
        } else if kind.is_symlink() {
            let target = hf_blob_target(&path, blobs)?;
            fs::remove_file(&path)?;
            fs::hard_link(target, path)?;
        }
    }
    Ok(())
}

async fn mx_download_direct(
    model_name: &str,
    provider: MxModelProvider,
    cache_dir: PathBuf,
    ignore_weights: bool,
    stream_hf: bool,
    filter: Option<std::sync::Arc<SnapshotFilter>>,
) -> anyhow::Result<PathBuf> {
    let offline = mx_envs::hf_offline();
    let _lock = if provider == MxModelProvider::HuggingFace && !offline {
        tokio::fs::create_dir_all(&cache_dir).await?;
        let repo = hf_hub::Repo::model(model_name.to_owned()).folder_name();
        let lock = match lock_hf_repo(&cache_dir, &repo).await {
            Ok(lock) => lock,
            Err(error) => {
                let repo = cache_dir.join(repo);
                return tokio::task::spawn_blocking(move || {
                    cached_snapshot_after_lock_error(&repo, ignore_weights, error)
                })
                .await?;
            }
        };
        if let Some(matches) = filter {
            // The provider writes refs/main while downloading. Verify a private checkout first.
            let staging = tempfile::Builder::new()
                .prefix(".mx-")
                .tempdir_in(&cache_dir)?;
            let download = mx::download_model_revision(
                model_name,
                provider,
                Some(staging.path().to_owned()),
                ignore_weights,
                None,
            )
            .await?;
            return tokio::task::spawn_blocking(move || {
                let _lock = lock;
                let result = (|| {
                    let staged_repo = staging.path().join(&repo);
                    let revision = download
                        .resolved_revision
                        .as_deref()
                        .context("Missing Hugging Face revision")?;
                    let snapshot = revision_snapshot(&staged_repo.join("snapshots"), revision)
                        .filter(|snapshot| *snapshot == download.path)
                        .context("Unexpected downloaded Hugging Face snapshot")?;
                    let blobs = staging.path().canonicalize()?.join(&repo).join("blobs");
                    materialize_hf_blobs(&snapshot, &blobs)?;
                    publish_hf_snapshot(
                        &staging,
                        &cache_dir,
                        &repo,
                        &snapshot,
                        Some(revision),
                        ignore_weights,
                        Some(matches.as_ref()),
                    )
                })();
                close_staging(staging);
                result
            })
            .await?;
        }
        Some(lock)
    } else {
        None
    };
    let path = mx::download_model(model_name, provider, Some(cache_dir), ignore_weights).await?;
    let shared_offline = offline
        && !stream_hf
        && !ignore_weights
        && MxModelProvider::resolve_provider_for_model_name(model_name, provider)
            == MxModelProvider::HuggingFace;
    if provider == MxModelProvider::HuggingFace && is_upgrade_snapshot(&path) {
        return recover_published_snapshot(path, None, ignore_weights, filter, shared_offline)
            .await;
    }
    if shared_offline || (stream_hf && offline && (filter.is_none() || !ignore_weights)) {
        // Checked metadata requests verify their own artifacts in from_hf_filtered.
        let check_safetensors = filter.is_none();
        let (path, ready) = tokio::task::spawn_blocking(move || {
            let ready = if shared_offline {
                shared_offline_snapshot_ready(&path, check_safetensors)
            } else {
                Ok(stream_model_ready(&path, ignore_weights)?
                    || (!ignore_weights && stream_weights_present(&path, false)?))
            };
            Ok::<_, anyhow::Error>((path, ready))
        })
        .await??;
        match ready {
            Ok(true) => return Ok(path),
            Err(error) if shared_offline => {
                return recover_published_snapshot(path, None, ignore_weights, filter, true)
                    .await
                    .or(Err(error));
            }
            Err(error) => return Err(error),
            Ok(false) => {}
        }
        let message = format!(
            "Hugging Face snapshot {} is incomplete for the requested download",
            path.display()
        );
        return recover_published_snapshot(path, None, ignore_weights, filter, shared_offline)
            .await
            .context(message);
    }
    Ok(path)
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

    #[test]
    fn shared_offline_weights_keep_opaque_exports() -> anyhow::Result<()> {
        let dir = TempDir::new()?;
        for name in ["README.md", "LICENSE", ".gitattributes"] {
            fs::write(dir.path().join(name), b"repository metadata")?;
        }
        assert_eq!(shared_offline_weights_ready(dir.path())?, Some(false));
        let first = dir.path().join("consolidated-00001-of-00002.safetensors");
        let second = dir.path().join("consolidated-00002-of-00002.safetensors");
        fs::write(
            &first,
            &SAFETENSORS_FIXTURE[..SAFETENSORS_FIXTURE.len() - 1],
        )?;
        fs::write(&second, ZERO_SAFETENSORS_FIXTURE)?;
        assert_eq!(shared_offline_weights_ready(dir.path())?, Some(false));
        let legacy = dir.path().join("legacy.safetensors");
        fs::write(&legacy, SAFETENSORS_FIXTURE)?;
        assert_eq!(shared_offline_weights_ready(dir.path())?, None);
        fs::remove_file(legacy)?;
        fs::write(&first, &SAFETENSORS_FIXTURE[..10])?;
        assert_eq!(shared_offline_weights_ready(dir.path())?, Some(false));
        fs::write(&first, EMPTY_SAFETENSORS_FIXTURE)?;
        assert_eq!(shared_offline_weights_ready(dir.path())?, Some(true));
        Ok(())
    }

    #[test]
    fn shared_offline_index_keeps_unsupported_references() -> anyhow::Result<()> {
        let dir = TempDir::new()?;
        for name in [
            "model.safetensors.index.json",
            "consolidated.safetensors.index.json",
        ] {
            let index = dir.path().join(name);
            for shard in [
                serde_json::json!("../weights.safetensors"),
                serde_json::json!("/weights.safetensors"),
                serde_json::json!("weights.bin"),
                serde_json::json!(null),
            ] {
                fs::write(
                    &index,
                    serde_json::to_vec(&serde_json::json!({
                        "weight_map": {"w": shard}
                    }))?,
                )?;
                assert_eq!(shared_offline_weights_ready(dir.path())?, None);
            }
            fs::write(&index, br#"{"weight_map":{"w":"missing.safetensors"}}"#)?;
            assert_eq!(shared_offline_weights_ready(dir.path())?, Some(false));
            fs::remove_file(index)?;
        }
        Ok(())
    }

    #[tokio::test]
    async fn shared_offline_recovery_keeps_opaque_refs_and_excludes_private() -> anyhow::Result<()>
    {
        let cache = TempDir::new()?;
        let repo = cache.path().join(STREAM_REPO);
        let snapshots = repo.join("snapshots");
        let bad = snapshots.join("aaa");
        let legacy = snapshots.join("bbb");
        let full = snapshots.join("ccc");
        let private = snapshots.join(format!("{UPGRADE_PREFIX}test"));
        fs::create_dir_all(&bad)?;
        fs::create_dir_all(&legacy)?;
        fs::create_dir_all(&private)?;
        fs::create_dir(repo.join("refs"))?;
        fs::write(repo.join("refs/main"), "aaa")?;
        fs::write(legacy.join("legacy.safetensors"), SAFETENSORS_FIXTURE)?;
        fs::write(private.join("model.safetensors"), SAFETENSORS_FIXTURE)?;
        assert_eq!(
            recover_published_snapshot(bad.clone(), None, false, None, true).await?,
            legacy
        );
        assert_eq!(
            recover_published_snapshot(private.clone(), None, false, None, true).await?,
            legacy
        );
        fs::create_dir(&full)?;
        fs::write(full.join("model.safetensors"), SAFETENSORS_FIXTURE)?;

        let ordinary_file = snapshots.join("ddd");
        fs::write(&ordinary_file, b"not a directory")?;
        assert!(!shared_offline_snapshot_ready(&ordinary_file, true)?);
        fs::write(bad.join("config.json"), b"{}")?;
        fs::write(bad.join("tokenizer.json"), b"{}")?;
        fs::write(bad.join("model.safetensors"), b"truncated")?;
        fs::write(bad.join("legacy.safetensors"), SAFETENSORS_FIXTURE)?;
        assert!(shared_offline_snapshot_ready(&bad, false)?);
        assert_eq!(
            recover_published_snapshot(bad.clone(), None, false, None, true).await?,
            legacy
        );

        fs::write(repo.join("refs/main"), "bbb")?;
        assert_eq!(
            recover_published_snapshot(bad.clone(), None, false, None, true).await?,
            legacy
        );
        fs::remove_dir_all(legacy)?;
        fs::remove_dir_all(full)?;
        fs::write(repo.join("refs/main"), "aaa")?;
        assert!(
            recover_published_snapshot(bad, None, false, None, true)
                .await
                .is_err()
        );
        Ok(())
    }

    #[cfg(unix)]
    #[serial_test::serial]
    #[test]
    fn shared_offline_recovers_before_returning_inspection_error() -> anyhow::Result<()> {
        let cache = TempDir::new()?;
        let old = build_hf_cache(cache.path(), "org/model", &["model.safetensors"]);
        let repo = cache.path().join(STREAM_REPO);
        fs::remove_file(repo.join("refs/main"))?;
        fs::File::open(&old)?.set_modified(std::time::SystemTime::UNIX_EPOCH)?;
        let bad = repo.join("snapshots/newest");
        fs::create_dir(&bad)?;
        std::os::unix::fs::symlink(
            "consolidated.safetensors",
            bad.join("consolidated.safetensors"),
        )?;
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()?
            .block_on(temp_env::async_with_vars(
                [("HF_HUB_OFFLINE", Some("1"))],
                async {
                    let fetch = || {
                        mx_download_direct(
                            "org/model",
                            MxModelProvider::HuggingFace,
                            cache.path().to_owned(),
                            false,
                            false,
                            None,
                        )
                    };
                    assert_eq!(fetch().await?, old);
                    fs::remove_dir_all(&old)?;
                    let error = fetch().await.unwrap_err();
                    assert_eq!(
                        error
                            .downcast_ref::<std::io::Error>()
                            .unwrap()
                            .raw_os_error(),
                        Some(libc::ELOOP),
                    );
                    Ok::<_, anyhow::Error>(())
                },
            ))
    }

    #[cfg(unix)]
    #[test]
    fn shared_offline_index_handles_non_regular_files() -> anyhow::Result<()> {
        let dir = TempDir::new()?;
        let _socket = std::os::unix::net::UnixListener::bind(
            dir.path().join("model.safetensors.index.json"),
        )?;
        assert_eq!(shared_offline_weights_ready(dir.path())?, None);
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn shared_offline_weights_preserve_known_io_errors() -> anyhow::Result<()> {
        let dir = TempDir::new()?;
        let broken = dir.path().join("consolidated.safetensors");
        std::os::unix::fs::symlink("consolidated.safetensors", &broken)?;
        let error = shared_offline_weights_ready(dir.path()).unwrap_err();
        assert_eq!(
            error
                .downcast_ref::<std::io::Error>()
                .unwrap()
                .raw_os_error(),
            Some(libc::ELOOP)
        );
        fs::write(dir.path().join("legacy.safetensors"), SAFETENSORS_FIXTURE)?;
        assert_eq!(shared_offline_weights_ready(dir.path())?, None);
        Ok(())
    }

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
            let bytes = if f.ends_with(".safetensors") {
                SAFETENSORS_FIXTURE
            } else {
                b"{}"
            };
            fs::write(snapshot_dir.join(f), bytes).unwrap();
        }
        snapshot_dir
    }

    // One F32 tensor named "w" with four bytes of payload.
    const SAFETENSORS_FIXTURE: &[u8] = b"\x36\0\0\0\0\0\0\0{\"w\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}\0\0\x80?";
    const SAFETENSORS_V_FIXTURE: &[u8] = b"\x36\0\0\0\0\0\0\0{\"v\":{\"dtype\":\"F32\",\"shape\":[1],\"data_offsets\":[0,4]}}\0\0\x80?";
    const EMPTY_SAFETENSORS_FIXTURE: &[u8] = b"\x02\0\0\0\0\0\0\0{}";
    const ZERO_SAFETENSORS_FIXTURE: &[u8] =
        b"\x36\0\0\0\0\0\0\0{\"w\":{\"dtype\":\"F32\",\"shape\":[0],\"data_offsets\":[0,0]}}";
    const STREAM_REPO: &str = "models--org--model";

    fn stage(root: &Path, revision: &str, files: &[(&str, &[u8])]) -> (TempDir, PathBuf) {
        let staging = tempfile::tempdir_in(root).unwrap();
        let snapshot = staging
            .path()
            .join(STREAM_REPO)
            .join("snapshots")
            .join(revision);
        fs::create_dir_all(&snapshot).unwrap();
        for (name, data) in files {
            let path = snapshot.join(name);
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            fs::write(path, data).unwrap();
        }
        (staging, snapshot)
    }

    #[test]
    fn partial_or_mismatched_stream_does_not_publish_a_revision() {
        let cache = TempDir::new().unwrap();
        let (staging, path) = stage(
            cache.path(),
            "aaa",
            &[
                ("config.json", b"{}"),
                ("tokenizer.json", b"{}"),
                (
                    "model.safetensors.index.json",
                    br#"{"weight_map":{"a":"missing.bin"}}"#,
                ),
            ],
        );
        for (resolved, metadata_only) in [(Some("bbb"), true), (Some("aaa"), false)] {
            assert!(
                publish_hf_snapshot(
                    &staging,
                    cache.path(),
                    STREAM_REPO,
                    &path,
                    resolved,
                    metadata_only,
                    None,
                )
                .is_err()
            );
            let repo = cache.path().join(STREAM_REPO);
            assert!(!repo.join("refs/main").exists());
            assert!(!repo.join("snapshots/aaa").exists());
        }
    }

    #[cfg(unix)]
    #[test]
    fn metadata_upgrade_keeps_native_files_and_ref_on_failure() -> anyhow::Result<()> {
        use std::os::unix::fs::{PermissionsExt, symlink};
        let cache = TempDir::new()?;
        let root = cache.path();
        let metadata: &[(&str, &[u8])] = &[("config.json", b"{}"), ("tokenizer.json", b"{}")];
        let (staging, path) = stage(root, "aaa", metadata);
        let live = publish_hf_snapshot(&staging, root, STREAM_REPO, &path, None, true, None)?;
        assert_eq!(
            Cache::new(root.to_owned())
                .model("org/model".into())
                .get("config.json"),
            Some(live.join("config.json"))
        );
        let repo = root.join(STREAM_REPO);
        fs::create_dir(repo.join("blobs"))?;
        fs::write(repo.join("blobs/sidecar"), b"native")?;
        symlink("../../blobs/sidecar", live.join("sidecar"))?;
        let blob = repo.join("blobs/weights");
        fs::write(&blob, b"old weights")?;
        fs::set_permissions(&blob, fs::Permissions::from_mode(0o000))?;
        symlink("../../blobs/weights", live.join("model.safetensors"))?;
        fs::set_permissions(live.join("config.json"), fs::Permissions::from_mode(0o600))?;
        let mut full = metadata.to_vec();
        full.push(("model.safetensors", SAFETENSORS_FIXTURE));
        full.push(("sidecar/conflict", b"conflict"));
        let (staging, path) = stage(root, "aaa", &full);
        assert!(
            publish_hf_snapshot(&staging, root, STREAM_REPO, &path, Some("aaa"), false, None)
                .is_err()
        );
        assert!(live.join("model.safetensors").is_symlink());
        assert_eq!(fs::read_to_string(repo.join("refs/main"))?, "aaa");

        full.pop();
        let nested = live.join("nested");
        fs::create_dir(&nested)?;
        fs::write(nested.join("sidecar"), b"retired contents")?;
        fs::set_permissions(&nested, fs::Permissions::from_mode(0o555))?;
        full.push(("nested/sidecar", b"current contents"));
        fs::set_permissions(&live, fs::Permissions::from_mode(0o555))?;
        let (staging, path) = stage(root, "aaa", &full);
        assert_eq!(
            publish_hf_snapshot(&staging, root, STREAM_REPO, &path, Some("aaa"), false, None)?,
            live
        );
        assert!(live.join("sidecar").is_symlink());
        assert_eq!(fs::read(live.join("sidecar"))?, b"native");
        assert_eq!(
            fs::metadata(live.join("config.json"))?.permissions().mode() & 0o777,
            0o600
        );
        let weight = live.join("model.safetensors");
        assert_eq!(fs::read(&weight)?, SAFETENSORS_FIXTURE);
        assert_eq!(weight.metadata()?.permissions().mode() & 0o777, 0o400);
        assert_eq!(blob.metadata()?.permissions().mode() & 0o777, 0);
        let private = staging.path().to_owned();
        close_staging(staging);
        assert!(!private.exists());
        assert_eq!(fs::read(nested.join("sidecar"))?, b"current contents");
        assert_eq!(nested.metadata()?.permissions().mode() & 0o777, 0o555);
        assert_eq!(live.metadata()?.permissions().mode() & 0o777, 0o555);
        assert_eq!(fs::read_dir(repo.join("snapshots"))?.count(), 1);
        // Let the test's cache TempDir remove its deliberately read-only fixture.
        fs::set_permissions(&live, fs::Permissions::from_mode(0o700))?;
        fs::set_permissions(&nested, fs::Permissions::from_mode(0o700))?;
        Ok(())
    }

    #[test]
    fn metadata_stream_replaces_an_unusable_main_ref() -> anyhow::Result<()> {
        let cache = TempDir::new()?;
        let root = cache.path();
        let old = build_hf_cache(root, "org/model", &[]);
        fs::write(old.join("config.json"), [])?;
        let (staging, snapshot) = stage(
            root,
            "aaa",
            &[("config.json", b"{}"), ("tokenizer.json", b"{}")],
        );
        let new = publish_hf_snapshot(&staging, root, STREAM_REPO, &snapshot, None, true, None)?;
        assert_eq!(
            cached_stream_snapshot(&root.join(STREAM_REPO), true)?,
            Some(new)
        );
        Ok(())
    }

    #[test]
    fn full_stream_repairs_invalid_main_ref() -> anyhow::Result<()> {
        let cache = TempDir::new()?;
        let root = cache.path();
        let repo = root.join(STREAM_REPO);
        fs::create_dir_all(repo.join("refs"))?;
        fs::write(repo.join("refs/main"), [0xff])?;
        let (staging, snapshot) = stage(
            root,
            "aaa",
            &[
                ("config.json", b"{}"),
                ("tokenizer.json", b"{}"),
                ("model.safetensors", SAFETENSORS_FIXTURE),
            ],
        );
        let installed =
            publish_hf_snapshot(&staging, root, STREAM_REPO, &snapshot, None, false, None)?;
        assert_eq!(cached_stream_snapshot(&repo, false)?, Some(installed));
        Ok(())
    }

    #[test]
    fn artifact_only_stream_is_checked_before_publication() -> anyhow::Result<()> {
        let cache = TempDir::new()?;
        let root = cache.path();
        let live = build_hf_cache(
            root,
            "org/model",
            &["config.json", "tokenizer.json", "model.safetensors"],
        );
        let repo = root.join(STREAM_REPO);
        let filename = "generation_config.json";
        let matches =
            move |path: &Path| fs::read(path.join(filename)).is_ok_and(|data| data == b"expected");
        for (contents, accepted) in [(b"rejected".as_slice(), false), (b"expected", true)] {
            let (staging, path) = stage(root, "aaa", &[(filename, contents)]);
            let result = publish_hf_snapshot(
                &staging,
                root,
                STREAM_REPO,
                &path,
                None,
                true,
                Some(&matches),
            );
            if accepted {
                result?;
            } else {
                assert!(result.unwrap_err().is::<SnapshotChecksumMismatch>());
            }
            assert_eq!(repo.join("snapshots/aaa").exists(), accepted);
            assert_eq!(cached_stream_snapshot(&repo, false)?.as_ref(), Some(&live));
        }
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn dangling_weight_index_does_not_hide_complete_numbered_shards() -> anyhow::Result<()> {
        let dir = TempDir::new()?;
        for part in 1..=2 {
            let shard = format!("consolidated-{part:05}-of-00002.safetensors");
            fs::write(dir.path().join(shard), SAFETENSORS_FIXTURE)?;
        }
        std::os::unix::fs::symlink(
            "missing-index",
            dir.path().join("model.safetensors.index.json"),
        )?;
        assert!(stream_weights_present(dir.path(), false)?);
        fs::remove_file(dir.path().join("consolidated-00002-of-00002.safetensors"))?;
        assert!(!stream_weights_present(dir.path(), false)?);
        Ok(())
    }

    #[test]
    fn indexed_stream_replaces_only_corrupt_standalone_safetensors() -> anyhow::Result<()> {
        let metadata: &[(&str, &[u8])] = &[("config.json", b"{}"), ("tokenizer.json", b"{}")];
        let index = "model.safetensors.index.json";
        let checkpoint = "model.safetensors";
        let header = br#"{"w":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
        let mut valid = (header.len() as u64).to_le_bytes().to_vec();
        valid.extend_from_slice(header);
        valid.extend_from_slice(&1.0f32.to_le_bytes());
        for (existing, corrupt) in [
            (b"".as_slice(), true),
            (b"x", true),
            (&valid[..valid.len() - 1], true),
            (EMPTY_SAFETENSORS_FIXTURE, true),
            (ZERO_SAFETENSORS_FIXTURE, false),
            (valid.as_slice(), false),
        ] {
            let cache = TempDir::new()?;
            let root = cache.path();
            let (staging, path) = stage(root, "aaa", metadata);
            fs::write(path.join(checkpoint), existing)?;
            fs::create_dir(path.join("local"))?;
            fs::write(path.join("local").join(checkpoint), [])?;
            let live = publish_hf_snapshot(&staging, root, STREAM_REPO, &path, None, true, None)?;

            let (staging, path) = stage(root, "aaa", metadata);
            let shard = "model-00001-of-00001.safetensors";
            fs::write(path.join(shard), &valid)?;
            fs::write(
                path.join(index),
                serde_json::to_vec(&serde_json::json!({"weight_map":{"w":shard}}))?,
            )?;
            publish_hf_snapshot(&staging, root, STREAM_REPO, &path, None, false, None)?;
            assert!(stream_index_complete(&live, index)?);
            assert!(live.join("local").join(checkpoint).is_file());
            if corrupt {
                assert!(!live.join(checkpoint).exists());
            } else {
                assert_eq!(fs::read(live.join(checkpoint))?, existing);
            }
        }
        Ok(())
    }

    #[test]
    fn full_stream_can_complete_the_confirmed_revision() -> anyhow::Result<()> {
        let index = "model.safetensors.index.json";
        let first = "model-00001-of-00002.safetensors";
        let second = "model-00002-of-00002.safetensors";
        for (previous, resolved, complete) in [
            (Some("aaa"), Some("aaa"), true),
            (Some("aaa"), None, false),
            (Some("bbb"), Some("aaa"), false),
            (None, Some("aaa"), false),
        ] {
            let cache = TempDir::new()?;
            let root = cache.path();
            let repo = root.join(STREAM_REPO);
            if let Some(revision) = previous {
                let prior = repo.join("snapshots").join(revision);
                fs::create_dir_all(&prior)?;
                fs::write(prior.join(second), SAFETENSORS_V_FIXTURE)?;
            }
            let (temp, snapshot) = stage(
                root,
                "aaa",
                &[
                    ("config.json", b"{}"),
                    ("tokenizer.json", b"{}"),
                    (first, SAFETENSORS_FIXTURE),
                ],
            );
            fs::write(
                snapshot.join(index),
                serde_json::to_vec(&serde_json::json!({"weight_map":{"w":first,"v":second}}))?,
            )?;
            let result =
                publish_hf_snapshot(&temp, root, STREAM_REPO, &snapshot, resolved, false, None);
            assert_eq!(result.is_ok(), complete, "{result:?}");
            assert_eq!(repo.join("refs/main").is_file(), complete);
            if let Ok(path) = result {
                assert!(stream_index_complete(&path, index)?);
            }
        }
        Ok(())
    }

    #[test]
    fn standalone_safetensors_require_tensors() -> anyhow::Result<()> {
        let header = br#"{"__metadata__":{"format":"pt"}}"#;
        let mut metadata_only = (header.len() as u64).to_le_bytes().to_vec();
        metadata_only.extend_from_slice(header);
        for name in ["model.safetensors", "consolidated.safetensors"] {
            for (bytes, ready) in [
                (EMPTY_SAFETENSORS_FIXTURE, false),
                (metadata_only.as_slice(), false),
                (ZERO_SAFETENSORS_FIXTURE, true),
                (SAFETENSORS_FIXTURE, true),
            ] {
                let cache = TempDir::new()?;
                let (staging, snapshot) = stage(
                    cache.path(),
                    "aaa",
                    &[
                        ("config.json", b"{}"),
                        ("tokenizer.json", b"{}"),
                        (name, bytes),
                    ],
                );
                assert!(stream_model_ready(&snapshot, true)?);
                assert_eq!(stream_model_ready(&snapshot, false)?, ready, "{name}");
                if ready {
                    publish_hf_snapshot(
                        &staging,
                        cache.path(),
                        STREAM_REPO,
                        &snapshot,
                        None,
                        false,
                        None,
                    )?;
                }
            }
        }

        let dir = TempDir::new()?;
        fs::write(
            dir.path().join("model.safetensors"),
            EMPTY_SAFETENSORS_FIXTURE,
        )?;
        discard_corrupt_safetensors(dir.path())?;
        assert!(dir.path().join("model.safetensors").is_file());
        assert!(!hf_safetensors_ready(dir.path())?);
        fs::write(
            dir.path().join("model.safetensors.index.json"),
            br#"{"weight_map":{"w":"part.safetensors"}}"#,
        )?;
        discard_corrupt_safetensors(dir.path())?;
        assert!(dir.path().join("model.safetensors").is_file());
        fs::write(dir.path().join("part.safetensors"), SAFETENSORS_FIXTURE)?;
        discard_corrupt_safetensors(dir.path())?;
        assert!(!dir.path().join("model.safetensors").exists());
        assert!(hf_safetensors_ready(dir.path())?);
        Ok(())
    }

    #[test]
    fn numbered_safetensors_require_a_tensor() -> anyhow::Result<()> {
        for prefix in ["model", "consolidated"] {
            for count in [1, 2] {
                let cache = TempDir::new()?;
                let (staging, snapshot) = stage(
                    cache.path(),
                    "aaa",
                    &[("config.json", b"{}"), ("tokenizer.json", b"{}")],
                );
                for part in 1..=count {
                    fs::write(
                        snapshot.join(format!("{prefix}-{part:05}-of-{count:05}.safetensors")),
                        EMPTY_SAFETENSORS_FIXTURE,
                    )?;
                }
                assert!(!stream_model_ready(&snapshot, false)?);
                assert!(
                    publish_hf_snapshot(
                        &staging,
                        cache.path(),
                        STREAM_REPO,
                        &snapshot,
                        None,
                        false,
                        None
                    )
                    .is_err()
                );
                assert!(!cache.path().join(STREAM_REPO).join("refs/main").exists());
                fs::write(
                    snapshot.join(format!("{prefix}-{count:05}-of-{count:05}.safetensors")),
                    ZERO_SAFETENSORS_FIXTURE,
                )?;
                assert!(stream_model_ready(&snapshot, false)?);
                let live = publish_hf_snapshot(
                    &staging,
                    cache.path(),
                    STREAM_REPO,
                    &snapshot,
                    None,
                    false,
                    None,
                )?;
                assert_eq!(
                    cached_stream_snapshot(&cache.path().join(STREAM_REPO), false)?,
                    Some(live)
                );
            }
        }
        Ok(())
    }

    #[test]
    fn empty_safetensors_shards_remain_usable() -> anyhow::Result<()> {
        for indexed in [false, true] {
            let cache = TempDir::new()?;
            let first = "consolidated-00001-of-00002.safetensors";
            let second = "consolidated-00002-of-00002.safetensors";
            let (staging, snapshot) = stage(
                cache.path(),
                "aaa",
                &[
                    ("config.json", b"{}"),
                    ("tokenizer.json", b"{}"),
                    (first, EMPTY_SAFETENSORS_FIXTURE),
                    (
                        second,
                        if indexed {
                            ZERO_SAFETENSORS_FIXTURE
                        } else {
                            SAFETENSORS_FIXTURE
                        },
                    ),
                ],
            );
            if indexed {
                fs::write(
                    snapshot.join("model.safetensors.index.json"),
                    serde_json::to_vec(&serde_json::json!({"weight_map":{"w":second}}))?,
                )?;
            }
            let live = publish_hf_snapshot(
                &staging,
                cache.path(),
                STREAM_REPO,
                &snapshot,
                None,
                false,
                None,
            )?;
            assert_eq!(
                cached_stream_snapshot(&cache.path().join(STREAM_REPO), false)?,
                Some(live)
            );
        }
        Ok(())
    }

    #[test]
    fn indexed_safetensors_require_assigned_tensors() -> anyhow::Result<()> {
        let dir = TempDir::new()?;
        let index = "model.safetensors.index.json";
        fs::write(dir.path().join("first.safetensors"), SAFETENSORS_FIXTURE)?;
        fs::write(dir.path().join("second.safetensors"), SAFETENSORS_V_FIXTURE)?;
        let mapping = serde_json::json!({"weight_map": {
            "w": "first.safetensors", "v": "first.safetensors"
        }});
        fs::write(dir.path().join(index), serde_json::to_vec(&mapping)?)?;
        assert!(!stream_index_complete(dir.path(), index)?);
        let mapping = serde_json::json!({"weight_map": {
            "w": "first.safetensors", "v": "second.safetensors"
        }});
        fs::write(dir.path().join(index), serde_json::to_vec(&mapping)?)?;
        assert!(stream_index_complete(dir.path(), index)?);
        Ok(())
    }

    #[test]
    fn metadata_merge_preserves_full_numbered_snapshot() -> anyhow::Result<()> {
        let cache = TempDir::new()?;
        let root = cache.path();
        let repo = root.join(STREAM_REPO);
        let metadata: &[(&str, &[u8])] = &[("config.json", b"{}"), ("tokenizer.json", b"{}")];
        let (staging, path) = stage(root, "aaa", metadata);
        let first = "consolidated-00001-of-00002.safetensors";
        let second = "consolidated-00002-of-00002.safetensors";
        fs::write(path.join(first), SAFETENSORS_FIXTURE)?;
        fs::write(path.join(second), SAFETENSORS_V_FIXTURE)?;
        let live = publish_hf_snapshot(&staging, root, STREAM_REPO, &path, None, false, None)?;

        let index = "consolidated.safetensors.index.json";
        let (staging, path) = stage(root, "aaa", metadata);
        fs::write(
            path.join(index),
            br#"{"weight_map":{"a":"missing.safetensors"}}"#,
        )?;
        let error =
            publish_hf_snapshot(&staging, root, STREAM_REPO, &path, None, true, None).unwrap_err();
        assert!(!error.is::<SnapshotInstallFailed>());
        assert!(!live.join(index).exists());
        assert_eq!(cached_stream_snapshot(&repo, false)?, Some(live.clone()));

        let (staging, path) = stage(root, "aaa", metadata);
        fs::write(
            path.join(index),
            serde_json::to_vec(&serde_json::json!({"weight_map":{"w":first,"v":second}}))?,
        )?;
        publish_hf_snapshot(&staging, root, STREAM_REPO, &path, None, true, None)?;
        assert_eq!(cached_stream_snapshot(&repo, false)?, Some(live.clone()));
        assert!(live.join(index).is_file());
        assert_eq!(fs::read(live.join(first))?, SAFETENSORS_FIXTURE);
        assert_eq!(fs::read(live.join(second))?, SAFETENSORS_V_FIXTURE);
        Ok(())
    }

    #[tokio::test]
    async fn private_upgrade_does_not_mask_published_snapshot() -> anyhow::Result<()> {
        let root = TempDir::new()?;
        let files = ["config.json", "spm.model", "model.safetensors"];
        let published = build_hf_cache(root.path(), "org/model", &files);
        let snapshots = published.parent().unwrap();
        fs::remove_file(snapshots.parent().unwrap().join("refs/main"))?;
        let private = snapshots.join(".dynamo-upgrade-test");
        fs::rename(&published, &private)?;
        assert!(
            recover_published_snapshot(private.clone(), None, false, None, false)
                .await
                .is_err()
        );
        fs::create_dir(&published)?;
        for file in files {
            let bytes = if file.ends_with(".safetensors") {
                SAFETENSORS_FIXTURE
            } else {
                b"valid fixture"
            };
            fs::write(published.join(file), bytes)?;
        }
        assert_eq!(
            recover_published_snapshot(private, None, false, None, false).await?,
            published
        );
        Ok(())
    }

    #[test]
    fn unpublished_upgrade_is_not_a_card_candidate() -> anyhow::Result<()> {
        let root = TempDir::new()?;
        let snapshots = root.path().join("snapshots");
        fs::create_dir(&snapshots)?;
        let sibling = tempfile::Builder::new()
            .prefix(UPGRADE_PREFIX)
            .tempdir_in(&snapshots)?;
        fs::write(sibling.path().join("config.json"), b"from upgrade")?;
        let matches = |path: &Path| path.join("config.json").is_file();
        assert_eq!(
            matching_stream_snapshot(root.path(), true, Some(&matches))?,
            None
        );
        Ok(())
    }

    #[test]
    fn streamed_tiktoken_layout_agrees_with_model_card() -> anyhow::Result<()> {
        let temp = TempDir::new()?;
        let dir = temp.path();
        fs::write(dir.join("config.json"), "{}")?;
        fs::write(dir.join("one.tiktoken"), b"one")?;
        assert!(stream_metadata_present(dir)?);
        fs::write(dir.join("two.tiktoken"), b"two")?;
        assert!(stream_metadata_present(dir).is_err());
        assert!(crate::model_card::TokenizerKind::from_disk(dir).is_err());
        for preferred in ["tiktoken.model", "tokenizer.json"] {
            fs::write(dir.join(preferred), "{}")?;
            assert!(stream_metadata_present(dir)?);
            assert!(crate::model_card::TokenizerKind::from_disk(dir)?.is_some());
            fs::remove_file(dir.join(preferred))?;
        }
        #[cfg(unix)]
        {
            std::os::unix::fs::symlink("tokenizer.json", dir.join("tokenizer.json"))?;
            fs::remove_file(dir.join("one.tiktoken"))?;
            fs::remove_file(dir.join("two.tiktoken"))?;
            assert!(hf_tokenizer_file(dir).unwrap_err().is::<std::io::Error>());
            fs::write(dir.join("tiktoken.model"), b"fallback")?;
            assert!(stream_metadata_present(dir)?);
            assert!(matches!(
                crate::model_card::TokenizerKind::from_disk(dir)?,
                Some(crate::model_card::TokenizerKind::TikTokenModel(_))
            ));
            fs::remove_file(dir.join("tiktoken.model"))?;
            fs::write(dir.join("one.tiktoken"), b"fallback")?;
            assert_eq!(hf_tokenizer_file(dir)?, Some(dir.join("one.tiktoken")));
            fs::remove_file(dir.join("tokenizer.json"))?;
        }
        fs::write(dir.join("tokenizer.json"), [])?;
        fs::write(dir.join("tiktoken.model"), b"fallback")?;
        assert!(hf_tokenizer_file(dir).is_err());
        fs::remove_file(dir.join("tokenizer.json"))?;
        fs::create_dir(dir.join("tokenizer.json"))?;
        assert!(stream_metadata_present(dir)?);
        assert!(matches!(
            crate::model_card::TokenizerKind::from_disk(dir)?,
            Some(crate::model_card::TokenizerKind::TikTokenModel(_))
        ));
        Ok(())
    }

    #[cfg(unix)]
    #[test]
    fn unreadable_stream_does_not_replace_live_files_or_ref() -> anyhow::Result<()> {
        use std::os::unix::fs::symlink;
        let cache = TempDir::new()?;
        let live = build_hf_cache(
            cache.path(),
            "org/model",
            &["config.json", "tokenizer.json"],
        );
        let repo = cache.path().join(STREAM_REPO);
        let revision = live.file_name().unwrap().to_str().unwrap();
        let (staging, snapshot) = stage(
            cache.path(),
            revision,
            &[("config.json", b"new"), ("tokenizer.json", b"{}")],
        );
        symlink("model.safetensors", snapshot.join("model.safetensors"))?;
        let old_config = fs::read(live.join("config.json"))?;
        fs::create_dir(repo.join("snapshots/previous"))?;
        fs::write(repo.join("refs/main"), b"previous")?;
        let error = publish_hf_snapshot(
            &staging,
            cache.path(),
            STREAM_REPO,
            &snapshot,
            None,
            false,
            None,
        )
        .unwrap_err();
        assert!(error.is::<std::io::Error>());
        assert!(!error.is::<SnapshotInstallFailed>());
        assert_eq!(fs::read(live.join("config.json"))?, old_config);
        assert_eq!(fs::read(repo.join("refs/main"))?, b"previous");
        assert!(!live.join("model.safetensors").is_symlink());
        Ok(())
    }

    #[test]
    fn read_only_lock_failure_reuses_only_complete_snapshots() -> anyhow::Result<()> {
        let cache = TempDir::new()?;
        let path = build_hf_cache(cache.path(), "org/model", &["config.json", "spm.model"]);
        let repo = cache.path().join(STREAM_REPO);
        let probe = |kind: ErrorKind, ignore_weights| {
            let error =
                anyhow::Error::new(std::io::Error::from(kind)).context("opening the cache lock");
            cached_snapshot_after_lock_error(&repo, ignore_weights, error)
        };
        assert_eq!(probe(ErrorKind::ReadOnlyFilesystem, true)?, path);
        assert!(probe(ErrorKind::PermissionDenied, false).is_err());
        fs::write(path.join("pytorch_model.bin"), b"weights")?;
        assert_eq!(probe(ErrorKind::PermissionDenied, false)?, path);
        assert!(probe(ErrorKind::InvalidData, false).is_err());
        Ok(())
    }

    #[serial_test::serial]
    #[tokio::test]
    async fn direct_hf_download_holds_the_publication_lock() {
        use tokio::io::{AsyncReadExt as _, AsyncWriteExt as _};
        let cache = TempDir::new().unwrap();
        let hub = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
        let endpoint = format!("http://{}", hub.local_addr().unwrap());
        let serve = async {
            let (mut socket, _) = hub.accept().await.unwrap();
            let mut request = [0; 1024];
            assert!(socket.read(&mut request).await.unwrap() > 0);
            let repo = cache.path().join(STREAM_REPO);
            mx_cache::CacheConfig::from_path(cache.path())
                .unwrap()
                .clear_model("org/model", MxModelProvider::HuggingFace)
                .unwrap();
            fs::create_dir(&repo).unwrap();
            let lock_path = cache
                .path()
                .join(".locks")
                .join(format!("{STREAM_REPO}.dynamo-stream.lock"));
            let lock = fs::File::options()
                .read(true)
                .write(true)
                .open(lock_path)
                .unwrap();
            assert!(matches!(lock.try_lock(), Err(fs::TryLockError::WouldBlock)));
            socket
                .write_all(
                    b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                )
                .await
                .unwrap();
        };
        let direct = mx_download_direct(
            "org/model",
            MxModelProvider::HuggingFace,
            cache.path().to_owned(),
            false,
            true,
            None,
        );
        let env = [
            ("HF_ENDPOINT", Some(endpoint.as_str())),
            ("HF_HUB_OFFLINE", Some("0")),
        ];
        let (result, ()) = temp_env::async_with_vars(
            env,
            tokio::time::timeout(std::time::Duration::from_secs(10), async {
                tokio::join!(direct, serve)
            }),
        )
        .await
        .unwrap();
        assert!(result.is_err());
    }

    #[test]
    fn published_sentencepiece_snapshot_is_reusable() -> anyhow::Result<()> {
        for (config, tokenizer, ignore_weights) in [
            ("config.json", "spm.model", true),
            ("config.json", "spm.model", false),
            ("params.json", "tokenizer.model.v3", false),
        ] {
            let cache = TempDir::new()?;
            if ignore_weights {
                let previous = build_hf_cache(cache.path(), "org/model", &[]);
                fs::write(previous.join("model.gguf"), gguf_fixture(None, None))?;
            }
            let (staging, snapshot) = stage(
                cache.path(),
                "aaa",
                &[(config, b"{}"), (tokenizer, b"tokenizer")],
            );
            if !ignore_weights {
                fs::write(snapshot.join("pytorch_model.bin"), b"weights")?;
            }
            let published = publish_hf_snapshot(
                &staging,
                cache.path(),
                STREAM_REPO,
                &snapshot,
                None,
                ignore_weights,
                None,
            )?;
            assert_eq!(
                cached_stream_snapshot(&cache.path().join(STREAM_REPO), ignore_weights)?,
                Some(published)
            );
        }
        Ok(())
    }

    #[test]
    fn incomplete_stream_is_not_published() -> anyhow::Result<()> {
        let cache = TempDir::new()?;
        let (staging, snapshot) = stage(
            cache.path(),
            "aaa",
            &[("config.json", b"{}"), ("tokenizer_config.json", b"{}")],
        );
        let publish = |ignore_weights| {
            publish_hf_snapshot(
                &staging,
                cache.path(),
                STREAM_REPO,
                &snapshot,
                None,
                ignore_weights,
                None,
            )
        };
        assert!(!publish(false).unwrap_err().is::<SnapshotInstallFailed>());
        assert!(publish(true).unwrap_err().is::<SnapshotInstallFailed>());
        fs::write(snapshot.join("pytorch_model.bin"), b"weights")?;
        assert!(publish(false).unwrap_err().is::<SnapshotInstallFailed>());
        assert!(!cache.path().join(STREAM_REPO).join("refs/main").exists());
        Ok(())
    }

    #[test]
    fn numbered_mistral_weights_publish_only_when_complete() -> anyhow::Result<()> {
        let cache = TempDir::new()?;
        let root = cache.path();
        let (staging, snapshot) = stage(
            root,
            "aaa",
            &[
                ("params.json", b"{}"),
                ("tekken.json", b"{}"),
                (
                    "consolidated-00001-of-00002.safetensors",
                    SAFETENSORS_FIXTURE,
                ),
                (
                    "consolidated-00002-of-00003.safetensors",
                    b"different total",
                ),
            ],
        );
        let standalone = snapshot.join("consolidated.safetensors");
        fs::write(&standalone, b"x")?;
        assert!(!stream_weights_present(&snapshot, false)?);
        fs::write(&standalone, SAFETENSORS_FIXTURE)?;
        assert!(stream_weights_present(&snapshot, false)?);
        fs::remove_file(standalone)?;
        let publish =
            || publish_hf_snapshot(&staging, root, STREAM_REPO, &snapshot, None, false, None);
        assert!(publish().is_err());
        assert!(!root.join(STREAM_REPO).join("refs/main").exists());
        let second = snapshot.join("consolidated-00002-of-00002.safetensors");
        fs::write(&second, b"")?;
        assert!(!stream_weights_present(&snapshot, false)?);
        fs::write(
            &second,
            &SAFETENSORS_V_FIXTURE[..SAFETENSORS_V_FIXTURE.len() - 1],
        )?;
        assert!(!stream_weights_present(&snapshot, false)?);
        fs::write(second, SAFETENSORS_V_FIXTURE)?;

        let index = snapshot.join("model.safetensors.index.json");
        fs::write(&index, r#"{"weight_map":{"w":"named.safetensors"}}"#)?;
        assert!(!stream_weights_present(&snapshot, false)?);
        let consolidated_index = snapshot.join("consolidated.safetensors.index.json");
        fs::rename(index, &consolidated_index)?;
        fs::write(snapshot.join("named.safetensors"), SAFETENSORS_FIXTURE)?;
        assert!(stream_weights_present(&snapshot, false)?);
        fs::remove_file(consolidated_index)?;

        let live = publish()?;
        assert_eq!(
            cached_stream_snapshot(&root.join(STREAM_REPO), false)?,
            Some(live)
        );
        Ok(())
    }

    #[test]
    fn empty_metadata_does_not_make_a_stream_ready() -> anyhow::Result<()> {
        for (config, tokenizer) in [
            ("config.json", "tokenizer.json"),
            ("config.json", "example.tiktoken"),
            ("params.json", "tokenizer.model.v3"),
        ] {
            let cache = TempDir::new()?;
            let dir = cache.path();
            fs::write(dir.join(config), [])?;
            fs::write(dir.join(tokenizer), b"tokenizer")?;
            let error = stream_metadata_present(dir).unwrap_err();
            assert_eq!(
                error.downcast::<std::io::Error>()?.kind(),
                ErrorKind::InvalidData
            );
            fs::write(dir.join(config), b"{}")?;
            assert!(stream_metadata_present(dir)?);
            fs::write(dir.join(tokenizer), [])?;
            let error = stream_metadata_present(dir).unwrap_err();
            assert_eq!(
                error.downcast::<std::io::Error>()?.kind(),
                ErrorKind::InvalidData
            );
        }
        Ok(())
    }

    #[test]
    fn auxiliary_export_does_not_satisfy_full_request() -> anyhow::Result<()> {
        let cache = TempDir::new()?;
        let snapshot = cache.path();
        fs::write(snapshot.join("config.json"), "{}")?;
        fs::write(snapshot.join("tokenizer.json"), "{}")?;
        // GPT-2 metadata can contain a TFLite export; training args are not weights either.
        fs::write(snapshot.join("64.tflite"), b"auxiliary export")?;
        for aux in [
            "training_args.bin",
            "optimizer.bin",
            "optimizer-00001-of-00001.bin",
            "adapter_model.safetensors",
            "adapter_model-00001-of-00001.safetensors",
        ] {
            fs::write(snapshot.join(aux), b"auxiliary state")?;
        }
        assert!(stream_model_ready(snapshot, true)?);
        assert!(!stream_model_ready(snapshot, false)?);
        Ok(())
    }

    fn gguf_fixture(split_count: Option<u16>, artifact_type: Option<&str>) -> Vec<u8> {
        // GGUF v3 with one 32-element Q4_0 block (a scale and 16 packed bytes).
        let mut gguf = b"GGUF".to_vec();
        gguf.extend(3_u32.to_le_bytes());
        gguf.extend(1_u64.to_le_bytes()); // tensor count
        let entries = u64::from(split_count.is_some()) + u64::from(artifact_type.is_some());
        gguf.extend(entries.to_le_bytes());
        if let Some(kind) = artifact_type {
            gguf.extend(12_u64.to_le_bytes());
            gguf.extend(b"general.type");
            gguf.extend(8_u32.to_le_bytes()); // STRING
            gguf.extend((kind.len() as u64).to_le_bytes());
            gguf.extend(kind.as_bytes());
        }
        if let Some(count) = split_count {
            gguf.extend(11_u64.to_le_bytes());
            gguf.extend(b"split.count");
            gguf.extend(2_u32.to_le_bytes()); // UINT16
            gguf.extend(count.to_le_bytes());
        }
        gguf.extend(1_u64.to_le_bytes());
        gguf.push(b'w');
        gguf.extend(1_u32.to_le_bytes());
        gguf.extend(32_u64.to_le_bytes());
        gguf.extend(2_u32.to_le_bytes()); // Q4_0
        gguf.extend(0_u64.to_le_bytes());
        gguf.resize(gguf.len().next_multiple_of(32), 0); // tensor data is 32-byte aligned
        gguf.extend([0; 18]);
        gguf
    }

    #[test]
    fn split_gguf_requires_matching_complete_parts() -> anyhow::Result<()> {
        use gguf_rs_lib::prelude::{
            GGUFBuilder,
            MetadataValue::{I32, U16},
        };

        let cache = TempDir::new()?;
        let dir = cache.path();
        let part = |index, tensors, tensor: &str| -> anyhow::Result<Vec<u8>> {
            let (bytes, _) = GGUFBuilder::new()
                .add_metadata("split.no", U16(index))
                .add_metadata("split.count", U16(2))
                .add_metadata("split.tensors.count", I32(tensors))
                .add_f32_tensor(tensor, vec![1], vec![1.0])?
                .build_to_bytes()?;
            Ok(bytes)
        };
        fs::write(dir.join("tiny-00001-of-00002.gguf"), part(0, 2, "a")?)?;
        assert!(!stream_model_ready(dir, false)?);
        let second = dir.join("tiny-00002-of-00002.gguf");
        for invalid in [part(0, 2, "b")?, part(1, 3, "b")?, part(1, 2, "a")?] {
            fs::write(&second, invalid)?;
            assert!(!stream_model_ready(dir, false)?);
        }
        let mut valid = part(1, 2, "b")?;
        fs::write(&second, &valid)?;
        assert!(stream_model_ready(dir, false)?);
        valid.truncate(valid.len() - 32);
        fs::write(second, valid)?;
        assert!(!stream_model_ready(dir, false)?);
        Ok(())
    }

    #[test]
    fn standalone_gguf_is_ready_without_sidecar_metadata() -> anyhow::Result<()> {
        let dir = TempDir::new()?;
        let snapshot = build_hf_cache(dir.path(), "org/model", &[]);
        let repo = dir.path().join(STREAM_REPO);
        let path = snapshot.join("model.gguf");
        for (count, kind, expected) in [
            (None, None, true),
            (Some(1), Some("model"), true),
            (Some(2), Some("model"), false),
            (None, Some("adapter"), false),
            (Some(0), None, true),
            (None, Some("imatrix"), false),
        ] {
            fs::write(&path, gguf_fixture(count, kind))?;
            assert_eq!(
                cached_stream_snapshot(&repo, false)?.as_deref(),
                expected.then_some(snapshot.as_path()),
                "{count:?}, {kind:?}"
            );
            assert!(cached_stream_snapshot(&repo, true)?.is_none());
        }
        fs::remove_file(&path)?;
        fs::write(snapshot.join("model.safetensors"), b"weights")?;
        assert!(cached_stream_snapshot(&repo, false)?.is_none());
        Ok(())
    }

    #[test]
    fn stream_index_requires_shards_in_the_index_format() -> anyhow::Result<()> {
        let dir = TempDir::new()?;
        let dir = dir.path();
        fs::create_dir(dir.join("weights"))?;
        fs::write(dir.join("config.json"), b"{}")?;
        fs::write(dir.join("weights/part.bin"), b"fixture")?;
        fs::write(dir.join("blob"), SAFETENSORS_FIXTURE)?;
        #[cfg(unix)]
        std::os::unix::fs::symlink("../blob", dir.join("weights/part.safetensors"))?;
        #[cfg(not(unix))]
        fs::copy(dir.join("blob"), dir.join("weights/part.safetensors"))?;
        for (index, same_format, other_format) in [
            (
                "model.safetensors.index.json",
                "weights/part.safetensors",
                "weights/part.bin",
            ),
            (
                "pytorch_model.bin.index.json",
                "weights/part.bin",
                "weights/part.safetensors",
            ),
        ] {
            for (name, expected) in [
                (same_format, true),
                (&format!("./{same_format}"), true),
                (&format!("weights/../{same_format}"), false),
                (dir.join(same_format).to_str().unwrap(), false),
                (other_format, false),
                ("config.json", false),
            ] {
                fs::write(
                    dir.join(index),
                    serde_json::json!({"weight_map":{"w":name}}).to_string(),
                )?;
                assert_eq!(
                    stream_index_complete(dir, index)?,
                    expected,
                    "{index}: {name}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn standalone_gguf_weights_accept_quantized_names() -> anyhow::Result<()> {
        let cache = TempDir::new()?;
        let snapshot = cache.path();
        fs::write(snapshot.join("config.json"), "{}")?;
        fs::write(snapshot.join("tokenizer.json"), "{}")?;
        let mut gguf = gguf_fixture(None, None);
        fs::write(
            snapshot.join("model.safetensors.index.json"),
            r#"{"weight_map":{"w":"missing-shard.safetensors"}}"#,
        )?;
        for (name, expected) in [
            ("model.gguf", true),
            ("tiny-Q4_K_M.gguf", true),
            ("Tree-of-Thought-Q4.gguf", true),
            ("tiny-00001-of-00001.gguf", true),
            ("tiny-00001-of-00002.gguf", false),
            ("mmproj-tiny-F16.gguf", false),
        ] {
            let path = snapshot.join(name);
            fs::write(&path, &gguf)?;
            assert_eq!(stream_model_ready(snapshot, false)?, expected, "{name}");
            fs::remove_file(path)?;
        }
        let path = snapshot.join("tokenizer.gguf");
        gguf[4..8].copy_from_slice(&2_u32.to_le_bytes());
        fs::write(&path, &gguf)?;
        assert!(stream_model_ready(snapshot, false)?); // little-endian GGUF v2
        for length in [25, 64, gguf.len() - 1] {
            fs::write(&path, &gguf[..length])?;
            assert!(
                !stream_model_ready(snapshot, false)?,
                "truncated at {length}"
            );
        }
        gguf.truncate(24);
        fs::write(&path, &gguf)?;
        assert!(!stream_model_ready(snapshot, false)?); // only a header was written
        gguf[8..16].fill(0);
        gguf.resize(32, 0); // padded empty GGUF with no tensors
        fs::write(path, gguf)?;
        assert!(!stream_model_ready(snapshot, false)?);
        Ok(())
    }

    #[test]
    fn onnx_model_bin_sidecar_requires_the_declared_tensor_range() -> anyhow::Result<()> {
        use prost::encoding::{WireType, encode_key, encode_varint};

        let message = |number, bytes: &[u8], parent: &mut Vec<u8>| {
            encode_key(number, WireType::LengthDelimited, parent);
            encode_varint(bytes.len() as u64, parent);
            parent.extend_from_slice(bytes);
        };
        // Model with one FLOAT initializer of shape [1] stored in model.bin.
        let mut tensor = b"\x08\x01\x10\x01\x42\x01w\x70\x01".to_vec();
        message(13, b"\x0a\x08location\x12\x09model.bin", &mut tensor);
        let mut graph = b"\x12\x01g".to_vec();
        message(5, &tensor, &mut graph);
        message(
            12,
            b"\x0a\x01w\x12\x0a\x0a\x08\x08\x01\x12\x04\x0a\x02\x08\x01",
            &mut graph,
        );
        let mut model = vec![0x08, 8]; // IR version
        message(7, &graph, &mut model);
        message(8, b"\x10\x0d", &mut model);

        let cache = TempDir::new()?;
        let snapshot = cache.path();
        let onnx = snapshot.join("model.onnx");
        let sidecar = snapshot.join("model.bin");
        fs::write(snapshot.join("config.json"), "{}")?;
        fs::write(snapshot.join("tokenizer.json"), "{}")?;
        fs::write(&onnx, model)?;
        fs::write(&sidecar, [0])?;
        assert!(!stream_model_ready(snapshot, false)?);
        assert!(stream_model_ready(snapshot, true)?);

        let alternative = snapshot.join("model.safetensors");
        fs::write(&alternative, SAFETENSORS_FIXTURE)?;
        assert!(stream_model_ready(snapshot, false)?);
        fs::remove_file(alternative)?;

        fs::write(&sidecar, 1_f32.to_le_bytes())?;
        assert!(stream_model_ready(snapshot, false)?);
        let quantized = snapshot.join("model_quantized.onnx");
        fs::rename(onnx, &quantized)?;
        assert!(stream_model_ready(snapshot, false)?);
        fs::write(&sidecar, [0])?;
        assert!(!stream_model_ready(snapshot, false)?);
        let nested = snapshot.join("onnx");
        fs::create_dir(&nested)?;
        let nested_sidecar = nested.join("model.bin");
        fs::rename(&sidecar, &nested_sidecar)?;
        fs::rename(quantized, nested.join("model_quantized.onnx"))?;
        assert!(!stream_model_ready(snapshot, false)?);
        fs::write(nested_sidecar, 1_f32.to_le_bytes())?;
        assert!(stream_model_ready(snapshot, false)?);
        Ok(())
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
            r#"{"weight_map": {"w": "model-00001-of-00002.safetensors", "v": "model-00002-of-00002.safetensors"}}"#,
        )
        .unwrap();

        let incomplete = get_cached_model_path_in(model, false, temp.path().to_path_buf());
        assert!(
            incomplete.is_none(),
            "sharded cache without shard files must NOT satisfy ignore_weights=false"
        );

        fs::write(
            snapshot.join("model-00001-of-00002.safetensors"),
            SAFETENSORS_FIXTURE,
        )
        .unwrap();
        fs::write(
            snapshot.join("model-00002-of-00002.safetensors"),
            SAFETENSORS_V_FIXTURE,
        )
        .unwrap();
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
    #[test]
    fn preferred_cache_probe_uses_blocking_pool() -> anyhow::Result<()> {
        use std::sync::mpsc;
        use std::time::Duration;

        let temp = TempDir::new()?;
        let model = "test-org/cache-probe-pool";
        let snapshot = build_hf_cache(
            temp.path(),
            model,
            &["config.json", "tokenizer.json", "model.safetensors"],
        );
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .max_blocking_threads(1)
            .build()?;
        let (started, ready) = mpsc::channel();
        let (release, released) = mpsc::channel();
        let held = runtime.spawn_blocking(move || {
            started.send(())?;
            released.recv_timeout(Duration::from_secs(5))?;
            anyhow::Ok(())
        });
        ready.recv_timeout(Duration::from_secs(5))?;
        runtime.block_on(temp_env::async_with_vars(
            [
                ("HF_HUB_CACHE", Some(temp.path().to_str().unwrap())),
                ("HF_HOME", None),
                ("MODEL_EXPRESS_CACHE_PATH", None),
                ("MODEL_EXPRESS_NO_SHARED_STORAGE", Some("0")),
                ("HF_HUB_OFFLINE", Some("1")),
                ("MODEL_EXPRESS_URL", Some("http://[")),
                ("HF_ENDPOINT", Some("http://[")),
            ],
            async {
                let mut fetch = Box::pin(from_hf(model, false));
                let pending = std::future::poll_fn(|cx| {
                    std::task::Poll::Ready(
                        std::future::Future::poll(fetch.as_mut(), cx).is_pending(),
                    )
                })
                .await;
                release.send(())?;
                assert!(
                    pending,
                    "cache inspection ran before a blocking thread was available"
                );
                assert_eq!(fetch.await?, snapshot);
                held.await??;
                anyhow::Ok(())
            },
        ))?;
        Ok(())
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
        let native_model = "test-org/shared-sentencepiece";
        let native_snapshot = build_hf_cache(
            temp.path(),
            native_model,
            &["config.json", "spm.model", "pytorch_model.bin"],
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
                (
                    env_model::model_express::MODEL_EXPRESS_NO_SHARED_STORAGE,
                    Some("0"),
                ),
                (
                    env_model::model_express::MODEL_EXPRESS_URL,
                    Some("http://["),
                ),
                ("HF_ENDPOINT", Some("http://[")),
            ],
            async {
                for (name, expected) in [(model, snapshot), (native_model, native_snapshot)] {
                    let path = from_hf(name, false).await.unwrap();
                    assert_eq!(path, expected);
                }
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
}
