// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    env,
    future::Future,
    io::Write,
    path::{Path, PathBuf},
    time::Duration,
};

use anyhow::Context;
use hf_hub::{
    Cache, Repo, RepoType,
    api::tokio::{Api, ApiBuilder, ApiError},
};

use dynamo_runtime::config::environment_names::model as env_model;

use super::is_offline_mode;

const LORA_DOWNLOAD_TIMEOUT_SECS: &str = "LORA_DOWNLOAD_TIMEOUT_SECS";
const DEFAULT_LORA_DOWNLOAD_TIMEOUT_SECS: u64 = 3600;
const HF_LOCK_RETRY_INTERVAL: Duration = Duration::from_millis(100);
const LORA_COMPLETE_MARKER: &str = ".dynamo_lora_complete";
const LORA_COMPLETE_MARKER_CONTENT: &[u8] = b"1\n";

/// A Hugging Face model repository and the revision requested by an `hf://` URI.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct HfRepoSpec {
    repo_id: String,
    revision: String,
}

impl HfRepoSpec {
    pub(crate) fn from_uri(uri: &str) -> anyhow::Result<Self> {
        let value = uri
            .strip_prefix("hf://")
            .with_context(|| format!("expected hf:// URI, got: {uri}"))?;

        if value.contains(['?', '#']) {
            anyhow::bail!("hf:// URI must not contain a query or fragment: {uri}");
        }

        let (repo_id, revision) = match value.rsplit_once('@') {
            Some((repo_id, revision)) => (repo_id, revision),
            None => (value, "main"),
        };

        validate_hf_relative_path(repo_id, "repository")?;
        validate_hf_relative_path(revision, "revision")?;

        Ok(Self {
            repo_id: repo_id.to_string(),
            revision: revision.to_string(),
        })
    }

    fn repo(&self) -> Repo {
        Repo::with_revision(self.repo_id.clone(), RepoType::Model, self.revision.clone())
    }
}

fn validate_hf_relative_path(value: &str, kind: &str) -> anyhow::Result<()> {
    if value.is_empty()
        || value.starts_with('/')
        || value.starts_with('\\')
        || value.contains('\\')
        || value
            .split('/')
            .any(|component| component.is_empty() || matches!(component, "." | ".."))
    {
        anyhow::bail!("invalid Hugging Face {kind}: {value:?}");
    }
    Ok(())
}

/// Validate a path received from the Hub before passing it to hf-hub, whose cache
/// writer joins sibling names directly beneath the snapshot directory.
fn validate_hf_repo_file(filename: &str) -> anyhow::Result<()> {
    validate_hf_relative_path(filename, "repository filename")?;
    if filename.eq_ignore_ascii_case(LORA_COMPLETE_MARKER) {
        anyhow::bail!("reserved Hugging Face repository filename: {filename:?}");
    }
    Ok(())
}

fn validate_hf_commit_sha(sha: &str) -> anyhow::Result<()> {
    if sha.len() != 40 || !sha.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        anyhow::bail!("invalid Hugging Face commit SHA: {sha:?}");
    }
    Ok(())
}

fn hf_home_dir_from_values(
    hf_home: Option<String>,
    xdg_cache_home: Option<String>,
    home: Option<String>,
    userprofile: Option<String>,
) -> PathBuf {
    if let Some(hf_home) = hf_home {
        return PathBuf::from(hf_home);
    }
    if let Some(xdg_cache_home) = xdg_cache_home {
        return PathBuf::from(xdg_cache_home).join("huggingface");
    }

    PathBuf::from(home.or(userprofile).unwrap_or_else(|| ".".to_string()))
        .join(".cache/huggingface")
}

fn hf_cache_dir_from_values(
    hf_hub_cache: Option<String>,
    hf_home: Option<String>,
    xdg_cache_home: Option<String>,
    home: Option<String>,
    userprofile: Option<String>,
) -> PathBuf {
    if let Some(cache_path) = hf_hub_cache {
        return PathBuf::from(cache_path);
    }

    hf_home_dir_from_values(hf_home, xdg_cache_home, home, userprofile).join("hub")
}

fn hf_token_path_from_values(
    hf_token_path: Option<String>,
    hf_home: Option<String>,
    xdg_cache_home: Option<String>,
    home: Option<String>,
    userprofile: Option<String>,
) -> PathBuf {
    hf_token_path.map(PathBuf::from).unwrap_or_else(|| {
        hf_home_dir_from_values(hf_home, xdg_cache_home, home, userprofile).join("token")
    })
}

pub(crate) fn huggingface_cache() -> Cache {
    Cache::new(hf_cache_dir_from_values(
        env::var(env_model::huggingface::HF_HUB_CACHE).ok(),
        env::var(env_model::huggingface::HF_HOME).ok(),
        env::var("XDG_CACHE_HOME").ok(),
        env::var("HOME").ok(),
        env::var("USERPROFILE").ok(),
    ))
}

fn huggingface_token() -> Option<String> {
    let token_path = hf_token_path_from_values(
        env::var(env_model::huggingface::HF_TOKEN_PATH).ok(),
        env::var(env_model::huggingface::HF_HOME).ok(),
        env::var("XDG_CACHE_HOME").ok(),
        env::var("HOME").ok(),
        env::var("USERPROFILE").ok(),
    );
    huggingface_token_from_values(
        env::var(env_model::huggingface::HF_TOKEN).ok(),
        env::var(env_model::huggingface::HUGGING_FACE_HUB_TOKEN).ok(),
        token_path,
    )
}

fn non_empty_token(token: Option<String>) -> Option<String> {
    token
        .map(|token| token.trim().to_string())
        .filter(|token| !token.is_empty())
}

fn huggingface_token_from_values(
    hf_token: Option<String>,
    legacy_token: Option<String>,
    token_path: PathBuf,
) -> Option<String> {
    non_empty_token(hf_token)
        .or_else(|| non_empty_token(legacy_token))
        .or_else(|| non_empty_token(std::fs::read_to_string(token_path).ok()))
}

fn is_complete_hf_snapshot(snapshot: &Path, anchor_file: &str) -> bool {
    snapshot.join(anchor_file).is_file()
        && std::fs::read(snapshot.join(LORA_COMPLETE_MARKER))
            .is_ok_and(|content| content == LORA_COMPLETE_MARKER_CONTENT)
}

pub(crate) fn cached_hf_snapshot(
    cache: &Cache,
    spec: &HfRepoSpec,
    anchor_file: &str,
) -> Option<PathBuf> {
    let repo = spec.repo();
    if let Some(snapshot) = cache
        .repo(repo.clone())
        .get(anchor_file)
        .and_then(|path| path.parent().map(Path::to_path_buf))
        .filter(|snapshot| is_complete_hf_snapshot(snapshot, anchor_file))
    {
        return Some(snapshot);
    }

    if validate_hf_commit_sha(&spec.revision).is_err() {
        return None;
    }

    let snapshot = cache
        .path()
        .join(repo.folder_name())
        .join("snapshots")
        .join(&spec.revision);
    is_complete_hf_snapshot(&snapshot, anchor_file).then_some(snapshot)
}

pub(crate) fn finalize_hf_snapshot(
    cache: &Cache,
    spec: &HfRepoSpec,
    snapshot: &Path,
) -> anyhow::Result<()> {
    let commit = snapshot
        .file_name()
        .and_then(|name| name.to_str())
        .context("Hugging Face snapshot path has no commit SHA")?;
    validate_hf_commit_sha(commit)?;

    let requested_repo = spec.repo();
    let expected_snapshot = cache
        .path()
        .join(requested_repo.folder_name())
        .join("snapshots")
        .join(commit);
    if snapshot != expected_snapshot {
        anyhow::bail!(
            "Hugging Face snapshot {} is outside the expected cache path {}",
            snapshot.display(),
            expected_snapshot.display()
        );
    }

    let mut marker = tempfile::NamedTempFile::new_in(snapshot).with_context(|| {
        format!(
            "creating Hugging Face completion marker in {}",
            snapshot.display()
        )
    })?;
    marker
        .write_all(LORA_COMPLETE_MARKER_CONTENT)
        .context("writing Hugging Face completion marker")?;
    marker
        .as_file()
        .sync_all()
        .context("syncing Hugging Face completion marker")?;
    marker
        .persist(snapshot.join(LORA_COMPLETE_MARKER))
        .map_err(|error| error.error)
        .context("publishing Hugging Face completion marker")?;

    cache
        .repo(requested_repo)
        .create_ref(commit)
        .with_context(|| {
            format!(
                "publishing Hugging Face cache ref {}@{}",
                spec.repo_id, spec.revision
            )
        })?;
    Ok(())
}

fn build_hf_api(cache: Cache) -> anyhow::Result<Api> {
    let mut builder = ApiBuilder::from_cache(cache)
        .with_token(huggingface_token())
        .with_progress(false);

    if let Ok(endpoint) = env::var(env_model::huggingface::HF_ENDPOINT)
        && !endpoint.trim().is_empty()
    {
        builder = builder.with_endpoint(endpoint.trim_end_matches('/').to_string());
    }

    builder.build().context("building Hugging Face Hub client")
}

fn hf_download_timeout() -> Duration {
    Duration::from_secs(
        env::var(LORA_DOWNLOAD_TIMEOUT_SECS)
            .ok()
            .and_then(|value| value.parse().ok())
            .unwrap_or(DEFAULT_LORA_DOWNLOAD_TIMEOUT_SECS),
    )
}

async fn with_hf_timeout<T>(
    timeout: Duration,
    operation: impl Future<Output = Result<T, ApiError>>,
    context: String,
) -> anyhow::Result<T> {
    let timeout_context = format!(
        "{context} timed out after {} seconds; configure {LORA_DOWNLOAD_TIMEOUT_SECS} to adjust the limit",
        timeout.as_secs()
    );
    tokio::time::timeout(timeout, operation)
        .await
        .context(timeout_context)?
        .with_context(|| context)
}

async fn with_hf_lock_retry<T, F, Fut>(
    timeout: Duration,
    mut operation: F,
    context: String,
) -> anyhow::Result<T>
where
    F: FnMut() -> Fut,
    Fut: Future<Output = Result<T, ApiError>>,
{
    let started = tokio::time::Instant::now();
    let timeout_context = format!(
        "{context} timed out after {} seconds; configure {LORA_DOWNLOAD_TIMEOUT_SECS} to adjust the limit",
        timeout.as_secs()
    );

    loop {
        let Some(remaining) = timeout.checked_sub(started.elapsed()) else {
            anyhow::bail!(timeout_context);
        };
        if remaining.is_zero() {
            anyhow::bail!(timeout_context);
        }

        match tokio::time::timeout(remaining, operation()).await {
            Err(_) => anyhow::bail!(timeout_context),
            Ok(Ok(value)) => return Ok(value),
            Ok(Err(ApiError::LockAcquisition(lock_path))) => {
                tracing::debug!(
                    path = %lock_path.display(),
                    "waiting for concurrent Hugging Face cache download"
                );
                let Some(remaining) = timeout.checked_sub(started.elapsed()) else {
                    anyhow::bail!(timeout_context);
                };
                tokio::time::sleep(HF_LOCK_RETRY_INTERVAL.min(remaining)).await;
            }
            Ok(Err(error)) => return Err(error).with_context(|| context.clone()),
        }
    }
}

struct CancelHfRuntimeOnDrop(Option<tokio::sync::oneshot::Sender<()>>);

impl CancelHfRuntimeOnDrop {
    fn disarm(&mut self) {
        self.0 = None;
    }
}

impl Drop for CancelHfRuntimeOnDrop {
    fn drop(&mut self) {
        if let Some(cancel) = self.0.take() {
            let _ = cancel.send(());
        }
    }
}

/// Run hf-hub operations on a dedicated runtime so timing out its parent
/// future also tears down any chunk tasks that hf-hub spawned internally.
async fn run_in_isolated_hf_runtime<T, F, Fut>(operation: F) -> anyhow::Result<T>
where
    T: Send + 'static,
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = anyhow::Result<T>> + 'static,
{
    let (cancel_tx, cancel_rx) = tokio::sync::oneshot::channel();
    let mut cancel_on_drop = CancelHfRuntimeOnDrop(Some(cancel_tx));
    let download = tokio::task::spawn_blocking(move || {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .context("building isolated Hugging Face download runtime")?;
        let result = runtime.block_on(async move {
            tokio::select! {
                biased;
                _ = cancel_rx => anyhow::bail!("Hugging Face download cancelled"),
                result = operation() => result,
            }
        });
        // Async chunk tasks are cancelled as the runtime shuts down. Do not
        // wait indefinitely for unrelated blocking-pool work such as DNS.
        runtime.shutdown_timeout(Duration::ZERO);
        result
    });
    let result = download.await;
    cancel_on_drop.disarm();
    result.context("joining isolated Hugging Face download runtime")?
}

/// Download one immutable repository snapshot into hf-hub's native cache layout.
///
/// The requested branch/tag is resolved once via `info()`. Every sibling is then
/// downloaded through the commit SHA. The caller validates the completed LoRA
/// before publishing the completion marker and mutable cache ref.
pub(crate) async fn download_hf_snapshot(
    cache: &Cache,
    spec: &HfRepoSpec,
) -> anyhow::Result<PathBuf> {
    if is_offline_mode() {
        anyhow::bail!(
            "HF_HUB_OFFLINE is enabled and hf://{}@{} is not fully cached",
            spec.repo_id,
            spec.revision
        );
    }

    let cache = cache.clone();
    let spec = spec.clone();
    run_in_isolated_hf_runtime(move || download_hf_snapshot_inner(cache, spec)).await
}

async fn download_hf_snapshot_inner(cache: Cache, spec: HfRepoSpec) -> anyhow::Result<PathBuf> {
    let api = build_hf_api(cache.clone())?;
    let download_timeout = hf_download_timeout();
    let requested_repo = spec.repo();
    let requested_api = api.repo(requested_repo.clone());
    let info_context = format!(
        "resolving Hugging Face repository {} at revision {}",
        spec.repo_id, spec.revision
    );
    let info = with_hf_timeout(download_timeout, requested_api.info(), info_context).await?;

    validate_hf_commit_sha(&info.sha)?;
    if info.siblings.is_empty() {
        anyhow::bail!(
            "Hugging Face repository {}@{} contains no files",
            spec.repo_id,
            spec.revision
        );
    }
    for sibling in &info.siblings {
        validate_hf_repo_file(&sibling.rfilename)?;
    }

    let pinned_repo = Repo::with_revision(spec.repo_id.clone(), RepoType::Model, info.sha.clone());
    for sibling in &info.siblings {
        let download_context = format!(
            "downloading {} from Hugging Face repository {}@{}",
            sibling.rfilename, spec.repo_id, info.sha
        );
        // `hf-hub` spawns one task per chunk and detaches those tasks when
        // this parent future is dropped. The dedicated runtime surrounding
        // this function is destroyed before the timeout error is returned,
        // which aborts every chunk task before callers can observe failure.
        let download_api = api.clone();
        let download_repo = pinned_repo.clone();
        let filename = sibling.rfilename.clone();
        with_hf_lock_retry(
            download_timeout,
            move || {
                let file_api = download_api.repo(download_repo.clone());
                let filename = filename.clone();
                async move { file_api.get(&filename).await }
            },
            download_context,
        )
        .await?;
    }

    let snapshot = cache
        .path()
        .join(requested_repo.folder_name())
        .join("snapshots")
        .join(&info.sha);
    if !snapshot.is_dir() {
        anyhow::bail!(
            "Hugging Face download completed without snapshot directory {}",
            snapshot.display()
        );
    }

    Ok(snapshot)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_hf_lora_uri_defaults_to_main_and_supports_revision() {
        let main = HfRepoSpec::from_uri("hf://codelion/Qwen3-0.6B-lora").unwrap();
        assert_eq!(main.repo_id, "codelion/Qwen3-0.6B-lora");
        assert_eq!(main.revision, "main");

        let tagged = HfRepoSpec::from_uri("hf://codelion/Qwen3-0.6B-lora@refs/pr/7").unwrap();
        assert_eq!(tagged.repo_id, "codelion/Qwen3-0.6B-lora");
        assert_eq!(tagged.revision, "refs/pr/7");
    }

    #[test]
    fn parse_hf_lora_uri_rejects_malformed_or_unsafe_values() {
        for uri in [
            "s3://bucket/adapter",
            "hf://",
            "hf://org/adapter@",
            "hf://org/adapter@../../outside",
            "hf://org/adapter?download=true",
            "hf://org/adapter#fragment",
        ] {
            assert!(HfRepoSpec::from_uri(uri).is_err(), "accepted {uri}");
        }
    }

    #[test]
    fn validates_hf_repo_sibling_paths() {
        assert!(validate_hf_repo_file("adapter_config.json").is_ok());
        assert!(validate_hf_repo_file("nested/tokenizer.json").is_ok());

        for path in [
            "",
            "../secret",
            "nested/../../secret",
            "/etc/passwd",
            ".dynamo_lora_complete",
            ".DYNAMO_LORA_COMPLETE",
        ] {
            assert!(validate_hf_repo_file(path).is_err(), "accepted {path}");
        }
    }

    #[test]
    fn validates_hf_commit_sha() {
        assert!(validate_hf_commit_sha("0123456789abcdef0123456789abcdef01234567").is_ok());

        for sha in [
            "abc123",
            "0123456789abcdef0123456789abcdef0123456g",
            "../../outside",
        ] {
            assert!(validate_hf_commit_sha(sha).is_err(), "accepted {sha}");
        }
    }

    #[test]
    fn hf_token_path_is_independent_from_hub_cache() {
        assert_eq!(
            hf_token_path_from_values(
                None,
                Some("/hf-home".to_string()),
                Some("/xdg".to_string()),
                Some("/home".to_string()),
                None,
            ),
            PathBuf::from("/hf-home/token")
        );
        assert_eq!(
            hf_token_path_from_values(
                Some("/custom/token".to_string()),
                Some("/hf-home".to_string()),
                Some("/xdg".to_string()),
                Some("/home".to_string()),
                None,
            ),
            PathBuf::from("/custom/token")
        );
    }

    #[test]
    fn hf_token_precedence_includes_legacy_environment_variable() {
        let directory = tempfile::tempdir().unwrap();
        let token_path = directory.path().join("token");
        std::fs::write(&token_path, "file-token\n").unwrap();

        assert_eq!(
            huggingface_token_from_values(
                Some("primary-token".to_string()),
                Some("legacy-token".to_string()),
                token_path.clone(),
            ),
            Some("primary-token".to_string())
        );
        assert_eq!(
            huggingface_token_from_values(
                None,
                Some("legacy-token".to_string()),
                token_path.clone(),
            ),
            Some("legacy-token".to_string())
        );
        assert_eq!(
            huggingface_token_from_values(None, None, token_path),
            Some("file-token".to_string())
        );
    }

    #[test]
    fn hf_cache_dir_matches_huggingface_environment_precedence() {
        assert_eq!(
            hf_cache_dir_from_values(
                Some("/hub-cache".to_string()),
                Some("/hf-home".to_string()),
                Some("/xdg".to_string()),
                Some("/home".to_string()),
                None,
            ),
            PathBuf::from("/hub-cache")
        );
        assert_eq!(
            hf_cache_dir_from_values(
                None,
                Some("/hf-home".to_string()),
                Some("/xdg".to_string()),
                Some("/home".to_string()),
                None,
            ),
            PathBuf::from("/hf-home/hub")
        );
        assert_eq!(
            hf_cache_dir_from_values(
                None,
                None,
                Some("/xdg".to_string()),
                Some("/home".to_string()),
                None,
            ),
            PathBuf::from("/xdg/huggingface/hub")
        );
    }

    #[serial_test::serial]
    #[test]
    fn hf_download_timeout_uses_lora_download_configuration() {
        temp_env::with_var("LORA_DOWNLOAD_TIMEOUT_SECS", Some("17"), || {
            assert_eq!(hf_download_timeout().as_secs(), 17);
        });
        temp_env::with_var("LORA_DOWNLOAD_TIMEOUT_SECS", Some("invalid"), || {
            assert_eq!(hf_download_timeout().as_secs(), 3600);
        });
    }

    #[tokio::test]
    async fn hf_timeout_reports_operation_context() {
        let error = with_hf_timeout(
            std::time::Duration::ZERO,
            std::future::pending::<Result<(), hf_hub::api::tokio::ApiError>>(),
            "testing Hugging Face timeout".to_string(),
        )
        .await
        .unwrap_err();

        assert!(
            error
                .to_string()
                .contains("testing Hugging Face timeout timed out after 0 seconds")
        );
    }

    #[tokio::test]
    async fn hf_lock_contention_is_retried() {
        let attempts = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let operation_attempts = attempts.clone();

        let result = with_hf_lock_retry(
            Duration::from_secs(1),
            move || {
                let attempt = operation_attempts.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                async move {
                    if attempt < 2 {
                        Err(ApiError::LockAcquisition(PathBuf::from("adapter.lock")))
                    } else {
                        Ok("downloaded")
                    }
                }
            },
            "downloading adapter".to_string(),
        )
        .await
        .unwrap();

        assert_eq!(result, "downloaded");
        assert_eq!(attempts.load(std::sync::atomic::Ordering::SeqCst), 3);
    }

    #[tokio::test]
    async fn isolated_hf_runtime_aborts_spawned_tasks_before_returning() {
        let directory = tempfile::tempdir().unwrap();
        let marker = directory.path().join("background-task-finished");
        let background_marker = marker.clone();

        let error = run_in_isolated_hf_runtime(move || async move {
            with_hf_timeout(
                Duration::from_millis(10),
                async move {
                    tokio::spawn(async move {
                        tokio::time::sleep(Duration::from_millis(100)).await;
                        std::fs::write(background_marker, b"finished").unwrap();
                    });
                    std::future::pending::<Result<(), ApiError>>().await
                },
                "testing isolated Hugging Face timeout".to_string(),
            )
            .await
        })
        .await
        .unwrap_err();

        assert!(error.to_string().contains("timed out after 0 seconds"));
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(!marker.exists(), "background task survived timeout");
    }

    #[tokio::test]
    async fn cancelling_hf_runtime_cancels_its_spawned_tasks() {
        let directory = tempfile::tempdir().unwrap();
        let marker = directory.path().join("cancelled-task-finished");
        let background_marker = marker.clone();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();

        let download = tokio::spawn(run_in_isolated_hf_runtime(move || async move {
            tokio::spawn(async move {
                tokio::time::sleep(Duration::from_millis(100)).await;
                std::fs::write(background_marker, b"finished").unwrap();
            });
            started_tx.send(()).unwrap();
            std::future::pending::<anyhow::Result<()>>().await
        }));

        started_rx.await.unwrap();
        download.abort();
        assert!(download.await.unwrap_err().is_cancelled());
        tokio::time::sleep(Duration::from_millis(200)).await;
        assert!(!marker.exists(), "background task survived cancellation");
    }
}
