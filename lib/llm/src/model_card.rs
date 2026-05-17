// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! # Model Deployment Card
//!
//! The ModelDeploymentCard (MDC) is the primary model configuration structure that will be available to any
//! component that needs to interact with the model or its dependent artifacts.
//!
//! The ModelDeploymentCard contains LLM model deployment configuration information:
//! - Display name and service name for the model
//! - Model information (ModelInfoType)
//! - Tokenizer configuration (TokenizerKind)
//! - Prompt formatter settings (PromptFormatterArtifact)

use std::fmt;
use std::path::{Path, PathBuf};
use std::sync::{Arc, OnceLock};

use crate::common::checked_file::CheckedFile;
use crate::entrypoint::RouterConfig;
use crate::local_model::runtime_config::ModelRuntimeConfig;
use crate::model_type::{ModelInput, ModelType};
use anyhow::{Context, Result};
use derive_builder::Builder;
use dynamo_runtime::{slug::Slug, storage::kv};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer as HfTokenizer;

use crate::preprocessor::media::{MediaDecoder, MediaFetcher};
use crate::protocols::TokenIdType;

/// Identify model deployment cards in the key-value store
pub const ROOT_PATH: &str = "v1/mdc";

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum ModelInfoType {
    HfConfigJson(CheckedFile),
}

impl ModelInfoType {
    pub fn checksum(&self) -> String {
        match self {
            ModelInfoType::HfConfigJson(c) => c.checksum().to_string(),
        }
    }

    pub fn is_local(&self) -> bool {
        match self {
            ModelInfoType::HfConfigJson(c) => c.is_local(),
        }
    }

    pub fn update_dir(&mut self, dir: &Path) {
        match self {
            ModelInfoType::HfConfigJson(c) => c.update_dir(dir),
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum TokenizerKind {
    HfTokenizerJson(CheckedFile),
    TikTokenModel(CheckedFile),
}

impl TokenizerKind {
    pub fn checksum(&self) -> String {
        match self {
            TokenizerKind::HfTokenizerJson(c) | TokenizerKind::TikTokenModel(c) => {
                c.checksum().to_string()
            }
        }
    }

    pub fn is_local(&self) -> bool {
        match self {
            TokenizerKind::HfTokenizerJson(c) | TokenizerKind::TikTokenModel(c) => c.is_local(),
        }
    }

    pub fn update_dir(&mut self, dir: &Path) {
        match self {
            TokenizerKind::HfTokenizerJson(c) | TokenizerKind::TikTokenModel(c) => {
                c.update_dir(dir)
            }
        }
    }
}

/// Supported types of prompt formatters.
///
/// We need a way to associate the prompt formatter template definition with an associated
/// data model which is expected for rendering.
///
/// All current prompt formatters are Jinja2 templates which use the OpenAI ChatCompletionRequest
/// format. However, we currently do not have a discovery path to know if the model supports tool use
/// unless we inspect the template.
///
/// TODO(): Add an enum for the PromptFormatDataModel with at minimum arms for:
/// - OaiChat
/// - OaiChatToolUse
#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum PromptFormatterArtifact {
    HfTokenizerConfigJson(CheckedFile),
    #[serde(rename = "hf_chat_template", alias = "hf_chat_template_jinja")]
    HfChatTemplateJinja {
        is_custom: bool,
        file: CheckedFile,
    },
    HfChatTemplateJson {
        is_custom: bool,
        file: CheckedFile,
    },
}

impl PromptFormatterArtifact {
    pub fn checksum(&self) -> String {
        match self {
            PromptFormatterArtifact::HfTokenizerConfigJson(c) => c.checksum().to_string(),
            PromptFormatterArtifact::HfChatTemplateJinja { file: c, .. }
            | PromptFormatterArtifact::HfChatTemplateJson { file: c, .. } => {
                c.checksum().to_string()
            }
        }
    }

    /// Is this file available locally
    pub fn is_local(&self) -> bool {
        match self {
            PromptFormatterArtifact::HfTokenizerConfigJson(c) => c.is_local(),
            PromptFormatterArtifact::HfChatTemplateJinja { file: c, .. }
            | PromptFormatterArtifact::HfChatTemplateJson { file: c, .. } => c.is_local(),
        }
    }

    pub fn update_dir(&mut self, dir: &Path) {
        match self {
            PromptFormatterArtifact::HfTokenizerConfigJson(c) => c.update_dir(dir),
            PromptFormatterArtifact::HfChatTemplateJinja { file: c, .. }
            | PromptFormatterArtifact::HfChatTemplateJson { file: c, .. } => c.update_dir(dir),
        }
    }

    pub fn is_custom(&self) -> bool {
        match self {
            PromptFormatterArtifact::HfTokenizerConfigJson(_) => false,
            PromptFormatterArtifact::HfChatTemplateJinja { is_custom, .. }
            | PromptFormatterArtifact::HfChatTemplateJson { is_custom, .. } => *is_custom,
        }
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "snake_case")]
pub enum PromptContextMixin {
    /// Support OAI Chat Messages and Tools
    OaiChat,

    /// Enables templates with `{{datetime}}` to be rendered with the current date and time.
    Llama3DateTime,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
#[serde(rename_all = "snake_case")]
pub enum GenerationConfig {
    HfGenerationConfigJson(CheckedFile),
}

impl GenerationConfig {
    pub fn checksum(&self) -> String {
        match self {
            GenerationConfig::HfGenerationConfigJson(c) => c.checksum().to_string(),
        }
    }

    pub fn is_local(&self) -> bool {
        match self {
            GenerationConfig::HfGenerationConfigJson(c) => c.is_local(),
        }
    }

    pub fn update_dir(&mut self, dir: &Path) {
        match self {
            GenerationConfig::HfGenerationConfigJson(c) => c.update_dir(dir),
        }
    }
}

/// Check if our model only has config fields for a Mistral-format model.
fn is_exclusively_mistral_model(directory: &Path) -> bool {
    !directory.join("config.json").exists() && directory.join("params.json").exists()
}

/// MDC cache: `blobs/<blake3>` + `by-slug/<slug>/<mdcsum>/<filename>`.
/// The `<mdcsum>` segment mirrors HF Hub's `snapshots/<rev>/` and
/// isolates worker sets that share a model name but publish different
/// file content.
fn mdc_cache_root() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".cache/dynamo/mdc")
}

fn mdc_blobs_dir() -> anyhow::Result<PathBuf> {
    let dir = mdc_cache_root().join("blobs");
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating MDC blobs dir {}", dir.display()))?;
    Ok(dir)
}

fn mdc_slug_dir(slug: &Slug, mdcsum: &str) -> anyhow::Result<PathBuf> {
    let dir = mdc_cache_root()
        .join("by-slug")
        .join(slug.to_string())
        .join(mdcsum);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating MDC slug dir {}", dir.display()))?;
    Ok(dir)
}

/// Per-call tmp path next to `dest`. pid + uuid suffix keeps tmps
/// disjoint across concurrent callers so `rename(2)` is the only
/// synchronization point — never `create(tmp)`.
fn unique_tmp_path(dest: &Path) -> PathBuf {
    let suffix = format!(
        "tmp.{}.{}",
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    );
    let name = dest
        .file_name()
        .map(|f| f.to_string_lossy().into_owned())
        .unwrap_or_default();
    let mut p = dest.to_path_buf();
    p.set_file_name(format!("{name}.{suffix}"));
    p
}

/// RAII guard that unlinks a tmp path on drop. `dismiss()` cancels the
/// cleanup once the tmp has been consumed by a successful rename. Drop
/// ignores ENOENT — safe under double-cleanup races.
struct TmpGuard {
    path: Option<PathBuf>,
}

impl TmpGuard {
    fn new(path: PathBuf) -> Self {
        Self { path: Some(path) }
    }
    fn dismiss(&mut self) {
        self.path = None;
    }
}

impl Drop for TmpGuard {
    fn drop(&mut self) {
        if let Some(p) = self.path.take() {
            let _ = std::fs::remove_file(&p);
        }
    }
}

/// Atomic publish: stage via `f(&tmp)`, then `rename(tmp -> dest)`.
/// Concurrent-safe because tmps are per-call (see [`unique_tmp_path`])
/// and `rename(2)` is atomic + overwrites. Tmp is unlinked on any
/// failure path including async cancellation — see [`TmpGuard`].
async fn stage_and_rename<F, Fut>(dest: &Path, f: F) -> anyhow::Result<()>
where
    F: FnOnce(PathBuf) -> Fut,
    Fut: std::future::Future<Output = anyhow::Result<()>>,
{
    let tmp = unique_tmp_path(dest);
    let mut guard = TmpGuard::new(tmp.clone());
    f(tmp.clone()).await?;
    tokio::fs::rename(&tmp, dest)
        .await
        .with_context(|| format!("renaming {} -> {}", tmp.display(), dest.display()))?;
    guard.dismiss();
    Ok(())
}

/// Sync sibling of [`stage_and_rename`] for cheap operations
/// (e.g., creating a symlink).
fn stage_and_rename_sync<F>(dest: &Path, f: F) -> anyhow::Result<()>
where
    F: FnOnce(&Path) -> anyhow::Result<()>,
{
    let tmp = unique_tmp_path(dest);
    let mut guard = TmpGuard::new(tmp.clone());
    f(&tmp)?;
    std::fs::rename(&tmp, dest)
        .with_context(|| format!("renaming {} -> {}", tmp.display(), dest.display()))?;
    guard.dismiss();
    Ok(())
}

/// Concurrent-safe via [`stage_and_rename_sync`].
fn symlink_force(target: &Path, link: &Path) -> anyhow::Result<()> {
    stage_and_rename_sync(link, |tmp| {
        #[cfg(unix)]
        std::os::unix::fs::symlink(target, tmp)
            .with_context(|| format!("symlinking {} -> {}", tmp.display(), target.display()))?;
        #[cfg(not(unix))]
        std::fs::copy(target, tmp)
            .map(|_| ())
            .with_context(|| format!("copying {} -> {}", target.display(), tmp.display()))?;
        Ok(())
    })
}

/// 1 GiB cap on metadata fetch — realistic files are <20 MiB. Bounds
/// disk usage if a worker advertises a bogus `CheckedFile.size`, and is
/// the fallback when `size` is absent on a `CheckedFile`.
const ABSOLUTE_MAX_METADATA_BYTES: u64 = 1024 * 1024 * 1024;

/// File extensions that identify model weights. Used by the hf:// sibling
/// harvest to skip weight blobs (which `hub::from_hf(_, ignore_weights=true)`
/// also already filters at download, but the snapshot dir may contain
/// pre-existing weights from a prior unrestricted pull).
///
/// `.safetensors.index.json` is correctly kept: extension is `.json`.
fn is_weight_file(path: &Path) -> bool {
    matches!(
        path.extension().and_then(|e| e.to_str()),
        Some("safetensors" | "bin" | "gguf" | "onnx" | "tflite" | "h5" | "pt" | "pth" | "msgpack")
    )
}

/// Parent directory of a `file://` URI when it resolves to a real local
/// directory. Returns `None` for any other scheme or unreachable path.
fn file_uri_parent(uri: &str) -> Option<PathBuf> {
    let url = url::Url::parse(uri).ok()?;
    if url.scheme() != "file" {
        return None;
    }
    let path = url.to_file_path().ok()?;
    let parent = path.parent()?;
    parent.is_dir().then(|| parent.to_path_buf())
}

/// Symlink non-weight files from `snapshot_dir` into `slug_dir`. Picks up
/// `preprocessor_config.json` and other sibling files that
/// `from_pretrained(slug_dir)` consumers need.
///
/// Names in `typed_filenames` are owned by the resolve loop's typed-slot
/// pass — never overwritten. Every other harvested sibling is re-linked
/// unconditionally so a re-registration with the same `mdcsum` but a
/// different upstream snapshot picks up fresh contents (mdcsum doesn't
/// cover harvested files).
fn harvest_siblings(
    snapshot_dir: &Path,
    slug_dir: &Path,
    typed_filenames: &std::collections::HashSet<String>,
) -> anyhow::Result<()> {
    let entries = match std::fs::read_dir(snapshot_dir) {
        Ok(e) => e,
        Err(e) => {
            tracing::debug!(
                snapshot = %snapshot_dir.display(),
                error = %e,
                "sibling harvest: snapshot dir unreadable, skipping",
            );
            return Ok(());
        }
    };
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() || is_weight_file(&path) {
            continue;
        }
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) if !n.is_empty() => n.to_owned(),
            _ => continue,
        };
        if typed_filenames.contains(&name) {
            continue;
        }
        let dst = slug_dir.join(&name);
        // Resolve through the canonical target so a downstream
        // `canonicalize` lands on a stable blob path rather than
        // chasing snapshot-dir symlinks. `symlink_force` is idempotent
        // and atomic via `stage_and_rename_sync`.
        let target = std::fs::canonicalize(&path).unwrap_or(path);
        symlink_force(&target, &dst)
            .with_context(|| format!("harvesting {} -> {}", target.display(), dst.display()))?;
        tracing::debug!(
            file = %name,
            target = %target.display(),
            "harvested sibling into slug_dir",
        );
    }
    Ok(())
}

/// Stage `uri` into `dest`, verifying staged bytes against `expected`
/// before publishing. Schemes: `http(s)`, `file`, `hf`. For `hf://`,
/// `hf_snapshots` must already contain the resolved repo path —
/// caller is expected to pre-resolve once per repo.
async fn resolve_uri(
    client: &reqwest::Client,
    uri: &str,
    expected: &CheckedFile,
    dest: &Path,
    hf_snapshots: &std::collections::HashMap<String, PathBuf>,
) -> anyhow::Result<()> {
    let cap = expected
        .size()
        .unwrap_or(ABSOLUTE_MAX_METADATA_BYTES)
        .min(ABSOLUTE_MAX_METADATA_BYTES);

    let parsed = url::Url::parse(uri).with_context(|| format!("parsing artifact uri: {uri}"))?;

    if dest.exists() {
        if CheckedFile::from_disk(dest).is_ok_and(|cf| cf.checksum() == expected.checksum()) {
            return Ok(());
        }
        tracing::warn!(dest = %dest.display(), "MDC cache blob failed re-verification; refetching");
        let _ = std::fs::remove_file(dest);
    }

    stage_and_rename(dest, |tmp| async move {
        match parsed.scheme() {
            "http" | "https" => stream_to_tmp(client, uri, &tmp, cap).await?,
            "file" => {
                let path = parsed
                    .to_file_path()
                    .map_err(|()| anyhow::anyhow!("invalid file:// uri: {uri}"))?;
                copy_to_tmp(&path, &tmp, cap).await?;
            }
            "hf" => {
                let (repo, filename) = parse_hf_uri(uri)?;
                let snapshot = hf_snapshots
                    .get(&repo)
                    .with_context(|| format!("hf snapshot not pre-resolved for {repo}"))?;
                copy_to_tmp(&snapshot.join(&filename), &tmp, cap).await?;
            }
            scheme => anyhow::bail!("unsupported artifact uri scheme: {scheme} (uri: {uri})"),
        }

        // Re-blake3 the staged bytes via the same `from_disk` path the
        // worker used at registration so the comparison is bit-identical.
        let actual = CheckedFile::from_disk(&tmp)?;
        if actual.checksum() != expected.checksum() {
            anyhow::bail!(
                "checksum mismatch for {uri}: expected {}, got {}",
                expected.checksum(),
                actual.checksum()
            );
        }
        Ok(())
    })
    .await
}

/// Stream HTTP body to `tmp`, capped at `cap` bytes. Pre-checks
/// `Content-Length` if present; otherwise the post-write check
/// catches overage.
async fn stream_to_tmp(
    client: &reqwest::Client,
    uri: &str,
    tmp: &Path,
    cap: u64,
) -> anyhow::Result<()> {
    use futures::TryStreamExt;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio_util::io::StreamReader;

    let response = client
        .get(uri)
        .send()
        .await
        .with_context(|| format!("fetching {uri}"))?;
    if !response.status().is_success() {
        anyhow::bail!("fetching {uri} returned status {}", response.status());
    }
    if let Some(content_length) = response.content_length()
        && content_length > cap
    {
        anyhow::bail!("{uri} reports {content_length} bytes, exceeds cap {cap}");
    }

    let stream = response.bytes_stream().map_err(std::io::Error::other);
    // `cap + 1` so a body of exactly `cap` passes but anything larger
    // trips the `written > cap` check below.
    let mut reader = StreamReader::new(stream).take(cap + 1);
    let mut file = tokio::fs::File::create(tmp)
        .await
        .with_context(|| format!("creating {}", tmp.display()))?;
    let written = tokio::io::copy(&mut reader, &mut file)
        .await
        .with_context(|| format!("streaming body from {uri} to {}", tmp.display()))?;
    file.flush()
        .await
        .with_context(|| format!("flushing {}", tmp.display()))?;
    if written > cap {
        anyhow::bail!("{uri} body exceeds cap {cap}");
    }
    Ok(())
}

async fn copy_to_tmp(src: &Path, tmp: &Path, cap: u64) -> anyhow::Result<()> {
    let metadata = tokio::fs::metadata(src)
        .await
        .with_context(|| format!("reading metadata for {}", src.display()))?;
    if metadata.len() > cap {
        anyhow::bail!(
            "{} is {} bytes, exceeds cap {}",
            src.display(),
            metadata.len(),
            cap
        );
    }
    tokio::fs::copy(src, tmp)
        .await
        .with_context(|| format!("copying {} -> {}", src.display(), tmp.display()))?;
    Ok(())
}

/// Parse `hf://repo[@rev]/filename` into `(repo[@rev], filename)`.
fn parse_hf_uri(uri: &str) -> anyhow::Result<(String, String)> {
    let body = uri
        .strip_prefix("hf://")
        .with_context(|| format!("expected hf:// scheme, got: {uri}"))?;
    let (repo, filename) = body
        .rsplit_once('/')
        .with_context(|| format!("hf:// uri must end in /filename, got: {uri}"))?;
    if repo.is_empty() || filename.is_empty() {
        anyhow::bail!("malformed hf:// uri: {uri}");
    }
    Ok((repo.to_string(), filename.to_string()))
}

fn checked_file_uri(
    cf: &CheckedFile,
    source: &str,
    local_model_path: Option<&Path>,
    is_custom: bool,
) -> anyhow::Result<String> {
    use std::borrow::Cow;

    // Coerce path-only into a synthetic file:// URL up front so the
    // scheme dispatch below covers both shapes uniformly. `absolute`
    // (not `canonicalize`) — worker paths often don't exist here.
    let url: Cow<url::Url> = if let Some(u) = cf.url() {
        Cow::Borrowed(u)
    } else {
        let Some(p) = cf.path() else {
            anyhow::bail!("CheckedFile has neither path nor url");
        };
        let abs = std::path::absolute(p)?;
        Cow::Owned(
            url::Url::from_file_path(&abs)
                .map_err(|()| anyhow::anyhow!("invalid file path: {}", abs.display()))?,
        )
    };

    match url.scheme() {
        "http" | "https" | "hf" => Ok(url.to_string()),
        "file" => {
            // worker location → --model-path → hf://. Basename + checksum preserved.
            // is_custom slots aren't published on HF, so rung 4 errors instead.
            let path = url
                .to_file_path()
                .map_err(|()| anyhow::anyhow!("invalid file uri: {url}"))?;
            let filename = path
                .file_name()
                .and_then(|f| f.to_str())
                .with_context(|| format!("no filename in file uri: {url}"))?;
            if path.exists() {
                return Ok(url.to_string());
            }
            if let Some(prefix) = local_model_path {
                let local = prefix.join(filename);
                if local.exists() {
                    return file_uri_for(&local);
                }
            }
            if is_custom {
                anyhow::bail!(
                    "custom file {filename} not reachable on this host \
                     (worker path {} missing, no --model-path overlay); \
                     custom files aren't published on HF, so ensure the \
                     file exists at the same path on every host (shared \
                     mount) or pass --model-path with the same basename",
                    path.display()
                );
            }
            Ok(format!("hf://{source}/{filename}"))
        }
        _ => Ok(url.to_string()),
    }
}

fn file_uri_for(p: &Path) -> anyhow::Result<String> {
    Ok(url::Url::from_file_path(std::path::absolute(p)?)
        .map_err(|()| anyhow::anyhow!("invalid file path: {}", p.display()))?
        .to_string())
}

fn uri_basename(uri: &str) -> anyhow::Result<String> {
    url::Url::parse(uri)
        .with_context(|| format!("parsing uri: {uri}"))?
        .path_segments()
        .and_then(|mut s| s.rfind(|s| !s.is_empty()))
        .map(String::from)
        .with_context(|| format!("no basename in uri: {uri}"))
}

fn pf_checked_file(p: &PromptFormatterArtifact) -> &CheckedFile {
    match p {
        PromptFormatterArtifact::HfTokenizerConfigJson(cf)
        | PromptFormatterArtifact::HfChatTemplateJinja { file: cf, .. }
        | PromptFormatterArtifact::HfChatTemplateJson { file: cf, .. } => cf,
    }
}

fn pf_checked_file_mut(p: &mut PromptFormatterArtifact) -> &mut CheckedFile {
    match p {
        PromptFormatterArtifact::HfTokenizerConfigJson(cf)
        | PromptFormatterArtifact::HfChatTemplateJinja { file: cf, .. }
        | PromptFormatterArtifact::HfChatTemplateJson { file: cf, .. } => cf,
    }
}

#[derive(Serialize, Deserialize, Clone, Debug, Builder, Default)]
pub struct ModelDeploymentCard {
    /// Human readable model name, e.g. "Meta Llama 3.1 8B Instruct"
    pub display_name: String,

    // Cache the Slugified display_name so we can share references to it
    slug: Slug,

    /// Original HuggingFace repository path for downloading model files.
    /// When `display_name` is customized (e.g., via `--served-model-name`),
    /// this field preserves the original repository path needed for downloads.
    /// Falls back to `display_name` if not set.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_path: Option<String>,

    /// Model information
    pub model_info: Option<ModelInfoType>,

    /// Tokenizer configuration
    pub tokenizer: Option<TokenizerKind>,

    /// Prompt Formatter configuration
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_formatter: Option<PromptFormatterArtifact>,

    /// chat template may be stored as a separate file instead of in `prompt_formatter`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub chat_template_file: Option<PromptFormatterArtifact>,

    /// Generation config - default sampling params
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub gen_config: Option<GenerationConfig>,

    /// Prompt Formatter Config
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prompt_context: Option<Vec<PromptContextMixin>>,

    /// Max context (in number of tokens) this model can handle
    pub context_length: u32,

    /// Size of a KV cache block.
    /// Passed to the engine, KV router, and trace replay hash path.
    pub kv_cache_block_size: u32,

    /// How many times a request can be migrated to another worker if the HTTP server lost
    /// connection to the current worker.
    pub migration_limit: u32,

    /// Specifies whether the model is a chat, completions, etc model.
    pub model_type: ModelType,

    /// Specifies the model input type.
    /// `Tokens` for engines that expect pre-processed input.
    /// `Text` for engines that take care of pre-processing themselves.
    pub model_input: ModelInput,

    /// LoRA metadata for routing
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora: Option<LoraInfo>,

    /// User-defined metadata for custom worker behavior
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_data: Option<serde_json::Value>,

    #[serde(default)]
    pub runtime_config: ModelRuntimeConfig,

    /// Media decoding configuration
    #[serde(default)]
    pub media_decoder: Option<MediaDecoder>,

    /// Media fetching configuration
    #[serde(default)]
    pub media_fetcher: Option<MediaFetcher>,

    /// Per-worker-set router configuration override.
    /// When set, the frontend watcher uses this instead of the global frontend router config.
    /// Falls back to the frontend-level config when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub router_config: Option<RouterConfig>,

    #[serde(skip, default)]
    checksum: OnceLock<String>,
}

/// LoRA adapter information for routing decisions
#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct LoraInfo {
    /// LoRA adapter name (e.g., "customer-123-v2")
    pub name: String,

    /// Maximum number of LoRA adapters that can be loaded at once on a single GPU
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub max_gpu_lora_count: Option<u32>,
}

impl ModelDeploymentCard {
    pub fn builder() -> ModelDeploymentCardBuilder {
        ModelDeploymentCardBuilder::default()
    }

    /// Create a ModelDeploymentCard where only the name is filled in.
    ///
    /// Single-process setups don't need an MDC to communicate model details, but it
    /// simplifies the code to assume we always have one. This is how you get one in those
    /// cases. A quasi-null object: <https://en.wikipedia.org/wiki/Null_object_pattern>
    pub fn with_name_only(name: &str) -> ModelDeploymentCard {
        ModelDeploymentCard {
            display_name: name.to_string(),
            slug: Slug::from_string(name),
            ..Default::default()
        }
    }

    /// Load a model deployment card from a JSON file
    pub fn load_from_json_file<P: AsRef<Path>>(file: P) -> std::io::Result<Self> {
        let contents = std::fs::read_to_string(&file)?;
        Ok(serde_json::from_str(&contents).inspect_err(|err| {
            crate::log_json_err(&file.as_ref().display().to_string(), &contents, err)
        })?)
    }

    /// Load a model deployment card from a JSON string
    pub fn load_from_json_str(contents: &str) -> Result<Self, anyhow::Error> {
        Ok(serde_json::from_str(contents)
            .inspect_err(|err| crate::log_json_err("unknown", contents, err))?)
    }

    //
    // Methods
    //

    /// Save the model deployment card to a JSON file
    pub fn save_to_json_file(&self, file: &str) -> Result<(), anyhow::Error> {
        std::fs::write(file, self.to_json()?)?;
        Ok(())
    }

    #[inline]
    pub fn name(&self) -> &str {
        &self.display_name
    }

    #[inline]
    pub fn slug(&self) -> &Slug {
        &self.slug
    }

    /// Serialize the model deployment card to a JSON string
    pub fn to_json(&self) -> Result<String, anyhow::Error> {
        Ok(serde_json::to_string(self)?)
    }

    pub fn mdcsum(&self) -> &str {
        self.checksum
            .get_or_init(|| {
                // Only include the important fields
                let mut bytes_to_hash: Vec<u8> = Vec::with_capacity(512);
                bytes_to_hash.extend(self.display_name.as_bytes());
                if let Some(source_path) = self.source_path.as_ref() {
                    bytes_to_hash.extend(source_path.as_bytes());
                }

                // The files can be either a URL or a local path, so we ignore that and hash their
                // checksum instead, which won't change wherever they are.

                if let Some(model_info) = self.model_info.as_ref() {
                    bytes_to_hash.extend(model_info.checksum().as_bytes());
                }
                if let Some(tokenizer) = self.tokenizer.as_ref() {
                    bytes_to_hash.extend(tokenizer.checksum().as_bytes());
                }
                if let Some(prompt_formatter) = self.prompt_formatter.as_ref() {
                    bytes_to_hash.extend(prompt_formatter.checksum().as_bytes());
                }
                if let Some(chat_template) = self.chat_template_file.as_ref() {
                    bytes_to_hash.extend(chat_template.checksum().as_bytes());
                }
                if let Some(gen_config) = self.gen_config.as_ref() {
                    bytes_to_hash.extend(gen_config.checksum().as_bytes());
                }

                if let Some(prompt_context_vec) = self.prompt_context.as_ref() {
                    // Paste it as the bytes of the debug format. It's a Vec of enum, so this should be
                    // fine. If the debug representation changes that only happens in a new release.
                    bytes_to_hash.extend(format!("{prompt_context_vec:?}").as_bytes());
                }
                bytes_to_hash.extend(self.context_length.to_be_bytes());
                bytes_to_hash.extend(self.kv_cache_block_size.to_be_bytes());

                if let Some(router_config) = self.router_config.as_ref()
                    && let Ok(bytes) = serde_json::to_vec(router_config)
                {
                    // Hash router_config separately so we extend bytes_to_hash with a
                    // fixed-size digest (32 bytes) rather than the full JSON payload.
                    // [gluo TODO] take checksum() approach that is the same as above,
                    // along with this effort, we should reorganize where RouterConfig
                    // should be defined.
                    bytes_to_hash.extend(blake3::hash(&bytes).as_bytes());
                }

                blake3::hash(&bytes_to_hash).to_string()
            })
            .as_ref()
    }

    /// Is this a full model card with tokenizer?
    /// There are cases where we have a placeholder card (see `with_name_only`).
    pub fn has_tokenizer(&self) -> bool {
        self.tokenizer.is_some()
    }

    /// Load the tokenizer as a generic, backend-agnostic `Tokenizer` trait object.
    /// This supports both HuggingFace `tokenizer.json` and tiktoken `.model`/`.tiktoken` files.
    ///
    /// When the `DYN_TOKENIZER=fastokens` env var is set, uses `fastokens` for encoding
    pub fn tokenizer(&self) -> anyhow::Result<crate::tokenizers::Tokenizer> {
        let use_fast = match std::env::var("DYN_TOKENIZER") {
            Ok(v) if v == "fastokens" => true,
            Ok(v) if v == "default" || v.is_empty() => false,
            Ok(v) => {
                tracing::warn!(
                    value = %v,
                    "Unrecognized DYN_TOKENIZER value, expected 'fastokens' or 'default'; falling back to default"
                );
                false
            }
            Err(_) => false,
        };

        match &self.tokenizer {
            Some(TokenizerKind::HfTokenizerJson(checked_file)) => {
                let p = checked_file.path().ok_or_else(|| {
                    anyhow::anyhow!("Tokenizer is URL-backed ({:?})", checked_file.url())
                })?;

                // Try fastokens backend if requested
                if use_fast {
                    if let Some(path_str) = p.to_str() {
                        match crate::tokenizers::FastTokenizer::from_file(path_str) {
                            Ok(fast) => {
                                tracing::info!("Using fastokens tokenizer backend");
                                return Ok(crate::tokenizers::Tokenizer::from(Arc::new(fast)));
                            }
                            Err(e) => {
                                tracing::warn!(
                                    %e,
                                    "Failed to load fastokens, falling back to HuggingFace"
                                );
                            }
                        }
                    } else {
                        tracing::warn!(
                            path = %p.display(),
                            "Tokenizer path contains non-UTF-8 characters, skipping fastokens; falling back to HuggingFace"
                        );
                    }
                }

                let hf = HfTokenizer::from_file(p)
                    .inspect_err(|err| {
                        if let Some(serde_err) = err.downcast_ref::<serde_json::Error>()
                            && let Ok(contents) = std::fs::read_to_string(p)
                        {
                            crate::log_json_err(&p.display().to_string(), &contents, serde_err);
                        }
                    })
                    .map_err(anyhow::Error::msg)
                    .with_context(|| p.display().to_string())?;
                Ok(crate::tokenizers::Tokenizer::from(Arc::new(
                    crate::tokenizers::HuggingFaceTokenizer::from_tokenizer(hf),
                )))
            }
            Some(TokenizerKind::TikTokenModel(checked_file)) => {
                let p = checked_file.path().ok_or_else(|| {
                    anyhow::anyhow!("Tokenizer is URL-backed ({:?})", checked_file.url())
                })?;
                let path_str = p.to_str().ok_or_else(|| {
                    anyhow::anyhow!("Tokenizer path contains invalid UTF-8: {}", p.display())
                })?;
                let tokenizer = crate::tokenizers::TikTokenTokenizer::from_file_auto(path_str)
                    .with_context(|| {
                        format!("Failed to load tiktoken tokenizer from {}", p.display())
                    })?;
                Ok(crate::tokenizers::Tokenizer::from(Arc::new(tokenizer)))
            }
            None => {
                anyhow::bail!(
                    "ModelDeploymentCard for '{}' does not have a tokenizer. \
                     Provide a supported tokenizer file (tokenizer.json, tiktoken.model, \
                     or *.tiktoken), use --use-<framework>-tokenizer to delegate \
                     tokenization to the backend, or use a non-Rust chat processor \
                     (e.g. --dyn-chat-processor vllm).",
                    self.display_name
                );
            }
        }
    }

    pub(crate) fn set_source_path(&mut self, source_path: PathBuf) {
        self.source_path = Some(source_path.display().to_string());
    }

    /// Allow user to override the name we register this model under.
    /// Corresponds to vllm's `--served-model-name`.
    pub fn set_name(&mut self, name: &str) {
        self.display_name = name.to_string();
        self.slug = Slug::from_string(name);
    }

    pub fn source_path(&self) -> &str {
        self.source_path.as_ref().unwrap_or(&self.display_name)
    }

    /// Build an in-memory ModelDeploymentCard from a folder containing config.json,
    /// tokenizer.json and tokenizer_config.json (i.e. a huggingface repo checkout).
    /// Optional custom template.
    pub fn load_from_disk(
        config_path: impl AsRef<Path>,
        custom_template_path: Option<&Path>,
    ) -> anyhow::Result<ModelDeploymentCard> {
        Self::from_local_path(config_path.as_ref(), custom_template_path)
    }

    pub fn requires_preprocessing(&self) -> bool {
        matches!(self.model_input, ModelInput::Tokens)
    }

    /// Iterate populated metadata slots in deterministic order:
    /// model_info, tokenizer, prompt_formatter, chat_template_file,
    /// gen_config. Each entry is `(file, is_custom)` — `is_custom` is
    /// only ever true for operator-supplied chat templates, which
    /// can't fall back to HF.
    ///
    /// TODO(gh-8749): external preprocessors (vllm/sglang) read
    /// `from_pretrained(slug_dir)` and may expect siblings outside the
    /// typed slots — `preprocessor_config.json`, `special_tokens_map.json`,
    /// `added_tokens.json`, etc. Add an `extra_files: Vec<CheckedFile>`
    /// MDC field so the worker can advertise everything in its model dir
    /// minus weights. Frontend pipeline is already generic over slot count.
    pub fn iter_metadata_files(&self) -> Vec<(&CheckedFile, bool)> {
        let mut out: Vec<(&CheckedFile, bool)> = Vec::with_capacity(5);
        if let Some(ModelInfoType::HfConfigJson(cf)) = self.model_info.as_ref() {
            out.push((cf, false));
        }
        if let Some(TokenizerKind::HfTokenizerJson(cf) | TokenizerKind::TikTokenModel(cf)) =
            self.tokenizer.as_ref()
        {
            out.push((cf, false));
        }
        if let Some(p) = self.prompt_formatter.as_ref() {
            out.push((pf_checked_file(p), p.is_custom()));
        }
        if let Some(c) = self.chat_template_file.as_ref() {
            out.push((pf_checked_file(c), c.is_custom()));
        }
        if let Some(GenerationConfig::HfGenerationConfigJson(cf)) = self.gen_config.as_ref() {
            out.push((cf, false));
        }
        out
    }

    /// Mutable mirror of [`Self::iter_metadata_files`].
    pub fn iter_metadata_files_mut(&mut self) -> Vec<(&mut CheckedFile, bool)> {
        let mut out: Vec<(&mut CheckedFile, bool)> = Vec::with_capacity(5);
        if let Some(ModelInfoType::HfConfigJson(cf)) = self.model_info.as_mut() {
            out.push((cf, false));
        }
        if let Some(TokenizerKind::HfTokenizerJson(cf) | TokenizerKind::TikTokenModel(cf)) =
            self.tokenizer.as_mut()
        {
            out.push((cf, false));
        }
        if let Some(p) = self.prompt_formatter.as_mut() {
            let is_custom = p.is_custom();
            out.push((pf_checked_file_mut(p), is_custom));
        }
        if let Some(c) = self.chat_template_file.as_mut() {
            let is_custom = c.is_custom();
            out.push((pf_checked_file_mut(c), is_custom));
        }
        if let Some(GenerationConfig::HfGenerationConfigJson(cf)) = self.gen_config.as_mut() {
            out.push((cf, false));
        }
        out
    }

    async fn resolve_metadata_files(
        &mut self,
        local_model_path: Option<&Path>,
    ) -> anyhow::Result<()> {
        let source = self.source_path().to_string();
        let mdcsum = self.mdcsum().to_string();
        let blobs = mdc_blobs_dir()?;
        let slug_dir = mdc_slug_dir(&self.slug, &mdcsum)?;

        let entries: Vec<(String, CheckedFile)> = self
            .iter_metadata_files()
            .into_iter()
            .map(|(cf, is_custom)| {
                Ok((
                    checked_file_uri(cf, &source, local_model_path, is_custom)?,
                    cf.clone(),
                ))
            })
            .collect::<anyhow::Result<_>>()?;

        // Pre-resolve hf:// repos once per unique repo; otherwise the
        // resolve loop would call hub::from_hf N times for one model.
        let mut hf_snapshots: std::collections::HashMap<String, PathBuf> =
            std::collections::HashMap::new();
        for (uri, _) in &entries {
            if uri.starts_with("hf://") {
                let (repo, _) = parse_hf_uri(uri)?;
                if let std::collections::hash_map::Entry::Vacant(e) = hf_snapshots.entry(repo) {
                    let repo_name = e.key().clone();
                    let snap = crate::hub::from_hf(&repo_name, /* ignore_weights = */ true)
                        .await
                        .with_context(|| format!("hub::from_hf({repo_name})"))?;
                    e.insert(snap);
                }
            }
        }

        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .timeout(std::time::Duration::from_secs(300))
            .build()
            .context("building http client for metadata fetch")?;
        for (uri, expected) in &entries {
            let filename = uri_basename(uri)?;
            // Hash is validated at MDC deserialize; safe as a path component.
            let blake3_hex = expected.checksum().hash();
            let blob = blobs.join(blake3_hex);
            tracing::debug!(filename = %filename, uri = %uri, blake3 = %blake3_hex, "resolving");
            resolve_uri(&client, uri, expected, &blob, &hf_snapshots).await?;
            symlink_force(&blob, &slug_dir.join(&filename))?;
        }
        tracing::debug!(
            display_name = %self.display_name,
            artifact_count = entries.len(),
            cache_root = %mdc_cache_root().display(),
            "resolved model metadata files",
        );

        // Harvest non-weight siblings (preprocessor_config.json, …) from
        // every snapshot dir we touched. Typed-slot basenames are passed
        // through so the harvest never overwrites them. No-op for `http://`.
        let typed_filenames: std::collections::HashSet<String> = entries
            .iter()
            .filter_map(|(uri, _)| uri_basename(uri).ok())
            .collect();
        let mut snapshot_dirs: std::collections::HashSet<PathBuf> =
            hf_snapshots.values().cloned().collect();
        for (uri, _) in &entries {
            if let Some(parent) = file_uri_parent(uri) {
                snapshot_dirs.insert(parent);
            }
        }
        for snap in &snapshot_dirs {
            harvest_siblings(snap, &slug_dir, &typed_filenames)?;
        }

        // Pass 3: rewrite cf.path to the cache symlink so downstream
        // tokenizer/config loaders read from a verified location.
        for (cf, _) in self.iter_metadata_files_mut() {
            cf.update_dir(&slug_dir);
        }
        Ok(())
    }

    /// Resolve every metadata `CheckedFile` through the cache: fetch,
    /// blake3-verify, content-address. `local_model_path` (frontend's
    /// `--model-path`) supplies a fallback directory for `file://`
    /// slots whose worker-published location is unreachable.
    pub async fn download_config(&mut self, local_model_path: Option<&Path>) -> anyhow::Result<()> {
        // TensorBased models don't use metadata files — backend handles
        // everything.
        if self.model_type.supports_tensor() {
            tracing::debug!(
                display_name = %self.display_name,
                "Skipping config download for TensorBased model"
            );
            return Ok(());
        }
        // Single resolve pipeline: every CheckedFile (URL or local
        // path, existing or missing) flows through resolve_uri,
        // blake3-verifies, lands in the MDC cache. No new/legacy split.
        self.resolve_metadata_files(local_model_path).await
    }

    /// Re-write all the local disk paths as a URL. Do this before publishing the MDC.
    /// The opposite of `move_to_url` is `update_dir`.
    pub fn move_to_url(&mut self, base_url: &str) -> anyhow::Result<()> {
        macro_rules! change {
            ($field:expr, $enum_variant:path) => {
                if let Some($enum_variant(src_file)) = $field.as_mut()
                    && let Some(filename) = src_file
                        .path()
                        .and_then(|p| p.file_name())
                        .and_then(|f| f.to_str())
                        .map(|f| f.to_string())
                {
                    let hf_url = url::Url::parse(base_url)
                        .and_then(|u| u.join(filename.as_ref()))
                        .context(filename)?;
                    src_file.move_to_url(hf_url);
                }
            };
        }

        // config.json
        change!(self.model_info, ModelInfoType::HfConfigJson);

        // generation_config.json
        change!(self.gen_config, GenerationConfig::HfGenerationConfigJson);

        // tokenizer_config.json
        change!(
            self.prompt_formatter,
            PromptFormatterArtifact::HfTokenizerConfigJson
        );

        // tokenizer.json or tiktoken.model
        change!(self.tokenizer, TokenizerKind::HfTokenizerJson);
        change!(self.tokenizer, TokenizerKind::TikTokenModel);

        // We only "move" the chat template if it came form the repo. If we have a custom template
        // file we cannot download that from HF.
        if let Some(
            PromptFormatterArtifact::HfChatTemplateJinja {
                file: src_file,
                is_custom,
            }
            | PromptFormatterArtifact::HfChatTemplateJson {
                file: src_file,
                is_custom,
            },
        ) = self.chat_template_file.as_mut()
        {
            if *is_custom {
                tracing::info!(
                    "Detected custom chat template. Ensure file exists in the same location on all hosts."
                );
            } else if let Some(filename) = src_file
                .path()
                .and_then(|p| p.file_name())
                .and_then(|f| f.to_str())
                .map(|f| f.to_string())
            {
                let hf_url = url::Url::parse(base_url)
                    .and_then(|u| u.join(filename.as_ref()))
                    .context(filename)?;
                src_file.move_to_url(hf_url);
            }
        }
        Ok(())
    }

    /// Creates a ModelDeploymentCard from a local directory path.
    ///
    /// Currently HuggingFace format is supported and following files are expected:
    /// - config.json: Model configuration in HuggingFace format
    /// - tokenizer.json: Tokenizer configuration in HuggingFace format
    /// - tokenizer_config.json: Optional prompt formatter configuration
    ///
    /// # Arguments
    /// * `local_root_dir` - Path to the local model directory
    ///
    /// # Errors
    /// Returns an error if:
    /// - The path doesn't exist or isn't a directory
    /// - The path contains invalid Unicode characters
    /// - Required model files are missing or invalid
    fn from_local_path(
        local_path: impl AsRef<Path>,
        custom_template_path: Option<&Path>,
    ) -> anyhow::Result<Self> {
        check_valid_local_repo_path(&local_path)?;
        Self::from_repo_checkout(&local_path, custom_template_path)
    }

    fn from_repo_checkout(
        local_path: impl AsRef<Path>,
        custom_template_path: Option<&Path>,
    ) -> anyhow::Result<Self> {
        let local_path = local_path.as_ref();

        // This is usually the right choice
        let context_length =
            crate::file_json_field(&local_path.join("config.json"), "max_position_embeddings")
                // But sometimes this is
                .or_else(|_| {
                    crate::file_json_field(
                        &local_path.join("tokenizer_config.json"),
                        "model_max_length",
                    )
                })
                // If neither of those are present let the engine default it
                .unwrap_or(0);

        let is_mistral_model = is_exclusively_mistral_model(local_path);

        let (model_info, tokenizer, gen_config, prompt_formatter) = if !is_mistral_model {
            (
                Some(ModelInfoType::from_disk(local_path)?),
                TokenizerKind::from_disk(local_path)?,
                GenerationConfig::from_disk(local_path).ok(),
                PromptFormatterArtifact::from_disk(local_path)?,
            )
        } else {
            (None, None, None, None)
        };

        // Load chat template - either custom or from repo
        let chat_template_file = if is_mistral_model {
            None
        } else if let Some(template_path) = custom_template_path {
            if !template_path.exists() {
                anyhow::bail!(
                    "Custom template file does not exist: {}",
                    template_path.display()
                );
            }

            // Verify the file is readable
            let _template_content = std::fs::read_to_string(template_path).with_context(|| {
                format!(
                    "Failed to read custom template file: {}",
                    template_path.display()
                )
            })?;

            Some(PromptFormatterArtifact::HfChatTemplateJinja {
                is_custom: custom_template_path.is_some(),
                file: CheckedFile::from_disk(template_path)?,
            })
        } else {
            PromptFormatterArtifact::chat_template_from_disk(local_path)?
        };

        // This gets replaced when we `set_name`
        let display_name = local_path.display().to_string();

        Ok(Self {
            slug: Slug::from_string(&display_name),
            display_name,
            source_path: None,
            model_info,
            tokenizer,
            gen_config,
            prompt_formatter,
            chat_template_file,
            prompt_context: None, // TODO - auto-detect prompt context
            context_length,
            kv_cache_block_size: 0, // set later
            migration_limit: 0,
            model_type: Default::default(),  // set later
            model_input: Default::default(), // set later
            lora: None,
            user_data: None,
            runtime_config: ModelRuntimeConfig::default(),
            media_decoder: None,
            media_fetcher: None,
            router_config: None,
            checksum: OnceLock::new(),
        })
    }
}

impl PartialEq for ModelDeploymentCard {
    fn eq(&self, other: &ModelDeploymentCard) -> bool {
        self.mdcsum() == other.mdcsum()
    }
}

/// A ModelDeploymentCard is published a single time per instance and never updated.
impl kv::Versioned for ModelDeploymentCard {
    fn revision(&self) -> u64 {
        0
    }

    fn set_revision(&mut self, _revision: u64) {}
}

impl fmt::Display for ModelDeploymentCard {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.slug())
    }
}
pub trait ModelInfo: Send + Sync {
    /// Model type
    fn model_type(&self) -> String;

    /// Token ID for the beginning of sequence (optional - not all models have it)
    fn bos_token_id(&self) -> Option<TokenIdType>;

    /// Token ID for the end of sequence
    fn eos_token_ids(&self) -> Vec<TokenIdType>;

    /// Maximum position embeddings / max sequence length
    /// TODO: This is only used in a single test, no other code. Remove?
    fn max_position_embeddings(&self) -> Option<usize>;

    /// Vocabulary size
    /// TODO: This is only used in a single test, no other code. Remove?
    fn vocab_size(&self) -> Option<usize>;
}

impl ModelInfoType {
    pub fn get_model_info(&self) -> Result<Arc<dyn ModelInfo>> {
        match self {
            Self::HfConfigJson(checked_file) => {
                let Some(path) = checked_file.path() else {
                    anyhow::bail!("model info is not a local path: {checked_file:?}");
                };
                Ok(HFConfig::from_json_file(path)?)
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HFConfig {
    /// denotes the mixin to the flattened data model which can be present
    /// in the config.json file
    architectures: Vec<String>,

    /// general model type
    model_type: String,

    text_config: Option<HFTextConfig>,

    // Sometimes it's inside HFTextConfig, sometimes it's here
    eos_token_id: Option<serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct HFTextConfig {
    // Optional - not all models have a bos_token_id
    bos_token_id: Option<TokenIdType>,

    eos_token_id: Option<serde_json::Value>,

    #[serde(default)]
    final_eos_token_ids: Vec<TokenIdType>,

    /// max sequence length
    max_position_embeddings: Option<usize>,

    /// number of layers in the model
    /// Optional because some multimodal models (e.g., LLaVA) don't include this in text_config
    num_hidden_layers: Option<usize>,

    /// number of attention heads in the model
    num_attention_heads: Option<usize>,

    /// Vocabulary size
    vocab_size: Option<usize>,
}

impl HFConfig {
    fn from_json_file<P: AsRef<Path>>(file: P) -> Result<Arc<dyn ModelInfo>> {
        let file_path = file.as_ref();
        let contents = std::fs::read_to_string(file_path)?;
        let mut config: Self = json_five::from_str(&contents)
            .inspect_err(|err| {
                tracing::error!(path=%file_path.display(), %err, "Failed to parse config.json as JSON5");
            })?;
        if config.text_config.is_none() {
            let text_config: HFTextConfig = json_five::from_str(&contents)
                .inspect_err(|err| {
                    tracing::error!(path=%file_path.display(), %err, "Failed to parse text config from config.json as JSON5");
                })?;
            config.text_config = Some(text_config);
        }

        let Some(text_config) = config.text_config.as_mut() else {
            anyhow::bail!(
                "Missing text config fields (model_type, eos_token_ids, etc) in config.json"
            );
        };

        let gencfg_path = file_path
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .join("generation_config.json");

        // bos_token_id is optional - not all models have it
        // Try to load from generation_config.json if not in config.json
        if text_config.bos_token_id.is_none() {
            text_config.bos_token_id =
                crate::file_json_field::<TokenIdType>(&gencfg_path, "bos_token_id").ok();
        }

        // TODO: refactor this when we switch to per-architecture tokenization
        // eos_token_id can appear in multiple places, and as suggested by HuggingFace
        // community that the priority should be:
        // 1. generation_config.json;
        // 2. config.json, or text_config field in config.json.
        // https://github.com/huggingface/transformers/issues/25395#issuecomment-1671863257
        let mut final_eos_token_ids: Vec<TokenIdType> = {
                // Firstly check the generation_config.json
                crate::file_json_field::<serde_json::Value>(&gencfg_path, "eos_token_id")
                .inspect_err(
                    |err| tracing::warn!(%err, "Missing eos_token_id in generation_config.json"),
                )
                .ok().and_then(|v| {
                    if v.is_number() {
                        v.as_number()
                            .and_then(|n| n.as_u64())
                            .map(|n| vec![n as TokenIdType])
                    } else if v.is_array() {
                        let arr = v.as_array().unwrap();
                        Some(
                            arr.iter()
                                .filter_map(|inner_v| {
                                    inner_v
                                        .as_number()
                                        .and_then(|n| n.as_u64())
                                        .map(|n| n as TokenIdType)
                                })
                                .collect(),
                        )
                    } else {
                        None
                    }
                })
            }.or_else(|| {
                // Check config.json and text_config
                config
                .eos_token_id
                .as_ref()
                .or(text_config.eos_token_id.as_ref())
                .and_then(|v| {
                    if v.is_number() {
                        v.as_number()
                            .and_then(|n| n.as_u64())
                            .map(|n| vec![n as TokenIdType])
                    } else {
                        serde_json::from_value(v.clone())
                            .map(Some)
                            .unwrap_or_else(|err| {
                                tracing::error!(
                                    ?v,
                                    path = %file_path.display(),
                                    "eos_token_id is not a number or an array, cannot deserialize: {err}",
                                );
                                None
                            })
                    }
                })
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "missing eos_token_id in config.json and generation_config.json, cannot load"
                )
            })?;
        // Also check tokenizer_config.json for the tokenizer's eos_token.
        // Some models (e.g. Qwen3.5) have text_config.eos_token_id = <|endoftext|>
        // but the tokenizer's eos_token is <|im_end|> — the token the model actually
        // emits to end generation. Merge the tokenizer's EOS into the set so both
        // are recognized as stop tokens.
        let tokenizer_cfg_path = file_path
            .parent()
            .unwrap_or_else(|| Path::new(""))
            .join("tokenizer_config.json");
        if let Ok(tokenizer_eos_id) =
            resolve_eos_token_id_from_tokenizer_config(&tokenizer_cfg_path)
            && !final_eos_token_ids.contains(&tokenizer_eos_id)
        {
            final_eos_token_ids.push(tokenizer_eos_id);
        }

        text_config.final_eos_token_ids = final_eos_token_ids;

        Ok(Arc::new(config))
    }
}

/// Resolve the tokenizer's `eos_token` to a token ID by reading `tokenizer_config.json`.
///
/// Reads the `eos_token` field (string) and looks it up in `added_tokens_decoder`
/// to find the corresponding token ID. This handles models where the tokenizer's
/// EOS token differs from `config.json`'s `eos_token_id`.
fn resolve_eos_token_id_from_tokenizer_config(path: &Path) -> anyhow::Result<TokenIdType> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read tokenizer_config.json: {:?}", path))?;
    let config: serde_json::Value = serde_json::from_str(&contents)
        .with_context(|| format!("Failed to parse tokenizer_config.json: {:?}", path))?;

    // Get eos_token — can be a plain string or a dict with a "content" field (older HF format)
    let eos_token_str = match config.get("eos_token") {
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(serde_json::Value::Object(obj)) => obj
            .get("content")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow::anyhow!("eos_token is an object without 'content' field"))?,
        _ => anyhow::bail!("eos_token not found or not a string in tokenizer_config.json"),
    };

    // Look up the token string in added_tokens_decoder to get its ID
    let added_tokens = config
        .get("added_tokens_decoder")
        .and_then(|v| v.as_object())
        .ok_or_else(|| {
            anyhow::anyhow!("added_tokens_decoder not found in tokenizer_config.json")
        })?;

    for (id_str, token_info) in added_tokens {
        let content = token_info
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if content == eos_token_str {
            let token_id: TokenIdType = id_str.parse().with_context(|| {
                format!(
                    "Failed to parse token ID '{}' from added_tokens_decoder",
                    id_str
                )
            })?;
            return Ok(token_id);
        }
    }

    anyhow::bail!(
        "eos_token '{}' not found in added_tokens_decoder",
        eos_token_str
    )
}

impl ModelInfo for HFConfig {
    fn model_type(&self) -> String {
        self.model_type.clone()
    }

    fn bos_token_id(&self) -> Option<TokenIdType> {
        self.text_config.as_ref().and_then(|tc| tc.bos_token_id)
    }

    fn eos_token_ids(&self) -> Vec<TokenIdType> {
        self.text_config
            .as_ref()
            .unwrap()
            .final_eos_token_ids
            .clone()
    }

    fn max_position_embeddings(&self) -> Option<usize> {
        self.text_config.as_ref().unwrap().max_position_embeddings
    }

    fn vocab_size(&self) -> Option<usize> {
        self.text_config.as_ref().unwrap().vocab_size
    }
}

impl ModelInfoType {
    pub fn from_disk(directory: &Path) -> Result<Self> {
        let f = CheckedFile::from_disk(directory.join("config.json")).with_context(|| {
            format!(
                "unable to extract config.json from directory {}",
                directory.display()
            )
        })?;
        Ok(Self::HfConfigJson(f))
    }
}

impl GenerationConfig {
    pub fn from_disk(directory: &Path) -> Result<Self> {
        let f = CheckedFile::from_disk(directory.join("generation_config.json")).with_context(
            || {
                format!(
                    "unable to extract generation_config from directory {}",
                    directory.display()
                )
            },
        )?;
        Ok(Self::HfGenerationConfigJson(f))
    }
}

impl PromptFormatterArtifact {
    pub fn from_disk(directory: &Path) -> Result<Option<Self>> {
        // we should only error if we expect a prompt formatter and it's not found
        // right now, we don't know when to expect it, so we just return Ok(Some/None)
        match CheckedFile::from_disk(directory.join("tokenizer_config.json")) {
            Ok(f) => Ok(Some(Self::HfTokenizerConfigJson(f))),
            Err(_) => Ok(None),
        }
    }

    pub fn chat_template_from_disk(directory: &Path) -> Result<Option<Self>> {
        // Try chat_template.jinja first (raw Jinja template)
        let jinja_path = directory.join("chat_template.jinja");
        if jinja_path.exists() {
            let f = CheckedFile::from_disk(&jinja_path)
                .with_context(|| format!("Failed to load {}", jinja_path.display()))?;
            return Ok(Some(Self::HfChatTemplateJinja {
                file: f,
                is_custom: false,
            }));
        }

        // Try chat_template.json (JSON with "chat_template" key, e.g. Qwen3-Omni)
        let json_path = directory.join("chat_template.json");
        if json_path.exists() {
            let f = CheckedFile::from_disk(&json_path)
                .with_context(|| format!("Failed to load {}", json_path.display()))?;
            return Ok(Some(Self::HfChatTemplateJson {
                file: f,
                is_custom: false,
            }));
        }

        Ok(None)
    }
}

impl TokenizerKind {
    /// Try to discover a tokenizer in the given directory.
    ///
    /// Returns `Ok(Some(..))` when a supported tokenizer is found,
    /// `Ok(None)` when no tokenizer files are present (e.g. models that
    /// ship only `vocab.json` + `merges.txt`), and `Err` for ambiguous
    /// layouts or filesystem failures that should be treated as hard errors.
    pub fn from_disk(directory: &Path) -> Result<Option<Self>> {
        // Helper: probe a single well-known file.  Returns Ok(None) when the
        // file simply does not exist, Ok(Some(..)) on success, and Err for
        // anything else (unreadable file, checksum failure, etc.).
        fn probe(path: std::path::PathBuf) -> Result<Option<CheckedFile>> {
            if !path.exists() {
                return Ok(None);
            }
            Ok(Some(CheckedFile::from_disk(path)?))
        }

        // 1. Try tokenizer.json (HuggingFace)
        if let Some(f) = probe(directory.join("tokenizer.json"))? {
            return Ok(Some(Self::HfTokenizerJson(f)));
        }

        // 2. Try tiktoken.model
        if let Some(f) = probe(directory.join("tiktoken.model"))? {
            return Ok(Some(Self::TikTokenModel(f)));
        }

        // 3. Search for any *.tiktoken file
        let tiktoken_files: Vec<_> = std::fs::read_dir(directory)
            .with_context(|| format!("Failed to read directory {}", directory.display()))?
            .collect::<std::io::Result<Vec<_>>>()
            .with_context(|| format!("Failed to iterate directory {}", directory.display()))?
            .into_iter()
            .filter(|entry| entry.path().extension().is_some_and(|e| e == "tiktoken"))
            .collect();

        if tiktoken_files.len() == 1 {
            let f = CheckedFile::from_disk(tiktoken_files[0].path())?;
            return Ok(Some(Self::TikTokenModel(f)));
        } else if tiktoken_files.len() > 1 {
            let names: Vec<_> = tiktoken_files
                .iter()
                .map(|e| e.path().display().to_string())
                .collect();
            anyhow::bail!(
                "Multiple .tiktoken files found in {}: {:?}. Cannot determine which to use.",
                directory.display(),
                names
            );
        }

        tracing::warn!(
            "No supported tokenizer found in {} \
             (expected tokenizer.json or a tiktoken file). \
             Features that depend on the Rust tokenizer will not be available.",
            directory.display()
        );
        Ok(None)
    }
}

/// Checks if the provided path is a valid local repository path.
///
/// # Arguments
/// * `path` - Path to validate
///
/// # Errors
/// Returns an error if the path doesn't exist or isn't a directory
fn check_valid_local_repo_path(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    if !path.exists() {
        return Err(anyhow::anyhow!(
            "Model path does not exist: {}",
            path.display()
        ));
    }

    if !path.is_dir() {
        return Err(anyhow::anyhow!(
            "Model path is not a directory: {}",
            path.display()
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::HFConfig;
    use std::collections::HashSet;
    use std::path::{Path, PathBuf};

    #[test]
    pub fn test_config_json_llama3() -> anyhow::Result<()> {
        let config_file = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-llama-3.1-8b-instruct/config.json");
        let config = HFConfig::from_json_file(&config_file)?;
        assert_eq!(config.bos_token_id(), Some(128000));
        // eos_token_ids can be in any order as long as the set is correct
        let eos_token_id_set: HashSet<_> = config.eos_token_ids().iter().cloned().collect();
        assert_eq!(eos_token_id_set, vec![128001, 128009].into_iter().collect());
        Ok(())
    }

    #[test]
    pub fn test_config_json_llama4() -> anyhow::Result<()> {
        let config_file = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/Llama-4-Scout-17B-16E-Instruct/config.json");
        let config = HFConfig::from_json_file(&config_file)?;
        assert_eq!(config.bos_token_id(), Some(200000));
        Ok(())
    }

    /// The Python JSON parser accepts `Infinity` as a numeric value. This is explicitly against the
    /// JSON spec, but inevitably people rely on it, so we have to allow it.
    /// We treat that file as JSON5 (a lenient superset of JSON) to be able to parse it.
    #[test]
    fn test_invalid_json_but_py_accepts_it() {
        dynamo_runtime::logging::init();
        let path = "tests/data/sample-models/NVIDIA-Nemotron-Nano-12B-v2-Base/config.json";
        let _ = HFConfig::from_json_file(path).unwrap();
    }

    /// Qwen3.5 models have text_config.eos_token_id = 248044 (<|endoftext|>) but the
    /// tokenizer's eos_token is <|im_end|> (248046). The model actually emits <|im_end|>
    /// to end generation. Verify that both are included in the resolved EOS set.
    #[test]
    fn test_config_json_qwen35_eos_from_tokenizer() -> anyhow::Result<()> {
        let config_file = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-qwen3.5-0.8B/config.json");
        let config = HFConfig::from_json_file(&config_file)?;
        let eos_token_id_set: HashSet<_> = config.eos_token_ids().iter().cloned().collect();
        // Must include both: 248044 (<|endoftext|>) from text_config and
        // 248046 (<|im_end|>) from tokenizer_config.json
        assert!(
            eos_token_id_set.contains(&248044),
            "Should contain text_config eos_token_id (248044 <|endoftext|>)"
        );
        assert!(
            eos_token_id_set.contains(&248046),
            "Should contain tokenizer eos_token (248046 <|im_end|>)"
        );
        Ok(())
    }

    fn test_cf(uri: &str, size: u64) -> super::CheckedFile {
        serde_json::from_value(serde_json::json!({
            "path": uri,
            "checksum": format!("blake3:{}", "0".repeat(64)),
            "size": size,
        }))
        .unwrap()
    }

    async fn assert_resolve_uri_rejects(body: &[u8], declared_size: u64, expected_err: &str) {
        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("GET", "/f")
            .with_status(200)
            .with_body(body)
            .create_async()
            .await;
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("f");
        let url = format!("{}/f", server.url());
        let result = super::resolve_uri(
            &reqwest::Client::new(),
            &url,
            &test_cf(&url, declared_size),
            &dest,
            &std::collections::HashMap::new(),
        )
        .await;
        let msg = result.expect_err("expected error").to_string();
        assert!(
            msg.contains(expected_err),
            "want `{expected_err}` in: {msg}"
        );
        assert!(!dest.exists(), "no file should be written");
    }

    #[tokio::test]
    async fn resolve_uri_http_rejects_checksum_mismatch() {
        assert_resolve_uri_rejects(b"hello world", 11, "checksum mismatch").await;
    }

    #[tokio::test]
    async fn resolve_uri_http_rejects_oversize_body() {
        assert_resolve_uri_rejects(b"x".repeat(35).as_slice(), 8, "exceeds cap").await;
    }

    /// Cache hit re-verifies the on-disk blob; mismatch → unlink + refetch.
    #[tokio::test]
    async fn resolve_uri_refetches_on_cache_hit_mismatch() {
        let body: &[u8] = b"valid-bytes-for-blob";
        let dir = tempfile::tempdir().unwrap();
        let valid = dir.path().join("valid");
        std::fs::write(&valid, body).unwrap();
        let cf = super::CheckedFile::from_disk(&valid).unwrap();

        let mut server = mockito::Server::new_async().await;
        let _m = server
            .mock("GET", "/f")
            .with_status(200)
            .with_body(body)
            .create_async()
            .await;
        let url = format!("{}/f", server.url());

        let dest = dir.path().join("blob");
        std::fs::write(&dest, b"corrupt-bytes").unwrap();

        super::resolve_uri(
            &reqwest::Client::new(),
            &url,
            &cf,
            &dest,
            &std::collections::HashMap::new(),
        )
        .await
        .expect("resolve_uri should refetch and succeed");

        let after = std::fs::read(&dest).unwrap();
        assert_eq!(after, body, "blob should have been replaced");
    }

    /// `parse_hf_uri` round-trip — `hf://repo/filename` parses into
    /// `(repo, filename)` and rejects malformed inputs.
    #[test]
    fn parse_hf_uri_roundtrip() {
        let (repo, filename) = super::parse_hf_uri("hf://Qwen/Qwen3-0.6B/tokenizer.json").unwrap();
        assert_eq!(repo, "Qwen/Qwen3-0.6B");
        assert_eq!(filename, "tokenizer.json");

        assert!(super::parse_hf_uri("hf://just-a-name").is_err());
        assert!(super::parse_hf_uri("https://example.com/x").is_err());
    }

    /// HF-cache-style snapshot of TinyLlama: per-file symlinks into a
    /// sibling `blobs/<hash>` dir. The symlink layout is what triggers
    /// the canonicalize trap `resolve_metadata_files` has to handle.
    fn hf_cache_fixture(workspace: &Path) -> anyhow::Result<PathBuf> {
        use std::hash::{Hash, Hasher};
        let snapshot = workspace.join("snapshots/abc");
        let blobs = workspace.join("blobs");
        std::fs::create_dir_all(&snapshot)?;
        std::fs::create_dir_all(&blobs)?;
        let src =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/sample-models/TinyLlama_v1.1");
        for entry in std::fs::read_dir(&src)? {
            let entry = entry?;
            let name = entry.file_name();
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            name.to_string_lossy().hash(&mut hasher);
            let blob = format!("blob-{:x}", hasher.finish());
            std::fs::copy(entry.path(), blobs.join(&blob))?;
            #[cfg(unix)]
            std::os::unix::fs::symlink(
                Path::new("../..").join("blobs").join(&blob),
                snapshot.join(name),
            )?;
        }
        Ok(snapshot)
    }

    #[tokio::test]
    #[serial_test::serial]
    async fn download_config_pipelines_local_files_through_cache() -> anyhow::Result<()> {
        let workspace = tempfile::tempdir()?;
        let snapshot = hf_cache_fixture(workspace.path())?;
        let home = tempfile::tempdir()?;
        let home_path = home.path().to_path_buf();

        temp_env::async_with_vars([("HOME", Some(home.path()))], async {
            let mut mdc = super::ModelDeploymentCard::load_from_disk(&snapshot, None)?;
            let slug = mdc.slug.clone();
            let mdcsum = mdc.mdcsum().to_string();
            mdc.download_config(None).await?;

            let blobs = home_path.join(".cache/dynamo/mdc/blobs");
            let snap = home_path
                .join(".cache/dynamo/mdc/by-slug")
                .join(slug.to_string())
                .join(&mdcsum);

            assert!(snap.join("config.json").exists());
            assert!(snap.join("tokenizer.json").exists());
            assert!(snap.join("generation_config.json").exists());

            // Sibling harvest: TinyLlama_v1.1 fixture ships
            // `special_tokens_map.json` and `tokenizer.model` outside the
            // typed slots — both must land in slug_dir for
            // `from_pretrained()` to see a complete model dir.
            assert!(snap.join("special_tokens_map.json").exists());
            assert!(snap.join("tokenizer.model").exists());

            for (cf, _) in mdc.iter_metadata_files() {
                let path = cf.path().expect("post-download local path");
                assert!(path.starts_with(&snap));
                assert!(std::fs::canonicalize(path)?.starts_with(&blobs));
            }
            Ok::<_, anyhow::Error>(())
        })
        .await
    }

    /// Build a `CheckedFile` whose wire `path` field is `repr` — parses
    /// as a URL when `repr` has a scheme, otherwise as a `PathBuf`.
    fn cf_for(repr: &str) -> super::CheckedFile {
        serde_json::from_value(serde_json::json!({
            "path": repr,
            "checksum": format!("blake3:{}", "0".repeat(64)),
            "size": 1u64,
        }))
        .unwrap()
    }

    #[test]
    fn checked_file_uri_passes_through_remote_urls() {
        let tmp = tempfile::tempdir().unwrap();
        for url in [
            "http://worker:8080/v1/metadata/slug/base/config.json",
            "hf://Qwen/Qwen3-0.6B/config.json",
        ] {
            let got =
                super::checked_file_uri(&cf_for(url), "Qwen/Qwen3-0.6B", Some(tmp.path()), false)
                    .unwrap();
            assert_eq!(got, url);
        }
    }

    #[test]
    fn checked_file_uri_uses_local_model_path_when_worker_path_unreachable() {
        let cf = cf_for("/nonexistent/worker/path/config.json");
        let local = tempfile::tempdir().unwrap();
        let local_cfg = local.path().join("config.json");
        std::fs::write(&local_cfg, b"").unwrap();
        let got =
            super::checked_file_uri(&cf, "Qwen/Qwen3-0.6B", Some(local.path()), false).unwrap();
        assert_eq!(
            got,
            url::Url::from_file_path(&local_cfg).unwrap().to_string()
        );
    }

    /// Rung 4: when worker path is missing and `--model-path` doesn't
    /// supply the basename, synthesize hf://; if `is_custom`, error
    /// with a clear operator-action message instead (HF doesn't host
    /// custom slots).
    #[test]
    fn checked_file_uri_rung_4_hf_fallback_or_custom_error() {
        let cf = cf_for("/nonexistent/worker/path/template.jinja");

        let got = super::checked_file_uri(&cf, "Qwen/Qwen3-0.6B", None, false).unwrap();
        assert_eq!(got, "hf://Qwen/Qwen3-0.6B/template.jinja");

        let err = super::checked_file_uri(&cf, "Qwen/Qwen3-0.6B", None, true)
            .expect_err("custom slot must error instead of falling back to HF");
        let msg = err.to_string();
        assert!(msg.contains("template.jinja"), "wrong error: {msg}");
        assert!(msg.contains("custom"), "wrong error: {msg}");
        assert!(
            msg.contains("--model-path") || msg.contains("shared mount"),
            "wrong error: {msg}"
        );
    }

    /// Dropping `stage_and_rename`'s future mid-await (caller cancellation)
    /// must still unlink the tmp file via the [`TmpGuard`] drop path.
    #[tokio::test]
    async fn stage_and_rename_unlinks_tmp_on_cancel() {
        let dir = tempfile::tempdir().unwrap();
        let dest = dir.path().join("dest");
        let leaked_before: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .collect();
        assert!(leaked_before.is_empty(), "tempdir starts empty");

        let fut = super::stage_and_rename(&dest, |tmp| async move {
            std::fs::write(&tmp, b"partial").unwrap();
            // Mimic a worker abort mid-fetch: yield, then never resume.
            std::future::pending::<()>().await;
            Ok(())
        });
        // Drop the future before it can complete (cancellation).
        {
            let _ = tokio::time::timeout(std::time::Duration::from_millis(50), fut).await;
        }

        // Give Drop a moment to run.
        tokio::task::yield_now().await;

        let leaked_after: Vec<_> = std::fs::read_dir(dir.path())
            .unwrap()
            .filter_map(Result::ok)
            .map(|e| e.path())
            .collect();
        assert!(
            leaked_after.is_empty(),
            "TmpGuard should have unlinked the tmp on cancel; leaked: {leaked_after:?}"
        );
        assert!(!dest.exists(), "dest must not exist after cancel");
    }

    /// Brings in the sibling that `lightseek-mm` needs.
    #[test]
    fn harvest_brings_in_non_weight_siblings() -> anyhow::Result<()> {
        let snap = tempfile::tempdir()?;
        let slug = tempfile::tempdir()?;
        std::fs::write(snap.path().join("preprocessor_config.json"), b"pre")?;
        std::fs::write(snap.path().join("tokenizer.model"), b"sp")?;
        // `.safetensors.index.json` is `.json`, not a weight — must be kept.
        std::fs::write(snap.path().join("model.safetensors.index.json"), b"idx")?;
        super::harvest_siblings(snap.path(), slug.path(), &Default::default())?;
        assert!(slug.path().join("preprocessor_config.json").exists());
        assert!(slug.path().join("tokenizer.model").exists());
        assert!(slug.path().join("model.safetensors.index.json").exists());
        Ok(())
    }

    /// Weight blobs stay out so the metadata cache doesn't bloat.
    #[test]
    fn harvest_skips_weight_blobs() -> anyhow::Result<()> {
        let snap = tempfile::tempdir()?;
        let slug = tempfile::tempdir()?;
        for weight in ["model.safetensors", "pytorch_model.bin", "model.gguf"] {
            std::fs::write(snap.path().join(weight), b"WEIGHTS")?;
        }
        super::harvest_siblings(snap.path(), slug.path(), &Default::default())?;
        for weight in ["model.safetensors", "pytorch_model.bin", "model.gguf"] {
            assert!(!slug.path().join(weight).exists());
        }
        Ok(())
    }

    /// Missing snapshot dir is best-effort: no error, no work.
    #[test]
    fn harvest_tolerates_missing_snapshot() -> anyhow::Result<()> {
        let slug = tempfile::tempdir()?;
        super::harvest_siblings(
            &slug.path().join("does-not-exist"),
            slug.path(),
            &Default::default(),
        )?;
        Ok(())
    }

    /// Names in `typed_filenames` survive a harvest pass even when the
    /// snapshot dir contains a different file at the same basename — the
    /// resolve loop's typed slots own those.
    #[test]
    fn harvest_preserves_typed_filenames() -> anyhow::Result<()> {
        let blob_dir = tempfile::tempdir()?;
        let snap = tempfile::tempdir()?;
        let slug = tempfile::tempdir()?;

        // Typed slot: blob in the dynamo cache; slug_dir links to it.
        let typed_blob = blob_dir.path().join("config-blob");
        std::fs::write(&typed_blob, b"typed-slot-content")?;
        super::symlink_force(&typed_blob, &slug.path().join("config.json"))?;

        // Snapshot dir has a different `config.json` — the stale-payload
        // case the harvest must NOT import over the typed slot.
        std::fs::write(snap.path().join("config.json"), b"STALE-DO-NOT-IMPORT")?;
        std::fs::write(snap.path().join("special_tokens_map.json"), b"st")?;

        let typed_filenames: std::collections::HashSet<String> =
            ["config.json".to_string()].into_iter().collect();
        super::harvest_siblings(snap.path(), slug.path(), &typed_filenames)?;

        // Content equality is portable: `symlink_force` degrades to a copy
        // on non-Unix, so we can't depend on `is_symlink()`.
        assert_eq!(
            std::fs::read(slug.path().join("config.json"))?,
            b"typed-slot-content"
        );
        assert!(slug.path().join("special_tokens_map.json").exists());
        Ok(())
    }
}
