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
use crate::protocols::tensor::TensorModelConfig;
use anyhow::{Context, Result};
use derive_builder::Builder;
use dynamo_kv_router::identity::{ExplicitIdentityMap, IndexerIdentitySpec};
use dynamo_runtime::{slug::Slug, storage::kv};
use serde::{Deserialize, Serialize};
use tokenizers::Tokenizer as HfTokenizer;

use crate::preprocessor::media::{MediaDecoder, MediaFetcher};
use crate::protocols::TokenIdType;

const DEFAULT_TOKENIZER_CACHE_BYTES: usize = 64 * 1024 * 1024;

fn append_indexer_identity_checksum(bytes: &mut Vec<u8>, spec: &IndexerIdentitySpec) {
    bytes.extend_from_slice(b"dynamo/model-card/indexer-identity/v1");
    append_identity_dimension(bytes, spec.semantics());
    append_identity_dimension(bytes, spec.routing_scope());
}

fn append_identity_dimension(bytes: &mut Vec<u8>, dimension: Option<&ExplicitIdentityMap>) {
    let Some(dimension) = dimension else {
        bytes.push(0);
        return;
    };
    bytes.push(1);
    bytes.extend_from_slice(&(dimension.entries().len() as u32).to_le_bytes());
    for (key, value) in dimension.entries() {
        append_framed_identity_value(bytes, key.as_bytes());
        append_framed_identity_value(bytes, value.as_bytes());
    }
}

fn append_framed_identity_value(bytes: &mut Vec<u8>, value: &[u8]) {
    let len = u32::try_from(value.len()).expect("validated identity values fit u32");
    bytes.extend_from_slice(&len.to_le_bytes());
    bytes.extend_from_slice(value);
}

fn tokenizer_cache_enabled(value: Option<&str>) -> bool {
    !matches!(value, Some("0"))
}

fn tokenizer_cache_bytes(value: Option<&str>) -> usize {
    value
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(DEFAULT_TOKENIZER_CACHE_BYTES)
}

fn tokenizer_cache_token_observer(model: &str) -> crate::tokenizers::CacheTokenUsageFn {
    let cached_tokens = dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_CACHED_TOKENS_TOTAL
        .with_label_values(&[model]);
    let uncached_tokens =
        dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_UNCACHED_TOKENS_TOTAL
            .with_label_values(&[model]);

    Arc::new(move |usage| {
        cached_tokens.inc_by(usage.cached_tokens as u64);
        uncached_tokens.inc_by(usage.uncached_tokens as u64);
    })
}

fn instrumented_tokenizer_cache(
    raw: Arc<dyn crate::tokenizers::traits::Tokenizer>,
    special_tokens: Vec<String>,
    cache_bytes: usize,
    cache_extend: bool,
    model: &str,
) -> Arc<dyn crate::tokenizers::traits::Tokenizer> {
    Arc::new(
        crate::tokenizers::CachedTokenizer::new(raw, special_tokens, cache_bytes)
            .with_extend(cache_extend)
            .with_observer(
                Arc::new(|| {
                    dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_HITS_TOTAL.inc();
                }),
                Arc::new(|| {
                    dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_MISSES_TOTAL.inc();
                }),
            )
            .with_token_observer(tokenizer_cache_token_observer(model)),
    )
}

/// Identify model deployment cards in the key-value store
pub const ROOT_PATH: &str = "v1/mdc";

/// Extract the set of atomic special-token strings from a HuggingFace tokenizer.
///
/// "Special" here means `AddedToken { special: true, .. }` — these are the only tokens
/// guaranteed atomic in BPE (won't be merged with surrounding bytes), so they are the
/// only safe boundary points for the L1 prefix cache.
fn extract_hf_special_tokens(hf: &HfTokenizer) -> Vec<String> {
    let added = hf.get_added_tokens_decoder();
    let mut out: Vec<String> = added
        .values()
        .filter(|t| t.special)
        .map(|t| t.content.clone())
        .collect();
    out.sort();
    out.dedup();
    out
}

/// serde `deserialize_with` that maps an explicitly-present value -- *including
/// an explicit JSON `null`* -- to `Some`. Paired with `#[serde(default)]` (which
/// supplies `None` only when the key is absent), this distinguishes "field
/// missing" from "field present but null/invalid". A plain `Option<Value>` would
/// collapse an explicit `null` into `None`, letting a malformed present
/// `max_position_embeddings` silently fall through to the next source instead of
/// surfacing the documented deserialization error.
fn deserialize_present_json_value<'de, D>(
    deserializer: D,
) -> Result<Option<serde_json::Value>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    serde_json::Value::deserialize(deserializer).map(Some)
}

/// Minimal projection of `config.json`, holding only the fields consulted when
/// deriving the architectural context length. Every other entry is skipped by
/// serde. This matters because some HF configs (e.g. Nemotron-H) serialize
/// fields such as `time_step_limit` with non-finite literals (`Infinity`,
/// `NaN`) that are valid JSON5 but not strict JSON; deserializing the whole
/// document into `serde_json::Value` would reject them, whereas projecting into
/// this struct via the JSON5 parser never materializes those unread fields.
///
/// The two consulted fields are captured as raw `serde_json::Value` (rather than
/// `u32`) so the original per-field error messages are preserved when a value is
/// present but not a valid integer. `deserialize_present_json_value` keeps an
/// explicit `null` distinguishable from an absent key (see its docs).
#[derive(Deserialize)]
struct ArchMaxContextConfig {
    #[serde(default, deserialize_with = "deserialize_present_json_value")]
    max_position_embeddings: Option<serde_json::Value>,
    text_config: Option<ArchMaxContextTextConfig>,
}

#[derive(Deserialize)]
struct ArchMaxContextTextConfig {
    #[serde(default, deserialize_with = "deserialize_present_json_value")]
    max_position_embeddings: Option<serde_json::Value>,
}

/// Resolve the static architectural context limit from local HF metadata.
///
/// `config.json` is authoritative when it declares `max_position_embeddings`;
/// multimodal configs may instead nest it under `text_config`; tokenizer
/// `model_max_length` is only a last resort. Missing fields fall through to the
/// next source, but malformed present fields return an error so bad model
/// metadata cannot silently resize request validation and planner limits.
///
/// This was added for the MiniMax-M3-VL : its multimodal
/// Hugging Face config keeps the language model context under
/// `config.json.text_config.max_position_embeddings`
fn architectural_max_context_length_from_repo(local_path: &Path) -> anyhow::Result<Option<u32>> {
    let tokenizer_context_length = || {
        crate::file_json_field(
            &local_path.join("tokenizer_config.json"),
            "model_max_length",
        )
        .ok()
    };

    let config_path = local_path.join("config.json");
    let config_json = match std::fs::read_to_string(&config_path) {
        Ok(config_json) => Some(config_json),
        Err(err) if err.kind() == std::io::ErrorKind::NotFound => None,
        Err(err) => {
            return Err(err).with_context(|| format!("Failed to read {}", config_path.display()));
        }
    };
    let Some(config_json) = config_json else {
        return Ok(tokenizer_context_length().filter(|context_length| *context_length > 0));
    };
    // Parse with the JSON5 parser (a lenient superset of JSON) into a minimal
    // projection, mirroring `HFConfig::from_json_file`. This tolerates non-finite
    // literals (`Infinity`, `NaN`) that HF configs may emit in fields we don't
    // read, which strict `serde_json` would reject.
    let config: ArchMaxContextConfig = json_five::from_str(&config_json)
        .with_context(|| format!("Failed to parse JSON from file: {}", config_path.display()))?;

    let context_length = match config.max_position_embeddings {
        Some(value) => Some(
            serde_json::from_value(value)
                .context("Failed to deserialize max_position_embeddings")?,
        ),
        None => match config
            .text_config
            .and_then(|text_config| text_config.max_position_embeddings)
        {
            Some(value) => Some(
                serde_json::from_value(value)
                    .context("Failed to deserialize text_config.max_position_embeddings")?,
            ),
            None => tokenizer_context_length(),
        },
    };

    Ok(context_length.filter(|context_length| *context_length > 0))
}

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

// `PromptContextMixin` is owned by the `dynamo-renderer` crate (it drives
// chat-template rendering); the MDC's `prompt_context` field is typed with it.
use dynamo_renderer::PromptContextMixin;

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

/// Per-MDC cache directory: `<root>/by-slug/<slug>/<mdcsum>/`.
/// Pure path computation; use [`mdc_local_dir`] when you need the
/// directory created.
fn mdc_local_path(slug: &Slug, mdcsum: &str) -> PathBuf {
    mdc_cache_root()
        .join("by-slug")
        .join(slug.to_string())
        .join(mdcsum)
}

fn mdc_local_dir(slug: &Slug, mdcsum: &str) -> anyhow::Result<PathBuf> {
    let dir = mdc_local_path(slug, mdcsum);
    std::fs::create_dir_all(&dir)
        .with_context(|| format!("creating MDC local dir {}", dir.display()))?;
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

/// File extensions that identify model weights. Callers: the frontend
/// hf:// sibling harvest below and `local_model::harvest_extra_files`.
pub(crate) fn is_weight_file(path: &Path) -> bool {
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

/// Symlink non-weight files from `snapshot_dir` into `local_dir`. Picks up
/// `preprocessor_config.json` and other sibling files that
/// `from_pretrained(local_dir)` consumers need.
///
/// Names in `typed_filenames` are owned by the resolve loop's typed-slot
/// pass — never overwritten. Every other harvested sibling is re-linked
/// unconditionally so a re-registration with the same `mdcsum` but a
/// different upstream snapshot picks up fresh contents (mdcsum doesn't
/// cover harvested files).
fn harvest_siblings(
    snapshot_dir: &Path,
    local_dir: &Path,
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
        let dst = local_dir.join(&name);
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
            "harvested sibling into local_dir",
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

    /// Architectural context maximum derived from model or tokenizer metadata.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub architectural_max_context_length: Option<u32>,

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

    /// Processing stage this worker handles (Prefill, Decode, Encode, Aggregated).
    /// Orthogonal to `model_type` (which describes endpoints exposed).
    ///
    /// Every worker must set this explicitly. `None` means the worker has
    /// not declared a worker type and is treated as misconfiguration:
    /// `Model::ws_type_and_needs` returns `None`, the serving-readiness
    /// gate refuses to vouch for the namespace, and `register_model`
    /// rejects such cards outright. The `Option<>` type and
    /// `#[serde(default)]` are kept so older cards still deserialize, but
    /// downstream readers treat them as not-ready.
    #[serde(default)]
    pub worker_type: Option<crate::worker_type::WorkerType>,

    /// Peer worker types this worker requires to serve traffic, in DNF form.
    /// The outer `Vec` is OR; each inner `Vec` is an AND-set of required
    /// worker types. Empty outer `Vec` means "no peers required."
    ///
    /// Examples:
    /// - Prefill worker: `[[Decode]]` — needs a Decode peer.
    /// - Encode worker: `[[Prefill, Decode], [Aggregated]]` — needs either a
    ///   P+D pair or a single Aggregated peer.
    #[serde(default)]
    pub needs: Vec<Vec<crate::worker_type::WorkerType>>,

    /// LoRA metadata for routing
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub lora: Option<LoraInfo>,

    /// Additional names this model responds to (aliases).
    /// Requests using any of these names will be routed to this worker.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub aliases: Vec<String>,

    /// User-defined metadata for custom worker behavior
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub user_data: Option<serde_json::Value>,

    #[serde(default)]
    pub runtime_config: ModelRuntimeConfig,

    /// Tensor model configuration for tensor-serving protocols.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub tensor_model_config: Option<TensorModelConfig>,

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

    /// Optional authoritative KV-indexer compatibility and isolation material.
    ///
    /// A present dimension replaces its component-derived default. Entry labels are deliberately
    /// uninterpreted so engine-specific compatibility facts do not expand this schema.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    #[builder(default)]
    pub indexer_identity: Option<IndexerIdentitySpec>,

    /// Sibling files (e.g. `preprocessor_config.json`) the worker
    /// advertises alongside the typed slots.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extra_files: Vec<CheckedFile>,

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
    /// Number of typed metadata slots (`model_info`, `tokenizer`,
    /// `prompt_formatter`, `chat_template_file`, `gen_config`). Used as
    /// a capacity hint for [`Self::iter_metadata_files`].
    const TYPED_SLOT_COUNT: usize = 5;

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

    /// Effective serving context: runtime engine limit, then architectural maximum.
    pub fn effective_context_length(&self) -> u32 {
        self.runtime_config
            .context_length
            .or(self.architectural_max_context_length)
            .unwrap_or(0)
    }

    /// Serialize the model deployment card to a JSON string
    pub fn to_json(&self) -> Result<String, anyhow::Error> {
        Ok(serde_json::to_string(self)?)
    }

    /// Per-MDC resolve directory. After `download_config` runs, every
    /// typed slot + harvested sibling is symlinked here for
    /// `from_pretrained(local_dir)` consumers. Pure path — does not
    /// create the directory; the resolve pipeline owns that.
    pub fn local_dir(&self) -> PathBuf {
        mdc_local_path(&self.slug, self.mdcsum())
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

                // extra_files: hash sorted (basename, checksum) pairs so
                // (a) workers with identical siblings produce the same
                // mdcsum regardless of `read_dir` order, and (b) the same
                // bytes under different filenames don't collide — otherwise
                // the frontend cache could serve a local_dir missing siblings.
                let mut extras: Vec<(&str, &str)> = self
                    .extra_files
                    .iter()
                    .map(|cf| (cf.basename().unwrap_or(""), cf.checksum().hash()))
                    .collect();
                extras.sort_unstable();
                for (name, h) in &extras {
                    bytes_to_hash.extend(name.as_bytes());
                    bytes_to_hash.push(0);
                    bytes_to_hash.extend(h.as_bytes());
                }

                if let Some(prompt_context_vec) = self.prompt_context.as_ref() {
                    // Paste it as the bytes of the debug format. It's a Vec of enum, so this should be
                    // fine. If the debug representation changes that only happens in a new release.
                    bytes_to_hash.extend(format!("{prompt_context_vec:?}").as_bytes());
                }
                bytes_to_hash.extend(self.effective_context_length().to_be_bytes());
                bytes_to_hash.extend(self.kv_cache_block_size.to_be_bytes());

                // worker_type/needs participate in the checksum so that a rolling
                // update that changes only those is correctly
                // rejected as incompatible with the existing WorkerSet (forcing
                // drain-and-redeploy) instead of silently joining and serving
                // stale readiness data.
                //
                // worker_type discriminator: 0 = None, then the variant ordinal.
                match self.worker_type {
                    None => bytes_to_hash.push(0),
                    Some(crate::worker_type::WorkerType::Prefill) => bytes_to_hash.push(1),
                    Some(crate::worker_type::WorkerType::Decode) => bytes_to_hash.push(2),
                    Some(crate::worker_type::WorkerType::Encode) => bytes_to_hash.push(3),
                    Some(crate::worker_type::WorkerType::Aggregated) => bytes_to_hash.push(4),
                }
                // needs is DNF: hash length(outer) || for each alt { length(inner) || each variant }
                bytes_to_hash.extend((self.needs.len() as u32).to_be_bytes());
                for alt in &self.needs {
                    bytes_to_hash.extend((alt.len() as u32).to_be_bytes());
                    for w in alt {
                        let v: u8 = match w {
                            crate::worker_type::WorkerType::Prefill => 1,
                            crate::worker_type::WorkerType::Decode => 2,
                            crate::worker_type::WorkerType::Encode => 3,
                            crate::worker_type::WorkerType::Aggregated => 4,
                        };
                        bytes_to_hash.push(v);
                    }
                }

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

                if let Some(identity) = self.indexer_identity.as_ref() {
                    append_indexer_identity_checksum(&mut bytes_to_hash, identity);
                }

                // Aliases participate in the checksum. Every worker in a
                // deployment carries the same static --served-model-name list,
                // so their checksums still match and they share one WorkerSet;
                // changing the alias list rolls a new WorkerSet (consistent per
                // set) rather than mutating a live one in place. `aliases` holds
                // only the alternate names (the primary is `display_name`, hashed
                // above), so order within the list still matters — hash in order.
                // Skipped entirely when empty so a card without aliases keeps the
                // same checksum as before this field existed (no spurious
                // WorkerSet split on upgrade); this is the last hashed field, so
                // omission is unambiguous.
                if !self.aliases.is_empty() {
                    bytes_to_hash.extend((self.aliases.len() as u32).to_be_bytes());
                    for alias in &self.aliases {
                        bytes_to_hash.extend((alias.len() as u32).to_be_bytes());
                        bytes_to_hash.extend(alias.as_bytes());
                    }
                }

                // TODO: Do we want any of user_data or runtime_config?

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
    /// Tokenizer backend controls:
    /// - `runtime_config.tokenizer_backend=fastokens` — use `fastokens` as the encoding backend
    /// - `DYN_TOKENIZER=fastokens` — fallback backend for callers without explicit runtime config
    /// - `DYN_TOKENIZER_CACHE=0` — disable the L1 prefix cache that records tokenizations
    ///   at special-token boundaries (enabled by default; any other value keeps it enabled)
    /// - `DYN_TOKENIZER_CACHE_BYTES=<n>` — L1 cache byte budget (default 64 MiB)
    /// - `DYN_TOKENIZER_CACHE_EXTEND=0` — disable partial-hit extension. By default
    ///   (when the cache is enabled) a partial hit also caches the new suffix so each
    ///   turn of a growing multi-turn conversation hits deeper than the last, keeping
    ///   per-turn tokenization cost flat instead of growing with history. Set to `0` to
    ///   fall back to the original hit-without-insert behavior.
    pub fn tokenizer(&self) -> anyhow::Result<crate::tokenizers::Tokenizer> {
        let use_fast = self
            .runtime_config
            .effective_tokenizer_backend()
            .is_fastokens();

        let cache_enabled =
            tokenizer_cache_enabled(std::env::var("DYN_TOKENIZER_CACHE").ok().as_deref());
        let cache_bytes =
            tokenizer_cache_bytes(std::env::var("DYN_TOKENIZER_CACHE_BYTES").ok().as_deref());
        // Partial-hit extension is on by default; disable with DYN_TOKENIZER_CACHE_EXTEND=0.
        let cache_extend = !matches!(
            std::env::var("DYN_TOKENIZER_CACHE_EXTEND").ok().as_deref(),
            Some("0")
        );

        let inner: Arc<dyn crate::tokenizers::traits::Tokenizer> = match &self.tokenizer {
            Some(TokenizerKind::HfTokenizerJson(checked_file)) => {
                let p = checked_file.path().ok_or_else(|| {
                    anyhow::anyhow!("Tokenizer is URL-backed ({:?})", checked_file.url())
                })?;

                // Load HF first — needed both for fallback and (if cache is on) for
                // extracting special-token strings. `FastTokenizer` does not re-expose
                // `get_added_tokens_decoder`, so we must capture specials from the raw
                // HF tokenizer before any swap.
                let mut hf = HfTokenizer::from_file(p)
                    .inspect_err(|err| {
                        if let Some(serde_err) = err.downcast_ref::<serde_json::Error>()
                            && let Ok(contents) = std::fs::read_to_string(p)
                        {
                            crate::log_json_err(&p.display().to_string(), &contents, serde_err);
                        }
                    })
                    .map_err(anyhow::Error::msg)
                    .with_context(|| p.display().to_string())?;
                // Apply the tokenizer_config.json special-token merge eagerly so
                // `extract_hf_special_tokens` below sees the same specials the
                // wrapped tokenizer will use. Without this the L1 prefix cache's
                // boundary list would diverge from the actual tokenizer
                // (e.g. Qwen2-VL's `<|image_pad|>` would be in the tokenizer
                // but missing from the cache specials), letting chat prefixes
                // straddle a special-token boundary and reducing hit rate.
                if let Some(model_dir) = p.parent() {
                    crate::tokenizers::hf::merge_special_tokens_from_config(&mut hf, model_dir);
                }

                // Disable any truncation baked into `tokenizer.json`: the HF
                // `tokenizers` crate honors it on `encode()`, silently clipping every
                // prompt (e.g. `stepfun-ai/Step-3.7-Flash-*` caps at 2048), unlike
                // Python `transformers`, which resets it on load. Match that: never
                // truncate implicitly; over-length prompts are rejected elsewhere.
                if hf.get_truncation().is_some() {
                    tracing::warn!(
                        "tokenizer.json declares a truncation config; disabling it so \
                         prompts are not silently clipped"
                    );
                    // Hard-fail rather than warn: if we can't clear it, the prompt
                    // would still be silently clipped, defeating the purpose.
                    hf.with_truncation(None)
                        .map_err(anyhow::Error::msg)
                        .context("failed to disable tokenizer.json truncation")?;
                }

                // Hold onto specials before any move of `hf`.
                let specials: Vec<String> = if cache_enabled {
                    extract_hf_special_tokens(&hf)
                } else {
                    Vec::new()
                };

                // Merge already applied above; just wrap.
                let wrap_hf =
                    |hf: HfTokenizer| crate::tokenizers::HuggingFaceTokenizer::from_tokenizer(hf);

                // Pick the inner backend.
                let raw: Arc<dyn crate::tokenizers::traits::Tokenizer> = if use_fast {
                    if let Some(path_str) = p.to_str() {
                        match crate::tokenizers::FastTokenizer::from_file(path_str) {
                            Ok(fast) => {
                                tracing::info!("Using fastokens tokenizer backend");
                                Arc::new(fast)
                            }
                            Err(e) => {
                                tracing::warn!(
                                    %e,
                                    "Failed to load fastokens, falling back to HuggingFace"
                                );
                                Arc::new(wrap_hf(hf))
                            }
                        }
                    } else {
                        tracing::warn!(
                            path = %p.display(),
                            "Tokenizer path contains non-UTF-8 characters, skipping fastokens; falling back to HuggingFace"
                        );
                        Arc::new(wrap_hf(hf))
                    }
                } else {
                    Arc::new(wrap_hf(hf))
                };

                if cache_enabled {
                    tracing::info!(
                        cache_bytes,
                        cache_extend,
                        specials = specials.len(),
                        "wrapping tokenizer in L1 prefix cache",
                    );
                    instrumented_tokenizer_cache(
                        raw,
                        specials,
                        cache_bytes,
                        cache_extend,
                        self.name(),
                    )
                } else {
                    raw
                }
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

                let specials = tokenizer.special_tokens().to_vec();
                let raw: Arc<dyn crate::tokenizers::traits::Tokenizer> = Arc::new(tokenizer);
                if cache_enabled {
                    tracing::info!(
                        cache_bytes,
                        cache_extend,
                        boundaries = specials.len(),
                        "wrapping tiktoken tokenizer in L1 prefix cache",
                    );
                    instrumented_tokenizer_cache(
                        raw,
                        specials,
                        cache_bytes,
                        cache_extend,
                        self.name(),
                    )
                } else {
                    raw
                }
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
        };

        Ok(crate::tokenizers::Tokenizer::from(inner))
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

    /// Set additional names (aliases) this model responds to.
    pub fn set_aliases(&mut self, aliases: Vec<String>) {
        self.aliases = aliases;
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
    /// gen_config, then any `extra_files` siblings the worker harvested
    /// (preprocessor_config.json, special_tokens_map.json, etc.). Each entry
    /// is `(file, is_custom)` — `is_custom` is only ever true for
    /// operator-supplied chat templates, which can't fall back to HF.
    pub fn iter_metadata_files(&self) -> Vec<(&CheckedFile, bool)> {
        let mut out: Vec<(&CheckedFile, bool)> =
            Vec::with_capacity(Self::TYPED_SLOT_COUNT + self.extra_files.len());
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
        for cf in &self.extra_files {
            out.push((cf, false));
        }
        out
    }

    /// Mutable mirror of [`Self::iter_metadata_files`].
    pub fn iter_metadata_files_mut(&mut self) -> Vec<(&mut CheckedFile, bool)> {
        let mut out: Vec<(&mut CheckedFile, bool)> =
            Vec::with_capacity(Self::TYPED_SLOT_COUNT + self.extra_files.len());
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
        for cf in &mut self.extra_files {
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
        let local_dir = mdc_local_dir(&self.slug, &mdcsum)?;

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
            symlink_force(&blob, &local_dir.join(&filename))?;
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
            harvest_siblings(snap, &local_dir, &typed_filenames)?;
        }

        // Pass 3: rewrite cf.path to the cache symlink so downstream
        // tokenizer/config loaders read from a verified location.
        for (cf, _) in self.iter_metadata_files_mut() {
            cf.update_dir(&local_dir);
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

        // Prefer the model config's architectural context length. Some
        // multimodal HF configs, including MiniMax-M3-VL, store the language
        // model config under `text_config`; tokenizer `model_max_length` can
        // be a much larger sentinel/default and must be a last resort.
        let architectural_max_context_length =
            architectural_max_context_length_from_repo(local_path)?;

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
            architectural_max_context_length,
            kv_cache_block_size: 0, // set later
            migration_limit: 0,
            model_type: Default::default(),  // set later
            model_input: Default::default(), // set later
            worker_type: Default::default(), // set later
            needs: Default::default(),       // set later
            lora: None,
            aliases: Vec::new(),
            user_data: None,
            runtime_config: ModelRuntimeConfig::default(),
            tensor_model_config: None,
            media_decoder: None,
            media_fetcher: None,
            router_config: None,
            indexer_identity: None,
            extra_files: Vec::new(),
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

        let model_dir = file_path.parent().unwrap_or_else(|| Path::new(""));
        let gencfg_path = model_dir.join("generation_config.json");

        // bos and eos resolve through the same chain, highest priority first:
        //   generation_config.json -> config.json -> tokenizer_config.json
        //   -> special_tokens_map.json
        // generation_config wins over config.json (HF convention); the tokenizer
        // rungs rescue models that ship the token only there, not in config.
        text_config.bos_token_id =
            crate::file_json_field::<TokenIdType>(&gencfg_path, "bos_token_id")
                .ok()
                .or(text_config.bos_token_id)
                .or_else(|| resolve_token_id_from_tokenizer_files(model_dir, "bos_token"));

        // Same chain as bos above, but eos may be a single id or a list.
        // TODO: refactor when we switch to per-architecture tokenization.
        let mut final_eos_token_ids: Vec<TokenIdType> = {
                crate::file_json_field::<serde_json::Value>(&gencfg_path, "eos_token_id")
                .inspect_err(
                    |err| tracing::debug!(%err, "eos_token_id not found in generation_config.json, will fall back"),
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
            .or_else(|| {
                resolve_token_id_from_tokenizer_files(model_dir, "eos_token").map(|id| vec![id])
            })
            .ok_or_else(|| {
                anyhow::anyhow!(
                    "missing eos_token_id in generation_config.json, config.json, \
                     tokenizer_config.json, and special_tokens_map.json — cannot load"
                )
            })?;

        // Some models (e.g. Qwen3.5) set eos in config.json but emit a different
        // token (<|im_end|>) at generation end; both must count as stop tokens.
        // Add the tokenizer's eos when it differs. Idempotent if already present.
        if let Ok(tokenizer_eos_id) = resolve_token_id_from_tokenizer_config(
            &model_dir.join("tokenizer_config.json"),
            "eos_token",
        ) && !final_eos_token_ids.contains(&tokenizer_eos_id)
        {
            final_eos_token_ids.push(tokenizer_eos_id);
        }

        text_config.final_eos_token_ids = final_eos_token_ids;

        Ok(Arc::new(config))
    }
}

/// Rungs 3-4 of the chain in `HFConfig::from_json_file`: resolve a token id
/// from the tokenizer artifacts, trying `tokenizer_config.json` then
/// `special_tokens_map.json`. `token_key` is the HF field, e.g. `"eos_token"`.
fn resolve_token_id_from_tokenizer_files(model_dir: &Path, token_key: &str) -> Option<TokenIdType> {
    if let Ok(id) =
        resolve_token_id_from_tokenizer_config(&model_dir.join("tokenizer_config.json"), token_key)
    {
        return Some(id);
    }
    resolve_token_id_from_special_tokens_map(model_dir, token_key).ok()
}

/// Read `<token_key>` from `tokenizer_config.json` (a string or
/// `{"content": ...}` object) and resolve its id via `added_tokens_decoder`,
/// falling back to `tokenizer.json:added_tokens`. Some models (e.g.
/// jina-embeddings-v5-omni) ship the token only as a bare string with no
/// `added_tokens_decoder`, keeping the id mapping solely in `tokenizer.json`.
fn resolve_token_id_from_tokenizer_config(
    path: &Path,
    token_key: &str,
) -> anyhow::Result<TokenIdType> {
    let config = read_json(path)
        .with_context(|| format!("Failed to read or parse tokenizer_config.json: {:?}", path))?;
    let token_str = extract_token_string(config.get(token_key), token_key)?;
    if let Some(added_tokens) = config
        .get("added_tokens_decoder")
        .and_then(|v| v.as_object())
        && let Ok(id) = lookup_id_in_added_tokens_decoder(added_tokens, &token_str)
    {
        return Ok(id);
    }
    let model_dir = path.parent().unwrap_or_else(|| Path::new(""));
    lookup_id_in_tokenizer_json(model_dir, &token_str).ok_or_else(|| {
        anyhow::anyhow!(
            "{token_key} '{token_str}' from tokenizer_config.json not found in \
             added_tokens_decoder or tokenizer.json added_tokens"
        )
    })
}

/// Read and JSON-parse a file, returning `None` if it is missing or invalid.
fn read_json(path: &Path) -> Option<serde_json::Value> {
    serde_json::from_str(&std::fs::read_to_string(path).ok()?).ok()
}

/// Look up `token_str`'s id in `tokenizer_config.json`'s `added_tokens_decoder`.
fn lookup_id_in_tokenizer_config(model_dir: &Path, token_str: &str) -> Option<TokenIdType> {
    let cfg = read_json(&model_dir.join("tokenizer_config.json"))?;
    let added = cfg.get("added_tokens_decoder")?.as_object()?;
    lookup_id_in_added_tokens_decoder(added, token_str).ok()
}

/// Look up `token_str`'s id in `tokenizer.json`'s `added_tokens` array.
fn lookup_id_in_tokenizer_json(model_dir: &Path, token_str: &str) -> Option<TokenIdType> {
    let tok = read_json(&model_dir.join("tokenizer.json"))?;
    tok.get("added_tokens")?
        .as_array()?
        .iter()
        .filter(|e| e.get("content").and_then(|v| v.as_str()) == Some(token_str))
        .find_map(|e| e.get("id").and_then(|v| v.as_u64()))
        .map(|id| id as TokenIdType)
}

/// `special_tokens_map.json` carries only the token string, so resolve its id
/// via `tokenizer_config.json` then `tokenizer.json`.
fn resolve_token_id_from_special_tokens_map(
    model_dir: &Path,
    token_key: &str,
) -> anyhow::Result<TokenIdType> {
    let stm = read_json(&model_dir.join("special_tokens_map.json"))
        .context("Failed to read or parse special_tokens_map.json")?;
    let token_str = extract_token_string(stm.get(token_key), token_key)?;

    lookup_id_in_tokenizer_config(model_dir, &token_str)
        .or_else(|| lookup_id_in_tokenizer_json(model_dir, &token_str))
        .ok_or_else(|| {
            anyhow::anyhow!(
                "{token_key} '{token_str}' from special_tokens_map.json not found in \
                 tokenizer_config.json added_tokens_decoder or tokenizer.json added_tokens"
            )
        })
}

/// Pull a token string out of a JSON field that may be `"<str>"` or
/// `{"content": "<str>", ...}` (the older HF format used in both
/// `tokenizer_config.json` and `special_tokens_map.json`).
fn extract_token_string(
    field: Option<&serde_json::Value>,
    token_key: &str,
) -> anyhow::Result<String> {
    match field {
        Some(serde_json::Value::String(s)) => Ok(s.clone()),
        Some(serde_json::Value::Object(obj)) => obj
            .get("content")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string())
            .ok_or_else(|| anyhow::anyhow!("{} is an object without 'content' field", token_key)),
        _ => anyhow::bail!("{} not found or not a string", token_key),
    }
}

fn lookup_id_in_added_tokens_decoder(
    added_tokens: &serde_json::Map<String, serde_json::Value>,
    token_str: &str,
) -> anyhow::Result<TokenIdType> {
    for (id_str, token_info) in added_tokens {
        let content = token_info
            .get("content")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if content == token_str {
            let token_id: TokenIdType = id_str.parse().with_context(|| {
                format!(
                    "Failed to parse token ID '{}' from added_tokens_decoder",
                    id_str
                )
            })?;
            return Ok(token_id);
        }
    }
    anyhow::bail!("token '{}' not found in added_tokens_decoder", token_str)
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
    use super::{HFConfig, ModelDeploymentCard};
    use std::collections::HashSet;
    use std::path::{Path, PathBuf};

    #[test]
    fn tokenizer_cache_token_observer_records_per_model_totals() {
        let model_a = "token-observer-test-model-a";
        let model_b = "token-observer-test-model-b";
        let cached_a = dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_CACHED_TOKENS_TOTAL
            .with_label_values(&[model_a]);
        let uncached_a =
            dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_UNCACHED_TOKENS_TOTAL
                .with_label_values(&[model_a]);
        let cached_b = dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_CACHED_TOKENS_TOTAL
            .with_label_values(&[model_b]);
        let uncached_b =
            dynamo_runtime::metrics::frontend_perf::TOKENIZER_CACHE_UNCACHED_TOKENS_TOTAL
                .with_label_values(&[model_b]);

        let before_a = (cached_a.get(), uncached_a.get());
        let before_b = (cached_b.get(), uncached_b.get());

        super::tokenizer_cache_token_observer(model_a)(crate::tokenizers::CacheTokenUsage {
            cached_tokens: 7,
            uncached_tokens: 5,
        });

        assert_eq!(
            (cached_a.get(), uncached_a.get()),
            (before_a.0 + 7, before_a.1 + 5)
        );
        assert_eq!((cached_b.get(), uncached_b.get()), before_b);

        super::tokenizer_cache_token_observer(model_b)(crate::tokenizers::CacheTokenUsage {
            cached_tokens: 3,
            uncached_tokens: 11,
        });

        assert_eq!(
            (cached_a.get(), uncached_a.get()),
            (before_a.0 + 7, before_a.1 + 5)
        );
        assert_eq!(
            (cached_b.get(), uncached_b.get()),
            (before_b.0 + 3, before_b.1 + 11)
        );
    }

    #[test]
    fn tokenizer_cache_is_enabled_by_default_and_disabled_only_by_zero() {
        assert!(super::tokenizer_cache_enabled(None));
        assert!(super::tokenizer_cache_enabled(Some("1")));
        assert!(!super::tokenizer_cache_enabled(Some("0")));
        assert!(super::tokenizer_cache_enabled(Some("true")));
    }

    #[test]
    fn tokenizer_cache_bytes_defaults_to_64_mib_and_accepts_valid_overrides() {
        assert_eq!(
            super::tokenizer_cache_bytes(None),
            super::DEFAULT_TOKENIZER_CACHE_BYTES
        );
        assert_eq!(super::tokenizer_cache_bytes(Some("1024")), 1024);
        assert_eq!(
            super::tokenizer_cache_bytes(Some("invalid")),
            super::DEFAULT_TOKENIZER_CACHE_BYTES
        );
    }

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

    /// Rung 3: model ships only `config.json` + `tokenizer_config.json`. No
    /// `generation_config.json`, no eos/bos in `config.json`. Both token ids
    /// must come from `tokenizer_config.json`'s `eos_token`/`bos_token` strings
    /// resolved through `added_tokens_decoder`.
    #[test]
    fn test_config_json_eos_bos_from_tokenizer_config_only() -> anyhow::Result<()> {
        let config_file = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-tokenizer-config-only/config.json");
        let config = HFConfig::from_json_file(&config_file)?;
        assert_eq!(config.bos_token_id(), Some(101));
        let eos: HashSet<_> = config.eos_token_ids().iter().cloned().collect();
        assert!(
            eos.contains(&100),
            "eos should resolve to 100 from tokenizer_config.json"
        );
        Ok(())
    }

    /// Rung 4: model ships only `config.json` + `special_tokens_map.json` +
    /// `tokenizer.json`. The token strings live in `special_tokens_map.json`
    /// and the id mapping in `tokenizer.json:added_tokens`. This is the rung
    /// that rescues models that don't duplicate eos/bos into
    /// `generation_config.json` or `config.json`.
    #[test]
    fn test_config_json_eos_bos_from_special_tokens_map() -> anyhow::Result<()> {
        let config_file = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-special-tokens-only/config.json");
        let config = HFConfig::from_json_file(&config_file)?;
        assert_eq!(config.bos_token_id(), Some(201));
        let eos: HashSet<_> = config.eos_token_ids().iter().cloned().collect();
        assert!(
            eos.contains(&200),
            "eos should resolve to 200 from special_tokens_map"
        );
        Ok(())
    }

    /// Rung 3, `tokenizer.json` fallback: `tokenizer_config.json` names the
    /// eos token as a bare string but ships NO `added_tokens_decoder`, so the
    /// string->id mapping lives only in `tokenizer.json:added_tokens`. With
    /// `generation_config.json` and `special_tokens_map.json` both absent, this
    /// is the only path that can recover the id. (bos is null in this model, so
    /// only eos is exercised here.)
    ///
    /// Models in the wild that need this: `jinaai/jina-embeddings-v5-omni-small`
    /// (https://huggingface.co/jinaai/jina-embeddings-v5-omni-small), reported
    /// in https://github.com/ai-dynamo/dynamo/issues/10805: its eos `<|im_end|>`
    /// resolves to 151645 from `tokenizer.json` alone. The fixture mirrors that
    /// layout (ids/strings kept identical to the real model).
    #[test]
    fn test_config_json_eos_bos_from_tokenizer_json_fallback() -> anyhow::Result<()> {
        let config_file = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data/sample-models/mock-tokenizer-json-fallback/config.json");
        let config = HFConfig::from_json_file(&config_file)?;
        let eos: HashSet<_> = config.eos_token_ids().iter().cloned().collect();
        assert!(
            eos.contains(&151645),
            "eos should resolve to 151645 (<|im_end|>) from tokenizer.json"
        );
        Ok(())
    }

    /// All four rungs miss → the error message must name every source so the
    /// operator knows what to add. Guards against the failure mode where
    /// only `generation_config.json` is mentioned.
    #[test]
    fn test_config_json_missing_eos_everywhere_lists_all_sources() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            br#"{"architectures":["FakeForCausalLM"],"model_type":"fake"}"#,
        )
        .unwrap();
        let err = match HFConfig::from_json_file(dir.path().join("config.json")) {
            Ok(_) => panic!("expected error when no eos source is available"),
            Err(e) => e,
        };
        let msg = err.to_string();
        for needle in [
            "generation_config.json",
            "config.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
        ] {
            assert!(
                msg.contains(needle),
                "error must name {needle} as a source it checked; got: {msg}"
            );
        }
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

            let blobs = std::fs::canonicalize(home_path.join(".cache/dynamo/mdc/blobs"))?;
            let snap = home_path
                .join(".cache/dynamo/mdc/by-slug")
                .join(slug.to_string())
                .join(&mdcsum);

            assert!(snap.join("config.json").exists());
            assert!(snap.join("tokenizer.json").exists());
            assert!(snap.join("generation_config.json").exists());

            // Sibling harvest: TinyLlama_v1.1 fixture ships
            // `special_tokens_map.json` and `tokenizer.model` outside the
            // typed slots — both must land in local_dir for
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

    /// Two MDCs with `extra_files` that share bytes but differ in basename
    /// must produce distinct mdcsums — otherwise the frontend cache would
    /// alias them and a local_dir built from one worker's harvest would be
    /// reused for another worker that needs a differently-named sibling.
    #[test]
    fn mdcsum_extras_distinguish_basename_at_equal_checksum() {
        let src =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/sample-models/TinyLlama_v1.1");
        let mut a = super::ModelDeploymentCard::load_from_disk(&src, None).unwrap();
        let mut b = super::ModelDeploymentCard::load_from_disk(&src, None).unwrap();
        a.extra_files.push(cf_for("/m/preprocessor_config.json"));
        b.extra_files.push(cf_for("/m/image_processor_config.json"));
        assert_ne!(a.mdcsum(), b.mdcsum());
    }

    /// Read-order independence: extras pushed in different order must
    /// hash the same (sort_unstable on (basename, checksum) pairs).
    #[test]
    fn mdcsum_extras_stable_across_read_order() {
        let src =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/sample-models/TinyLlama_v1.1");
        let mut a = super::ModelDeploymentCard::load_from_disk(&src, None).unwrap();
        let mut b = super::ModelDeploymentCard::load_from_disk(&src, None).unwrap();
        let cf1 = cf_for("/m/a.json");
        let cf2 = cf_for("/m/b.json");
        a.extra_files.extend([cf1.clone(), cf2.clone()]);
        b.extra_files.extend([cf2, cf1]);
        assert_eq!(a.mdcsum(), b.mdcsum());
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

    #[test]
    fn local_dir_computes_expected_path() {
        // Sentinel for cache-layout drift: the public `local_dir()` must
        // stay in lockstep with `mdc_local_path`. Resolve-pipeline
        // integration is covered by the vllm/sglang serve tests.
        let card = ModelDeploymentCard::with_name_only("Qwen/Qwen3-0.6B");
        assert_eq!(
            card.local_dir(),
            super::mdc_local_path(card.slug(), card.mdcsum())
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

    /// Brings in the sibling that `mm-routing` needs.
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

        // Typed slot: blob in the dynamo cache; local_dir links to it.
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

#[cfg(test)]
mod ownership_tests {
    use super::*;

    #[test]
    fn architectural_context_prefers_config_then_text_config_then_tokenizer() -> anyhow::Result<()>
    {
        let dir = tempfile::tempdir()?;

        std::fs::write(
            dir.path().join("config.json"),
            r#"{"max_position_embeddings": 4096}"#,
        )?;
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"model_max_length": 131072}"#,
        )?;
        assert_eq!(
            architectural_max_context_length_from_repo(dir.path())?,
            Some(4096)
        );

        std::fs::write(
            dir.path().join("config.json"),
            r#"{"text_config": {"max_position_embeddings": 8192}}"#,
        )?;
        assert_eq!(
            architectural_max_context_length_from_repo(dir.path())?,
            Some(8192)
        );

        std::fs::write(dir.path().join("config.json"), r#"{"model_type": "mock"}"#)?;
        assert_eq!(
            architectural_max_context_length_from_repo(dir.path())?,
            Some(131072)
        );

        std::fs::write(dir.path().join("config.json"), r#"{"text_config": {}}"#)?;
        assert_eq!(
            architectural_max_context_length_from_repo(dir.path())?,
            Some(131072)
        );

        std::fs::remove_file(dir.path().join("config.json"))?;
        assert_eq!(
            architectural_max_context_length_from_repo(dir.path())?,
            Some(131072)
        );

        Ok(())
    }

    #[test]
    fn architectural_context_errors_on_malformed_present_fields() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("tokenizer_config.json"),
            r#"{"model_max_length": 131072}"#,
        )?;

        std::fs::write(
            dir.path().join("config.json"),
            r#"{"max_position_embeddings": "not-a-number"}"#,
        )?;
        let err = architectural_max_context_length_from_repo(dir.path()).unwrap_err();
        assert!(
            err.to_string()
                .contains("Failed to deserialize max_position_embeddings"),
            "{err:?}"
        );

        std::fs::write(
            dir.path().join("config.json"),
            r#"{"text_config": {"max_position_embeddings": "not-a-number"}}"#,
        )?;
        let err = architectural_max_context_length_from_repo(dir.path()).unwrap_err();
        assert!(
            err.to_string()
                .contains("Failed to deserialize text_config.max_position_embeddings"),
            "{err:?}"
        );

        // An explicit null is a present-but-malformed value: it must error, not
        // silently fall through to text_config/tokenizer (regression guard for
        // the Option<Value> projection collapsing null into None).
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"max_position_embeddings": null}"#,
        )?;
        let err = architectural_max_context_length_from_repo(dir.path()).unwrap_err();
        assert!(
            err.to_string()
                .contains("Failed to deserialize max_position_embeddings"),
            "{err:?}"
        );

        std::fs::write(
            dir.path().join("config.json"),
            r#"{"text_config": {"max_position_embeddings": null}}"#,
        )?;
        let err = architectural_max_context_length_from_repo(dir.path()).unwrap_err();
        assert!(
            err.to_string()
                .contains("Failed to deserialize text_config.max_position_embeddings"),
            "{err:?}"
        );

        Ok(())
    }

    #[test]
    fn architectural_context_tolerates_non_finite_literals() -> anyhow::Result<()> {
        // Some HF configs (e.g. Nemotron-H) serialize fields such as
        // `time_step_limit` with the bare literal `Infinity`, which is valid
        // JSON5 but not strict JSON. Deriving max_position_embeddings must not
        // choke on such literals in fields we never read.
        let dir = tempfile::tempdir()?;
        std::fs::write(
            dir.path().join("config.json"),
            r#"{"max_position_embeddings": 262144, "time_step_limit": [0.0, Infinity]}"#,
        )?;
        assert_eq!(
            architectural_max_context_length_from_repo(dir.path())?,
            Some(262144)
        );

        Ok(())
    }

    #[test]
    fn effective_context_prefers_runtime_then_architecture_then_unknown() {
        let mut card = ModelDeploymentCard::with_name_only("model");
        assert_eq!(card.effective_context_length(), 0);

        card.architectural_max_context_length = Some(32_768);
        assert_eq!(card.effective_context_length(), 32_768);

        card.runtime_config.context_length = Some(8_192);
        assert_eq!(card.effective_context_length(), 8_192);

        card.runtime_config.context_length = Some(0);
        assert_eq!(card.effective_context_length(), 0);
    }

    #[test]
    fn tensor_config_serializes_at_card_top_level() {
        let mut card = ModelDeploymentCard::with_name_only("tensor");
        card.tensor_model_config = Some(TensorModelConfig {
            name: "tensor".to_string(),
            ..Default::default()
        });

        let value = serde_json::to_value(&card).unwrap();
        assert_eq!(value["tensor_model_config"]["name"], "tensor");
        assert!(value["runtime_config"].get("tensor_model_config").is_none());

        let parsed: ModelDeploymentCard = serde_json::from_value(value).unwrap();
        assert_eq!(
            parsed
                .tensor_model_config
                .as_ref()
                .map(|config| config.name.as_str()),
            Some("tensor")
        );
    }
}

#[cfg(test)]
mod worker_type_tests {
    //! Tests for the `worker_type` / `needs` fields on `ModelDeploymentCard`.
    //! See `docs/proposals/health-disagg-readiness.md`.

    use super::*;
    use crate::worker_type::WorkerType;
    use std::collections::BTreeMap;

    #[test]
    fn default_card_has_no_worker_type_and_no_needs() {
        let card = ModelDeploymentCard::with_name_only("test-model");
        assert_eq!(card.worker_type, None);
        assert!(card.needs.is_empty());
    }

    #[test]
    fn serde_round_trip_default() {
        let card = ModelDeploymentCard::with_name_only("test-model");
        let json = serde_json::to_string(&card).unwrap();
        let back: ModelDeploymentCard = serde_json::from_str(&json).unwrap();
        assert_eq!(back.worker_type, None);
        assert!(back.needs.is_empty());
    }

    #[test]
    fn serde_round_trip_decode_needs_prefill() {
        let mut card = ModelDeploymentCard::with_name_only("test-model");
        card.worker_type = Some(WorkerType::Decode);
        card.needs = vec![vec![WorkerType::Prefill]];
        let json = serde_json::to_string(&card).unwrap();
        let back: ModelDeploymentCard = serde_json::from_str(&json).unwrap();
        assert_eq!(back.worker_type, Some(WorkerType::Decode));
        assert_eq!(back.needs, vec![vec![WorkerType::Prefill]]);
    }

    #[test]
    fn serde_round_trip_aggregated_needs_encode() {
        // E-PD pattern: an aggregated worker with --route-to-encoder.
        let mut card = ModelDeploymentCard::with_name_only("test-model");
        card.worker_type = Some(WorkerType::Aggregated);
        card.needs = vec![vec![WorkerType::Encode]];
        let json = serde_json::to_string(&card).unwrap();
        let back: ModelDeploymentCard = serde_json::from_str(&json).unwrap();
        assert_eq!(back.worker_type, Some(WorkerType::Aggregated));
        assert_eq!(back.needs, vec![vec![WorkerType::Encode]]);
    }

    #[test]
    fn serde_round_trip_encode_needs_dnf() {
        // Encode worker: needs (Prefill AND Decode) OR Aggregated.
        let mut card = ModelDeploymentCard::with_name_only("test-model");
        card.worker_type = Some(WorkerType::Encode);
        card.needs = vec![
            vec![WorkerType::Prefill, WorkerType::Decode],
            vec![WorkerType::Aggregated],
        ];
        let json = serde_json::to_string(&card).unwrap();
        let back: ModelDeploymentCard = serde_json::from_str(&json).unwrap();
        assert_eq!(back.worker_type, Some(WorkerType::Encode));
        assert_eq!(back.needs.len(), 2);
        assert_eq!(back.needs[0], vec![WorkerType::Prefill, WorkerType::Decode]);
        assert_eq!(back.needs[1], vec![WorkerType::Aggregated]);
    }

    /// mdcsum must cover `worker_type` and `needs` so that a rolling update
    /// which changes only those produces a different checksum,
    /// triggering the drain-and-redeploy path in `watcher.rs` instead of
    /// silently joining an existing WorkerSet with a stale card.
    ///
    /// Note: `mdcsum()` caches its result on first call via `OnceLock`, so
    /// each case builds a fresh card rather than mutating one and re-hashing.
    #[test]
    fn mdcsum_covers_worker_type_and_needs() {
        fn hash(worker_type: Option<WorkerType>, needs: Vec<Vec<WorkerType>>) -> String {
            let mut card = ModelDeploymentCard::with_name_only("model");
            card.worker_type = worker_type;
            card.needs = needs;
            card.mdcsum().to_string()
        }

        let baseline = hash(None, vec![]);
        let prefill_only = hash(Some(WorkerType::Prefill), vec![]);
        let decode_only = hash(Some(WorkerType::Decode), vec![]);
        assert_ne!(baseline, prefill_only, "worker_type must change mdcsum");
        assert_ne!(
            prefill_only, decode_only,
            "swapping worker_type must change mdcsum"
        );

        let prefill_with_decode = hash(Some(WorkerType::Prefill), vec![vec![WorkerType::Decode]]);
        let prefill_with_decode_encode = hash(
            Some(WorkerType::Prefill),
            vec![vec![WorkerType::Decode, WorkerType::Encode]],
        );
        assert_ne!(
            prefill_only, prefill_with_decode,
            "adding needs must change mdcsum"
        );
        assert_ne!(
            prefill_with_decode, prefill_with_decode_encode,
            "extending an AND-set must change mdcsum"
        );

        let encode_dnf = hash(
            Some(WorkerType::Encode),
            vec![
                vec![WorkerType::Prefill, WorkerType::Decode],
                vec![WorkerType::Aggregated],
            ],
        );
        let encode_single_alt = hash(
            Some(WorkerType::Encode),
            vec![vec![WorkerType::Prefill, WorkerType::Decode]],
        );
        assert_ne!(
            encode_dnf, encode_single_alt,
            "adding an OR alternative must change mdcsum"
        );
    }

    /// Serde back-compat: an old-format card (no `worker_type` / `needs`
    /// keys in the JSON payload) must deserialize with both fields defaulted
    /// (`None` and empty `Vec`) — this is an attribute of the
    /// `#[serde(default)]` contract and is independent of how readers
    /// subsequently interpret the missing values. Construction of the test
    /// payload strips the new keys from a fresh serialization so the test
    /// tracks schema drift rather than a hand-rolled JSON literal.
    #[test]
    fn backward_compat_missing_fields_default_to_none_and_empty() {
        let mut card = ModelDeploymentCard::with_name_only("test-model");
        card.worker_type = Some(WorkerType::Prefill);
        card.needs = vec![vec![WorkerType::Decode]];
        let mut value: serde_json::Value = serde_json::to_value(&card).unwrap();
        let obj = value.as_object_mut().unwrap();
        assert!(
            obj.remove("worker_type").is_some(),
            "precondition: serialized card must carry worker_type"
        );
        assert!(
            obj.remove("needs").is_some(),
            "precondition: serialized card must carry needs"
        );
        let stripped = serde_json::to_string(&value).unwrap();
        let back: ModelDeploymentCard = serde_json::from_str(&stripped).unwrap();
        assert_eq!(back.worker_type, None);
        assert!(back.needs.is_empty());
    }

    #[test]
    fn indexer_identity_changes_mdcsum_and_missing_field_is_backward_compatible() {
        fn spec(value: &str) -> IndexerIdentitySpec {
            IndexerIdentitySpec::new(
                Some(
                    ExplicitIdentityMap::new(BTreeMap::from([(
                        "weights".to_string(),
                        value.to_string(),
                    )]))
                    .unwrap(),
                ),
                None,
            )
        }

        let baseline = ModelDeploymentCard::with_name_only("model");
        let mut explicit = ModelDeploymentCard::with_name_only("model");
        explicit.indexer_identity = Some(spec("revision-a"));
        assert_ne!(baseline.mdcsum(), explicit.mdcsum());

        let mut serialized = serde_json::to_value(&explicit).unwrap();
        assert!(
            serialized
                .as_object_mut()
                .unwrap()
                .remove("indexer_identity")
                .is_some()
        );
        let restored: ModelDeploymentCard = serde_json::from_value(serialized).unwrap();
        assert!(restored.indexer_identity.is_none());
    }
}
