// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::{
    env,
    path::{Path, PathBuf},
};

use anyhow::Context;
use hf_hub::{
    Cache, Repo, RepoType,
    api::tokio::{Api, ApiBuilder},
};

use dynamo_runtime::config::environment_names::model as env_model;

use super::is_offline_mode;

/// A Hugging Face model repository and the revision requested by an `hf://` URI.
#[derive(Debug, Eq, PartialEq)]
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
    validate_hf_relative_path(filename, "repository filename")
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
    env::var(env_model::huggingface::HF_TOKEN)
        .ok()
        .filter(|token| !token.trim().is_empty())
        .or_else(|| {
            let path = hf_token_path_from_values(
                env::var(env_model::huggingface::HF_TOKEN_PATH).ok(),
                env::var(env_model::huggingface::HF_HOME).ok(),
                env::var("XDG_CACHE_HOME").ok(),
                env::var("HOME").ok(),
                env::var("USERPROFILE").ok(),
            );
            std::fs::read_to_string(path)
                .ok()
                .map(|token| token.trim().to_string())
                .filter(|token| !token.is_empty())
        })
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
    snapshot.join(anchor_file).is_file().then_some(snapshot)
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

/// Download one immutable repository snapshot into hf-hub's native cache layout.
///
/// The requested branch/tag is resolved once via `info()`. Every sibling is then
/// downloaded through the commit SHA, and the mutable ref is published only after
/// the whole snapshot succeeds.
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

    let api = build_hf_api(cache.clone())?;
    let requested_repo = spec.repo();
    let info = api
        .repo(requested_repo.clone())
        .info()
        .await
        .with_context(|| {
            format!(
                "resolving Hugging Face repository {} at revision {}",
                spec.repo_id, spec.revision
            )
        })?;

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
    let pinned_api = api.repo(pinned_repo);
    for sibling in &info.siblings {
        pinned_api.get(&sibling.rfilename).await.with_context(|| {
            format!(
                "downloading {} from Hugging Face repository {}@{}",
                sibling.rfilename, spec.repo_id, info.sha
            )
        })?;
    }

    cache
        .repo(requested_repo.clone())
        .create_ref(&info.sha)
        .with_context(|| {
            format!(
                "publishing Hugging Face cache ref {}@{}",
                spec.repo_id, spec.revision
            )
        })?;

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

        for path in ["", "../secret", "nested/../../secret", "/etc/passwd"] {
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
}
