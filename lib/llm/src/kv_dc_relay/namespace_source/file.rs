// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transport-independent source files, usable with or without an operator.

use super::{NamespaceScope, NamespaceSelection, NamespaceSource, NamespaceUpdates};
use anyhow::{Context, Result, ensure};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::time::Duration;
use std::{collections::HashSet, path::PathBuf};
use tokio::io::AsyncReadExt;
use tokio_util::sync::CancellationToken;

const MAX_SOURCES_BYTES: u64 = 1024 * 1024;

#[derive(Debug, Clone)]
pub struct KvDcRelaySourcesFile {
    pub path: PathBuf,
    pub connection_revision: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(crate) struct RelaySource {
    pub namespace: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "camelCase", deny_unknown_fields)]
pub(crate) struct SourcesDocument {
    pub version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub connection_revision: Option<String>,
    pub sources: Vec<RelaySource>,
    #[serde(skip)]
    pub revision: String,
}

impl SourcesDocument {
    pub(crate) fn canonicalize(&mut self, connection: Option<&str>) -> Result<()> {
        ensure!(self.version == 1, "unsupported sources format version");
        ensure!(
            self.connection_revision.as_deref() == connection
                && connection.is_none_or(|id| !id.is_empty() && id.len() <= 512),
            "sources connection revision does not match the running Relay"
        );
        ensure!(self.sources.len() <= 4096, "sources limit exceeded");
        let mut namespaces = HashSet::new();
        for source in &self.sources {
            ensure!(
                source.namespace.len() <= 512
                    && !source.namespace.is_empty()
                    && source.namespace.trim() == source.namespace,
                "invalid source namespace"
            );
            ensure!(
                namespaces.insert(&source.namespace),
                "duplicate source namespace"
            );
        }
        self.sources.sort_by(|a, b| a.namespace.cmp(&b.namespace));
        // Match Go encoding/json, including HTML and JavaScript-safe escaping.
        // Absent connectionRevision is omitted in both implementations.
        let canonical = serde_json::to_string(self)?
            .replace('&', "\\u0026")
            .replace('<', "\\u003c")
            .replace('>', "\\u003e")
            .replace('\u{2028}', "\\u2028")
            .replace('\u{2029}', "\\u2029");
        self.revision = format!("{:x}", Sha256::digest(canonical.as_bytes()));
        Ok(())
    }
}

impl KvDcRelaySourcesFile {
    pub(crate) async fn read(&self) -> Result<SourcesDocument> {
        let file = tokio::fs::File::open(&self.path)
            .await
            .context("sources file is unavailable")?;
        let mut bytes = Vec::new();
        file.take(MAX_SOURCES_BYTES + 1)
            .read_to_end(&mut bytes)
            .await
            .context("cannot read sources file")?;
        ensure!(
            bytes.len() as u64 <= MAX_SOURCES_BYTES,
            "sources file size limit exceeded"
        );
        let mut document: SourcesDocument =
            serde_json::from_slice(&bytes).map_err(|_| anyhow::anyhow!("invalid sources JSON"))?;
        document.canonicalize(self.connection_revision.as_deref())?;
        Ok(document)
    }
}

impl From<SourcesDocument> for NamespaceSelection {
    fn from(document: SourcesDocument) -> Self {
        Self {
            scope: NamespaceScope::Namespaces(
                document
                    .sources
                    .into_iter()
                    .map(|source| source.namespace)
                    .collect(),
            ),
            revision: Some(document.revision),
        }
    }
}

impl NamespaceSource for KvDcRelaySourcesFile {
    fn updates(self: Box<Self>, cancel: CancellationToken) -> NamespaceUpdates {
        Box::pin(async_stream::stream! {
            let mut poll = tokio::time::interval(Duration::from_secs(1));
            poll.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Delay);
            loop {
                tokio::select! {
                    _ = cancel.cancelled() => break,
                    _ = poll.tick() => {}
                }
                let result = tokio::select! {
                    _ = cancel.cancelled() => break,
                    result = tokio::time::timeout(Duration::from_secs(10), self.read()) => result,
                };
                // File parsing errors omit the document contents.
                yield match result {
                    Ok(Ok(document)) => Ok(document.into()),
                    Ok(Err(error)) => Err(error),
                    Err(_) => Err(anyhow::anyhow!("sources file read timed out")),
                };
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_canonical_revisions() {
        #[derive(Deserialize)]
        struct Fixture {
            document: SourcesDocument,
            canonical: String,
            revision: String,
        }
        let fixtures: Vec<Fixture> = serde_json::from_str(include_str!(
            "../../../tests/fixtures/relay-sources-canonical.json"
        ))
        .unwrap();
        for fixture in fixtures {
            let mut document = fixture.document;
            let connection = document.connection_revision.clone();
            document.canonicalize(connection.as_deref()).unwrap();
            assert_eq!(document.revision, fixture.revision);
            assert_eq!(
                format!("{:x}", Sha256::digest(fixture.canonical.as_bytes())),
                fixture.revision
            );
        }
    }

    #[test]
    fn source_format_canonicalization_and_guard() {
        let raw = r#"{"sources":[{"namespace":"b"},{"namespace":"a"}],"version":1}"#;
        let mut doc: SourcesDocument = serde_json::from_str(raw).unwrap();
        doc.canonicalize(None).unwrap();
        let revision = doc.revision.clone();
        doc.sources.reverse();
        doc.canonicalize(None).unwrap();
        assert_eq!(doc.revision, revision);
        assert!(doc.canonicalize(Some("connection")).is_err());
        doc.connection_revision = Some("connection".into());
        assert!(doc.canonicalize(None).is_err());
        assert!(doc.canonicalize(Some("other")).is_err());
        doc.canonicalize(Some("connection")).unwrap();
        assert_ne!(doc.revision, revision);
        doc.sources.push(doc.sources[0].clone());
        assert!(doc.canonicalize(Some("connection")).is_err());
        assert!(
            serde_json::from_str::<SourcesDocument>(
                r#"{"version":1,"sources":[{"namespace":"a","id":"old"}]}"#
            )
            .is_err()
        );
        assert!(
            serde_json::from_str::<SourcesDocument>(
                r#"{"version":1,"revision":"old","sources":[]}"#
            )
            .is_err()
        );
    }
    #[test]
    fn file_namespaces_accept_discovery_names_and_reject_surrounding_whitespace() {
        use crate::kv_dc_relay::KvDcRelayDiscoveryConfig;

        for namespace in [
            "model.a",
            "team/model",
            "model name",
            "モデル",
            "",
            " ",
            " a",
            "a ",
            "a\n",
        ] {
            let discovery = KvDcRelayDiscoveryConfig {
                namespaces: vec![namespace.into()],
                ..Default::default()
            };
            let mut document: SourcesDocument = serde_json::from_value(serde_json::json!({
                "version": 1,
                "sources": [{"namespace": namespace}],
            }))
            .unwrap();
            assert_eq!(
                document.canonicalize(None).is_ok(),
                discovery.validate().is_ok(),
                "{namespace:?}"
            );
        }
    }
}
