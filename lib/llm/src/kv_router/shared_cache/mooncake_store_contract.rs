// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;

use anyhow::{Context, ensure};
use serde::{Deserialize, Serialize};

use crate::local_model::runtime_config::ModelRuntimeConfig;

pub(crate) const RUNTIME_KEY: &str = "vllm_mooncake_store";
const VLLM_REVISION: &str = "1085b64425a9e6f5ca52876ad32e55fda5665f4e";
const MAX_GROUPS: usize = 64;
const MAX_PREFIXES: usize = 1024;
const MAX_PREFIX_BYTES: usize = 4096;

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Descriptor {
    schema_version: u32,
    adapter: String,
    vllm_revision: String,
    hash: HashContract,
    input: InputContract,
    main_event_group: usize,
    main_event_block_size: u32,
    gpu_to_store_group: Vec<Option<usize>>,
    coordinator: Coordinator,
    groups: Vec<Group>,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct HashContract {
    algorithm: String,
    digest_encoding: String,
    gpu_event_hash: String,
    seed_policy: String,
    key_separator: String,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct InputContract {
    text_only: bool,
    normalized_namespaces: bool,
    lora: bool,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Coordinator {
    lcm_block_size: u32,
    speculative: bool,
    drop_blocks: bool,
    partial_hash_hits: bool,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(rename_all = "snake_case")]
pub(super) enum GroupKind {
    FullAttention,
    SlidingWindow,
    Mamba,
}

#[derive(Clone, Debug, PartialEq, Eq, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct Group {
    group_id: usize,
    pub(super) kind: GroupKind,
    spec: String,
    manager: String,
    pub(super) block_size: u32,
    hash_block_size: u32,
    key_prefixes: Vec<String>,
    pub(super) sliding_window: Option<u32>,
    mamba_cache_mode: Option<String>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ValidatedContract {
    descriptor: Descriptor,
    pub(super) alignment: u32,
    pub(super) prefixes: HashMap<String, u16>,
    pub(super) group_prefixes: Vec<Vec<u16>>,
}

impl ValidatedContract {
    pub(crate) fn from_runtime(runtime: &ModelRuntimeConfig) -> anyhow::Result<Option<Self>> {
        runtime
            .get_engine_specific::<Descriptor>(RUNTIME_KEY)?
            .map(Self::validate)
            .transpose()
    }

    #[cfg(test)]
    pub(super) fn runtime_config(&self) -> ModelRuntimeConfig {
        let mut runtime = ModelRuntimeConfig::new();
        runtime
            .set_engine_specific(RUNTIME_KEY, &self.descriptor)
            .unwrap();
        runtime
    }

    pub(crate) fn main_event_block_size(&self) -> u32 {
        self.descriptor.main_event_block_size
    }

    pub(super) fn groups(&self) -> &[Group] {
        &self.descriptor.groups
    }

    fn validate(mut descriptor: Descriptor) -> anyhow::Result<Self> {
        ensure!(
            descriptor.schema_version == 1,
            "unsupported Mooncake Store schema"
        );
        ensure!(
            descriptor.adapter == "vllm-1085b644",
            "unsupported Mooncake Store adapter"
        );
        ensure!(
            descriptor.vllm_revision == VLLM_REVISION,
            "unsupported vLLM revision"
        );
        let hash = &descriptor.hash;
        ensure!(
            hash.algorithm == "sha256"
                && hash.digest_encoding == "hex"
                && hash.gpu_event_hash == "low64"
                && hash.seed_policy == "pythonhashseed-0"
                && hash.key_separator == "@",
            "unsupported Mooncake Store hash contract"
        );
        ensure!(
            descriptor.input.text_only
                && descriptor.input.normalized_namespaces
                && descriptor.input.lora,
            "unsupported Mooncake Store input contract"
        );
        let b = descriptor.main_event_block_size;
        let s = descriptor.coordinator.lcm_block_size;
        ensure!(b > 0 && s > 0, "zero Mooncake Store block span");
        ensure!(
            !descriptor.coordinator.speculative && !descriptor.coordinator.drop_blocks,
            "unsupported speculative or block-drop semantics"
        );
        ensure!(
            !descriptor.groups.is_empty() && descriptor.groups.len() <= MAX_GROUPS,
            "unsupported Mooncake Store group count"
        );
        ensure!(
            !descriptor.gpu_to_store_group.is_empty()
                && descriptor.gpu_to_store_group.len() <= MAX_GROUPS,
            "unsupported GPU group projection"
        );
        let projected: Vec<_> = descriptor
            .gpu_to_store_group
            .iter()
            .flatten()
            .copied()
            .collect();
        ensure!(
            projected == (0..descriptor.groups.len()).collect::<Vec<_>>(),
            "GPU group projection is not ordered and complete"
        );
        let main_group = descriptor
            .gpu_to_store_group
            .get(descriptor.main_event_group)
            .copied()
            .flatten()
            .context("main event group is not projected")?;
        ensure!(
            descriptor.groups[main_group].kind == GroupKind::FullAttention
                && descriptor.groups[main_group].block_size == b,
            "main GPU event span does not identify a full-attention store group"
        );
        ensure!(
            descriptor
                .groups
                .iter()
                .position(|g| g.kind == GroupKind::FullAttention)
                == Some(main_group),
            "main event group is not the first full-attention group"
        );
        let hash_span = descriptor.groups[0].hash_block_size;
        ensure!(
            hash_span > 0 && b.is_multiple_of(hash_span),
            "unobservable hash span"
        );
        let mut alignment = checked_lcm(b, s)?;
        let mut prefixes = HashMap::new();
        let mut group_prefixes = Vec::new();
        for (idx, group) in descriptor.groups.iter_mut().enumerate() {
            ensure!(
                group.group_id == idx,
                "store groups are not ordered and contiguous"
            );
            ensure!(
                group.block_size > 0
                    && group.hash_block_size == hash_span
                    && group.block_size.is_multiple_of(hash_span)
                    && s.is_multiple_of(group.block_size),
                "invalid physical, hash, or coordinator spans"
            );
            alignment = checked_lcm(alignment, group.block_size)?;
            match group.kind {
                GroupKind::FullAttention => {
                    ensure!(
                        group.spec == "FullAttentionSpec"
                            && group.manager == "FullAttentionManager"
                            && group.sliding_window.is_none()
                            && group.mamba_cache_mode.is_none(),
                        "unsupported full-attention semantics"
                    );
                    ensure!(
                        group.block_size.is_multiple_of(b),
                        "unobservable full-attention boundaries"
                    );
                }
                GroupKind::SlidingWindow => {
                    ensure!(
                        group.spec == "SlidingWindowSpec"
                            && group.manager == "SlidingWindowManager"
                            && group.mamba_cache_mode.is_none(),
                        "unsupported sliding-window semantics"
                    );
                    let window = group.sliding_window.context("missing sliding window")?;
                    ensure!(window > 1, "unsupported sliding window");
                    ensure!(
                        group.block_size.is_multiple_of(b)
                            || (window - 1).div_ceil(group.block_size) == 1,
                        "unobservable interior sliding-window boundaries"
                    );
                }
                GroupKind::Mamba => {
                    ensure!(
                        group.spec == "MambaSpec"
                            && group.manager == "MambaManager"
                            && group.mamba_cache_mode.as_deref() == Some("align")
                            && group.sliding_window.is_none(),
                        "unsupported Mamba semantics"
                    );
                }
            }
            ensure!(
                !group.key_prefixes.is_empty(),
                "missing required object prefixes"
            );
            group.key_prefixes.sort();
            let mut ids = Vec::new();
            for prefix in &group.key_prefixes {
                ensure!(
                    !prefix.is_empty()
                        && prefix.len() <= MAX_PREFIX_BYTES
                        && !prefix.ends_with('@'),
                    "invalid Mooncake Store object prefix"
                );
                ensure!(
                    prefixes.len() < MAX_PREFIXES,
                    "too many required object prefixes"
                );
                let id = prefixes.len() as u16;
                ensure!(
                    prefixes.insert(prefix.clone(), id).is_none(),
                    "duplicate required object prefix"
                );
                ids.push(id);
            }
            group_prefixes.push(ids);
        }
        // Prefix IDs are assigned after sorting, so equivalent rank orderings compare equal.
        Ok(Self {
            descriptor,
            alignment,
            prefixes,
            group_prefixes,
        })
    }
}

fn checked_lcm(a: u32, b: u32) -> anyhow::Result<u32> {
    let (mut x, mut y) = (a, b);
    while y != 0 {
        (x, y) = (y, x % y);
    }
    (a / x)
        .checked_mul(b)
        .context("Mooncake Store alignment overflow")
}

#[cfg(test)]
pub(super) fn test_contract(
    groups: &[(GroupKind, u32, Option<u32>)],
    b: u32,
    ranks: usize,
) -> ValidatedContract {
    let groups = groups
        .iter()
        .enumerate()
        .map(|(idx, &(kind, block_size, sliding_window))| {
            let (spec, manager) = match kind {
                GroupKind::FullAttention => ("FullAttentionSpec", "FullAttentionManager"),
                GroupKind::SlidingWindow => ("SlidingWindowSpec", "SlidingWindowManager"),
                GroupKind::Mamba => ("MambaSpec", "MambaManager"),
            };
            Group {
                group_id: idx,
                kind,
                spec: spec.into(),
                manager: manager.into(),
                block_size,
                hash_block_size: groups.iter().fold(b, |gcd, (_, size, _)| {
                    let (mut a, mut b) = (gcd, *size);
                    while b != 0 {
                        (a, b) = (b, a % b);
                    }
                    a
                }),
                key_prefixes: (0..ranks)
                    .map(|r| format!("deployment@g{idx}@r{r}"))
                    .collect(),
                sliding_window,
                mamba_cache_mode: (kind == GroupKind::Mamba).then(|| "align".into()),
            }
        })
        .collect::<Vec<_>>();
    let s = groups
        .iter()
        .fold(b, |a, g| checked_lcm(a, g.block_size).unwrap());
    ValidatedContract::validate(Descriptor {
        schema_version: 1,
        adapter: "vllm-1085b644".into(),
        vllm_revision: VLLM_REVISION.into(),
        hash: HashContract {
            algorithm: "sha256".into(),
            digest_encoding: "hex".into(),
            gpu_event_hash: "low64".into(),
            seed_policy: "pythonhashseed-0".into(),
            key_separator: "@".into(),
        },
        input: InputContract {
            text_only: true,
            normalized_namespaces: true,
            lora: true,
        },
        main_event_group: groups
            .iter()
            .position(|g| g.kind == GroupKind::FullAttention)
            .unwrap(),
        main_event_block_size: b,
        gpu_to_store_group: (0..groups.len()).map(Some).collect(),
        coordinator: Coordinator {
            lcm_block_size: s,
            speculative: false,
            drop_blocks: false,
            partial_hash_hits: true,
        },
        groups,
    })
    .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    #[test]
    fn rejects_unsupported_contracts_and_geometry() {
        let valid = test_contract(
            &[
                (GroupKind::FullAttention, 32, None),
                (GroupKind::SlidingWindow, 8, Some(8)),
            ],
            32,
            2,
        );
        let mut invalid = valid.descriptor.clone();
        invalid.groups[1].sliding_window = Some(16);
        assert!(ValidatedContract::validate(invalid).is_err());
        for mutate in [
            |d: &mut Descriptor| d.schema_version = 2,
            |d: &mut Descriptor| d.hash.seed_policy = "random".into(),
            |d: &mut Descriptor| d.groups[0].manager = "UnknownManager".into(),
            |d: &mut Descriptor| d.coordinator.speculative = true,
            |d: &mut Descriptor| d.coordinator.drop_blocks = true,
            |d: &mut Descriptor| d.input.text_only = false,
            |d: &mut Descriptor| {
                let prefix = d.groups[0].key_prefixes[0].clone();
                d.groups[0].key_prefixes.push(prefix);
            },
            |d: &mut Descriptor| d.gpu_to_store_group[0] = None,
            |d: &mut Descriptor| d.main_event_block_size = 16,
            |d: &mut Descriptor| d.groups[0].hash_block_size = 0,
        ] {
            let mut invalid = valid.descriptor.clone();
            mutate(&mut invalid);
            assert!(ValidatedContract::validate(invalid).is_err());
        }
    }

    #[test]
    fn canonical_prefixes_and_nullable_projection() {
        let valid = test_contract(&[(GroupKind::FullAttention, 16, None)], 16, 2);
        let mut reordered = valid.descriptor.clone();
        reordered.groups[0].key_prefixes.reverse();
        assert_eq!(valid, ValidatedContract::validate(reordered).unwrap());
        let mut projected = valid.descriptor.clone();
        projected.gpu_to_store_group = vec![None, Some(0), None];
        projected.main_event_group = 1;
        assert!(ValidatedContract::validate(projected).is_ok());
        assert!(checked_lcm(u32::MAX, u32::MAX - 1).is_err());
        assert_eq!(valid.prefixes.values().collect::<HashSet<_>>().len(), 2);
    }
}
