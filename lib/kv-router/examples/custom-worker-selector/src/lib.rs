// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal runtime-loaded worker-selector plugin.

use dynamo_worker_selector_plugin_api::{
    RouterRole, Selection, SelectionInput, WorkerSelectorPlugin, export_worker_selector_plugin,
};

enum Strategy {
    MostCached,
    UseDefault,
}

impl WorkerSelectorPlugin for Strategy {
    fn from_config(config: &[u8], _router_role: RouterRole) -> Result<Self, String> {
        match config {
            b"" | b"most-cached" => Ok(Self::MostCached),
            b"use-default" => Ok(Self::UseDefault),
            _ => Err("expected most-cached or use-default".into()),
        }
    }

    fn select(&mut self, input: SelectionInput<'_>) -> Result<Selection, String> {
        match self {
            Self::UseDefault => Ok(Selection::UseDefault),
            Self::MostCached => input
                .candidates()
                .iter()
                .enumerate()
                .max_by_key(|(_, candidate)| {
                    (
                        candidate.cached_tokens,
                        candidate.worker_id,
                        candidate.dp_rank,
                    )
                })
                .map(|(index, _)| Selection::Candidate(index))
                .ok_or_else(|| "no eligible candidates".to_string()),
        }
    }
}

export_worker_selector_plugin!(Strategy);
