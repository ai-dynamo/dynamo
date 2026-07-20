// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Minimal runtime-loaded worker-selector plugin.

use dynamo_worker_selector_plugin_api::{
    CandidateInputs, RouterRole, Selection, SelectionInput, WorkerSelectorPlugin,
    export_worker_selector_plugin,
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

    fn required_candidate_inputs(&self) -> CandidateInputs {
        match self {
            Self::MostCached => CandidateInputs::CACHED_TOKENS,
            Self::UseDefault => CandidateInputs::NONE,
        }
    }

    fn select(&mut self, input: SelectionInput<'_>) -> Result<Selection, String> {
        match self {
            Self::UseDefault => Ok(Selection::UseDefault),
            Self::MostCached => {
                let cached_tokens = input
                    .cached_tokens()
                    .ok_or_else(|| "cached-token input is unavailable".to_string())?;
                (0..input.candidate_count())
                    .max_by_key(|&index| {
                        (
                            cached_tokens[index],
                            input.worker_ids()[index],
                            input.dp_ranks()[index],
                        )
                    })
                    .map(Selection::Candidate)
                    .ok_or_else(|| "no eligible candidates".to_string())
            }
        }
    }
}

export_worker_selector_plugin!(Strategy);
