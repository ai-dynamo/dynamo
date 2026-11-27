// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! A picker that keeps placements fixed while measuring decode-progress reads.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use clap::ValueEnum;
use dynamo_kv_router::plugins::experimental::decode_progress::{
    DECODE_PROGRESS, DecodeProgressByRow,
};
use dynamo_kv_router::plugins::worker_selection::{
    WorkerInputView, WorkerInputs, WorkerPicker, WorkerSelectionPolicy, WorkerSelectionPolicyError,
    WorkerSelectionRequest,
};
use dynamo_router_protocol::worker::WorkerWithDpRank;
use dynamo_scheduler::host::WorkerSelectionPolicyFactory;
use serde::Serialize;

use super::corpus::Corpus;

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, ValueEnum)]
#[serde(rename_all = "snake_case")]
pub enum ProgressMode {
    Off,
    Collect,
    Read,
}

impl ProgressMode {
    pub fn collects(self) -> bool {
        matches!(self, Self::Collect | Self::Read)
    }
}

pub struct ProgressProbe {
    pub mode: ProgressMode,
    pub output_limit: u32,
    available: AtomicU64,
    unavailable: AtomicU64,
}

impl ProgressProbe {
    pub fn new(mode: ProgressMode, corpus: &Corpus) -> Self {
        // A single synthetic cap covers the replay. It does not expose each trace request's
        // final observed length as its cap.
        let output_limit = corpus
            .requests
            .iter()
            .map(|request| request.output_length)
            .max()
            .unwrap_or(1)
            .max(1);
        Self {
            mode,
            output_limit,
            available: AtomicU64::new(0),
            unavailable: AtomicU64::new(0),
        }
    }

    pub fn factory(self: &Arc<Self>, corpus: &Arc<Corpus>) -> WorkerSelectionPolicyFactory {
        let probe = Arc::clone(self);
        let corpus = Arc::clone(corpus);
        Arc::new(move |context| {
            WorkerSelectionPolicy::new(
                context.worker_type().as_str(),
                Vec::new(),
                Box::new(ProbePicker {
                    probe: Arc::clone(&probe),
                    corpus: Arc::clone(&corpus),
                }),
            )
        })
    }

    pub fn reads(&self) -> (u64, u64) {
        match self.mode {
            ProgressMode::Read => {
                let available = self.available.load(Ordering::Relaxed);
                (
                    available + self.unavailable.load(Ordering::Relaxed),
                    available,
                )
            }
            ProgressMode::Off | ProgressMode::Collect => (0, 0),
        }
    }
}

struct ProbePicker {
    probe: Arc<ProgressProbe>,
    corpus: Arc<Corpus>,
}

impl WorkerPicker for ProbePicker {
    fn required_worker_inputs(&self) -> WorkerInputs {
        if self.probe.mode.collects() {
            DECODE_PROGRESS | WorkerInputs::CACHE
        } else {
            WorkerInputs::CACHE
        }
    }

    fn pick(
        &self,
        request: &WorkerSelectionRequest<'_>,
        input: WorkerInputView<'_>,
    ) -> Result<usize, WorkerSelectionPolicyError> {
        let request = request
            .info()
            .request_id()
            .and_then(|id| id.strip_prefix("req-"))
            .and_then(|index| index.parse::<usize>().ok())
            .and_then(|index| self.corpus.requests.get(index))
            .ok_or_else(|| WorkerSelectionPolicyError::failed("invalid replay request ID"))?;
        let params = &self.corpus.params;
        let rank = request.lane as usize % params.ranks();
        let worker = WorkerWithDpRank::new(
            (rank / params.dp_size as usize + 1) as u64,
            (rank % params.dp_size as usize) as u32,
        );
        let row = input
            .candidates()
            .iter()
            .position(|candidate| candidate.worker() == worker)
            .ok_or_else(|| WorkerSelectionPolicyError::failed("replay lane rank is ineligible"))?;
        input
            .cache()
            .and_then(|cache| cache.get(row))
            .ok_or_else(|| WorkerSelectionPolicyError::failed("missing replay cache row"))?;
        let available =
            self.probe.mode == ProgressMode::Read && input.decode_progress(row).is_some();
        // Charge all arms the same coverage bookkeeping.
        let counter = if available {
            &self.probe.available
        } else {
            &self.probe.unavailable
        };
        counter.fetch_add(1, Ordering::Relaxed);
        Ok(row)
    }
}
