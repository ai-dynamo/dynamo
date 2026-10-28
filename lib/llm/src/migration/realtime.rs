// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use dynamo_protocols::types::realtime::{
    DynamoRealtimeClientEvent, RealtimeClientEventSessionUpdate, RealtimeServerEvent,
};
use tokio::time::Instant;

use super::{AttemptBudget, is_migratable};

mod journal;
use journal::Journal;

const RECOVERY_TIMEOUT: Duration = Duration::from_secs(5);

#[derive(Debug, thiserror::Error)]
#[error("realtime worker stream ended unexpectedly")]
pub(crate) struct UnexpectedEnd;

pub(crate) struct RealtimeRecovery {
    journal: Option<Journal>,
    attempts: AttemptBudget,
    deadline: Option<Instant>,
}

impl RealtimeRecovery {
    pub(crate) fn new(session: &RealtimeClientEventSessionUpdate, migration_limit: u32) -> Self {
        Self {
            journal: (migration_limit > 0)
                .then(|| Journal::new(session))
                .flatten(),
            attempts: AttemptBudget::new(migration_limit),
            deadline: None,
        }
    }

    pub(crate) fn start_attempt(&mut self) -> anyhow::Result<()> {
        self.attempts
            .take()
            .map(|_| ())
            .ok_or_else(|| anyhow::anyhow!("realtime migration limit exhausted"))
    }

    pub(crate) fn retry(
        &mut self,
        error: anyhow::Error,
    ) -> anyhow::Result<DynamoRealtimeClientEvent> {
        let is_retryable = is_migratable(error.as_ref()) || error.is::<UnexpectedEnd>();
        let Some(journal) = self.journal.as_mut().filter(|j| j.can_recover()) else {
            return Err(error);
        };
        if !is_retryable || !self.attempts.has_remaining() {
            return Err(error);
        }
        let recovery_end = *self
            .deadline
            .get_or_insert_with(|| Instant::now() + RECOVERY_TIMEOUT);
        anyhow::ensure!(Instant::now() < recovery_end, "realtime recovery timed out");
        tracing::info!(
            attempt = self.attempts.next(),
            "Recovering realtime transcription worker"
        );
        journal
            .start_replay()
            .ok_or_else(|| anyhow::anyhow!("realtime replay unavailable"))
    }

    pub(crate) fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    pub(crate) fn is_replaying(&self) -> bool {
        self.deadline.is_some()
    }

    pub(crate) fn next_replay(&mut self) -> anyhow::Result<Option<DynamoRealtimeClientEvent>> {
        let Some(deadline) = self.deadline else {
            return Ok(None);
        };
        let journal = self
            .journal
            .as_mut()
            .ok_or_else(|| anyhow::anyhow!("realtime replay unavailable"))?;
        anyhow::ensure!(journal.can_recover(), "realtime replay disabled by worker");
        let event = journal.next_replay();
        if event.is_none() && !journal.waiting_for_session {
            let elapsed = RECOVERY_TIMEOUT - deadline.saturating_duration_since(Instant::now());
            tracing::info!(
                attempt = self.attempts.last(),
                elapsed_ms = elapsed.as_millis(),
                "Realtime transcription worker recovered"
            );
            self.deadline = None;
        }
        Ok(event)
    }

    pub(crate) fn record(&mut self, event: &DynamoRealtimeClientEvent) {
        if let Some(journal) = self.journal.as_mut() {
            journal.record(event);
        }
    }

    pub(crate) fn reconcile(&mut self, event: &mut RealtimeServerEvent) -> anyhow::Result<bool> {
        match self.journal.as_mut() {
            Some(journal) => journal.reconcile(event).map_err(anyhow::Error::msg),
            None => Ok(true),
        }
    }

    pub(crate) fn stream_ended(&self) -> anyhow::Result<()> {
        if self.journal.as_ref().is_some_and(Journal::can_recover) {
            Err(UnexpectedEnd.into())
        } else {
            Ok(())
        }
    }
}
