// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::VecDeque;
use std::io::{self, Write};
use std::sync::Arc;

use dynamo_protocols::types::realtime::{
    DynamoRealtimeClientEvent, RealtimeAudioFormats, RealtimeClientEvent,
    RealtimeClientEventSessionUpdate, RealtimeServerEvent, Session,
};

use super::budget::{Budget, Reservation};

const MAX_RETAINED_BYTES: usize = 16 * 1024 * 1024;
const MAX_RETAINED_EVENTS: usize = 4096;

#[derive(Default)]
struct Turn {
    events: Vec<RealtimeClientEvent>,
    next_event: usize,
    bytes: usize,
    is_committed: bool,
    is_cleared: bool,
    commit_acknowledged: bool,
    has_terminal: bool,
    client_item: Option<String>,
    worker_item: Option<String>,
    prefix: String,
    replay_remaining: usize,
}

/// Only unfinished, ordered ASR turns belong here. Workers own inference state.
pub(super) struct Journal {
    session: Option<RealtimeClientEventSessionUpdate>,
    turns: VecDeque<Turn>,
    previous_item: Option<(String, String)>,
    reservation: Reservation,
    events: usize,
    max_bytes: usize,
    max_events: usize,
    pub(super) is_acknowledged: bool,
    pub(super) waiting_for_session: bool,
}

fn eligible(session: &Session) -> bool {
    matches!(session, Session::RealtimeTranscriptionSession(s)
        if s.audio.input.turn_detection.is_none()
            && matches!(s.audio.input.format, RealtimeAudioFormats::PCMAudioFormat { .. }))
}

impl Journal {
    pub(super) fn new(
        session: &RealtimeClientEventSessionUpdate,
        budget: &Arc<Budget>,
    ) -> Option<Self> {
        if !eligible(&session.session) {
            return None;
        }
        let bytes = retained_size(session);
        let mut reservation = budget.reservation();
        if bytes > MAX_RETAINED_BYTES || !reservation.try_grow(bytes) {
            tracing::debug!("Realtime recovery disabled: session exceeds retention limit");
            return None;
        }
        Some(Self {
            session: Some(session.clone()),
            turns: VecDeque::new(),
            previous_item: None,
            reservation,
            events: 1,
            max_bytes: MAX_RETAINED_BYTES,
            max_events: MAX_RETAINED_EVENTS,
            is_acknowledged: false,
            waiting_for_session: true,
        })
    }

    pub(super) fn can_recover(&self) -> bool {
        self.session.is_some() && self.is_acknowledged
    }

    pub(super) fn disable(&mut self, reason: &'static str) {
        let Some(session) = self.session.take() else {
            return;
        };
        tracing::debug!(reason, "Realtime recovery disabled");
        let mut released = retained_size(&session);
        drop(session);
        // Keep turn order so a later clear cannot suppress an earlier turn's output.
        for turn in &mut self.turns {
            turn.events = Vec::new();
            // Keep only reconciliation still needed by output already in flight.
            if turn.replay_remaining == 0 {
                turn.prefix = String::new();
            }
            let retained = turn.prefix.capacity()
                + turn.client_item.as_ref().map_or(0, String::capacity)
                + turn.worker_item.as_ref().map_or(0, String::capacity);
            released += turn.bytes - retained;
            turn.bytes = retained;
        }
        self.reservation.release(released);
        self.release_empty_queue();
        self.events = 0;
    }

    fn release_empty_queue(&mut self) {
        if self.turns.is_empty() {
            let bytes = self.turns.capacity() * size_of::<Turn>();
            self.turns = VecDeque::new();
            self.reservation.release(bytes);
        }
    }

    pub(super) fn record(&mut self, event: &DynamoRealtimeClientEvent) {
        let DynamoRealtimeClientEvent::OpenAI(event) = event else {
            self.disable("unsupported client event");
            return;
        };
        if !matches!(
            event,
            RealtimeClientEvent::InputAudioBufferAppend(_)
                | RealtimeClientEvent::InputAudioBufferCommit(_)
                | RealtimeClientEvent::InputAudioBufferClear(_)
        ) {
            self.disable("session changed or unsupported client event");
            return;
        }

        let has_open_turn = self
            .turns
            .back()
            .is_some_and(|t| !t.is_committed && !t.is_cleared);
        if matches!(event, RealtimeClientEvent::InputAudioBufferCommit(_))
            && has_open_turn
            && let Some(turn) = self.turns.back_mut()
        {
            turn.is_committed = true;
        }
        if matches!(event, RealtimeClientEvent::InputAudioBufferClear(_))
            && has_open_turn
            && let Some(turn) = self.turns.back_mut()
        {
            // Reuse the allocated slot so even an exhausted budget can suppress
            // stale output until the worker acknowledges the clear.
            let turn = std::mem::replace(
                turn,
                Turn {
                    is_cleared: true,
                    ..Default::default()
                },
            );
            if self.session.is_some() {
                self.events -= turn.events.len();
            }
            let bytes = turn.bytes;
            drop(turn);
            self.reservation.release(bytes);
        }
        if self.session.is_none() {
            if matches!(event, RealtimeClientEvent::InputAudioBufferCommit(_))
                && let Some(turn) = self.turns.back_mut()
                && !turn.is_cleared
            {
                turn.is_committed = true;
            }
            return;
        }

        let needs_turn = !has_open_turn;
        let queue_growth = if needs_turn {
            capacity_growth(self.turns.len(), self.turns.capacity())
        } else {
            0
        };
        let event_growth = if needs_turn {
            capacity_growth(0, 0)
        } else {
            self.turns.back().map_or(0, |turn| {
                capacity_growth(turn.events.len(), turn.events.capacity())
            })
        };
        let turn_bytes = retained_size(event) - size_of_val(event)
            + event_growth * size_of::<RealtimeClientEvent>();
        let bytes = turn_bytes + queue_growth * size_of::<Turn>();
        if self.reservation.bytes.saturating_add(bytes) > self.max_bytes
            || self.events >= self.max_events
            || !self.reservation.try_grow(bytes)
        {
            self.disable("retention limit exceeded");
            return;
        }
        self.events += 1;
        if needs_turn {
            if queue_growth > 0 {
                self.turns.reserve_exact(queue_growth);
            }
            self.turns.push_back(Turn::default());
        }
        if let Some(turn) = self.turns.back_mut() {
            turn.bytes += turn_bytes;
            if event_growth > 0 {
                turn.events.reserve_exact(event_growth);
            }
            turn.is_committed = matches!(event, RealtimeClientEvent::InputAudioBufferCommit(_));
            turn.is_cleared = matches!(event, RealtimeClientEvent::InputAudioBufferClear(_));
            if let Some(event) = clone_audio_event(event) {
                turn.events.push(event);
            }
        }
    }

    pub(super) fn start_replay(&mut self) -> Option<DynamoRealtimeClientEvent> {
        let session = self.session.as_ref()?.clone();
        self.waiting_for_session = true;
        if let Some((worker, client)) = self.previous_item.take() {
            let bytes = worker.capacity() + client.capacity();
            drop((worker, client));
            self.reservation.release(bytes);
        }
        for turn in &mut self.turns {
            turn.next_event = 0;
            if let Some(id) = turn.worker_item.take() {
                let bytes = id.capacity();
                drop(id);
                turn.bytes -= bytes;
                self.reservation.release(bytes);
            }
            turn.replay_remaining = turn.prefix.len();
        }
        Some(RealtimeClientEvent::SessionUpdate(session).into())
    }

    pub(super) fn next_replay(&mut self) -> Option<DynamoRealtimeClientEvent> {
        if self.waiting_for_session {
            return None;
        }
        for turn in &mut self.turns {
            if let Some(event) = turn.events.get(turn.next_event) {
                turn.next_event += 1;
                return clone_audio_event(event).map(Into::into);
            }
        }
        None
    }

    fn finish_turn(&mut self) {
        if let Some(mut turn) = self.turns.pop_front() {
            if self.session.is_some() {
                self.events -= turn.events.len();
            }
            if let (Some(worker), Some(client)) = (turn.worker_item.take(), turn.client_item.take())
            {
                if let Some((worker, client)) = self.previous_item.take() {
                    let bytes = worker.capacity() + client.capacity();
                    drop((worker, client));
                    self.reservation.release(bytes);
                }
                turn.bytes -= worker.capacity() + client.capacity();
                self.previous_item = Some((worker, client));
            }
            let bytes = turn.bytes;
            drop(turn);
            self.reservation.release(bytes);
            self.release_empty_queue();
        }
    }

    /// Called immediately before writing to the socket; a failed write ends the session.
    pub(super) fn reconcile(
        &mut self,
        event: &mut RealtimeServerEvent,
    ) -> Result<bool, &'static str> {
        use RealtimeServerEvent::*;
        match event {
            SessionCreated(_) => return Ok(false),
            SessionUpdated(updated) => {
                let is_duplicate = self.is_acknowledged && self.waiting_for_session;
                self.waiting_for_session = false;
                self.is_acknowledged = true;
                if !eligible(&updated.session) {
                    self.disable("worker acknowledged an unsupported session");
                }
                return Ok(!is_duplicate);
            }
            Error(_) => {
                self.disable("worker application error");
                // An ordered error can cancel replay before the replacement assigns an item ID.
                let mut released = 0;
                self.turns.retain(|turn| {
                    if turn.client_item.is_some() && turn.worker_item.is_none() {
                        released += turn.bytes;
                        false
                    } else {
                        true
                    }
                });
                self.reservation.release(released);
                self.release_empty_queue();
                return Ok(true);
            }
            InputAudioBufferCleared(_) => {
                if self.turns.front().is_some_and(|t| t.is_cleared) {
                    self.finish_turn();
                } else if self.session.is_some() {
                    return Err("out-of-order clear acknowledgement");
                }
                return Ok(true);
            }
            _ => {}
        }
        if let InputAudioBufferCommitted(e) = event
            && let Some(previous) = &mut e.previous_item_id
            && let Some((worker, client)) = &self.previous_item
            && previous == worker
        {
            *previous = client.clone();
        }
        let (item_id, delta, is_terminal, is_commit) = match event {
            InputAudioBufferCommitted(e) => (&mut e.item_id, None, false, true),
            ConversationItemInputAudioTranscriptionDelta(e) => {
                (&mut e.item_id, Some(&mut e.delta), false, false)
            }
            ConversationItemInputAudioTranscriptionCompleted(e) => {
                (&mut e.item_id, None, true, false)
            }
            ConversationItemInputAudioTranscriptionFailed(e) => (&mut e.item_id, None, true, false),
            _ => {
                self.disable("unsupported worker event");
                return Ok(true);
            }
        };
        // A recoverable application error may abort a turn without a final event.
        // Once replay is disabled, a new item retires that stale binding.
        while self.session.is_none()
            && self
                .turns
                .front()
                .is_some_and(|turn| turn.worker_item.as_ref().is_some_and(|id| id != item_id))
        {
            if let Some(turn) = self.turns.pop_front() {
                let bytes = turn.bytes;
                drop(turn);
                self.reservation.release(bytes);
            }
            self.release_empty_queue();
        }
        let Some(turn) = self.turns.front_mut() else {
            if self.session.is_some() {
                return Err("transcription output without an input turn");
            }
            return Ok(true);
        };
        if turn.is_cleared || (turn.has_terminal && !is_commit) {
            return Ok(false);
        }
        if self.session.is_none() && turn.client_item.is_none() {
            // No identity translation is needed, but queued clear markers still
            // require this turn's terminal output before they become active.
            turn.has_terminal |= is_terminal;
            if turn.has_terminal && turn.is_committed {
                self.finish_turn();
            }
            return Ok(true);
        }
        let binding_bytes = usize::from(turn.worker_item.is_none()) * item_id.len()
            + usize::from(turn.client_item.is_none()) * item_id.len();
        if binding_bytes > 0
            && (self.reservation.bytes.saturating_add(binding_bytes) > self.max_bytes
                || !self.reservation.try_grow(binding_bytes))
        {
            let needs_translation = turn.client_item.is_some();
            self.disable("item identity retention limit exceeded");
            return if needs_translation {
                Err("realtime replay identity budget exhausted")
            } else {
                self.reconcile(event)
            };
        }
        let turn = self.turns.front_mut().ok_or("missing transcription turn")?;
        match &turn.worker_item {
            Some(bound) if bound != item_id => return Err("out-of-order transcription output"),
            None => {
                turn.worker_item = Some(item_id.clone());
                let bytes = item_id.len();
                turn.bytes += bytes;
            }
            _ => {}
        }
        if let Some(original) = &turn.client_item {
            *item_id = original.clone();
        } else {
            turn.client_item = Some(item_id.clone());
            turn.bytes += item_id.len();
        }

        let mut forward = true;
        if is_commit {
            if !turn.is_committed {
                return Err("commit acknowledgement without a client commit");
            }
            forward = !turn.commit_acknowledged;
            turn.commit_acknowledged = true;
        }
        if let Some(delta) = delta {
            let offset = turn.prefix.len() - turn.replay_remaining;
            let matched = turn.replay_remaining.min(delta.len());
            if turn.prefix.as_bytes()[offset..offset + matched] != delta.as_bytes()[..matched] {
                return Err("replayed transcription diverged");
            }
            // Equal UTF-8 prefixes of complete strings always end on a character boundary.
            turn.replay_remaining -= matched;
            delta.drain(..matched);
            forward = !delta.is_empty();
            if self.session.is_some() {
                let needed = turn.prefix.len().saturating_add(delta.len());
                let capacity = if needed > turn.prefix.capacity() {
                    needed.max(turn.prefix.capacity().saturating_mul(2)).max(8)
                } else {
                    turn.prefix.capacity()
                };
                let added = capacity - turn.prefix.capacity();
                if self.reservation.bytes.saturating_add(added) > self.max_bytes
                    || !self.reservation.try_grow(added)
                {
                    self.disable("transcript retention limit exceeded");
                } else {
                    if added > 0 {
                        turn.prefix.reserve_exact(capacity - turn.prefix.len());
                    }
                    turn.prefix.push_str(delta);
                    turn.bytes += added;
                }
            } else if turn.replay_remaining == 0 {
                let bytes = turn.prefix.capacity();
                turn.prefix = String::new();
                turn.bytes -= bytes;
                self.reservation.release(bytes);
            }
        }
        if is_terminal {
            // Finals may revise an interim hypothesis; they are not delta prefixes.
            let turn = self.turns.front_mut().ok_or("missing terminal turn")?;
            turn.has_terminal = true;
            if !turn.is_committed {
                self.disable("worker completed an open turn");
            }
        }
        if self
            .turns
            .front()
            .is_some_and(|t| t.has_terminal && t.is_committed)
        {
            self.finish_turn();
        }
        Ok(forward)
    }
}

fn capacity_growth(len: usize, capacity: usize) -> usize {
    if len == capacity { capacity.max(4) } else { 0 }
}

fn clone_audio_event(event: &RealtimeClientEvent) -> Option<RealtimeClientEvent> {
    use RealtimeClientEvent::*;
    Some(match event {
        InputAudioBufferAppend(event) => InputAudioBufferAppend(event.clone()),
        InputAudioBufferCommit(event) => InputAudioBufferCommit(event.clone()),
        InputAudioBufferClear(event) => InputAudioBufferClear(event.clone()),
        _ => return None,
    })
}

fn retained_size(value: &impl serde::Serialize) -> usize {
    struct Counter(usize);
    impl Write for Counter {
        fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
            self.0 = self.0.saturating_add(bytes.len());
            Ok(bytes.len())
        }
        fn flush(&mut self) -> io::Result<()> {
            Ok(())
        }
    }
    let mut count = Counter(size_of_val(value));
    if serde_json::to_writer(&mut count, value).is_err() {
        return usize::MAX;
    }
    count.0
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::{Value, json};

    fn configuration() -> RealtimeClientEventSessionUpdate {
        serde_json::from_value(json!({"session": {
            "type": "transcription", "audio": {"input": {
                "format": {"type": "audio/pcm", "rate": 24000},
                "transcription": {"model": "asr"}, "turn_detection": null
            }}
        }}))
        .unwrap()
    }

    fn record(journal: &mut Journal, value: Value) {
        journal.record(&serde_json::from_value(value).unwrap());
    }

    fn server(journal: &mut Journal, value: Value) -> Result<Option<Value>, &'static str> {
        let mut event = serde_json::from_value(value).unwrap();
        journal
            .reconcile(&mut event)
            .map(|forward| forward.then(|| serde_json::to_value(event).unwrap()))
    }

    fn acknowledge(journal: &mut Journal) -> Option<Value> {
        server(
            journal,
            json!({
                "type": "session.updated", "event_id": "setup",
                "session": configuration().session
            }),
        )
        .unwrap()
    }

    fn delta(item: &str, text: &str) -> Value {
        json!({"type": "conversation.item.input_audio_transcription.delta",
            "event_id": "delta", "item_id": item, "content_index": 0, "delta": text})
    }

    fn committed(item: &str) -> Value {
        json!({"type": "input_audio_buffer.committed", "event_id": "commit",
            "item_id": item, "previous_item_id": null})
    }

    fn completed(item: &str, text: &str) -> Value {
        json!({"type": "conversation.item.input_audio_transcription.completed",
            "event_id": "done", "item_id": item, "content_index": 0,
            "transcript": text, "usage": {"type": "duration", "seconds": 1.0}})
    }

    #[test]
    fn replay_preserves_unicode_ids_and_acknowledgements_but_allows_final_revisions() {
        let mut journal =
            Journal::new(&configuration(), &Arc::new(Budget::new(MAX_RETAINED_BYTES))).unwrap();
        assert!(!journal.can_recover());
        assert!(acknowledge(&mut journal).is_some());
        record(
            &mut journal,
            json!({"type": "input_audio_buffer.append", "audio": "AAAA"}),
        );
        assert!(
            server(&mut journal, delta("original", "hé🙂"))
                .unwrap()
                .is_some()
        );
        record(&mut journal, json!({"type": "input_audio_buffer.commit"}));
        assert!(
            server(&mut journal, committed("original"))
                .unwrap()
                .is_some()
        );

        let bytes = journal.reservation.bytes;
        for replacement in ["replacement", "another"] {
            let setup = serde_json::to_value(journal.start_replay().unwrap()).unwrap();
            assert!(setup["session"]["audio"]["input"]["turn_detection"].is_null());
            assert!(journal.next_replay().is_none());
            assert!(acknowledge(&mut journal).is_none());
            assert!(matches!(
                journal.next_replay(),
                Some(DynamoRealtimeClientEvent::OpenAI(
                    RealtimeClientEvent::InputAudioBufferAppend(_)
                ))
            ));
            assert!(matches!(
                journal.next_replay(),
                Some(DynamoRealtimeClientEvent::OpenAI(
                    RealtimeClientEvent::InputAudioBufferCommit(_)
                ))
            ));
            assert!(journal.next_replay().is_none());
            assert!(
                server(&mut journal, delta(replacement, "h"))
                    .unwrap()
                    .is_none()
            );
            assert!(
                server(&mut journal, delta(replacement, "é🙂"))
                    .unwrap()
                    .is_none()
            );
            assert!(
                server(&mut journal, committed(replacement))
                    .unwrap()
                    .is_none()
            );
        }
        assert!(
            journal.reservation.bytes <= bytes + 16,
            "replacing worker IDs must release old storage"
        );
        let event = server(&mut journal, delta("another", " next"))
            .unwrap()
            .unwrap();
        assert_eq!(event["item_id"], "original");
        assert_eq!(event["delta"], " next");
        let event = server(&mut journal, completed("another", "A corrected final."))
            .unwrap()
            .unwrap();
        assert_eq!(event["item_id"], "original");
        assert_eq!(event["transcript"], "A corrected final.");
        assert!(journal.turns.is_empty());
        assert_eq!(journal.events, 1);
        assert_eq!(
            journal.reservation.bytes,
            retained_size(&configuration()) + "another".len() + "original".len()
        );
    }

    #[test]
    fn queued_turns_clear_and_idle_replay_retain_only_unfinished_input() {
        let mut journal =
            Journal::new(&configuration(), &Arc::new(Budget::new(MAX_RETAINED_BYTES))).unwrap();
        acknowledge(&mut journal);
        record(
            &mut journal,
            json!({"type": "input_audio_buffer.append", "audio": "first"}),
        );
        record(&mut journal, json!({"type": "input_audio_buffer.commit"}));
        record(
            &mut journal,
            json!({"type": "input_audio_buffer.append", "audio": "discarded"}),
        );
        record(&mut journal, json!({"type": "input_audio_buffer.clear"}));
        record(
            &mut journal,
            json!({"type": "input_audio_buffer.append", "audio": "open"}),
        );
        journal.start_replay().unwrap();
        acknowledge(&mut journal);
        let mut replay = Vec::new();
        while let Some(event) = journal.next_replay() {
            replay.push(serde_json::to_value(event).unwrap());
        }
        assert_eq!(replay.len(), 4);
        assert_eq!(replay[0]["audio"], "first");
        assert_eq!(replay[1]["type"], "input_audio_buffer.commit");
        assert_eq!(replay[2]["type"], "input_audio_buffer.clear");
        assert_eq!(replay[3]["audio"], "open");
        server(&mut journal, committed("first")).unwrap();
        server(&mut journal, completed("first", "first")).unwrap();
        assert!(
            server(&mut journal, delta("cleared", "stale"))
                .unwrap()
                .is_none()
        );
        server(
            &mut journal,
            json!({"type": "input_audio_buffer.cleared", "event_id": "clear"}),
        )
        .unwrap();
        assert_eq!(
            server(&mut journal, delta("open", "new")).unwrap().unwrap()["item_id"],
            "open"
        );
        record(&mut journal, json!({"type": "input_audio_buffer.commit"}));
        server(&mut journal, committed("open")).unwrap();
        server(&mut journal, completed("open", "new")).unwrap();
        journal.start_replay().unwrap();
        acknowledge(&mut journal);
        assert!(
            journal.next_replay().is_none(),
            "idle recovery must not replay history"
        );
        assert_eq!(journal.events, 1);
    }

    #[test]
    fn divergence_fails_and_retention_or_application_errors_disable_recovery() {
        let mut journal =
            Journal::new(&configuration(), &Arc::new(Budget::new(MAX_RETAINED_BYTES))).unwrap();
        acknowledge(&mut journal);
        record(
            &mut journal,
            json!({"type": "input_audio_buffer.append", "audio": "AAAA"}),
        );
        server(&mut journal, delta("old", "hé🙂")).unwrap();
        journal.start_replay().unwrap();
        acknowledge(&mut journal);
        assert!(server(&mut journal, delta("new", "hé🙃")).is_err());

        for event_limit in [true, false] {
            let mut journal =
                Journal::new(&configuration(), &Arc::new(Budget::new(MAX_RETAINED_BYTES))).unwrap();
            acknowledge(&mut journal);
            if event_limit {
                journal.max_events = 1;
            } else {
                journal.max_bytes = journal.reservation.bytes + 1;
            }
            record(
                &mut journal,
                json!({"type": "input_audio_buffer.append", "audio": "AAAA"}),
            );
            assert!(!journal.can_recover());
            assert_eq!(journal.events, 0);
            assert!(journal.turns.iter().all(|t| t.events.is_empty()));
        }
        let mut journal =
            Journal::new(&configuration(), &Arc::new(Budget::new(MAX_RETAINED_BYTES))).unwrap();
        acknowledge(&mut journal);
        assert!(
            server(
                &mut journal,
                json!({"type": "error", "event_id": "invalid",
            "error": {"type": "invalid_request_error", "message": "invalid audio"}})
            )
            .unwrap()
            .is_some()
        );
        assert!(!journal.can_recover());
    }

    #[test]
    fn disabling_recovery_preserves_inflight_output_and_early_finals() {
        let mut journal =
            Journal::new(&configuration(), &Arc::new(Budget::new(MAX_RETAINED_BYTES))).unwrap();
        acknowledge(&mut journal);
        journal.max_events = 2;
        record(
            &mut journal,
            json!({"type": "input_audio_buffer.append", "audio": "AAAA"}),
        );
        server(&mut journal, delta("old", "hé")).unwrap();
        journal.start_replay().unwrap();
        acknowledge(&mut journal);
        let event = server(&mut journal, delta("new", "héllo"))
            .unwrap()
            .unwrap();
        assert_eq!(event["delta"], "llo");
        assert_eq!(event["item_id"], "old");
        record(&mut journal, json!({"type": "input_audio_buffer.commit"}));
        assert!(
            !journal.can_recover(),
            "commit exceeds the injected event limit"
        );
        server(
            &mut journal,
            json!({"type": "error", "event_id": "invalid", "error": {
                "type": "invalid_request_error", "message": "unsupported event"
            }}),
        )
        .unwrap();
        assert_eq!(
            server(&mut journal, committed("new")).unwrap().unwrap()["item_id"],
            "old"
        );
        assert_eq!(
            server(&mut journal, completed("new", "hello"))
                .unwrap()
                .unwrap()["item_id"],
            "old"
        );
        let mut ack = committed("next");
        ack["previous_item_id"] = json!("new");
        assert_eq!(
            server(&mut journal, ack).unwrap().unwrap()["previous_item_id"],
            "old"
        );

        let mut journal =
            Journal::new(&configuration(), &Arc::new(Budget::new(MAX_RETAINED_BYTES))).unwrap();
        acknowledge(&mut journal);
        record(
            &mut journal,
            json!({"type": "input_audio_buffer.append", "audio": "AAAA"}),
        );
        assert!(
            server(&mut journal, completed("early", "finished before commit"))
                .unwrap()
                .is_some()
        );
        assert!(!journal.can_recover());
        assert!(
            server(&mut journal, completed("early", "duplicate"))
                .unwrap()
                .is_none()
        );
        record(&mut journal, json!({"type": "input_audio_buffer.commit"}));
        assert!(server(&mut journal, committed("early")).unwrap().is_some());
        assert!(journal.turns.is_empty());
    }

    #[test]
    fn application_error_allows_a_new_turn_after_the_worker_cancels_an_open_turn() {
        for stage in ["original", "replacement-unbound", "replacement-bound"] {
            let mut journal =
                Journal::new(&configuration(), &Arc::new(Budget::new(MAX_RETAINED_BYTES))).unwrap();
            acknowledge(&mut journal);
            record(
                &mut journal,
                json!({"type": "input_audio_buffer.append", "audio": "AAAA"}),
            );
            server(&mut journal, delta("original", "partial")).unwrap();
            if stage != "original" {
                journal.start_replay().unwrap();
                acknowledge(&mut journal);
                while journal.next_replay().is_some() {}
            }
            if stage == "replacement-bound" {
                assert!(
                    server(&mut journal, delta("replacement", "partial"))
                        .unwrap()
                        .is_none()
                );
            }
            server(
                &mut journal,
                json!({"type": "error", "event_id": "invalid", "error": {
                    "type": "invalid_request_error", "code": "invalid_audio",
                    "message": "input audio exceeds 60 seconds"
                }}),
            )
            .unwrap();
            assert!(!journal.can_recover());
            record(
                &mut journal,
                json!({"type": "input_audio_buffer.append", "audio": "AAAA"}),
            );
            record(&mut journal, json!({"type": "input_audio_buffer.commit"}));
            for value in [
                committed("fresh"),
                delta("fresh", "next"),
                completed("fresh", "next"),
            ] {
                let mut event: RealtimeServerEvent = serde_json::from_value(value).unwrap();
                let expected = serde_json::to_value(&event).unwrap();
                assert!(journal.reconcile(&mut event).unwrap());
                assert_eq!(serde_json::to_value(event).unwrap(), expected);
            }
            assert!(journal.turns.is_empty());
        }
    }

    #[test]
    fn shared_budget_exhaustion_preserves_live_output_and_releases_retention() {
        let budget = Arc::new(Budget::new(64 * 1024));
        let mut first = Journal::new(&configuration(), &budget).unwrap();
        let mut second = Journal::new(&configuration(), &budget).unwrap();
        acknowledge(&mut first);
        acknowledge(&mut second);
        record(
            &mut first,
            json!({"type": "input_audio_buffer.append", "audio": "A".repeat(32 * 1024)}),
        );
        server(&mut first, delta("first", "hello")).unwrap();
        let first_bytes = first.reservation.bytes;
        record(
            &mut second,
            json!({"type": "input_audio_buffer.append", "audio": "B".repeat(32 * 1024)}),
        );
        assert!(!second.can_recover());
        assert!(first.can_recover());
        assert_eq!(budget.used(), first_bytes);
        assert!(
            server(&mut second, completed("second", "still healthy"))
                .unwrap()
                .is_some()
        );
        record(&mut first, json!({"type": "input_audio_buffer.commit"}));
        server(&mut first, completed("first", "hello")).unwrap();
        assert_eq!(
            budget.used(),
            retained_size(&configuration()) + 2 * "first".len()
        );
        record(
            &mut first,
            json!({"type": "input_audio_buffer.append", "audio": "A".repeat(32 * 1024)}),
        );
        record(&mut first, json!({"type": "input_audio_buffer.clear"}));
        server(
            &mut first,
            json!({"type": "input_audio_buffer.cleared", "event_id": "clear"}),
        )
        .unwrap();
        assert_eq!(
            budget.used(),
            retained_size(&configuration()) + 2 * "first".len()
        );
        drop(first);
        drop(second);
        assert_eq!(budget.used(), 0);
        assert!(Journal::new(&configuration(), &Arc::new(Budget::new(0))).is_none());
    }

    #[test]
    fn disabled_replay_keeps_prefix_and_identity_reservations_until_reconciled() {
        let budget = Arc::new(Budget::new(64 * 1024));
        let mut journal = Journal::new(&configuration(), &budget).unwrap();
        acknowledge(&mut journal);
        record(
            &mut journal,
            json!({"type": "input_audio_buffer.append", "audio": "AAAA"}),
        );
        server(&mut journal, delta("original", "hello")).unwrap();
        journal.start_replay().unwrap();
        acknowledge(&mut journal);
        assert!(
            server(&mut journal, delta("replacement", "he"))
                .unwrap()
                .is_none()
        );
        journal.disable("test pressure");
        let turn = journal.turns.front().unwrap();
        assert_eq!(
            budget.used(),
            journal.turns.capacity() * size_of::<Turn>()
                + turn.prefix.capacity()
                + "original".len()
                + "replacement".len()
        );
        let before = budget.used();
        let event = server(&mut journal, delta("replacement", "llo world"))
            .unwrap()
            .unwrap();
        assert_eq!(event["delta"], " world");
        assert_eq!(event["item_id"], "original");
        assert!(budget.used() < before);
        record(&mut journal, json!({"type": "input_audio_buffer.commit"}));
        server(&mut journal, completed("replacement", "hello world")).unwrap();
        assert_eq!(budget.used(), "original".len() + "replacement".len());
        drop(journal);
        assert_eq!(budget.used(), 0);
    }

    #[test]
    fn clear_keeps_suppressing_stale_output_when_retention_is_exhausted() {
        let budget = Arc::new(Budget::new(64 * 1024));
        let mut journal = Journal::new(&configuration(), &budget).unwrap();
        acknowledge(&mut journal);
        record(
            &mut journal,
            json!({"type": "input_audio_buffer.append", "audio": "AAAA"}),
        );
        server(&mut journal, delta("old", "partial")).unwrap();
        journal.max_events = 1;
        record(&mut journal, json!({"type": "input_audio_buffer.clear"}));
        assert!(!journal.can_recover());
        assert!(
            server(&mut journal, delta("old", "stale"))
                .unwrap()
                .is_none()
        );
        server(
            &mut journal,
            json!({"type": "input_audio_buffer.cleared", "event_id": "clear"}),
        )
        .unwrap();
        assert_eq!(budget.used(), 0);
        assert!(
            server(&mut journal, completed("fresh", "new turn"))
                .unwrap()
                .is_some()
        );
    }

    #[test]
    fn disabling_recovery_keeps_unbound_turns_ahead_of_clear_markers() {
        for reason in ["session update", "application error", "identity budget"] {
            let budget = Arc::new(Budget::new(64 * 1024));
            let mut journal = Journal::new(&configuration(), &budget).unwrap();
            acknowledge(&mut journal);
            record(
                &mut journal,
                json!({"type": "input_audio_buffer.append", "audio": "first"}),
            );
            record(&mut journal, json!({"type": "input_audio_buffer.commit"}));
            record(
                &mut journal,
                json!({"type": "input_audio_buffer.append", "audio": "discarded"}),
            );
            record(&mut journal, json!({"type": "input_audio_buffer.clear"}));
            match reason {
                "session update" => {
                    journal.record(&RealtimeClientEvent::SessionUpdate(configuration()).into())
                }
                "application error" => {
                    server(
                        &mut journal,
                        json!({"type": "error", "event_id": "invalid", "error": {
                            "type": "invalid_request_error", "message": "unsupported event"
                        }}),
                    )
                    .unwrap();
                }
                _ => journal.max_bytes = journal.reservation.bytes,
            }
            if reason != "identity budget" {
                assert!(server(&mut journal, committed("first")).unwrap().is_some());
                assert!(
                    server(&mut journal, delta("first", "hello"))
                        .unwrap()
                        .is_some()
                );
            }
            // Also cover a terminal event that itself exhausts the identity budget.
            assert!(
                server(&mut journal, completed("first", "hello"))
                    .unwrap()
                    .is_some()
            );
            assert!(!journal.can_recover());
            assert_eq!(budget.used(), journal.turns.capacity() * size_of::<Turn>());
            assert!(
                server(&mut journal, delta("discarded", "stale"))
                    .unwrap()
                    .is_none()
            );
            assert!(
                server(
                    &mut journal,
                    json!({"type": "input_audio_buffer.cleared", "event_id": "clear"}),
                )
                .unwrap()
                .is_some()
            );
            assert_eq!(budget.used(), 0);
            assert!(
                server(&mut journal, completed("fresh", "next turn"))
                    .unwrap()
                    .is_some()
            );
        }
    }
}
