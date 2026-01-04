// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::time::Duration;

use anyhow::Result;
use crossterm::event::{Event, KeyCode, KeyEvent, KeyModifiers};
use tokio::sync::mpsc;
use tokio_util::sync::CancellationToken;

#[derive(Debug, Clone, Copy)]
pub enum InputEvent {
    Quit,
    MoveUp,
    MoveDown,
    MoveLeft,
    MoveRight,
    Refresh,
}

pub fn spawn_input_listener(
    tx: mpsc::Sender<InputEvent>,
    cancel: CancellationToken,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let tx = tx;
        loop {
            if cancel.is_cancelled() {
                break;
            }

            let maybe_event = match tokio::task::spawn_blocking(|| poll_crossterm_event()).await {
                Ok(Ok(event)) => event,
                Ok(Err(err)) => {
                    tracing::warn!(error = %err, "input listener failed to poll crossterm event");
                    continue;
                }
                Err(err) => {
                    tracing::warn!(error = %err, "input listener task join error");
                    continue;
                }
            };

            if let Some(event) = maybe_event {
                if let Some(input) = map_event(event) {
                    if tx.send(input).await.is_err() {
                        break;
                    }
                }
            }
        }
    })
}

fn poll_crossterm_event() -> Result<Option<Event>> {
    if crossterm::event::poll(Duration::from_millis(100))? {
        Ok(Some(crossterm::event::read()?))
    } else {
        Ok(None)
    }
}

fn map_event(event: Event) -> Option<InputEvent> {
    match event {
        Event::Key(KeyEvent {
            code,
            modifiers,
            kind: _,
            state: _,
        }) => match code {
            KeyCode::Esc => Some(InputEvent::Quit),
            KeyCode::Char('c') if modifiers.contains(KeyModifiers::CONTROL) => {
                Some(InputEvent::Quit)
            }
            KeyCode::Char('q') | KeyCode::Char('Q') => Some(InputEvent::Quit),
            KeyCode::Char('r') | KeyCode::Char('R') => Some(InputEvent::Refresh),
            KeyCode::Up | KeyCode::Char('k') => Some(InputEvent::MoveUp),
            KeyCode::Down | KeyCode::Char('j') => Some(InputEvent::MoveDown),
            KeyCode::Left | KeyCode::Char('h') => Some(InputEvent::MoveLeft),
            KeyCode::Right | KeyCode::Char('l') => Some(InputEvent::MoveRight),
            _ => None,
        },
        _ => None,
    }
}
