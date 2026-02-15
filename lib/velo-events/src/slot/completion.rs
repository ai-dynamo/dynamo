// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use super::waiter::EventAwaiter;
use crate::status::EventPoison;

pub(crate) type PoisonArc = Arc<EventPoison>;

#[derive(Clone, Debug)]
pub(crate) enum CompletionKind {
    Triggered,
    Poisoned(PoisonArc),
}

impl CompletionKind {
    pub(crate) fn as_result(&self) -> Result<(), EventPoison> {
        match self {
            Self::Triggered => Ok(()),
            Self::Poisoned(poison) => Err((**poison).clone()),
        }
    }
}

pub(crate) enum WaitRegistration {
    Ready,
    Pending(EventAwaiter),
    Poisoned(PoisonArc),
}
