// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// pub mod leader;

mod onboard;

pub use onboard::OnboardingSession;

pub type SessionId = uuid::Uuid;
