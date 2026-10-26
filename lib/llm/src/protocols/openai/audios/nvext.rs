// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use serde::{Deserialize, Serialize};
use utoipa::ToSchema;

/// NVIDIA extensions to the Audio Speech API
#[derive(ToSchema, Serialize, Deserialize, Default, Debug, Clone)]
pub struct NvExt {
    /// Annotations for SSE stream events
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub annotations: Option<Vec<String>>,

    /// Internal frontend-to-worker compatibility signal.
    ///
    /// New frontends set this before forwarding `/v1/audio/speech`. When absent
    /// or false, workers must return one aggregated response so older frontends
    /// do not decode only the first chunk during rolling upgrades.
    ///
    /// TODO(v1.7): Remove after v1.4 leaves the N-2 compatibility window.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub frontend_accepts_audio_chunks: Option<bool>,

    /// Classifier-free guidance scale (Audex only, hence an extension rather
    /// than a top-level OpenAI field). Unset or 1.0 decodes unguided; higher
    /// values follow the prompt more closely. Declared here because serde drops
    /// unknown `nvext` keys, so without the field the client's value never
    /// reaches the worker and guidance is silently never applied.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cfg_scale: Option<f64>,
}
