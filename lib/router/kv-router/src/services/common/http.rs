// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! JSON response envelopes shared by the standalone HTTP services.

use std::fmt;

use axum::Json;
use axum::extract::rejection::JsonRejection;
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};

pub(crate) fn json_ok(status: StatusCode) -> Response {
    (status, Json(serde_json::json!({"status": "ok"}))).into_response()
}

pub(crate) fn json_error(status: StatusCode, error: impl fmt::Display) -> Response {
    (
        status,
        Json(serde_json::json!({"error": error.to_string()})),
    )
        .into_response()
}

pub(crate) fn json_rejection(error: JsonRejection) -> Response {
    json_error(error.status(), error.body_text())
}
