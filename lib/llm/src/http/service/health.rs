// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use super::{DeploymentState, RouteDoc};
use axum::{http::Method, http::StatusCode, response::IntoResponse, routing::get, Router};
use std::sync::Arc;

pub fn health_check_router(
    state: Arc<DeploymentState>,
    path_override: Option<String>,
) -> (Vec<RouteDoc>, Router) {
    let path = path_override.unwrap_or_else(|| "/health".to_string());
    let doc = RouteDoc::new(Method::GET, &path);

    let router = Router::new()
        .route(&path, get(health_handler))
        .with_state(state);

    (vec![doc], router)
}

async fn health_handler() -> impl IntoResponse {
    return (StatusCode::OK, "OK");
}
