// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

pub use crate::pipeline::network::tcp::{client, server};

/// Opaque TCP request path for an endpoint instance, shared by discovery and ingress.
/// Escape separators and literal percent signs so field boundaries remain unambiguous.
pub(crate) fn instance_path(endpoint: &crate::protocols::EndpointId, instance_id: u64) -> String {
    let escape = |field: &str| field.replace('%', "%25").replace('/', "%2F");
    format!(
        "{instance_id:x}/{}/{}/{}",
        escape(&endpoint.namespace),
        escape(&endpoint.component),
        escape(&endpoint.name),
    )
}
