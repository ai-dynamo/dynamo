// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Helper binary to generate the Dynamo HTTP frontend OpenAPI specification.
//!
//! This allows CI and documentation tooling to obtain the exact same
//! OpenAPI document that is served at `/openapi.json` by the frontend
//! without having to start the HTTP service and scrape the endpoint.
//!
//! Usage (from the repository root):
//! ```bash
//! cargo run -p dynamo-llm --bin generate-frontend-openapi
//! ```
//! The generated spec will be written to:
//!   `docs/frontends/openapi.json`
//!
//! This is a native, partially resolved document: `x-dynamo-schema-import` marks
//! dependency schema slots that require version-pinned offline composition. Those
//! slots do not establish the imported fields' validation constraints.
//! On a `text/event-stream` media type, `x-dynamo-sse-data-schema: true` means its
//! `schema` describes each successful JSON data payload, not the entire SSE body,
//! error/annotation events, or the literal `[DONE]` terminator. Consumers must
//! explicitly support this convention; ordinary JSON Schema validation does not
//! validate framing, event ordering, or termination.

use std::fs;
use std::path::PathBuf;
use std::thread;

use anyhow::Context as _;

use dynamo_llm::http::service::{openapi_docs, service_v2::HttpService};

/// Stack size for the generator thread (8 MB).
/// The utoipa schema derivation for deeply nested OpenAI types requires
/// additional stack space due to recursive type expansion.
const GENERATOR_STACK_SIZE: usize = 8 * 1024 * 1024;

/// Run [`generate_openapi`] on the larger-stack thread required for schema generation.
///
/// Returns its result, or an error if thread creation fails or the thread panics.
fn main() -> anyhow::Result<()> {
    // Spawn a thread with a larger stack to handle deeply nested schema generation
    let handle = thread::Builder::new()
        .stack_size(GENERATOR_STACK_SIZE)
        .spawn(generate_openapi)
        .context("failed to spawn generator thread")?;

    handle
        .join()
        .map_err(|e| anyhow::anyhow!("generator thread panicked: {:?}", e))?
}

/// Export the native document to disk without starting a listener or loading a model.
///
/// Uses the current directory as the output root; run from the repository root to
/// update its `docs/frontends/openapi.json`. Creates parent directories and overwrites
/// that file non-atomically. Returns `Ok(())` after writing it and printing its path.
///
/// # Errors
///
/// Propagates service-construction, serialization, and I/O errors. A failed write may
/// leave a truncated file; created directories are not removed on failure.
///
/// # Panics
///
/// Panics if compiled schema invariants are violated during document generation.
fn generate_openapi() -> anyhow::Result<()> {
    // Build an HttpService instance with all standard OpenAI-compatible
    // frontend endpoints enabled so that the generated OpenAPI document
    // reflects the full surface area exposed to users.
    //
    // This does NOT start any network listeners; it only builds the router
    // graph and associated route documentation.
    let http_service = HttpService::builder()
        .enable_chat_endpoints(true)
        .enable_cmpl_endpoints(true)
        .enable_embeddings_endpoints(true)
        .enable_responses_endpoints(true)
        .enable_anthropic_endpoints(true)
        .enable_batch_endpoints(true)
        .build()
        .context("failed to build HttpService for OpenAPI generation")?;

    let route_docs = http_service.route_docs().to_vec();
    let openapi = openapi_docs::generate_openapi_spec(&route_docs);

    // Write the spec to a stable location relative to the repository root.
    let out_dir = PathBuf::from("docs/frontends");
    let out_path = out_dir.join("openapi.json");

    fs::create_dir_all(&out_dir)
        .with_context(|| format!("failed to create OpenAPI output directory: {out_dir:?}"))?;

    let json =
        serde_json::to_string_pretty(&openapi).context("failed to serialize OpenAPI spec")?;

    fs::write(&out_path, json)
        .with_context(|| format!("failed to write OpenAPI spec to: {out_path:?}"))?;

    println!(
        "Generated Dynamo frontend OpenAPI specification at {}",
        out_path.display()
    );

    Ok(())
}
