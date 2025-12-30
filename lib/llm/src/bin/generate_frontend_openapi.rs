// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Helper binary to generate the Dynamo HTTP frontend OpenAPI specification.
//!
//! This allows CI, documentation tooling, and NIM to obtain the exact same
//! OpenAPI document that is served at `/openapi.json` by the frontend
//! without having to start the HTTP service and scrape the endpoint.
//!
//! Usage (from the repository root):
//! ```bash
//! cargo run -p dynamo-llm --bin generate-frontend-openapi
//! ```
//! The generated spec will be written to:
//!   `docs/frontends/openapi.json`

use std::fs;
use std::path::PathBuf;
use std::thread;

use anyhow::Context as _;

use dynamo_llm::http::service::{openapi_docs, service_v2::HttpService};

/// Stack size for the generator thread (8 MB).
/// The utoipa schema derivation for deeply nested OpenAI types requires
/// additional stack space due to recursive type expansion.
const GENERATOR_STACK_SIZE: usize = 8 * 1024 * 1024;

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
