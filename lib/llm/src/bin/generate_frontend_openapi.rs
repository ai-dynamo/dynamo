// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Helper binary to generate the Dynamo HTTP frontend OpenAPI specification.
//!
//! This allows CI and documentation tooling to obtain the frontend API schemas
//! without starting the HTTP service and scraping the endpoint. File mode also
//! lists `/docs` and `/openapi.json`: those documentation routes are registered
//! after the served document's route list is captured, so the documents differ there.
//!
//! Usage (from the repository root):
//! ```bash
//! cargo run -p dynamo-llm --bin generate-frontend-openapi
//! ```
//! The generated spec will be written to:
//!   `docs/frontends/openapi.json`
//!
//! Alternatively, `--serve 0.0.0.0:8000` starts the actual HttpService router
//! with in-memory discovery and no workers. Query `/openapi.json` over HTTP.
//! This mode needs neither model weights nor a GPU and does not test inference.
//!
//! This is a native, partially resolved document: `x-dynamo-schema-import` marks
//! dependency schema slots that require version-pinned offline composition. Those
//! slots do not establish the imported fields' validation constraints.
//! OpenAPI 3.2 `text/event-stream.itemSchema` describes parsed SSE data events:
//! chunks, errors, `[DONE]`, and optional named chat dispatch events. JSON payloads
//! use string `data` with `contentSchema`, which consumers must explicitly validate.
//! Comments/keep-alives are ignored by SSE parsing; event ordering, termination,
//! and disconnect behavior require separate lifecycle checks.
//! Unlike a configured service, this full-surface export includes both optional
//! named dispatch event variants; HTTP only advertises enabled dispatches.

use std::fs;
use std::net::SocketAddr;
use std::path::PathBuf;
use std::thread;

use anyhow::Context as _;

use dynamo_llm::http::service::{openapi_docs, service_v2::HttpService};

/// Stack size for the generator thread (8 MB).
/// The utoipa schema derivation for deeply nested OpenAI types requires
/// additional stack space due to recursive type expansion.
const GENERATOR_STACK_SIZE: usize = 8 * 1024 * 1024;

/// Run [`generate_openapi`] in file mode or the CLI-selected `--serve IP:PORT` mode.
///
/// Returns its result, or an error for invalid arguments, thread creation failure,
/// or a generator-thread panic.
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args().skip(1).collect();
    let listen = match args.as_slice() {
        [] => None,
        [flag, address] if flag == "--serve" => Some(address.parse::<SocketAddr>()?),
        _ => anyhow::bail!("usage: generate-frontend-openapi [--serve IP:PORT]"),
    };
    // Spawn a thread with a larger stack to handle deeply nested schema generation
    let handle = thread::Builder::new()
        .stack_size(GENERATOR_STACK_SIZE)
        .spawn(move || generate_openapi(listen))
        .context("failed to spawn generator thread")?;

    handle
        .join()
        .map_err(|e| anyhow::anyhow!("generator thread panicked: {:?}", e))?
}

/// Write the native schema to disk or serve it through a workerless frontend.
///
/// With `listen = None`, write `docs/frontends/openapi.json` under the current
/// directory, creating parents and overwriting the file non-atomically.
/// With `Some(address)`, bind there until server termination; Ctrl-C requests
/// shutdown. Serving mode writes no file and has no inference workers.
/// Returns `Ok(())` after the file is written or the server stops successfully.
///
/// # Errors
///
/// Propagates service-construction, schema-invariant, serialization, I/O, runtime,
/// signal, and server errors. A failed write can leave a truncated file or new directories.
fn generate_openapi(listen: Option<SocketAddr>) -> anyhow::Result<()> {
    // Build an HttpService instance with all standard OpenAI-compatible
    // frontend endpoints enabled so that the generated OpenAPI document
    // reflects the full surface area exposed to users.
    //
    // This does NOT start any network listeners; it only builds the router
    // graph and associated route documentation.
    let builder = HttpService::builder()
        .enable_chat_endpoints(true)
        .enable_cmpl_endpoints(true)
        .enable_embeddings_endpoints(true)
        .enable_responses_endpoints(true)
        .enable_anthropic_endpoints(true)
        .enable_batch_endpoints(true);
    let builder = if let Some(address) = listen {
        builder.host(address.ip().to_string()).port(address.port())
    } else {
        builder
    };
    let http_service = builder
        .build()
        .context("failed to build HttpService for OpenAPI generation")?;

    if let Some(address) = listen {
        println!("Serving frontend OpenAPI at http://{address}/openapi.json (no workers)");
        return tokio::runtime::Builder::new_multi_thread()
            .enable_all()
            .thread_stack_size(GENERATOR_STACK_SIZE)
            .build()?
            .block_on(async {
                let cancellation = tokio_util::sync::CancellationToken::new();
                let server = http_service.run(cancellation.clone());
                tokio::pin!(server);
                tokio::select! {
                    result = &mut server => result,
                    signal = tokio::signal::ctrl_c() => {
                        signal?;
                        cancellation.cancel();
                        server.await
                    }
                }
            });
    }

    let route_docs = http_service.route_docs().to_vec();
    let openapi = openapi_docs::generate_openapi_spec_with_options(
        &route_docs,
        openapi_docs::OpenApiOptions {
            streaming_tool_dispatch: true,
            streaming_reasoning_dispatch: true,
            ..Default::default()
        },
    )
    .map_err(anyhow::Error::msg)
    .context("failed to generate OpenAPI document")?;

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
