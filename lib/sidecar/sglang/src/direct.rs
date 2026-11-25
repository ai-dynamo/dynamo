// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! SGLang engines a frontend dials directly, with no sidecar beside them.

use std::sync::Arc;

use dynamo_backend_common::LLMEngine;
use dynamo_backend_common::direct::{DirectEngineFactory, DirectEngineOptions, DirectEngineSpec};
use futures::FutureExt;

use crate::engine::SglangSidecarEngine;

const BINARY: &str = "dynamo-sglang-sidecar";

/// Build each listed engine with the SGLang sidecar's own constructor, which
/// asks the engine for its role, model, parsers, listeners and KV publishers.
///
/// A prefill engine's bootstrap host is the host the frontend dials it at: the
/// sidecar would otherwise fall back to its own address, which in direct mode
/// is the frontend's.
pub fn direct_engine_factory(options: DirectEngineOptions) -> DirectEngineFactory {
    Arc::new(move |spec: &DirectEngineSpec| {
        let options = options.clone();
        let mut argv = options.sidecar_args(BINARY, spec);
        if let Some(host) = engine_host(&spec.url) {
            argv.extend(["--bootstrap-host".to_string(), host]);
        }
        async move {
            let (_, discovered) = SglangSidecarEngine::try_from_args_async(argv)
                .map_err(|error| error.into_dynamo())?;
            let (engine, mut config) = discovered.await?;
            options.apply(&mut config);
            Ok((Arc::new(engine) as Arc<dyn LLMEngine>, config))
        }
        .boxed()
    })
}

/// The host part of an engine address, given with or without a scheme.
fn engine_host(url: &str) -> Option<String> {
    let url = url.trim();
    let with_scheme = if url.contains("://") {
        url.to_string()
    } else {
        format!("http://{url}")
    };
    let host = url::Url::parse(&with_scheme).ok()?.host_str()?.to_string();
    Some(
        host.trim_start_matches('[')
            .trim_end_matches(']')
            .to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_engine_host_is_its_address_without_scheme_or_port() {
        assert_eq!(engine_host("10.0.0.7:30001").as_deref(), Some("10.0.0.7"));
        assert_eq!(
            engine_host("http://engine-0:30001/").as_deref(),
            Some("engine-0")
        );
        assert_eq!(engine_host("[fd00::7]:30001").as_deref(), Some("fd00::7"));
    }
}
