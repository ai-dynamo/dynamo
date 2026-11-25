// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Carries a request's metrics handle on its pipeline context: the HTTP and gRPC services attach
//! it when they admit the request, and the preprocessor reads it back to count prompt tokens.

use std::sync::Arc;

use dynamo_http_server::metrics::RequestMetrics;
use dynamo_runtime::{engine::Data, pipeline::Context};

const REQUEST_METRICS_KEY: &str = "frontend.request_metrics";

pub fn attach<T: Data>(metrics: &Arc<RequestMetrics>, context: &mut Context<T>) {
    context.insert_arc(REQUEST_METRICS_KEY, metrics.clone());
}

pub fn from_context<T: Data>(context: &Context<T>) -> Option<Arc<RequestMetrics>> {
    context
        .get_optional(REQUEST_METRICS_KEY)
        .expect("request metrics have a fixed registry type")
}

#[cfg(test)]
mod tests {
    use super::*;
    use dynamo_http_server::metrics::{Endpoint, Metrics};

    #[test]
    fn the_attached_handle_is_read_back() {
        let metrics = Arc::new(Metrics::new());
        let guard = metrics.create_inflight_guard("model", Endpoint::Completions, true, "");
        let mut context = Context::new(());
        assert!(from_context(&context).is_none());
        attach(guard.request_metrics(), &mut context);
        assert!(Arc::ptr_eq(
            guard.request_metrics(),
            &from_context(&context).unwrap()
        ));
    }
}
