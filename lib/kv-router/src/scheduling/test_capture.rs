// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing_subscriber::{Layer, layer::Context, prelude::*, registry::LookupSpan};

#[derive(Clone, Default)]
pub(super) struct Fields(pub HashMap<String, String>);

impl tracing::field::Visit for Fields {
    fn record_debug(&mut self, field: &tracing::field::Field, value: &dyn std::fmt::Debug) {
        self.0.insert(field.name().into(), format!("{value:?}"));
    }

    fn record_str(&mut self, field: &tracing::field::Field, value: &str) {
        self.0.insert(field.name().into(), value.into());
    }
}

#[derive(Clone, Default)]
pub(super) struct Capture(Arc<Mutex<Vec<(&'static str, Fields)>>>);

impl Capture {
    pub fn install() -> Self {
        let capture = Self::default();
        tracing::subscriber::set_global_default(
            tracing_subscriber::registry().with(capture.clone()),
        )
        .unwrap();
        capture
    }

    pub fn fields(&self, name: &str, request: &str) -> HashMap<String, String> {
        let spans = self.0.lock().unwrap();
        let matches: Vec<_> = spans
            .iter()
            .filter(|(n, fields)| {
                *n == name
                    && fields
                        .0
                        .get("dynamo.request.id")
                        .is_some_and(|id| id == request)
            })
            .collect();
        assert_eq!(matches.len(), 1, "{name} for {request}");
        matches[0].1.0.clone()
    }
}

impl<S: tracing::Subscriber + for<'a> LookupSpan<'a>> Layer<S> for Capture {
    fn on_new_span(
        &self,
        attrs: &tracing::span::Attributes<'_>,
        id: &tracing::Id,
        ctx: Context<'_, S>,
    ) {
        let mut fields = Fields::default();
        attrs.record(&mut fields);
        ctx.span(id).unwrap().extensions_mut().insert(fields);
    }

    fn on_record(&self, id: &tracing::Id, values: &tracing::span::Record<'_>, ctx: Context<'_, S>) {
        let span = ctx.span(id).unwrap();
        values.record(span.extensions_mut().get_mut::<Fields>().unwrap());
    }

    fn on_close(&self, id: tracing::Id, ctx: Context<'_, S>) {
        let span = ctx.span(&id).unwrap();
        let fields = span.extensions_mut().remove::<Fields>().unwrap();
        self.0.lock().unwrap().push((span.name(), fields));
    }
}

// Lifecycle configuration is process-cached; never mutate it in the test runner.
pub(super) fn isolated(test: &str) -> bool {
    const CHILD: &str = "DYNAMO_ROUTER_TELEMETRY_TEST";
    if std::env::var(CHILD).as_deref() == Ok(test) {
        return false;
    }
    let status = std::process::Command::new(std::env::current_exe().unwrap())
        .args(["--exact", test, "--nocapture"])
        .env(CHILD, test)
        .env("DYN_LIFECYCLE_TRACE_ENABLED", "true")
        .env("DYN_LIFECYCLE_TRACE_MODE", "investigation")
        .status()
        .unwrap();
    assert!(status.success(), "{test}");
    true
}
