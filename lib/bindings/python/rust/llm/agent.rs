// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use std::sync::Arc;

use dynamo_agents::trace::DEFAULT_TOOL_EVENTS_TOPIC;
use dynamo_runtime::transports::event_plane::EventPublisher;
use pythonize::depythonize;
use serde_json::Value;

use super::*;
use crate::DistributedRuntime;

#[pyclass]
pub(crate) struct AgentTraceEventPublisher {
    inner: Arc<EventPublisher>,
}

#[pymethods]
impl AgentTraceEventPublisher {
    /// Create a namespace-scoped publisher for normalized agent trace records.
    ///
    /// The record payload is serialized through Dynamo's event-plane codec; callers
    /// pass the JSON-shaped dict documented in docs/agents/agent-context.md.
    #[new]
    #[pyo3(signature = (runtime, namespace, topic=None))]
    fn new(
        runtime: DistributedRuntime,
        namespace: String,
        topic: Option<String>,
    ) -> PyResult<Self> {
        let namespace = runtime.inner.namespace(namespace).map_err(to_pyerr)?;
        let topic = topic.unwrap_or_else(|| DEFAULT_TOOL_EVENTS_TOPIC.to_string());
        let publisher = runtime
            .inner
            .runtime()
            .secondary()
            .block_on(EventPublisher::for_namespace(&namespace, topic))
            .map_err(to_pyerr)?;

        Ok(Self {
            inner: Arc::new(publisher),
        })
    }

    fn publish<'p>(
        &self,
        py: Python<'p>,
        record: Bound<'p, PyAny>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let record: Value = depythonize(&record).map_err(to_pyerr)?;
        let publisher = self.inner.clone();

        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            publisher.publish(&record).await.map_err(to_pyerr)?;
            Ok(())
        })
    }
}
