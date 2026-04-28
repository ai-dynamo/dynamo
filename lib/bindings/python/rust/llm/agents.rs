// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Python bindings for Dynamo-owned agent event ingress.

use pyo3::prelude::*;

use super::*;
use crate::Endpoint;
use crate::to_pyerr;

/// Relay that bridges local agent tool events from ZMQ to the Dynamo event plane.
#[pyclass]
pub(crate) struct AgentToolEventRelay {
    inner: llm_rs::agents::trace::AgentToolEventRelay,
}

#[pymethods]
impl AgentToolEventRelay {
    /// Create a relay for msgpack-encoded AgentTraceRecord events.
    ///
    /// Args:
    ///     endpoint: Dynamo component endpoint (provides runtime + discovery).
    ///     zmq_endpoint: Local ZMQ PUB address to subscribe to.
    ///     zmq_topic: Optional ZMQ topic filter. Defaults to all topics.
    ///     namespace: Optional Dynamo event-plane namespace override.
    ///     topic: Optional Dynamo event-plane topic override.
    #[new]
    #[pyo3(signature = (endpoint, zmq_endpoint, zmq_topic=None, namespace=None, topic=None))]
    fn new(
        endpoint: Endpoint,
        zmq_endpoint: String,
        zmq_topic: Option<String>,
        namespace: Option<String>,
        topic: Option<String>,
    ) -> PyResult<Self> {
        let component = endpoint.inner.component().clone();
        let inner = llm_rs::agents::trace::AgentToolEventRelay::new(
            component,
            zmq_endpoint,
            zmq_topic,
            namespace,
            topic,
        )
        .map_err(to_pyerr)?;
        Ok(Self { inner })
    }

    /// Shut down the relay task.
    fn shutdown(&self) {
        self.inner.shutdown();
    }
}
