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

use super::*;

#[pyclass(subclass)]
pub(crate) struct NatsQueue {
    inner: Arc<Mutex<crate::rs::transports::nats::NatsQueue>>,
}

#[pymethods]
impl NatsQueue {
    #[new]
    #[pyo3(signature = (stream_name, nats_server, dequeue_timeout))]
    fn new(stream_name: String, nats_server: String, dequeue_timeout: f64) -> PyResult<Self> {
        let inner = Arc::new(Mutex::new(crate::rs::transports::nats::NatsQueue::new(
            stream_name,
            nats_server,
            std::time::Duration::from_secs(dequeue_timeout as u64),
        )));
        Ok(Self { inner })
    }

    fn connect<'p>(&mut self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let queue = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            queue.lock().await.connect().await.map_err(to_pyerr)?;
            Ok(())
        })
    }

    fn ensure_connection<'p>(&mut self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let queue = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            queue
                .lock()
                .await
                .ensure_connection()
                .await
                .map_err(to_pyerr)?;
            Ok(())
        })
    }

    fn close<'p>(&mut self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let queue = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            queue.lock().await.close().await.map_err(to_pyerr)?;
            Ok(())
        })
    }

    fn enqueue_task<'p>(
        &mut self,
        py: Python<'p>,
        task_data: Py<PyBytes>,
    ) -> PyResult<Bound<'p, PyAny>> {
        let bytes = task_data.as_bytes(py).to_vec();

        let queue = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            queue
                .lock()
                .await
                .enqueue_task(bytes.into())
                .await
                .map_err(to_pyerr)?;
            Ok(())
        })
    }

    fn dequeue_task<'p>(&mut self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let queue = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            Ok(queue
                .lock()
                .await
                .dequeue_task()
                .await
                .map_err(to_pyerr)?
                .map(|bytes| bytes.to_vec()))
        })
    }

    fn get_queue_size<'p>(&mut self, py: Python<'p>) -> PyResult<Bound<'p, PyAny>> {
        let queue = self.inner.clone();
        pyo3_async_runtimes::tokio::future_into_py(py, async move {
            queue.lock().await.get_queue_size().await.map_err(to_pyerr)
        })
    }
}
