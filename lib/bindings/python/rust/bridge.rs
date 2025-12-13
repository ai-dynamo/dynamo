use std::{
    sync::{OnceLock, mpsc},
    thread::{self},
};

use crossbeam_channel::{Sender, bounded};
use tokio::sync::oneshot;

use pyo3::{PyErr, prelude::*, types::PyAny};
use pythonize::{depythonize, pythonize};
use serde::{Serialize, de::DeserializeOwned};

type Job = Box<dyn FnOnce() + Send + 'static>;

pub struct Bridge {
    tx: Sender<Job>,
}

static PYTHON_BRIDGE: OnceLock<Bridge> = OnceLock::new();

impl Bridge {
    pub fn global() -> &'static Bridge {
        // TODO: in the ideal case
        // not performing as a global singleton (GIL is a global singleton)

        PYTHON_BRIDGE.get_or_init(|| {
            let (tx, rx) = bounded::<Job>(1024);

            const NUM_THREADS: usize = 1;
            for _ in 0..NUM_THREADS {
                let rx = rx.clone();

                thread::spawn(move || {
                    while let Ok(job) = rx.recv() {
                        job();
                    }
                });
            }

            Bridge { tx }
        })
    }

    pub async fn to_py<T>(&self, value: T) -> Result<Py<PyAny>, PyErr>
    where
        T: Serialize + Send + 'static,
    {
        let (resp_tx, resp_rx) = oneshot::channel();

        self.tx
            .send(Box::new(move || {
                Python::with_gil(|py| {
                    let res: Result<Py<PyAny>, String> = pythonize(py, &value)
                        .map(|obj| obj.into())
                        .map_err(|e| e.to_string());
                    let _ = resp_tx.send(res);
                });
            }))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        match resp_rx.await {
            Ok(Ok(obj)) => Ok(obj),
            Ok(Err(msg)) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(msg)),
            Err(_) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "bridge dropped",
            )),
        }
    }

    pub async fn from_py<T>(&self, obj: Py<PyAny>) -> Result<T, PyErr>
    where
        T: DeserializeOwned + Send + 'static,
    {
        let (resp_tx, resp_rx) = oneshot::channel();

        self.tx
            .send(Box::new(move || {
                Python::with_gil(|py| {
                    let bound = obj.bind(py);
                    let res: Result<T, String> = depythonize(bound).map_err(|e| e.to_string());
                    let _ = resp_tx.send(res);
                    // `obj` (and any temporaries) are dropped here while GIL is held.
                });
            }))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        match resp_rx.await {
            Ok(Ok(v)) => Ok(v),
            Ok(Err(msg)) => Err(PyErr::new::<pyo3::exceptions::PyValueError, _>(msg)),
            Err(_) => Err(PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(
                "bridge dropped",
            )),
        }
    }

    pub async fn with_gil<T, F>(&self, f: F) -> Result<T, PyErr>
    where
        T: Send + 'static,
        F: FnOnce(Python<'_>) -> Result<T, PyErr> + Send + 'static,
    {
        let (resp_tx, resp_rx) = oneshot::channel::<Result<T, PyErr>>();

        self.tx
            .send(Box::new(move || {
                Python::with_gil(|py| {
                    let _ = resp_tx.send(f(py));
                });
            }))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        resp_rx
            .await
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("bridge dropped"))?
    }

    /// Sync/blocking version (blocks the caller thread)
    pub fn sync_with_gil<T, F>(&self, f: F) -> Result<T, PyErr>
    where
        T: Send + 'static,
        F: FnOnce(Python<'_>) -> Result<T, PyErr> + Send + 'static,
    {
        // Avoid deadlock if called from the bridge thread itself

        let (resp_tx, resp_rx) = mpsc::channel::<Result<T, PyErr>>();

        self.tx
            .send(Box::new(move || {
                Python::with_gil(|py| {
                    let _ = resp_tx.send(f(py));
                });
            }))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>(e.to_string()))?;

        resp_rx
            .recv()
            .map_err(|_| PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("bridge dropped"))?
    }
}
