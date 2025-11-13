use dynamo_llm::block_manager::offload::filter::OffloadFilter;
use dynamo_llm::tokens::SequenceHash;
use pyo3::prelude::*;

#[derive(Debug)]
pub struct PyOffloadFilter {
    py_lambda: Py<PyAny>,
}

impl PyOffloadFilter {
    pub fn new(py_lambda: Py<PyAny>) -> Self {
        Self { py_lambda }
    }

    fn check_python_should_offload(&self, py: Python<'_>, sequence_hash: SequenceHash) -> bool {
        self.py_lambda
            .call1(py, (sequence_hash,))
            .map(|should_offload| {
                should_offload.extract::<bool>(py).unwrap_or_else(|e| {
                    tracing::error!("Error extracting result from Python offload filter: {}", e);
                    false
                })
            })
            .unwrap_or_else(|e| {
                tracing::error!("Error calling Python offload filter: {}", e);
                false
            })
    }
}

impl OffloadFilter for PyOffloadFilter {
    fn should_offload(&self, sequence_hash: SequenceHash) -> bool {
        Python::with_gil(|py| self.check_python_should_offload(py, sequence_hash))
    }
    // TODO(jthomson04): Maybe we want to support batching here?
}
