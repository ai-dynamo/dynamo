use crate::llm::block_manager::vllm::KvbmRequest;

use super::*;
use pyo3::wrap_pymodule;

use crate::llm::block_manager::BlockManager as PyBlockManager;

#[pyclass]
pub struct KvConnectorLeader {
    slots: HashMap<String, LeaderSlot>,
    block_manager: PyBlockManager,
}

#[pymethods]
impl KvConnectorLeader {
    #[new]
    #[pyo3(signature = (worker_id, block_manager))]
    pub fn new(worker_id: String, block_manager: PyBlockManager) -> Self {
        tracing::info!(
            "KvConnectorLeader initialized with worker_id: {}",
            worker_id
        );
        Self {
            slots: HashMap::new(),
            block_manager,
        }
    }

    pub fn get_block_manager(&self) -> PyBlockManager {
        self.block_manager.clone()
    }

    pub fn get_num_new_matched_tokens(
        &self,
        request_id: String,
        request_num_tokens: u64,
        num_computed_tokens: u64,
    ) -> (u64, bool) {
        tracing::debug!(
            request_id,
            "request_num_tokens: {request_num_tokens}; num_computed_tokens: {num_computed_tokens}"
        );
        assert!(
            self.slots.contains_key(&request_id),
            "Slot not found for request_id: {request_id}"
        );
        (0, false)
    }

    /// We drop the need to pass in the KvCacheBlocks and the num_external_tokens as they are captured
    /// statefully in the [`VllmLeaderKvCacheManagerAndConnector::get_num_new_matched_tokens`] function.
    pub fn update_state_after_alloc(
        &mut self,
        request_id: String,
        block_ids: Vec<BlockId>,
        num_external_tokens: u64,
    ) {
        tracing::debug!(
            request_id,
            "block_ids: {block_ids:?}; num_external_tokens: {num_external_tokens}"
        );

        assert!(
            self.slots.contains_key(&request_id),
            "Slot not found for request_id: {request_id}"
        );
    }

    pub fn build_connector_metadata(&self, scheduler_output: SchedulerOutput) -> PyResult<Vec<u8>> {
        tracing::debug!("Building connector metadata");
        tracing::debug!("scheduler_output: {scheduler_output:#?}");
        scheduler_output.serialize()
    }

    pub fn request_finished(&mut self, request_id: String, block_ids: Vec<BlockId>) -> bool {
        tracing::debug!("Request finished: {request_id}; block_ids: {block_ids:?}");
        true
    }

    pub fn has_slot(&self, request_id: String) -> bool {
        self.slots.contains_key(&request_id)
    }

    pub fn create_slot(&mut self, request: KvbmRequest, tokens: Vec<u32>) -> PyResult<()> {
        self.slots.insert(request.request_id, LeaderSlot {});
        Ok(())
    }
}

struct LeaderSlot {}
