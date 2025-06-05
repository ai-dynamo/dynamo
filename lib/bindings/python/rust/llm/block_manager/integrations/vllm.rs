use pyo3::prelude::*;

use dynamo_llm::block_manager as bm;

use crate::llm::block_manager::BlockManager as PyBlockManager;
use crate::to_pyerr;

use super::utils::*;

#[pyclass]
struct DynamoVllmKvCacheManager {
    block_manager: PyBlockManager,
}

impl DynamoVllmKvCacheManager {
    #[inline(always)]
    fn block_manager(&self) -> &bm::KvBlockManager<bm::BasicMetadata> {
        self.block_manager.get_block_manager()
    }
}

#[pymethods]
impl DynamoVllmKvCacheManager {
    #[new]
    #[pyo3(signature = (block_manager))]
    fn new(block_manager: PyBlockManager) -> PyResult<Self> {
        Ok(Self { block_manager })
    }

    fn get_computed_blocks(&self, request: KvRequest) -> PyResult<DynamoKvBlockList> {
        let sequence_hashes = request.sequence_hashes();
        let blocks = self
            .block_manager()
            .device()
            .unwrap()
            .match_sequence_hashes_blocking(&sequence_hashes)
            .map_err(to_pyerr)?;

        Ok(DynamoKvBlockList::new(BlockListType::Immutable(blocks)))
    }
}

// struct SlotUpdate {
//     /// The request ID.
//     request_id: RequestID,

//     /// External state about the number of computed tokens in the request.
//     /// This should match the slots expectation.
//     request_num_computed_tokens: usize,

//     /// The number of new tokens which advances the sequence state.
//     /// This is the number of tokens which will be computed in the near future.
//     /// When [BaseKvCacheManager::update_slot] is called again, these tokens will be committed.
//     num_new_tokens: usize,

//     /// The number of new computed tokens in the request.
//     /// The `num_new_tokens / block_size` should be equal to the length of the `new_computed_blocks`,
//     /// it may have a remainder for the partial block state.
//     num_new_computed_tokens: Option<usize>,

//     /// The new computed blocks which advance the sequence state.
//     new_computed_blocks: Option<BlockList<DeviceStorage, BasicMetadata>>,

//     /// The number of lookahead blocks to cache.
//     num_lookahead_blocks: Option<usize>,

//     /// Whether to delay caching the blocks.
//     delay_cache_blocks: Option<bool>,
// }

// struct Slot {
//     num_computed_tokens: usize,
//     sequence_blocks: Vec<bm::block::ImmutableBlock<DeviceStorageType, bm::BasicMetadata>>,
//     inflight_blocks: Vec<bm::block::MutableBlock<DeviceStorage, BasicMetadata>>,
// }
