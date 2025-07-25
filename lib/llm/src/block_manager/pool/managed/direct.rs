use super::*;

/// Calls the [`State`] methods directly, bypassing the progress engine
pub struct DirectAccess<S: Storage, L: LocalityProvider, M: BlockMetadata> {
    state: Arc<Mutex<State<S, L, M>>>,
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> Clone for DirectAccess<S, L, M> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

impl<S: Storage, L: LocalityProvider, M: BlockMetadata> DirectAccess<S, L, M> {
    pub fn new(state: Arc<Mutex<State<S, L, M>>>) -> Self {
        Self { state }
    }

    /// Allocate a set of blocks from the pool.
    pub fn allocate_blocks(&self, count: usize) -> BlockPoolResult<Vec<MutableBlock<S, L, M>>> {
        let mut state = self.state.lock().unwrap();
        state.allocate_blocks(count)
    }

    /// Register a set of blocks with the pool.
    pub fn register_blocks(
        &self,
        blocks: Vec<MutableBlock<S, L, M>>,
    ) -> Result<ImmutableBlocks<S, L, M>, BlockPoolError> {
        let mut blocks = blocks;
        let mut immutable = Vec::with_capacity(blocks.len());
        let mut publish_handles: Option<Publisher> = None;
        let mut start_position = 0;

        loop {
            let mut state = self.state.lock().unwrap();

            if publish_handles.is_none() {
                publish_handles = Some(state.publisher());
            }

            match state.try_register_blocks_direct(
                &mut blocks,
                &mut immutable,
                publish_handles.as_mut().unwrap(),
                start_position,
            ) {
                Ok(_) => break,
                Err(e) => match e.retry_or_err() {
                    Ok(restart_position) => {
                        start_position = restart_position;
                    }
                    Err(pool_err) => return Err(pool_err),
                },
            }

            // under the extremely rare chance we need to retry, we need to drop the state
            drop(state);
            tracing::debug!("register_blocks_direct: retrying");
            std::thread::yield_now();
        }

        Ok(immutable)
    }

    /// Match a set of sequence hashes to existing blocks in the pool.
    pub fn match_sequence_hashes(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> BlockPoolResult<ImmutableBlocks<S, L, M>> {
        let mut immutable_blocks = Vec::new();
        let mut start_position = 0;

        loop {
            let mut state = self.state.lock().unwrap();
            match state.try_match_sequence_hashes_direct(
                sequence_hashes,
                &mut immutable_blocks,
                start_position,
            ) {
                Ok(_) => break,
                Err(e) => match e.retry_or_err() {
                    Ok(restart_position) => {
                        start_position = restart_position;
                    }
                    Err(e) => return Err(e),
                },
            }

            drop(state);
            tracing::debug!("match_sequence_hashes_direct: retrying");
            std::thread::yield_now();
        }

        Ok(immutable_blocks)
    }

    // pub fn touch_blocks(&self, sequence_hashes: &[SequenceHash]) -> BlockPoolResult<()> {
    //     let mut state = self.state.lock().unwrap();
    //     state.touch_blocks(sequence_hashes)
    // }

    pub fn add_blocks(&self, blocks: Vec<Block<S, L, M>>) -> Result<(), BlockPoolError> {
        let mut state = self.state.lock().unwrap();
        state.inactive.add_blocks(blocks)
    }

    pub fn try_return_block(&self, mut block: OwnedBlock<S, L, M>) -> BlockPoolResult<()> {
        let raw_block = block
            .try_take_block(private::PrivateToken)
            .ok_or(BlockPoolError::NotReturnable)?;

        let mut state = self.state.lock().unwrap();
        state.return_block(raw_block);

        Ok(())
    }

    pub fn status(&self) -> Result<BlockPoolStatus, BlockPoolError> {
        let state = self.state.lock().unwrap();
        Ok(state.status())
    }

    pub fn reset(&self) -> Result<(), BlockPoolError> {
        let mut state = self.state.lock().unwrap();
        state.inactive.reset()
    }

    pub fn reset_blocks(
        &self,
        sequence_hashes: &[SequenceHash],
    ) -> Result<ResetBlocksResponse, BlockPoolError> {
        let mut state = self.state.lock().unwrap();
        Ok(state.try_reset_blocks(sequence_hashes))
    }
}
