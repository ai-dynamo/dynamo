// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

use reuse::AvailableBlocks;

/// Manages the reservation and priority reuse of kv blocks for a single storage type,
/// e.g. a GPU, host memory.
pub struct KvStorageManager {
    available_blocks: AvailableBlocks,
    inflight_blocks: ReservedBlocks,
    block_size: usize,
}

impl KvStorageManager {
    pub async fn new(block_size: usize) -> Self {
        Self {
            available_blocks: AvailableBlocks::new().await,
            inflight_blocks: ReservedBlocks::new(block_size),
            block_size,
        }
    }

    pub async fn prepare_prefill_sequence(&mut self, tokens: Tokens) -> Result<PrefillMatched> {
        log::debug!("adding request with {} tokens", tokens.len());

        let seq = tokens.into_sequence(self.block_size);
        let (blocks, tail_block) = seq.into_parts();
        log::debug!(
            "request translates to {} blocks; remaining tokens: {}",
            blocks.len(),
            tail_block.tokens().len()
        );

        // first match blocks to inflight blocks
        let mut inflight_blocks = self.inflight_blocks.match_token_blocks(&blocks)?;
        log::debug!("matched {} inflight blocks", inflight_blocks.len());

        // shift the blocks to the left by the number of inflight blocks
        let unmatched_blocks = &blocks[inflight_blocks.len()..];
        let unmatched_hashes = unmatched_blocks
            .iter()
            .map(|b| b.sequence_hash())
            .collect::<Vec<_>>();

        // match the remaining blocks to freed gpu blocks (available_blocks)
        let unregistered_blocks = self.available_blocks.match_blocks(unmatched_hashes).await?;
        log::debug!("matched {} freed blocks", unregistered_blocks.len());

        // the blocks from the freed blocks pool must be registered as inflight blocks
        // todo - we might have to register the list of unregistered blocks as a single transaction
        for block in unregistered_blocks {
            inflight_blocks.push(self.inflight_blocks.register(block)?);
        }

        // the remaining blocks are the unmatched blocks
        let remaining_blocks = blocks.into_iter().skip(inflight_blocks.len()).collect();

        Ok(PrefillMatched {
            inflight_blocks,
            remaining_blocks,
            tail_block,
        })
    }

    pub async fn prepare_prefill_offload(
        &mut self,
        matched: PrefillMatched,
    ) -> Result<PrefillOffload> {
        let (inflight_blocks, remaining_blocks, tail_block) = matched.dissolve();

        let mut blocks_to_reuse = self
            .available_blocks
            .take_blocks(remaining_blocks.len() as u32 + 1)
            .await?;

        if blocks_to_reuse.len() != remaining_blocks.len() + 1 {
            raise!(
                "expected {} blocks, got {}",
                remaining_blocks.len() + 1,
                blocks_to_reuse.len()
            );
        }

        // update the blocks_to_reuse with the token block from remaining_blocks
        let complete_prefill_blocks: Vec<UniqueBlock> = remaining_blocks
            .into_iter()
            .map(|b| {
                let mut block = blocks_to_reuse.pop().unwrap();
                block.update_token_block(b);
                block
            })
            .collect();

        assert_eq!(blocks_to_reuse.len(), 1);
        let tail_kv_block = blocks_to_reuse.pop().unwrap();

        let tail_prefill_block = PartialKvBlock {
            token_block: tail_block,
            kv_block: tail_kv_block,
        };

        Ok(PrefillOffload {
            inflight_blocks,
            complete_prefill_blocks,
            tail_prefill_block,
        })
    }
}

#[derive(Dissolve)]
pub struct PartialKvBlock {
    token_block: PartialTokenBlock,
    kv_block: UniqueBlock,
}

#[derive(Dissolve)]
pub struct PrefillMatched {
    inflight_blocks: Vec<ReservedBlock>,
    remaining_blocks: Vec<TokenBlock>,
    tail_block: PartialTokenBlock,
}

#[derive(Dissolve)]
pub struct PrefillOffload {
    inflight_blocks: Vec<ReservedBlock>,
    complete_prefill_blocks: Vec<UniqueBlock>,
    tail_prefill_block: PartialKvBlock,
}

// #[cfg(test)]
// mod tests {
//     use super::*;
//     use dynamo_runtime::logging::init;

//     #[tokio::test]
//     async fn test() {
//         init();

//         let mut manager = KvStorageManager::new(2);

//         for _ in 0..100 {
//             manager.available_blocks.insert(KvBlock::default());
//         }

//         let tokens = Tokens::from([0_i32, 1, 2, 3, 4, 5, 6, 7, 8].as_ref());

//         // this is good for the scheduler to make a local decision as it now knows how many
//         // net-new blocks need to be prefilled
//         let sequence = manager.prepare_prefill_sequence(tokens).unwrap();

//         assert_eq!(sequence.inflight_blocks.len(), 0);
//         assert_eq!(sequence.remaining_blocks.len(), 4);
//     }
// }
