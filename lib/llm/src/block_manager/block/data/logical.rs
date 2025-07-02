// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

pub mod lw_sharded;

pub trait Parallelism: Send + Sync + 'static + std::fmt::Debug {
    type Output<S: Storage>: BlockDataExt<S> + std::fmt::Debug;
}

/// Individual block storage - cannot be cloned to ensure uniqueness
#[derive(Debug)]
pub struct LogicalBlockData<S: Storage, P: Parallelism> {
    block_id: BlockId,
    block_set_id: usize,
    worker_id: WorkerID,
    #[allow(dead_code)]
    parallelism: P,
    storage_type: StorageType,
    storage: std::marker::PhantomData<S>,
}

impl<S: Storage, P: Parallelism> LogicalBlockData<S, P> {
    pub fn new(
        block_id: BlockId,
        block_set_id: usize,
        worker_id: WorkerID,
        parallelism: P,
        storage_type: StorageType,
    ) -> Self {
        Self {
            block_id,
            block_set_id,
            worker_id,
            parallelism,
            storage_type,
            storage: std::marker::PhantomData,
        }
    }
}

impl<S: Storage, P: Parallelism> BlockDataExt<S> for LogicalBlockData<S, P> {
    fn block_id(&self) -> BlockId {
        self.block_id
    }

    fn block_set_id(&self) -> usize {
        self.block_set_id
    }

    fn worker_id(&self) -> WorkerID {
        self.worker_id
    }

    fn storage_type(&self) -> &StorageType {
        &self.storage_type
    }

    fn is_fully_contiguous(&self) -> bool {
        unimplemented!()
    }

    fn num_layers(&self) -> usize {
        unimplemented!()
    }

    fn page_size(&self) -> usize {
        unimplemented!()
    }

    fn num_outer_dims(&self) -> usize {
        unimplemented!()
    }

    fn num_inner_dims(&self) -> usize {
        unimplemented!()
    }

    fn is_local(&self) -> Option<&dyn BlockDataViews<S>> {
        None
    }

    fn is_local_mut(&mut self) -> Option<&mut dyn BlockDataViews<S>> {
        None
    }
}
