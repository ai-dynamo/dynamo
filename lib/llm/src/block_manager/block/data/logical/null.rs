// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

use super::*;

#[derive(Debug, Clone)]
pub struct NullResources;

impl LogicalResources for NullResources {
    fn handle_transfer<RB, WB>(
        &self,
        _sources: &[RB],
        _targets: &mut [WB],
        _ctx: Arc<TransferContext>,
    ) -> Result<oneshot::Receiver<()>, TransferError>
    where
        RB: ReadableBlock + WriteToStrategy<WB> + storage::Local,
        <RB as StorageTypeProvider>::StorageType: NixlDescriptor,
        <WB as StorageTypeProvider>::StorageType: NixlDescriptor,
        RB: BlockDataProvider<Locality = Logical<Self>>,
        WB: WritableBlock + BlockDataProviderMut<Locality = Logical<Self>>,
    {
        panic!("Null resources cannot be used for transfers");
    }
}
