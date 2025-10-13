// SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! This module implements the `WriteToStrategy` and `ReadFromStrategy` traits
//! for the common storage types.

use super::*;

impl WriteToStrategy<DiskStorage> for DiskStorage {
    #[inline(always)]
    fn write_to_strategy() -> TransferStrategy {
        TransferStrategy::Nixl(NixlTransfer::Write)
    }
}

impl WriteToStrategy<SystemStorage> for DiskStorage {
    #[inline(always)]
    fn write_to_strategy() -> TransferStrategy {
        TransferStrategy::Nixl(NixlTransfer::Read)
    }
}

impl WriteToStrategy<PinnedStorage> for DiskStorage {
    #[inline(always)]
    fn write_to_strategy() -> TransferStrategy {
        TransferStrategy::Nixl(NixlTransfer::Read)
    }
}

impl WriteToStrategy<DeviceStorage> for DiskStorage {
    #[inline(always)]
    fn write_to_strategy() -> TransferStrategy {
        TransferStrategy::Nixl(NixlTransfer::Read)
    }
}

impl WriteToStrategy<DiskStorage> for SystemStorage {
    #[inline(always)]
    fn write_to_strategy() -> TransferStrategy {
        TransferStrategy::Nixl(NixlTransfer::Write)
    }
}

impl WriteToStrategy<SystemStorage> for SystemStorage {
    #[inline(always)]
    fn write_to_strategy() -> TransferStrategy {
        TransferStrategy::Memcpy
    }
}

impl WriteToStrategy<PinnedStorage> for SystemStorage {
    #[inline(always)]
    fn write_to_strategy() -> TransferStrategy {
        TransferStrategy::Memcpy
    }
}

impl WriteToStrategy<DeviceStorage> for SystemStorage {
    #[inline(always)]
    fn write_to_strategy() -> TransferStrategy {
        TransferStrategy::CudaBlockingH2D
    }
}

impl WriteToStrategy<DiskStorage> for PinnedStorage {
    #[inline(always)]
    fn write_to_strategy() -> TransferStrategy {
        TransferStrategy::Nixl(NixlTransfer::Write)
    }
}

impl WriteToStrategy<SystemStorage> for PinnedStorage {
    #[inline(always)]
    fn write_to_strategy() -> TransferStrategy {
        TransferStrategy::Memcpy
    }
}

impl WriteToStrategy<PinnedStorage> for PinnedStorage {
    #[inline(always)]
    fn write_to_strategy() -> TransferStrategy {
        TransferStrategy::Memcpy
    }
}

impl WriteToStrategy<DeviceStorage> for PinnedStorage {
    #[inline(always)]
    fn write_to_strategy() -> TransferStrategy {
        TransferStrategy::CudaAsyncH2D
    }
}

impl WriteToStrategy<DiskStorage> for DeviceStorage {
    #[inline(always)]
    fn write_to_strategy() -> TransferStrategy {
        TransferStrategy::Nixl(NixlTransfer::Write)
    }
}

impl WriteToStrategy<SystemStorage> for DeviceStorage {
    #[inline(always)]
    fn write_to_strategy() -> TransferStrategy {
        TransferStrategy::CudaBlockingD2H
    }
}

impl WriteToStrategy<PinnedStorage> for DeviceStorage {
    #[inline(always)]
    fn write_to_strategy() -> TransferStrategy {
        TransferStrategy::CudaAsyncD2H
    }
}

impl WriteToStrategy<DeviceStorage> for DeviceStorage {
    #[inline(always)]
    fn write_to_strategy() -> TransferStrategy {
        TransferStrategy::CudaAsyncD2D
    }
}

impl<S: Storage + Local> WriteToStrategy<NixlStorage> for S {
    #[inline(always)]
    fn write_to_strategy() -> TransferStrategy {
        TransferStrategy::Nixl(NixlTransfer::Write)
    }
}

impl<S> ReadFromStrategy<S> for SystemStorage
where
    S: WriteToStrategy<SystemStorage> + Storage + Local,
{
    #[inline(always)]
    fn read_from_strategy() -> TransferStrategy {
        S::write_to_strategy()
    }
}

impl<S> ReadFromStrategy<S> for PinnedStorage
where
    S: WriteToStrategy<PinnedStorage> + Storage + Local,
{
    #[inline(always)]
    fn read_from_strategy() -> TransferStrategy {
        S::write_to_strategy()
    }
}

impl<S> ReadFromStrategy<S> for DeviceStorage
where
    S: WriteToStrategy<DeviceStorage> + Storage + Local,
{
    #[inline(always)]
    fn read_from_strategy() -> TransferStrategy {
        S::write_to_strategy()
    }
}

impl<S: Storage + Local> ReadFromStrategy<NixlStorage> for S {
    #[inline(always)]
    fn read_from_strategy() -> TransferStrategy {
        TransferStrategy::Nixl(NixlTransfer::Read)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_to_strategy() {
        // System to ...
        assert_eq!(
            <SystemStorage as WriteToStrategy<SystemStorage>>::write_to_strategy(),
            TransferStrategy::Memcpy
        );

        assert_eq!(
            <SystemStorage as WriteToStrategy<PinnedStorage>>::write_to_strategy(),
            TransferStrategy::Memcpy
        );

        assert_eq!(
            <SystemStorage as WriteToStrategy<DeviceStorage>>::write_to_strategy(),
            TransferStrategy::CudaBlockingH2D
        );

        assert_eq!(
            <SystemStorage as WriteToStrategy<NixlStorage>>::write_to_strategy(),
            TransferStrategy::Nixl(NixlTransfer::Write)
        );

        // Pinned to ...
        assert_eq!(
            <PinnedStorage as WriteToStrategy<SystemStorage>>::write_to_strategy(),
            TransferStrategy::Memcpy
        );
        assert_eq!(
            <PinnedStorage as WriteToStrategy<PinnedStorage>>::write_to_strategy(),
            TransferStrategy::Memcpy
        );
        assert_eq!(
            <PinnedStorage as WriteToStrategy<DeviceStorage>>::write_to_strategy(),
            TransferStrategy::CudaAsyncH2D
        );
        assert_eq!(
            <PinnedStorage as WriteToStrategy<NixlStorage>>::write_to_strategy(),
            TransferStrategy::Nixl(NixlTransfer::Write)
        );

        // Device to ...
        assert_eq!(
            <DeviceStorage as WriteToStrategy<SystemStorage>>::write_to_strategy(),
            TransferStrategy::CudaBlockingD2H
        );
        assert_eq!(
            <DeviceStorage as WriteToStrategy<PinnedStorage>>::write_to_strategy(),
            TransferStrategy::CudaAsyncD2H
        );
        assert_eq!(
            <DeviceStorage as WriteToStrategy<DeviceStorage>>::write_to_strategy(),
            TransferStrategy::CudaAsyncD2D
        );
        assert_eq!(
            <DeviceStorage as WriteToStrategy<NixlStorage>>::write_to_strategy(),
            TransferStrategy::Nixl(NixlTransfer::Write)
        );

        // Nixl to ... should fail to compile
        // assert_eq!(
        //     <NixlStorage as WriteToStrategy<SystemStorage>>::write_to_strategy(),
        //     TransferStrategy::Invalid
        // );
        // assert_eq!(
        //     <NixlStorage as WriteToStrategy<PinnedStorage>>::write_to_strategy(),
        //     TransferStrategy::Invalid
        // );
        // assert_eq!(
        //     <NixlStorage as WriteToStrategy<DeviceStorage>>::write_to_strategy(),
        //     TransferStrategy::Invalid
        // );
        // assert_eq!(
        //     <NixlStorage as WriteToStrategy<NixlStorage>>::write_to_strategy(),
        //     TransferStrategy::Invalid
        // );
    }

    #[test]
    fn read_from_strategy() {
        // System to ...
        assert_eq!(
            <SystemStorage as ReadFromStrategy<SystemStorage>>::read_from_strategy(),
            TransferStrategy::Memcpy
        );

        assert_eq!(
            <SystemStorage as ReadFromStrategy<PinnedStorage>>::read_from_strategy(),
            TransferStrategy::Memcpy
        );

        assert_eq!(
            <SystemStorage as ReadFromStrategy<DeviceStorage>>::read_from_strategy(),
            TransferStrategy::CudaBlockingD2H
        );

        assert_eq!(
            <SystemStorage as ReadFromStrategy<NixlStorage>>::read_from_strategy(),
            TransferStrategy::Nixl(NixlTransfer::Read)
        );

        // Pinned to ...
        assert_eq!(
            <PinnedStorage as ReadFromStrategy<SystemStorage>>::read_from_strategy(),
            TransferStrategy::Memcpy
        );

        assert_eq!(
            <PinnedStorage as ReadFromStrategy<PinnedStorage>>::read_from_strategy(),
            TransferStrategy::Memcpy
        );

        assert_eq!(
            <PinnedStorage as ReadFromStrategy<DeviceStorage>>::read_from_strategy(),
            TransferStrategy::CudaAsyncD2H
        );

        assert_eq!(
            <PinnedStorage as ReadFromStrategy<NixlStorage>>::read_from_strategy(),
            TransferStrategy::Nixl(NixlTransfer::Read)
        );

        // Device to ...
        assert_eq!(
            <DeviceStorage as ReadFromStrategy<SystemStorage>>::read_from_strategy(),
            TransferStrategy::CudaBlockingH2D
        );

        assert_eq!(
            <DeviceStorage as ReadFromStrategy<PinnedStorage>>::read_from_strategy(),
            TransferStrategy::CudaAsyncH2D
        );

        assert_eq!(
            <DeviceStorage as ReadFromStrategy<DeviceStorage>>::read_from_strategy(),
            TransferStrategy::CudaAsyncD2D
        );

        assert_eq!(
            <DeviceStorage as ReadFromStrategy<NixlStorage>>::read_from_strategy(),
            TransferStrategy::Nixl(NixlTransfer::Read)
        );

        // Nixl to ... should fail to compile
        // assert_eq!(
        //     <NixlStorage as ReadFromStrategy<SystemStorage>>::read_from_strategy(),
        //     TransferStrategy::Invalid
        // );
        //
        // assert_eq!(
        //     <NixlStorage as ReadFromStrategy<PinnedStorage>>::read_from_strategy(),
        //     TransferStrategy::Invalid
        // );
        //
        // assert_eq!(
        //     <NixlStorage as ReadFromStrategy<DeviceStorage>>::read_from_strategy(),
        //     TransferStrategy::Invalid
        // );
        //
        // assert_eq!(
        //     <NixlStorage as ReadFromStrategy<NixlStorage>>::read_from_strategy(),
        //     TransferStrategy::Invalid
        // );
    }
}
