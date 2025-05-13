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

//! This module implements the `WriteToStrategy` and `ReadFromStrategy` traits
//! for the common storage types.

use super::*;

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
        TransferStrategy::NixlWrite
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
        TransferStrategy::NixlRead
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
            TransferStrategy::NixlWrite
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
            TransferStrategy::NixlWrite
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
            TransferStrategy::NixlWrite
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
            TransferStrategy::NixlRead
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
            TransferStrategy::NixlRead
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
            TransferStrategy::NixlRead
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
