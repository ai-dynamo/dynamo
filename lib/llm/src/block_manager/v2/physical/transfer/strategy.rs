// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Transfer strategy selection based on source and destination storage locations.

use crate::block_manager::v2::memory::StorageKind;

use super::TransferCapabilities;
use crate::block_manager::v2::physical::{layout::PhysicalLayout, transfer::TransferContext};

/// Transfer strategy to use for copying memory between locations.
///
/// The strategy is determined by the source and destination storage locations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TransferStrategy {
    /// CPU memcpy (for host-to-host transfers)
    Memcpy,

    /// CUDA async host-to-device transfer
    CudaAsyncH2D,

    /// CUDA async device-to-host transfer
    CudaAsyncD2H,

    /// CUDA async device-to-device transfer
    CudaAsyncD2D,

    /// CUDA blocking host-to-device transfer
    CudaBlockingH2D,

    /// CUDA blocking device-to-host transfer
    CudaBlockingD2H,

    /// NIXL read operation (pull from remote)
    NixlRead,

    /// NIXL write operation (push to remote)
    NixlWrite,

    /// NIXL write (flipped local and remote order)
    /// This is needed for some NIXL backends.
    /// For example, the POSIX backend requires that host memory
    /// always be the "local" descriptor list, regardless of whether
    /// it's a read or write.
    NixlWriteFlipped,

    /// NIXL read (flipped local and remote order)
    NixlReadFlipped,

    /// Invalid/unsupported transfer
    Invalid,
}

/// Plan for executing a transfer, either direct or via bounce buffer.
///
/// Some transfers require staging through host memory when direct paths
/// are not enabled via capabilities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TransferPlan {
    /// Direct single-hop transfer using the specified strategy.
    Direct(TransferStrategy),

    /// Two-hop transfer requiring a bounce buffer in host memory.
    ///
    /// This is used when:
    /// - Device → Remote (without GPU RDMA)
    /// - Disk → Remote
    /// - Device ↔ Disk (without GDS)
    TwoHop {
        /// First hop strategy (src → bounce)
        first: TransferStrategy,

        /// Bounce buffer location (always Pinned for best performance)
        bounce_location: StorageKind,

        /// Second hop strategy (bounce → dst)
        second: TransferStrategy,
    },
}

pub(crate) fn select_strategy(
    src: &PhysicalLayout,
    dst: &PhysicalLayout,
    ctx: &TransferContext,
) -> anyhow::Result<TransferPlan> {
    let is_src_local = src.nixl_metadata().agent_name() == ctx.nixl_agent().name();
    let is_dst_local = dst.nixl_metadata().agent_name() == ctx.nixl_agent().name();

    if !is_src_local && !is_dst_local {
        return Err(anyhow::anyhow!(
            "Both src and dst are remote - this is not supported."
        ));
    }

    if is_src_local && is_dst_local {
        return Ok(select_direct_strategy(
            src.location(),
            dst.location(),
            false,
            ctx.capabilities(),
        ));
    }

    select_remote_strategy_v2(
        src.location(),
        is_src_local,
        dst.location(),
        is_dst_local,
        ctx.capabilities(),
    )
}

/// Select the appropriate transfer plan based on source and destination locations.
///
/// # Arguments
/// * `src` - Source storage location (always local)
/// * `dst` - Destination storage location (can be local or remote)
/// * `dst_is_remote` - Whether destination is on a remote node
/// * `capabilities` - Transfer capability flags
///
/// # Returns
/// A transfer plan (direct or two-hop)
///
/// # Conservative Default Policy
///
/// With default capabilities (all disabled):
/// - Device can only transfer to/from Host
/// - Disk can only transfer to/from Host
/// - Host can transfer to Device, Disk, or Remote
/// - Device ↔ Device is allowed (native CUDA)
///
/// Transfers that would violate this policy are staged through host:
/// - Device → Remote: Device → Host → Remote (2 hops)
/// - Disk → Remote: Disk → Host → Remote (2 hops)
/// - Device ↔ Disk: Device → Host → Disk (2 hops)
///
/// # Optional Direct Paths
///
/// - `allow_gds`: Enables Disk ↔ Device direct transfers
/// - `allow_gpu_rdma`: Enables Device → Remote direct transfers
fn select_direct_strategy(
    src: StorageKind,
    dst: StorageKind,
    dst_is_remote: bool,
    capabilities: &TransferCapabilities,
) -> TransferPlan {
    use StorageKind::*;
    use TransferStrategy::*;

    // Handle remote destination
    if dst_is_remote {
        return select_remote_strategy(src, capabilities);
    }

    // Local-to-local transfers
    match (src, dst) {
        // Host ↔ Host - direct memcpy
        (System, System) | (System, Pinned) | (Pinned, System) | (Pinned, Pinned) => {
            TransferPlan::Direct(Memcpy)
        }

        // Host → Device - direct CUDA
        (System, Device(_)) => TransferPlan::Direct(CudaBlockingH2D),
        (Pinned, Device(_)) => TransferPlan::Direct(CudaAsyncH2D),

        // Device → Host - direct CUDA
        (Device(_), System) => TransferPlan::Direct(CudaBlockingD2H),
        (Device(_), Pinned) => TransferPlan::Direct(CudaAsyncD2H),

        // Device ↔ Device - direct CUDA
        (Device(_), Device(_)) => TransferPlan::Direct(CudaAsyncD2D),

        // Host ↔ Disk - direct NIXL
        (System, Disk(_)) | (Pinned, Disk(_)) => TransferPlan::Direct(NixlWrite),
        (Disk(_), System) | (Disk(_), Pinned) => TransferPlan::Direct(NixlReadFlipped),

        // Disk ↔ Disk - NIXL doesn't seem to support direct transfers here.
        // Leaving this as two-hop for now.
        (Disk(_), Disk(_)) => TransferPlan::TwoHop {
            first: NixlReadFlipped,
            bounce_location: Pinned,
            second: NixlWrite,
        },

        // Device ↔ Disk - check GDS capability
        (Device(_), Disk(_)) => {
            if capabilities.allows_device_disk_direct() {
                // Direct GDS transfer
                TransferPlan::Direct(NixlWrite)
            } else {
                // Stage through host: Device → Pinned → Disk
                TransferPlan::TwoHop {
                    first: CudaAsyncD2H,
                    bounce_location: Pinned,
                    second: NixlWrite,
                }
            }
        }
        (Disk(_), Device(_)) => {
            if capabilities.allows_device_disk_direct() {
                // Direct GDS transfer
                TransferPlan::Direct(NixlRead)
            } else {
                // Stage through host: Disk → Pinned → Device
                TransferPlan::TwoHop {
                    first: NixlReadFlipped,
                    bounce_location: Pinned,
                    second: CudaAsyncH2D,
                }
            }
        }
    }
}

/// Select transfer strategy for remote destination.
fn select_remote_strategy(src: StorageKind, capabilities: &TransferCapabilities) -> TransferPlan {
    use StorageKind::*;
    use TransferStrategy::*;

    match src {
        // Host → Remote - direct NIXL
        System | Pinned => TransferPlan::Direct(NixlWrite),

        // Device → Remote - check GPU RDMA capability
        Device(_) => {
            if capabilities.allows_device_remote_direct() {
                // Direct GPU RDMA transfer
                TransferPlan::Direct(NixlWrite)
            } else {
                // Stage through host: Device → Pinned → Remote
                TransferPlan::TwoHop {
                    first: CudaAsyncD2H,
                    bounce_location: Pinned,
                    second: NixlWrite,
                }
            }
        }

        // Disk → Remote - always stage through host
        Disk(_) => TransferPlan::TwoHop {
            first: NixlWrite,
            bounce_location: Pinned,
            second: NixlWrite,
        },
    }
}

fn select_remote_strategy_v2(
    src: StorageKind,
    is_src_local: bool,
    dst: StorageKind,
    is_dst_local: bool,
    capabilities: &TransferCapabilities,
) -> anyhow::Result<TransferPlan> {
    // We only support System, Pinned and Device for remote transfers.
    // Later we might support staged/bounce buffer transfers.

    if matches!(src, StorageKind::Disk(_)) | matches!(dst, StorageKind::Disk(_)) {
        return Err(anyhow::anyhow!(
            "Neither local nor remote disk transfers are supported over NIXL at this time."
        ));
    }

    if !capabilities.allow_gpu_rdma
        && (matches!(src, StorageKind::Device(_)) || matches!(dst, StorageKind::Device(_)))
    {
        return Err(anyhow::anyhow!(
            "GPU RDMA is disabled - this transfer requires GPU RDMA."
        ));
    }

    if is_src_local && !is_dst_local {
        return Ok(TransferPlan::Direct(TransferStrategy::NixlWrite));
    }

    if is_dst_local && !is_src_local {
        return Ok(TransferPlan::Direct(TransferStrategy::NixlReadFlipped));
    }

    unreachable!("Both src and dst are remote - this is not supported.");
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_caps() -> TransferCapabilities {
        TransferCapabilities::default()
    }

    #[test]
    fn test_host_to_host_transfers() {
        let caps = default_caps();
        assert_eq!(
            select_direct_strategy(StorageKind::System, StorageKind::System, false, &caps),
            TransferPlan::Direct(TransferStrategy::Memcpy)
        );
        assert_eq!(
            select_direct_strategy(StorageKind::System, StorageKind::Pinned, false, &caps),
            TransferPlan::Direct(TransferStrategy::Memcpy)
        );
        assert_eq!(
            select_direct_strategy(StorageKind::Pinned, StorageKind::System, false, &caps),
            TransferPlan::Direct(TransferStrategy::Memcpy)
        );
        assert_eq!(
            select_direct_strategy(StorageKind::Pinned, StorageKind::Pinned, false, &caps),
            TransferPlan::Direct(TransferStrategy::Memcpy)
        );
    }

    #[test]
    fn test_host_to_device_transfers() {
        let caps = default_caps();
        // System (unpinned) to device should be blocking
        assert_eq!(
            select_direct_strategy(StorageKind::System, StorageKind::Device(0), false, &caps),
            TransferPlan::Direct(TransferStrategy::CudaBlockingH2D)
        );

        // Pinned to device should be async
        assert_eq!(
            select_direct_strategy(StorageKind::Pinned, StorageKind::Device(0), false, &caps),
            TransferPlan::Direct(TransferStrategy::CudaAsyncH2D)
        );
    }

    #[test]
    fn test_device_to_host_transfers() {
        let caps = default_caps();
        // Device to system should be blocking
        assert_eq!(
            select_direct_strategy(StorageKind::Device(0), StorageKind::System, false, &caps),
            TransferPlan::Direct(TransferStrategy::CudaBlockingD2H)
        );

        // Device to pinned should be async
        assert_eq!(
            select_direct_strategy(StorageKind::Device(0), StorageKind::Pinned, false, &caps),
            TransferPlan::Direct(TransferStrategy::CudaAsyncD2H)
        );
    }

    #[test]
    fn test_device_to_device_transfers() {
        let caps = default_caps();
        assert_eq!(
            select_direct_strategy(StorageKind::Device(0), StorageKind::Device(1), false, &caps),
            TransferPlan::Direct(TransferStrategy::CudaAsyncD2D)
        );
        assert_eq!(
            select_direct_strategy(StorageKind::Device(3), StorageKind::Device(3), false, &caps),
            TransferPlan::Direct(TransferStrategy::CudaAsyncD2D)
        );
    }

    #[test]
    fn test_disk_to_host_transfers() {
        let caps = default_caps();
        // Disk to host - direct NIXL
        assert_eq!(
            select_direct_strategy(StorageKind::Disk(42), StorageKind::System, false, &caps),
            TransferPlan::Direct(TransferStrategy::NixlReadFlipped)
        );
        assert_eq!(
            select_direct_strategy(StorageKind::Disk(42), StorageKind::Pinned, false, &caps),
            TransferPlan::Direct(TransferStrategy::NixlReadFlipped)
        );
    }

    #[test]
    fn test_host_to_disk_transfers() {
        let caps = default_caps();
        // Host to disk - direct NIXL
        assert_eq!(
            select_direct_strategy(StorageKind::System, StorageKind::Disk(42), false, &caps),
            TransferPlan::Direct(TransferStrategy::NixlWrite)
        );
        assert_eq!(
            select_direct_strategy(StorageKind::Pinned, StorageKind::Disk(42), false, &caps),
            TransferPlan::Direct(TransferStrategy::NixlWrite)
        );
    }

    #[test]
    fn test_device_to_disk_without_gds() {
        let caps = default_caps(); // GDS disabled
        // Device → Disk should use bounce buffer
        let plan =
            select_direct_strategy(StorageKind::Device(0), StorageKind::Disk(42), false, &caps);
        match plan {
            TransferPlan::TwoHop {
                first,
                bounce_location,
                second,
            } => {
                assert_eq!(first, TransferStrategy::CudaAsyncD2H);
                assert_eq!(bounce_location, StorageKind::Pinned);
                assert_eq!(second, TransferStrategy::NixlWrite);
            }
            _ => panic!("Expected TwoHop plan"),
        }
    }

    #[test]
    fn test_disk_to_device_without_gds() {
        let caps = default_caps(); // GDS disabled
        // Disk → Device should use bounce buffer
        let plan =
            select_direct_strategy(StorageKind::Disk(42), StorageKind::Device(0), false, &caps);
        match plan {
            TransferPlan::TwoHop {
                first,
                bounce_location,
                second,
            } => {
                assert_eq!(first, TransferStrategy::NixlReadFlipped);
                assert_eq!(bounce_location, StorageKind::Pinned);
                assert_eq!(second, TransferStrategy::CudaAsyncH2D);
            }
            _ => panic!("Expected TwoHop plan"),
        }
    }

    #[test]
    fn test_device_to_disk_with_gds() {
        let caps = TransferCapabilities::default().with_gds(true);
        // Device → Disk should be direct with GDS
        assert_eq!(
            select_direct_strategy(StorageKind::Device(0), StorageKind::Disk(42), false, &caps),
            TransferPlan::Direct(TransferStrategy::NixlWrite)
        );
    }

    #[test]
    fn test_disk_to_device_with_gds() {
        let caps = TransferCapabilities::default().with_gds(true);
        // Disk → Device should be direct with GDS
        assert_eq!(
            select_direct_strategy(StorageKind::Disk(42), StorageKind::Device(0), false, &caps),
            TransferPlan::Direct(TransferStrategy::NixlRead)
        );
    }

    #[test]
    fn test_host_to_remote() {
        let caps = default_caps();
        // Host → Remote - always direct
        assert_eq!(
            select_direct_strategy(StorageKind::System, StorageKind::System, true, &caps),
            TransferPlan::Direct(TransferStrategy::NixlWrite)
        );
        assert_eq!(
            select_direct_strategy(StorageKind::Pinned, StorageKind::Pinned, true, &caps),
            TransferPlan::Direct(TransferStrategy::NixlWrite)
        );
    }

    #[test]
    fn test_device_to_remote_without_rdma() {
        let caps = default_caps(); // GPU RDMA disabled
        // Device → Remote should use bounce buffer
        let plan = select_direct_strategy(StorageKind::Device(0), StorageKind::System, true, &caps);
        match plan {
            TransferPlan::TwoHop {
                first,
                bounce_location,
                second,
            } => {
                assert_eq!(first, TransferStrategy::CudaAsyncD2H);
                assert_eq!(bounce_location, StorageKind::Pinned);
                assert_eq!(second, TransferStrategy::NixlWrite);
            }
            _ => panic!("Expected TwoHop plan"),
        }
    }

    #[test]
    fn test_device_to_remote_with_rdma() {
        let caps = TransferCapabilities::default().with_gpu_rdma(true);
        // Device → Remote should be direct with GPU RDMA
        assert_eq!(
            select_direct_strategy(StorageKind::Device(0), StorageKind::Device(0), true, &caps),
            TransferPlan::Direct(TransferStrategy::NixlWrite)
        );
    }

    #[test]
    fn test_disk_to_remote() {
        let caps = default_caps();
        // Disk → Remote always uses bounce buffer
        let plan = select_direct_strategy(StorageKind::Disk(42), StorageKind::System, true, &caps);
        match plan {
            TransferPlan::TwoHop {
                first,
                bounce_location,
                second,
            } => {
                assert_eq!(first, TransferStrategy::NixlWrite);
                assert_eq!(bounce_location, StorageKind::Pinned);
                assert_eq!(second, TransferStrategy::NixlWrite);
            }
            _ => panic!("Expected TwoHop plan"),
        }
    }
}
