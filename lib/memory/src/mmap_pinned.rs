// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `mmap` + `cuMemHostRegister` host pinned storage with NUMA `mbind` and
//! best-effort hugepages.
//!
//! Companion to [`crate::pinned::PinnedStorage`] (which uses `cuMemHostAlloc`,
//! no hugepage / mbind control). This path exists because hugepage-backed
//! pinned memory is unreachable through `cuMemHostAlloc` — the kernel only
//! honors `MAP_HUGETLB` on caller-owned `mmap` regions. Use this for the
//! kvbm-service host-memory pool; use [`crate::PinnedStorage`] for legacy
//! per-GPU allocations.
//!
//! Pipeline per allocation (caller's thread should already be pinned to a
//! CPU in the target node's cpulist for correct first-touch):
//!
//! 1. `mmap(NULL, len, RW, MAP_PRIVATE | MAP_ANON [| MAP_HUGETLB|MAP_HUGE_2MB])`
//!    — try explicit hugetlb, fall back per [`HugepageMode`].
//! 2. `mbind(addr, len, MPOL_BIND, nodemask({target}), …, MPOL_MF_STRICT)`
//!    — belt-and-suspenders against the kernel migrating away from the
//!    target node, even after first-touch lands the page locally.
//! 3. First-touch one byte per (huge)page from the calling thread.
//! 4. `cuMemHostRegister(ptr, len, DEVICEMAP)` to pin + map to CUDA.
//!
//! Drop ordering inside [`MmappedPinnedStorage`] guarantees
//! `cuMemHostUnregister` runs before `munmap`, and the [`Arc<CudaContext>`]
//! outlives both so the unregister has a context to bind.

use std::any::Any;
use std::sync::Arc;

use cudarc::driver::CudaContext;
use nix::libc;
use serde::{Deserialize, Serialize};

use crate::numa::NumaNode;
use crate::{MemoryDescriptor, Result, StorageError, StorageKind, actions, nixl::NixlDescriptor};

/// Hugepage allocation policy for one allocation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum HugepageMode {
    /// Don't ask for hugepages. Plain anon `mmap`, no `MADV_HUGEPAGE`.
    #[default]
    Disabled,
    /// Try `MAP_HUGETLB` first; fall back to anon + `MADV_HUGEPAGE`; fall
    /// back to plain anon. Records the actual tier landed on the slab.
    BestEffort,
    /// Only `MAP_HUGETLB`. Allocation fails if hugepages aren't reserved.
    Required,
}

/// Which page-backing tier this slab actually landed on. Surfaced to the
/// pool so a fallback can be reported in metrics + `/v1/pool`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HugepageTier {
    /// `mmap(MAP_HUGETLB | MAP_HUGE_<size>)` succeeded.
    Explicit {
        /// Page size requested via the `MAP_HUGE_*` flag, in bytes.
        page_size: usize,
    },
    /// Plain anon `mmap` + `madvise(MADV_HUGEPAGE)`. Kernel may or may not
    /// have actually promoted the pages — observable via `/proc/<pid>/smaps`.
    Thp,
    /// Plain anon `mmap`, no hugepage hints.
    None,
}

/// Options for [`MmappedPinnedStorage::allocate`].
#[derive(Debug, Clone)]
pub struct MmappedPinnedOptions {
    /// Requested user-visible size in bytes. The actual mapping may be
    /// rounded up to the effective page size and reported via
    /// [`MmappedPinnedStorage::mapped_len`].
    pub size: usize,
    /// Target NUMA node for the pages. **Must be a host-CPU node** —
    /// memory-only nodes are out of scope (see
    /// [`crate::resources::NumaNodeRole::HostCpu`]). `mbind(MPOL_BIND)` is
    /// applied against this node.
    pub numa_node: NumaNode,
    /// Hugepage strategy.
    pub hugepage_mode: HugepageMode,
    /// Page size to request from the explicit hugetlb pool. `None` => use
    /// the system default (`/proc/meminfo Hugepagesize:`).
    pub hugepage_size: Option<usize>,
    /// CUDA device ordinal whose context is used for `cuMemHostRegister`.
    /// Any visible GPU works — the registration is device-portable.
    pub ctx_device_id: u32,
}

/// `mmap`-backed host pinned memory registered with CUDA via
/// `cuMemHostRegister`, with NUMA placement enforced by `mbind`.
#[derive(Debug)]
pub struct MmappedPinnedStorage {
    // RAII guard — only used for its Drop impl (cuMemHostUnregister).
    // Field order is load-bearing: `register_guard` is declared before
    // `mmap_guard` so the unregister runs before munmap.
    #[allow(dead_code)]
    register_guard: HostRegisterGuard,
    mmap_guard: MmapGuard,
    hugepage_tier: HugepageTier,
    numa_node: NumaNode,
    ctx: Arc<CudaContext>,
}

unsafe impl Send for MmappedPinnedStorage {}
unsafe impl Sync for MmappedPinnedStorage {}

impl MmappedPinnedStorage {
    /// Allocate one slab according to `opt`. **Caller must already be on a
    /// CPU in `opt.numa_node`'s cpulist** so the first-touch policy lands
    /// pages on the right node. The [`crate::numa::worker_pool::NumaWorkerPool`]
    /// is the production caller and handles thread pinning.
    pub fn allocate(opt: MmappedPinnedOptions) -> Result<Self> {
        if opt.size == 0 {
            return Err(StorageError::AllocationFailed(
                "zero-sized allocations are not supported".into(),
            ));
        }
        let MmappedPinnedOptions {
            size,
            numa_node,
            hugepage_mode,
            hugepage_size,
            ctx_device_id,
        } = opt;

        let page_size = hugepage_size.unwrap_or_else(default_hugepage_size);
        let (mmap_guard, hugepage_tier) = mmap_with_tier(size, page_size, hugepage_mode)?;

        mbind_to_node(mmap_guard.ptr, mmap_guard.len, numa_node).map_err(|e| {
            StorageError::AllocationFailed(format!(
                "mbind(MPOL_BIND) on node {}: {}",
                numa_node.0, e
            ))
        })?;

        first_touch(mmap_guard.ptr, mmap_guard.len, hugepage_tier);

        let ctx = crate::device::cuda_context(ctx_device_id)?;
        ctx.bind_to_thread().map_err(StorageError::Cuda)?;

        let register_guard = HostRegisterGuard::new(mmap_guard.ptr, mmap_guard.len, ctx.clone())?;

        Ok(Self {
            register_guard,
            mmap_guard,
            hugepage_tier,
            numa_node,
            ctx,
        })
    }

    /// User-requested size in bytes. May be less than [`Self::mapped_len`]
    /// when rounded up for hugepages.
    pub fn size(&self) -> usize {
        // Size and mapped_len are identical when no rounding occurred; we
        // track the request through mmap_guard.len which equals the rounded
        // mapping length — there's no separate user-size field because the
        // pool treats the rounded length as the actual capacity.
        self.mmap_guard.len
    }

    /// Length actually mapped (size rounded up to the effective page size).
    pub fn mapped_len(&self) -> usize {
        self.mmap_guard.len
    }

    /// Which page-backing tier this slab landed on.
    pub fn hugepage_tier(&self) -> HugepageTier {
        self.hugepage_tier
    }

    /// NUMA node the pages were bound to via `mbind(MPOL_BIND)`.
    pub fn numa_node(&self) -> NumaNode {
        self.numa_node
    }

    /// CUDA context used for registration.
    pub fn ctx(&self) -> &Arc<CudaContext> {
        &self.ctx
    }

    /// Pointer to the start of the mapping.
    ///
    /// # Safety
    /// Caller must not retain the pointer past this storage's drop.
    pub unsafe fn as_ptr(&self) -> *const u8 {
        self.mmap_guard.ptr as *const u8
    }
}

impl MemoryDescriptor for MmappedPinnedStorage {
    fn addr(&self) -> usize {
        self.mmap_guard.ptr
    }

    fn size(&self) -> usize {
        self.mmap_guard.len
    }

    fn storage_kind(&self) -> StorageKind {
        StorageKind::Pinned
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn nixl_descriptor(&self) -> Option<NixlDescriptor> {
        None
    }
}

impl crate::nixl::NixlCompatible for MmappedPinnedStorage {
    fn nixl_params(&self) -> (*const u8, usize, nixl_sys::MemType, u64) {
        (
            self.mmap_guard.ptr as *const u8,
            self.mmap_guard.len,
            nixl_sys::MemType::Dram,
            0,
        )
    }
}

impl actions::Memset for MmappedPinnedStorage {
    fn memset(&mut self, value: u8, offset: usize, size: usize) -> Result<()> {
        let end = offset
            .checked_add(size)
            .ok_or_else(|| StorageError::OperationFailed("memset: offset overflow".into()))?;
        if end > self.mmap_guard.len {
            return Err(StorageError::OperationFailed(
                "memset: offset + size > storage size".into(),
            ));
        }
        unsafe {
            let ptr = (self.mmap_guard.ptr as *mut u8).add(offset);
            std::ptr::write_bytes(ptr, value, size);
        }
        Ok(())
    }
}

// =============================================================================
// internal: tier ladder, mmap, mbind, first-touch, register guard
// =============================================================================

/// `MPOL_MF_STRICT` from `<numaif.h>` — kernel returns `EIO` if a page can't
/// be placed on the requested node. Not exposed by `libc`, so define inline.
const MPOL_MF_STRICT: libc::c_uint = 1;

#[derive(Debug)]
struct MmapGuard {
    ptr: usize,
    len: usize,
}

impl Drop for MmapGuard {
    fn drop(&mut self) {
        if self.ptr == 0 || self.len == 0 {
            return;
        }
        // SAFETY: we own the mapping; no outstanding pointers because Drop
        // runs only when the owning MmappedPinnedStorage is being destroyed.
        let r = unsafe { libc::munmap(self.ptr as *mut libc::c_void, self.len) };
        if r != 0 {
            tracing::warn!(
                "munmap({:#x}, {}) failed: {}",
                self.ptr,
                self.len,
                std::io::Error::last_os_error()
            );
        }
    }
}

#[derive(Debug)]
struct HostRegisterGuard {
    ptr: usize,
    ctx: Arc<CudaContext>,
}

impl HostRegisterGuard {
    fn new(ptr: usize, len: usize, ctx: Arc<CudaContext>) -> Result<Self> {
        // SAFETY: the mapping is live for the duration of this Self
        // (declared after MmapGuard in MmappedPinnedStorage, drops first).
        let cu = unsafe {
            cudarc::driver::sys::cuMemHostRegister_v2(
                ptr as *mut std::ffi::c_void,
                len,
                cudarc::driver::sys::CU_MEMHOSTREGISTER_DEVICEMAP,
            )
        };
        cu.result().map_err(StorageError::Cuda)?;
        Ok(Self { ptr, ctx })
    }
}

impl Drop for HostRegisterGuard {
    fn drop(&mut self) {
        if let Err(e) = self.ctx.bind_to_thread() {
            tracing::warn!("bind CUDA context for cuMemHostUnregister: {:?}", e);
        }
        let cu = unsafe {
            cudarc::driver::sys::cuMemHostUnregister(self.ptr as *mut std::ffi::c_void)
        };
        if let Err(e) = cu.result() {
            tracing::warn!("cuMemHostUnregister({:#x}): {:?}", self.ptr, e);
        }
    }
}

fn mmap_with_tier(
    size: usize,
    page_size: usize,
    mode: HugepageMode,
) -> Result<(MmapGuard, HugepageTier)> {
    match mode {
        HugepageMode::Disabled => mmap_anon(size).map(|g| (g, HugepageTier::None)),
        HugepageMode::Required => mmap_explicit(size, page_size)
            .map(|g| (g, HugepageTier::Explicit { page_size }))
            .map_err(|e| {
                StorageError::AllocationFailed(format!(
                    "explicit hugetlb required but unavailable ({} bytes, {} byte pages): {}",
                    size, page_size, e
                ))
            }),
        HugepageMode::BestEffort => match mmap_explicit(size, page_size) {
            Ok(g) => Ok((g, HugepageTier::Explicit { page_size })),
            Err(huge_err) => {
                tracing::info!(
                    "explicit MAP_HUGETLB unavailable for {} bytes ({}); trying THP",
                    size,
                    huge_err
                );
                match mmap_anon_thp(size) {
                    Ok(g) => Ok((g, HugepageTier::Thp)),
                    Err(thp_err) => {
                        tracing::info!(
                            "THP path failed for {} bytes ({}); falling back to plain anon",
                            size,
                            thp_err
                        );
                        mmap_anon(size).map(|g| (g, HugepageTier::None))
                    }
                }
            }
        },
    }
}

fn mmap_anon(size: usize) -> Result<MmapGuard> {
    let len = round_up(size, system_page_size());
    do_mmap(len, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, 0)
}

fn mmap_anon_thp(size: usize) -> Result<MmapGuard> {
    let len = round_up(size, system_page_size());
    let g = do_mmap(len, libc::MAP_PRIVATE | libc::MAP_ANONYMOUS, 0)?;
    // Best-effort hint; ignore failures (e.g. THP disabled at boot).
    let _ = unsafe {
        libc::madvise(
            g.ptr as *mut libc::c_void,
            g.len,
            libc::MADV_HUGEPAGE,
        )
    };
    Ok(g)
}

fn mmap_explicit(size: usize, page_size: usize) -> Result<MmapGuard> {
    let len = round_up(size, page_size);
    let huge_flag = map_huge_flag_for(page_size).ok_or_else(|| {
        StorageError::AllocationFailed(format!(
            "no MAP_HUGE_* flag known for page size {} bytes",
            page_size
        ))
    })?;
    do_mmap(
        len,
        libc::MAP_PRIVATE | libc::MAP_ANONYMOUS | libc::MAP_HUGETLB,
        huge_flag,
    )
}

fn do_mmap(len: usize, base_flags: libc::c_int, extra_flags: libc::c_int) -> Result<MmapGuard> {
    let flags = base_flags | extra_flags;
    let ptr = unsafe {
        libc::mmap(
            std::ptr::null_mut(),
            len,
            libc::PROT_READ | libc::PROT_WRITE,
            flags,
            -1,
            0,
        )
    };
    if ptr == libc::MAP_FAILED {
        return Err(StorageError::AllocationFailed(format!(
            "mmap len={} flags={:#x}: {}",
            len,
            flags,
            std::io::Error::last_os_error()
        )));
    }
    Ok(MmapGuard {
        ptr: ptr as usize,
        len,
    })
}

/// Bind `[addr, addr+len)` to a single NUMA node via `mbind(MPOL_BIND)`.
fn mbind_to_node(addr: usize, len: usize, node: NumaNode) -> std::result::Result<(), String> {
    if node.0 >= 64 {
        return Err(format!(
            "mbind nodemask supports up to 64 nodes; got node id {}",
            node.0
        ));
    }
    let nodemask: u64 = 1u64 << node.0;
    // maxnode is the bit count the kernel reads from nodemask. 64 covers a
    // single u64 mask.
    let r = unsafe {
        libc::syscall(
            libc::SYS_mbind,
            addr,
            len,
            libc::MPOL_BIND as usize,
            &nodemask as *const u64 as usize,
            64usize,
            MPOL_MF_STRICT as usize,
        )
    };
    if r != 0 {
        return Err(format!(
            "mbind: {}",
            std::io::Error::last_os_error()
        ));
    }
    Ok(())
}

/// Touch one byte per page so first-touch lands pages on the calling
/// thread's CPU's NUMA node. The caller pins the thread before invoking.
fn first_touch(addr: usize, len: usize, tier: HugepageTier) {
    let step = match tier {
        HugepageTier::Explicit { page_size } => page_size,
        HugepageTier::Thp | HugepageTier::None => system_page_size(),
    };
    let mut off = 0usize;
    while off < len {
        unsafe {
            std::ptr::write_volatile((addr as *mut u8).add(off), 0);
        }
        off = off.saturating_add(step);
    }
    // Ensure the last byte is touched if size isn't an exact page multiple.
    if len > 0 {
        unsafe {
            std::ptr::write_volatile((addr as *mut u8).add(len - 1), 0);
        }
    }
}

fn system_page_size() -> usize {
    match unsafe { libc::sysconf(libc::_SC_PAGESIZE) } {
        n if n > 0 => n as usize,
        _ => 4096,
    }
}

fn default_hugepage_size() -> usize {
    crate::hugepage::HugepageInfo::discover().default_size_bytes.max(2 * 1024 * 1024)
}

fn round_up(size: usize, page: usize) -> usize {
    if page == 0 {
        return size;
    }
    size.div_ceil(page).saturating_mul(page)
}

/// Map a desired page size onto the `MAP_HUGE_*` flag the kernel expects.
/// Encoded as `log2(page_size) << MAP_HUGE_SHIFT` — same scheme the kernel
/// uses for `MAP_HUGE_2MB`, `MAP_HUGE_1GB`, and the Grace 64K-page kernel's
/// 512 MiB hugepages.
fn map_huge_flag_for(page_size: usize) -> Option<libc::c_int> {
    if page_size == 0 || !page_size.is_power_of_two() {
        return None;
    }
    let log2 = page_size.trailing_zeros() as libc::c_int;
    // Sanity range: 64 KiB (2^16) up to 16 GiB (2^34). MAP_HUGE_MASK is
    // 6 bits on Linux, so encoding values above 63 would overflow.
    if !(16..=34).contains(&log2) {
        return None;
    }
    Some(log2 << libc::MAP_HUGE_SHIFT)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_up_basic() {
        assert_eq!(round_up(1, 4096), 4096);
        assert_eq!(round_up(4096, 4096), 4096);
        assert_eq!(round_up(4097, 4096), 8192);
        assert_eq!(round_up(0, 4096), 0);
        assert_eq!(round_up(123, 0), 123);
    }

    #[test]
    fn map_huge_flag_covers_common_sizes() {
        assert_eq!(map_huge_flag_for(2 * 1024 * 1024), Some(libc::MAP_HUGE_2MB));
        assert_eq!(
            map_huge_flag_for(1024 * 1024 * 1024),
            Some(libc::MAP_HUGE_1GB)
        );
        // 4 KiB is below the hugepage range.
        assert!(map_huge_flag_for(4096).is_none());
        // Grace 64K-page kernel default `Hugepagesize:` is 512 MiB.
        assert_eq!(
            map_huge_flag_for(512 * 1024 * 1024),
            Some(29 << libc::MAP_HUGE_SHIFT)
        );
        // 32 MiB hugepages exist on some kernels — should round-trip.
        assert_eq!(
            map_huge_flag_for(32 * 1024 * 1024),
            Some(25 << libc::MAP_HUGE_SHIFT)
        );
        // Non-power-of-two is rejected.
        assert!(map_huge_flag_for(3 * 1024 * 1024).is_none());
    }

    #[test]
    fn hugepage_tier_serializes() {
        let json = serde_json::to_string(&HugepageTier::Explicit {
            page_size: 2 * 1024 * 1024,
        })
        .unwrap();
        assert!(json.contains("Explicit"));
        let parsed: HugepageTier = serde_json::from_str(&json).unwrap();
        assert_eq!(
            parsed,
            HugepageTier::Explicit {
                page_size: 2 * 1024 * 1024
            }
        );
    }

    /// `mmap` an anon region (no hugetlb, no CUDA register) and confirm we
    /// can read + write it. Exercises the mmap path without touching CUDA
    /// or NUMA so it runs on any Linux host.
    #[test]
    fn mmap_anon_basic_rw() {
        let g = mmap_anon(8192).unwrap();
        unsafe {
            for i in 0..8192 {
                std::ptr::write_volatile((g.ptr as *mut u8).add(i), (i % 256) as u8);
            }
            for i in 0..8192 {
                let v = std::ptr::read_volatile((g.ptr as *const u8).add(i));
                assert_eq!(v, (i % 256) as u8);
            }
        }
    }
}
