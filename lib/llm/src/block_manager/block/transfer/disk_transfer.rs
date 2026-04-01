// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! NIXL POSIX/GDS-MT disk-backend transfer implementation.

use super::*;
use super::remote::{RemoteBlockDescriptor, RemoteKey, RemoteTransferDirection};
use crate::block_manager::config::RemoteTransferContext;
use crate::block_manager::storage::RemoteDiskStorage;
use crate::block_manager::storage::nixl::NixlRegisterableStorage;
use dynamo_runtime::config::environment_names::kvbm::remote_disk as env;
use nixl_sys::{Agent as NixlAgent, MemType, MemoryRegion, NixlDescriptor, OptArgs, XferDescList, XferOp};
use once_cell::sync::Lazy;
use parking_lot::Mutex as SyncMutex;
use std::collections::HashMap;
use std::time::Duration;
use tokio::sync::Mutex as AsyncMutex;
use tokio_util::sync::CancellationToken;

// ── FD cache ─────────────────────────────────────────────────────────────────

const DEFAULT_REMOTE_DISK_FD_CACHE_MAX_ENTRIES: usize = 131_072;
const DEFAULT_O_DIRECT_ALIGNMENT_FALLBACK: usize = 4096;

static REMOTE_DISK_FD_CACHE_MAX_ENTRIES: Lazy<usize> = Lazy::new(|| {
    std::env::var(env::DYN_KVBM_REMOTE_DISK_FD_CACHE_MAX_ENTRIES)
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(DEFAULT_REMOTE_DISK_FD_CACHE_MAX_ENTRIES)
});

static REMOTE_DISK_FD_CACHE: Lazy<AsyncMutex<RemoteDiskFdCache>> =
    Lazy::new(|| AsyncMutex::new(RemoteDiskFdCache::new(*REMOTE_DISK_FD_CACHE_MAX_ENTRIES)));

#[derive(Debug, Clone, Copy)]
struct RemoteDiskAlignmentConfig {
    quantum: usize,
    validate: bool,
}

static REMOTE_DISK_ALIGNMENT_CONFIG: Lazy<RemoteDiskAlignmentConfig> = Lazy::new(|| {
    let page_size = nix::unistd::sysconf(nix::unistd::SysconfVar::PAGE_SIZE)
        .ok()
        .flatten()
        .and_then(|v| usize::try_from(v).ok())
        .filter(|v| *v > 0)
        .unwrap_or(DEFAULT_O_DIRECT_ALIGNMENT_FALLBACK);

    let alignment_override = std::env::var(env::DYN_KVBM_REMOTE_DISK_ALIGNMENT_BYTES)
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .filter(|v| *v > 0);

    let quantum = alignment_override.unwrap_or(page_size);
    let validate = std::env::var(env::DYN_KVBM_REMOTE_DISK_VALIDATE_ALIGNMENT)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    tracing::info!(
        target: "kvbm-g4",
        page_size,
        alignment_quantum = quantum,
        validate_alignment = validate,
        "remote disk alignment config initialized"
    );

    RemoteDiskAlignmentConfig { quantum, validate }
});

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct RemoteDiskFdCacheKey {
    path: String,
    use_odirect: bool,
}

#[derive(Debug)]
struct RemoteDiskFdCacheEntry {
    storage: Arc<SyncMutex<RemoteDiskStorage>>,
    last_access_tick: u64,
}

#[derive(Debug)]
struct RemoteDiskFdCache {
    entries: HashMap<RemoteDiskFdCacheKey, RemoteDiskFdCacheEntry>,
    access_tick: u64,
    max_entries: usize,
}

impl RemoteDiskFdCache {
    fn new(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            access_tick: 0,
            max_entries,
        }
    }

    fn next_tick(&mut self) -> u64 {
        self.access_tick = self.access_tick.wrapping_add(1);
        self.access_tick
    }

    fn get(&mut self, key: &RemoteDiskFdCacheKey) -> Option<Arc<SyncMutex<RemoteDiskStorage>>> {
        let tick = self.next_tick();
        let entry = self.entries.get_mut(key)?;
        entry.last_access_tick = tick;
        Some(entry.storage.clone())
    }

    fn insert(
        &mut self,
        key: RemoteDiskFdCacheKey,
        storage: Arc<SyncMutex<RemoteDiskStorage>>,
    ) -> Arc<SyncMutex<RemoteDiskStorage>> {
        if self.max_entries > 0 && self.entries.len() >= self.max_entries {
            self.evict_lru();
        }

        let tick = self.next_tick();
        self.entries.insert(
            key,
            RemoteDiskFdCacheEntry {
                storage: storage.clone(),
                last_access_tick: tick,
            },
        );
        storage
    }

    fn evict_lru(&mut self) {
        let Some(lru_key) = self
            .entries
            .iter()
            .min_by_key(|(_, entry)| entry.last_access_tick)
            .map(|(k, _)| k.clone())
        else {
            return;
        };

        self.entries.remove(&lru_key);
    }

    fn clear(&mut self) {
        self.entries.clear();
        self.access_tick = 0;
    }
}

/// Drop all cached remote-disk file descriptors and their NIXL registrations.
///
/// Call this between benchmark sweep points or whenever the NIXL agent/backend
/// changes so that subsequent transfers re-open and re-register files against
/// the current agent.
pub async fn clear_remote_disk_fd_cache() {
    REMOTE_DISK_FD_CACHE.lock().await.clear();
}

static FD_OPEN_DURATIONS: Lazy<parking_lot::Mutex<Vec<Duration>>> =
    Lazy::new(|| parking_lot::Mutex::new(Vec::new()));

/// Drain and return all recorded per-file open+register durations.
pub fn take_fd_open_durations() -> Vec<Duration> {
    std::mem::take(&mut *FD_OPEN_DURATIONS.lock())
}

async fn get_or_open_remote_disk_storage(
    agent: &NixlAgent,
    path: &str,
    block_size: usize,
    create: bool,
    use_odirect: bool,
    preallocate: bool,
) -> Result<Arc<SyncMutex<RemoteDiskStorage>>, TransferError> {
    let key = RemoteDiskFdCacheKey {
        path: path.to_string(),
        use_odirect,
    };

    {
        let mut cache = REMOTE_DISK_FD_CACHE.lock().await;
        if let Some(storage) = cache.get(&key) {
            return Ok(storage);
        }
    }

    let fd_start = std::time::Instant::now();
    let mut storage =
        RemoteDiskStorage::open(path, block_size, create, use_odirect, preallocate).map_err(|e| {
            TransferError::ExecutionError(format!(
                "Failed to {} RemoteDiskStorage at {}: {:?}",
                if create { "create" } else { "open" },
                path,
                e
            ))
        })?;
    storage.nixl_register(agent, None).map_err(|e| {
        TransferError::ExecutionError(format!("Failed to register disk storage {}: {:?}", path, e))
    })?;
    FD_OPEN_DURATIONS.lock().push(fd_start.elapsed());

    let storage = Arc::new(SyncMutex::new(storage));

    let mut cache = REMOTE_DISK_FD_CACHE.lock().await;
    if let Some(existing) = cache.get(&key) {
        return Ok(existing);
    }

    Ok(cache.insert(key, storage))
}

// ── Public entry point ────────────────────────────────────────────────────────

/// Execute a disk transfer via NIXL (POSIX or GDS-MT backend).
///
/// `pub(crate)` — called exclusively from [`super::nixl::execute_remote_transfer`].
pub(crate) async fn execute_disk_transfer<LB>(
    direction: RemoteTransferDirection,
    descriptors: &[RemoteBlockDescriptor],
    local_blocks: &[LB],
    block_size: usize,
    ctx: &RemoteTransferContext,
    cancel_token: &CancellationToken,
) -> Result<(), TransferError>
where
    LB: ReadableBlock + WritableBlock + Local,
    <LB as StorageTypeProvider>::StorageType: NixlDescriptor,
{
    let nixl_agent_arc = ctx.nixl_agent();
    let agent = nixl_agent_arc
        .as_ref()
        .as_ref()
        .ok_or_else(|| TransferError::ExecutionError("NIXL agent not available".to_string()))?;
    execute_disk_transfer_nixl(
        agent,
        direction,
        descriptors,
        local_blocks,
        block_size,
        ctx,
        cancel_token,
    )
    .await
}

// ── NIXL POSIX / GDS-MT path ──────────────────────────────────────────────────

async fn execute_disk_transfer_nixl<LB>(
    agent: &NixlAgent,
    direction: RemoteTransferDirection,
    descriptors: &[RemoteBlockDescriptor],
    local_blocks: &[LB],
    block_size: usize,
    ctx: &RemoteTransferContext,
    cancel_token: &CancellationToken,
) -> Result<(), TransferError>
where
    LB: ReadableBlock + WritableBlock + Local,
    <LB as StorageTypeProvider>::StorageType: NixlDescriptor,
{
    let num_blocks = descriptors.len();
    let op = if matches!(direction, RemoteTransferDirection::Offload) {
        "write"
    } else {
        "read"
    };
    let base = ctx.base_path().unwrap_or("(none)");
    tracing::info!(
        target: "kvbm-diag",
        direction = op,
        base_path = base,
        num_blocks,
        block_size,
        "Disk transfer starting"
    );

    // For Offload (write): create files
    // For Onboard (read): open existing files
    let create_files = matches!(direction, RemoteTransferDirection::Offload);

    // Determine per-direction backend from transfer flags.
    //
    // Offload: use GDS_MT when DISK_FLAG_GDS_WRITE is set, else POSIX.
    // Onboard: use GDS_MT when DISK_FLAG_GDS_READ is set AND the backend is
    //          available; fall back to POSIX otherwise.
    use crate::block_manager::config::{DISK_FLAG_GDS_READ, DISK_FLAG_GDS_WRITE};
    let flags = ctx.disk_transfer_flags();
    let gds_write = flags & DISK_FLAG_GDS_WRITE != 0;
    let gds_read = flags & DISK_FLAG_GDS_READ != 0;

    // Resolve the GDS_MT backend handle once (None if not loaded in agent).
    let gds_backend = agent.get_backend("GDS_MT");

    // Optional: allow POSIX backend to also open files with O_DIRECT.
    // This is independent from backend selection (GDS_MT vs POSIX).
    let posix_odirect = std::env::var(env::DYN_KVBM_REMOTE_DISK_O_DIRECT)
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false);

    // Determine whether this direction will use GDS backend.
    let use_gds_backend = if create_files {
        gds_write
    } else {
        gds_read && gds_backend.is_some()
    };

    // Enable O_DIRECT when using GDS backend, and optionally for POSIX when
    // DYN_KVBM_REMOTE_DISK_O_DIRECT is set.
    let use_odirect = use_gds_backend || (!use_gds_backend && posix_odirect);

    let alignment_cfg = *REMOTE_DISK_ALIGNMENT_CONFIG;
    if use_odirect && alignment_cfg.validate && !block_size.is_multiple_of(alignment_cfg.quantum) {
        return Err(TransferError::ExecutionError(format!(
            "O_DIRECT alignment validation failed: block_size must be {}-byte aligned (got {}). \
             Set {}=false to disable validation.",
            alignment_cfg.quantum, block_size, env::DYN_KVBM_REMOTE_DISK_VALIDATE_ALIGNMENT
        )));
    }

    // Use a scope block to ensure all non-Send types are dropped before await
    // (OptArgs contains NonNull which is !Send)
    let (xfer_req, still_pending, _disk_storages, bounce_storage, _bounce_reg) = {
        let worker_id = ctx.worker_id() as usize;
        let world_size = ctx.world_size();

        let mut file_paths: Vec<String> = Vec::with_capacity(num_blocks);
        for desc in descriptors.iter() {
            let file_path = match desc.key() {
                RemoteKey::Disk(disk_key) => {
                    let hash = desc.sequence_hash().ok_or_else(|| {
                        TransferError::ExecutionError(
                            "Disk descriptor missing sequence_hash metadata".to_string(),
                        )
                    })?;
                    let base = ctx.base_path().unwrap_or(&disk_key.path);
                    format!("{}/{:016x}_{}_{}", base, hash, worker_id, world_size)
                }
                _ => {
                    return Err(TransferError::IncompatibleTypes(
                        "Expected Disk key for disk storage transfer".to_string(),
                    ));
                }
            };
            file_paths.push(file_path);
        }

        if let Some(first) = file_paths.first() {
            tracing::info!(
                target: "kvbm-diag",
                direction = op,
                first_file = %first,
                num_blocks,
                "Disk transfer files (showing first)"
            );
        }

        let disk_storages = {
            let mut storages: Vec<Arc<SyncMutex<RemoteDiskStorage>>> =
                Vec::with_capacity(num_blocks);
            for path in &file_paths {
                let storage = get_or_open_remote_disk_storage(
                    agent,
                    path,
                    block_size,
                    create_files,
                    use_odirect,
                    use_gds_backend,
                )
                .await?;
                storages.push(storage);
            }
            storages
        };

        // Allocate a contiguous, page-aligned bounce buffer when requested.
        // Useful when user block buffers are not page-aligned (required for O_DIRECT).
        let use_bounce_buf = std::env::var(env::DYN_KVBM_NIXL_BOUNCE_BUFFER)
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);

        let (bounce_storage, _bounce_reg_handle): (
            Option<nixl_sys::SystemStorage>,
            Option<nixl_sys::RegistrationHandle>,
        ) = if use_bounce_buf {
            let buf = nixl_sys::SystemStorage::new(num_blocks * block_size).map_err(|e| {
                TransferError::ExecutionError(format!("Failed to allocate bounce buffer: {:?}", e))
            })?;
            let handle = agent.register_memory(&buf, None).map_err(|e| {
                TransferError::ExecutionError(format!(
                    "Failed to register bounce buffer: {:?}",
                    e
                ))
            })?;
            (Some(buf), Some(handle))
        } else {
            (None, None)
        };

        // Build transfer descriptor lists for disk
        let mut src_dl = XferDescList::new(MemType::Dram).map_err(|e| {
            TransferError::ExecutionError(format!("Failed to create src_dl: {:?}", e))
        })?;
        let mut dst_dl = XferDescList::new(MemType::File).map_err(|e| {
            TransferError::ExecutionError(format!("Failed to create dst_dl: {:?}", e))
        })?;

        for (i, (block, disk_storage)) in local_blocks.iter().zip(disk_storages.iter()).enumerate() {
            let addr = if let Some(ref storage) = bounce_storage {
                unsafe { MemoryRegion::as_ptr(storage) as usize + i * block_size }
            } else {
                let block_view = block.block_data().block_view()?;
                unsafe { block_view.as_ptr() as usize }
            };

            if bounce_storage.is_none() && use_odirect && alignment_cfg.validate && !addr.is_multiple_of(alignment_cfg.quantum)
            {
                return Err(TransferError::ExecutionError(format!(
                    "O_DIRECT alignment validation failed: host buffer address must be {}-byte aligned; got 0x{:x}. \
                     Set {}=false to disable validation.",
                    alignment_cfg.quantum, addr, env::DYN_KVBM_REMOTE_DISK_VALIDATE_ALIGNMENT
                )));
            }

            // Add DRAM source descriptor
            let _ = src_dl.add_desc(addr, block_size, 0);

            // Add FILE destination descriptor using the actual file descriptor
            let fd = disk_storage.lock().fd();
            let _ = dst_dl.add_desc(0, block_size, fd);
        }

        // Determine the transfer operation
        let xfer_op = match direction {
            RemoteTransferDirection::Offload => XferOp::Write,
            RemoteTransferDirection::Onboard => XferOp::Read,
        };

        // Build OptArgs inside scope block so it's dropped before any await.
        // OptArgs contains NonNull which is !Send; it must not be held across await.
        // Offload: always POSIX when gds_write=false; GDS_MT when true.
        // Onboard: GDS_MT if available, else let NIXL fall through to POSIX.
        let opt_args: Option<OptArgs> = {
            let backend_name = if create_files {
                if gds_write {
                    Some("GDS_MT")
                } else {
                    Some("POSIX")
                }
            } else if gds_read {
                if gds_backend.is_some() {
                    Some("GDS_MT")
                } else {
                    Some("POSIX")
                }
            } else {
                Some("POSIX")
            };
            backend_name.and_then(|name| {
                let backend = agent.get_backend(name)?;
                let mut opt = OptArgs::new().ok()?;
                opt.add_backend(&backend).ok()?;
                Some(opt)
            })
        };
        let opt_ref = opt_args.as_ref();

        // Create transfer request, pinning the backend via OptArgs when set.
        let agent_name = agent.name();
        let xfer_req = agent
            .create_xfer_req(xfer_op, &src_dl, &dst_dl, &agent_name, opt_ref)
            .map_err(|e| {
                TransferError::ExecutionError(format!("Failed to create xfer_req: {:?}", e))
            })?;

        let still_pending = agent.post_xfer_req(&xfer_req, opt_ref).map_err(|e| {
            TransferError::ExecutionError(format!("Failed to post xfer_req: {:?}", e))
        })?;

        (xfer_req, still_pending, disk_storages, bounce_storage, _bounce_reg_handle)
    };

    if still_pending {
        use tracing::Instrument;
        let nixl_span = tracing::info_span!(
            "nixl_io",
            otel.name = match direction {
                RemoteTransferDirection::Onboard => "kvbm.nixl_read",
                RemoteTransferDirection::Offload => "kvbm.nixl_write",
            },
            description = "NIXL disk I/O (post + completion wait)",
            num_blocks,
            direction = op,
        );
        super::nixl::poll_transfer_completion(agent, &xfer_req, cancel_token)
            .instrument(nixl_span)
            .await?;
    }

    // For onboard with bounce buffer: copy from bounce into actual block views.
    if let Some(ref storage) = bounce_storage {
        let src_base = unsafe { MemoryRegion::as_ptr(storage) };
        for (i, block) in local_blocks.iter().enumerate() {
            let block_view = block.block_data().block_view()?;
            // Safety: NIXL has completed the transfer; the backing memory is
            // exclusively owned by this transfer at this point. We write through
            // the raw pointer because &[LB] does not give us &mut access.
            let dst = unsafe { block_view.as_ptr() as *mut u8 };
            unsafe {
                std::ptr::copy_nonoverlapping(src_base.add(i * block_size), dst, block_size);
            }
        }
    }

    tracing::debug!(
        "Disk transfer complete: {} blocks, direction={:?}, bounce={}",
        num_blocks,
        direction,
        bounce_storage.is_some(),
    );

    Ok(())
}
