// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Runtime-loaded worker-selector plugins.
//!
//! Plugins are trusted native code loaded when a KV router is constructed. The built-in
//! selector does not pass through this module.

use std::{
    cell::RefCell,
    collections::HashMap,
    env,
    ffi::c_void,
    fmt,
    mem::MaybeUninit,
    path::{Path, PathBuf},
    ptr::NonNull,
};

use anyhow::{Context, bail, ensure};
use dynamo_worker_selector_plugin_api::{
    ABI_VERSION_V1, ByteSliceV1, CAPACITY_UNAVAILABLE, ENTRYPOINT_SYMBOL_V1,
    INPUT_HAS_EXPECTED_OUTPUT_TOKENS, INPUT_HAS_REQUEST_ID, INPUT_HAS_SESSION_ID,
    INPUT_HAS_SHARED_CACHE_HITS, INPUT_TRACKS_PREFILL_TOKENS, ROUTER_ROLE_DECODE,
    ROUTER_ROLE_PREFILL, SELECTION_MODE_QUERY_ONLY, SELECTION_MODE_TRACKED,
    SELECTION_MODE_TRACKED_WITH_ADMISSION, STATUS_ERROR, STATUS_OK, STATUS_USE_DEFAULT,
    WorkerSelectorCandidateV1, WorkerSelectorCreateV1, WorkerSelectorDestroyV1,
    WorkerSelectorEntrypointV1, WorkerSelectorErrorBufferV1, WorkerSelectorInputV1,
    WorkerSelectorPluginHeaderV1, WorkerSelectorPluginV1, WorkerSelectorSelectV1,
};
use libloading::Library;

use super::{
    KvSchedulerError, RoutingEligibility, ScheduleMode, SchedulingRequest,
    config::KvRouterConfig,
    selector::{DefaultWorkerSelector, WorkerSelector},
};
use crate::{
    TargetWorkerSelector,
    protocols::{WorkerConfigLike, WorkerId, WorkerWithDpRank},
};

pub const DYN_ROUTER_WORKER_SELECTOR_PLUGIN: &str = "DYN_ROUTER_WORKER_SELECTOR_PLUGIN";
pub const DYN_ROUTER_WORKER_SELECTOR_CONFIG: &str = "DYN_ROUTER_WORKER_SELECTOR_CONFIG";

const ERROR_BUFFER_SIZE: usize = 1024;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimePluginRouterRole {
    Decode,
    Prefill,
}

impl RuntimePluginRouterRole {
    fn as_abi(self) -> u32 {
        match self {
            Self::Decode => ROUTER_ROLE_DECODE,
            Self::Prefill => ROUTER_ROLE_PREFILL,
        }
    }

    fn worker_type(self) -> &'static str {
        match self {
            Self::Decode => "decode",
            Self::Prefill => "prefill",
        }
    }
}

#[derive(Clone, Copy)]
struct PluginCallbacks {
    create: WorkerSelectorCreateV1,
    select: WorkerSelectorSelectV1,
    destroy: WorkerSelectorDestroyV1,
}

/// Configuration for one runtime worker-selector plugin.
#[derive(Clone, Debug)]
pub struct RuntimePluginConfig {
    path: PathBuf,
    config: Vec<u8>,
}

impl RuntimePluginConfig {
    /// Capture plugin configuration from the process environment without loading native code.
    pub fn from_env() -> anyhow::Result<Option<Self>> {
        let path = env::var_os(DYN_ROUTER_WORKER_SELECTOR_PLUGIN);
        let config = match env::var(DYN_ROUTER_WORKER_SELECTOR_CONFIG) {
            Ok(value) => Some(value.into_bytes()),
            Err(env::VarError::NotPresent) => None,
            Err(env::VarError::NotUnicode(_)) => {
                bail!("{DYN_ROUTER_WORKER_SELECTOR_CONFIG} must contain valid Unicode")
            }
        };

        let Some(path) = path else {
            return if config.is_some() {
                Err(anyhow::anyhow!(
                    "{DYN_ROUTER_WORKER_SELECTOR_CONFIG} requires {DYN_ROUTER_WORKER_SELECTOR_PLUGIN}"
                ))
            } else {
                Ok(None)
            };
        };
        Ok(Some(Self {
            path: PathBuf::from(path),
            config: config.unwrap_or_default(),
        }))
    }

    /// Load the captured plugin configuration.
    ///
    /// # Safety
    ///
    /// The configured library must implement ABI v1 exactly.
    pub unsafe fn load(
        &self,
        router_role: RuntimePluginRouterRole,
        kv_router_config: KvRouterConfig,
    ) -> anyhow::Result<RuntimePluginSelector> {
        unsafe {
            RuntimePluginSelector::load(&self.path, &self.config, router_role, kv_router_config)
        }
    }
}

/// A trusted native worker selector loaded from a `cdylib`.
pub struct RuntimePluginSelector {
    state: NonNull<c_void>,
    candidate_scratch: RefCell<Vec<WorkerSelectorCandidateV1>>,
    callbacks: PluginCallbacks,
    _library: Library,
    path: PathBuf,
    default_fallback: DefaultWorkerSelector,
}

impl RuntimePluginSelector {
    /// Load a trusted plugin from an absolute path.
    ///
    /// # Safety
    ///
    /// The library must implement ABI v1 exactly. Loading native code and assigning a type to its
    /// exported symbol cannot be verified by the host. The library remains loaded until the
    /// selector destroys its plugin state.
    unsafe fn load(
        path: &Path,
        config: &[u8],
        router_role: RuntimePluginRouterRole,
        kv_router_config: KvRouterConfig,
    ) -> anyhow::Result<Self> {
        let path = path.to_path_buf();
        let (library, callbacks) = unsafe { load_plugin(&path) }?;

        let mut state = std::ptr::null_mut();
        let mut error_bytes = [MaybeUninit::uninit(); ERROR_BUFFER_SIZE];
        let mut error = error_buffer(&mut error_bytes);
        let status = unsafe {
            (callbacks.create)(
                ByteSliceV1::from_slice(config),
                router_role.as_abi(),
                &mut state,
                &mut error,
            )
        };
        if status != STATUS_OK {
            bail!(
                "worker-selector plugin {path:?} failed to initialize (status {status}): {}",
                error_message(&error_bytes, &error)
            );
        }
        let state = NonNull::new(state)
            .with_context(|| format!("worker-selector plugin {path:?} returned a null state"))?;

        Ok(Self {
            state,
            candidate_scratch: RefCell::new(Vec::new()),
            callbacks,
            _library: library,
            path,
            default_fallback: DefaultWorkerSelector::new(
                Some(kv_router_config),
                router_role.worker_type(),
            ),
        })
    }
}

impl fmt::Debug for RuntimePluginSelector {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("RuntimePluginSelector")
            .field("path", &self.path)
            .finish_non_exhaustive()
    }
}

// SAFETY: The queue actor owns a selector and serializes calls to its plugin state. ABI v1 requires
// that state to be movable between threads, and the selector owns its library handle.
unsafe impl Send for RuntimePluginSelector {}

impl Drop for RuntimePluginSelector {
    fn drop(&mut self) {
        unsafe { (self.callbacks.destroy)(self.state.as_ptr()) };
    }
}

impl<C: WorkerConfigLike> TargetWorkerSelector<C> for RuntimePluginSelector {
    fn select_worker(
        &self,
        workers: &HashMap<WorkerId, C>,
        request: &SchedulingRequest,
        eligibility: RoutingEligibility<'_>,
        block_size: u32,
    ) -> Result<WorkerWithDpRank, KvSchedulerError> {
        let has_tier_overlap = !request.overlap.tier_overlap_blocks.device.is_empty()
            || !request.overlap.tier_overlap_blocks.host_pinned.is_empty()
            || !request.overlap.tier_overlap_blocks.disk.is_empty();
        let mut candidates = self.candidate_scratch.borrow_mut();
        candidates.clear();
        eligibility.for_each_eligible_worker_rank(workers, |worker, config| {
            let effective_overlap = request.effective_overlap_blocks_for(worker);
            let device_overlap = request
                .overlap
                .tier_overlap_blocks
                .device
                .get(&worker)
                .copied()
                .unwrap_or_else(|| {
                    if has_tier_overlap {
                        0
                    } else {
                        effective_overlap.round().max(0.0) as usize
                    }
                });
            let load = request
                .worker_loads
                .get(&worker)
                .copied()
                .unwrap_or_default();
            let stable_routing_id = config
                .stable_routing_id()
                .map(|value| ByteSliceV1::from_slice(value.as_bytes()))
                .unwrap_or_else(ByteSliceV1::empty);
            candidates.push(WorkerSelectorCandidateV1 {
                worker_id: worker.worker_id,
                dp_rank: worker.dp_rank,
                _padding: 0,
                cached_tokens: to_u64(request.effective_cached_tokens_for(worker)),
                effective_overlap_blocks: effective_overlap,
                device_overlap_blocks: to_u64(device_overlap),
                host_pinned_overlap_blocks: to_u64(
                    request
                        .overlap
                        .tier_overlap_blocks
                        .host_pinned
                        .get(&worker)
                        .copied()
                        .unwrap_or(0),
                ),
                disk_overlap_blocks: to_u64(
                    request
                        .overlap
                        .tier_overlap_blocks
                        .disk
                        .get(&worker)
                        .copied()
                        .unwrap_or(0),
                ),
                shared_cache_beyond_device_blocks: request
                    .shared_cache_hits
                    .as_ref()
                    .map(|hits| {
                        u64::from(hits.hits_beyond(device_overlap.min(u32::MAX as usize) as u32))
                    })
                    .unwrap_or(0),
                active_prefill_tokens: to_u64(load.active_prefill_tokens),
                active_decode_blocks: to_u64(load.active_decode_blocks),
                additional_active_blocks: to_u64(load.additional_active_blocks),
                stable_routing_id,
                preferred_taint_multiplier: request
                    .routing_constraints
                    .preferred_taint_multiplier(config.taints())
                    .unwrap_or(1.0),
                total_kv_blocks: config.total_kv_blocks().unwrap_or(CAPACITY_UNAVAILABLE),
                max_num_batched_tokens: config
                    .max_num_batched_tokens()
                    .unwrap_or(CAPACITY_UNAVAILABLE),
            });
        });
        if candidates.is_empty() {
            return Err(KvSchedulerError::NoEndpoints);
        }

        let mut flags = 0;
        let request_id = request.mode.request_id();
        if request_id.is_some() {
            flags |= INPUT_HAS_REQUEST_ID;
        }
        if request.session_id.is_some() {
            flags |= INPUT_HAS_SESSION_ID;
        }
        if request.expected_output_tokens.is_some() {
            flags |= INPUT_HAS_EXPECTED_OUTPUT_TOKENS;
        }
        if request.track_prefill_tokens {
            flags |= INPUT_TRACKS_PREFILL_TOKENS;
        }
        if request.shared_cache_hits.is_some() {
            flags |= INPUT_HAS_SHARED_CACHE_HITS;
        }
        let selection_mode = match &request.mode {
            ScheduleMode::QueryOnly { .. } => SELECTION_MODE_QUERY_ONLY,
            ScheduleMode::Tracked { .. } => SELECTION_MODE_TRACKED,
            ScheduleMode::TrackedWithAdmission { .. } => SELECTION_MODE_TRACKED_WITH_ADMISSION,
        };
        let input = WorkerSelectorInputV1 {
            struct_size: size_of::<WorkerSelectorInputV1>() as u32,
            candidate_size: size_of::<WorkerSelectorCandidateV1>() as u32,
            flags,
            block_size,
            selection_mode,
            isl_tokens: to_u64(request.isl_tokens),
            expected_output_tokens: u64::from(request.expected_output_tokens.unwrap_or_default()),
            request_id: request_id
                .map(|value| ByteSliceV1::from_slice(value.as_bytes()))
                .unwrap_or_else(ByteSliceV1::empty),
            session_id: request
                .session_id
                .as_deref()
                .map(|value| ByteSliceV1::from_slice(value.as_bytes()))
                .unwrap_or_else(ByteSliceV1::empty),
            candidates: candidates.as_ptr(),
            candidate_count: candidates.len(),
        };

        let mut selected_index = usize::MAX;
        let mut error_bytes = [MaybeUninit::uninit(); ERROR_BUFFER_SIZE];
        let mut error = error_buffer(&mut error_bytes);
        let status = unsafe {
            (self.callbacks.select)(self.state.as_ptr(), &input, &mut selected_index, &mut error)
        };
        if status == STATUS_USE_DEFAULT {
            return self
                .default_fallback
                .select_worker(workers, request, eligibility, block_size)
                .map(|result| result.worker);
        }
        if status != STATUS_OK {
            return Err(KvSchedulerError::InitFailed(format!(
                "worker-selector plugin failed (status {status}): {}",
                error_message(&error_bytes, &error)
            )));
        }
        let selected = candidates.get(selected_index).ok_or_else(|| {
            KvSchedulerError::InitFailed(format!(
                "worker-selector plugin failed (status {STATUS_ERROR}): plugin returned candidate index {selected_index} for {} candidates",
                candidates.len()
            ))
        })?;
        Ok(WorkerWithDpRank::new(selected.worker_id, selected.dp_rank))
    }
}

unsafe fn load_plugin(path: &Path) -> anyhow::Result<(Library, PluginCallbacks)> {
    ensure!(
        path.is_absolute(),
        "{DYN_ROUTER_WORKER_SELECTOR_PLUGIN} must be an absolute path, got {path:?}"
    );
    let library = unsafe { open_library(path) }
        .with_context(|| format!("failed to load worker-selector plugin {path:?}"))?;
    let entrypoint = unsafe {
        *library
            .get::<WorkerSelectorEntrypointV1>(ENTRYPOINT_SYMBOL_V1)
            .with_context(|| {
                format!(
                    "worker-selector plugin {path:?} does not export dynamo_worker_selector_plugin_v1"
                )
            })?
    };
    let descriptor = unsafe { entrypoint() };
    let callbacks = unsafe { validate_descriptor(descriptor) }
        .with_context(|| format!("invalid worker-selector plugin descriptor from {path:?}"))?;
    Ok((library, callbacks))
}

#[cfg(unix)]
unsafe fn open_library(path: &Path) -> Result<Library, libloading::Error> {
    use libloading::os::unix::{Library as UnixLibrary, RTLD_LOCAL, RTLD_NOW};

    unsafe { UnixLibrary::open(Some(path), RTLD_NOW | RTLD_LOCAL) }.map(Library::from)
}

#[cfg(not(unix))]
unsafe fn open_library(path: &Path) -> Result<Library, libloading::Error> {
    unsafe { Library::new(path) }
}

unsafe fn validate_descriptor(
    descriptor: *const WorkerSelectorPluginV1,
) -> anyhow::Result<PluginCallbacks> {
    ensure!(!descriptor.is_null(), "plugin returned a null descriptor");
    let header = unsafe { &*descriptor.cast::<WorkerSelectorPluginHeaderV1>() };
    ensure!(
        header.abi_version == ABI_VERSION_V1,
        "plugin uses ABI {}; expected {ABI_VERSION_V1}",
        header.abi_version
    );
    let required = size_of::<WorkerSelectorPluginV1>() as u32;
    ensure!(
        header.struct_size >= required,
        "plugin descriptor is {} bytes; ABI v1 requires at least {required}",
        header.struct_size
    );
    let descriptor = unsafe { &*descriptor };
    Ok(PluginCallbacks {
        create: descriptor
            .create
            .context("plugin is missing its create callback")?,
        select: descriptor
            .select
            .context("plugin is missing its select callback")?,
        destroy: descriptor
            .destroy
            .context("plugin is missing its destroy callback")?,
    })
}

fn error_buffer(bytes: &mut [MaybeUninit<u8>; ERROR_BUFFER_SIZE]) -> WorkerSelectorErrorBufferV1 {
    WorkerSelectorErrorBufferV1 {
        ptr: bytes.as_mut_ptr().cast(),
        capacity: bytes.len(),
        written: 0,
    }
}

fn error_message(
    bytes: &[MaybeUninit<u8>; ERROR_BUFFER_SIZE],
    error: &WorkerSelectorErrorBufferV1,
) -> String {
    let written = error.written.min(bytes.len());
    if written == 0 {
        return "plugin returned no error message".to_string();
    }
    // SAFETY: ABI v1 requires a plugin to initialize exactly the `written` prefix before returning
    // a non-success status. The host never reads this storage on the success path.
    let initialized = unsafe { std::slice::from_raw_parts(bytes.as_ptr().cast::<u8>(), written) };
    String::from_utf8_lossy(initialized).into_owned()
}

fn to_u64(value: usize) -> u64 {
    u64::try_from(value).unwrap_or(u64::MAX)
}

#[cfg(test)]
mod tests;
