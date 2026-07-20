// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stable C ABI and safe Rust authoring API for Dynamo worker-selector plugins.
//!
//! Plugins are trusted native code. ABI pointers are borrowed for the duration of a callback;
//! no Rust-owned allocation crosses the dynamic-library boundary.

use std::{
    ffi::c_void,
    mem::{align_of, size_of},
    ops::{BitOr, BitOrAssign},
    slice, str,
};

pub const ABI_VERSION_V1: u32 = 1;
pub const ENTRYPOINT_SYMBOL_V1: &[u8] = b"dynamo_worker_selector_plugin_v1\0";

pub const STATUS_OK: i32 = 0;
pub const STATUS_INVALID_INPUT: i32 = 1;
pub const STATUS_ERROR: i32 = 2;
pub const STATUS_PANICKED: i32 = 3;
/// Delegate to the host's default selector; candidate_index_out is ignored.
pub const STATUS_USE_DEFAULT: i32 = 4;

/// ABI encoding for an unavailable optional candidate capacity.
pub const CAPACITY_UNAVAILABLE: u64 = u64::MAX;

pub const INPUT_HAS_REQUEST_ID: u32 = 1 << 0;
pub const INPUT_HAS_SESSION_ID: u32 = 1 << 1;
pub const INPUT_HAS_EXPECTED_OUTPUT_TOKENS: u32 = 1 << 2;
pub const INPUT_TRACKS_PREFILL_TOKENS: u32 = 1 << 3;
pub const INPUT_HAS_SHARED_CACHE_HITS: u32 = 1 << 4;

pub const ROUTER_ROLE_UNKNOWN: u32 = 0;
pub const ROUTER_ROLE_DECODE: u32 = 1;
pub const ROUTER_ROLE_PREFILL: u32 = 2;

pub const SELECTION_MODE_QUERY_ONLY: u32 = 0;
pub const SELECTION_MODE_TRACKED: u32 = 1;
pub const SELECTION_MODE_TRACKED_WITH_ADMISSION: u32 = 2;

/// Candidate storage format used by [`WorkerSelectorInputV1`].
pub const CANDIDATE_FORMAT_COLUMNAR_V1: u32 = 1;

/// Candidate inputs a plugin needs for selection.
///
/// Dynamo materializes only the requested inputs. Any non-empty requirement includes
/// [`Self::IDENTITY`] so a returned index always maps to a worker and data-parallel rank.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CandidateInputs(u64);

impl CandidateInputs {
    pub const NONE: Self = Self(0);
    pub const IDENTITY: Self = Self(1 << 0);
    pub const CACHED_TOKENS: Self = Self(1 << 1);
    pub const CACHE_TIERS: Self = Self(1 << 2);
    pub const LOAD: Self = Self(1 << 3);
    pub const CAPACITY: Self = Self(1 << 4);
    pub const ROUTING: Self = Self(1 << 5);
    /// Dynamo's complete per-candidate cost before temperature sampling.
    pub const DEFAULT_COST: Self = Self(1 << 6);
    /// KV overlap credit used by Dynamo's default cost, measured in blocks.
    pub const DEFAULT_KV_OVERLAP: Self = Self(1 << 7);
    /// Projected decode load used by Dynamo's default cost, measured in blocks.
    pub const DEFAULT_DECODE_LOAD: Self = Self(1 << 8);
    pub const ALL: Self = Self((1 << 9) - 1);

    pub const fn bits(self) -> u64 {
        self.0
    }

    pub const fn from_bits(bits: u64) -> Option<Self> {
        if bits & !Self::ALL.0 == 0 {
            Some(Self(bits))
        } else {
            None
        }
    }

    pub const fn contains(self, other: Self) -> bool {
        self.0 & other.0 == other.0
    }

    pub const fn with_identity(self) -> Self {
        if self.0 == 0 {
            self
        } else {
            Self(self.0 | Self::IDENTITY.0)
        }
    }
}

impl BitOr for CandidateInputs {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for CandidateInputs {
    fn bitor_assign(&mut self, rhs: Self) {
        self.0 |= rhs.0;
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum RouterRole {
    Unknown,
    Decode,
    Prefill,
}

impl RouterRole {
    fn from_abi(value: u32) -> Self {
        match value {
            ROUTER_ROLE_DECODE => Self::Decode,
            ROUTER_ROLE_PREFILL => Self::Prefill,
            _ => Self::Unknown,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SelectionMode {
    Unknown,
    QueryOnly,
    Tracked,
    TrackedWithAdmission,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ByteSliceV1 {
    pub ptr: *const u8,
    pub len: usize,
}

impl ByteSliceV1 {
    pub const fn empty() -> Self {
        Self {
            ptr: std::ptr::null(),
            len: 0,
        }
    }

    pub fn from_slice(value: &[u8]) -> Self {
        Self {
            ptr: value.as_ptr(),
            len: value.len(),
        }
    }

    unsafe fn as_slice<'a>(self) -> Result<&'a [u8], &'static str> {
        if self.len == 0 {
            return Ok(&[]);
        }
        if self.ptr.is_null() {
            return Err("non-empty byte slice has a null pointer");
        }
        if self.len > isize::MAX as usize {
            return Err("byte slice is too large");
        }
        // SAFETY: The ABI contract requires the producer to keep this range alive for the call.
        Ok(unsafe { slice::from_raw_parts(self.ptr, self.len) })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WorkerSelectorCacheTiersV1 {
    pub effective_overlap_blocks: f64,
    pub device_overlap_blocks: u64,
    pub host_pinned_overlap_blocks: u64,
    pub disk_overlap_blocks: u64,
    pub shared_cache_beyond_device_blocks: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkerSelectorLoadV1 {
    pub active_prefill_tokens: u64,
    pub active_decode_blocks: u64,
    pub additional_active_blocks: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorkerSelectorCapacityV1 {
    /// Total KV capacity, or CAPACITY_UNAVAILABLE.
    pub total_kv_blocks: u64,
    /// Maximum batched-token capacity, or CAPACITY_UNAVAILABLE.
    pub max_num_batched_tokens: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct WorkerSelectorRoutingV1 {
    pub stable_routing_id: ByteSliceV1,
    /// Stock routing's multiplicative preferred-taint cost adjustment. `1.0` is neutral.
    pub preferred_taint_multiplier: f64,
}

impl WorkerSelectorCapacityV1 {
    pub fn total_kv_blocks(&self) -> Option<u64> {
        (self.total_kv_blocks != CAPACITY_UNAVAILABLE).then_some(self.total_kv_blocks)
    }

    pub fn max_num_batched_tokens(&self) -> Option<u64> {
        (self.max_num_batched_tokens != CAPACITY_UNAVAILABLE).then_some(self.max_num_batched_tokens)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct WorkerSelectorCandidateColumnsV1 {
    pub struct_size: u32,
    pub _reserved: u32,
    pub provided_inputs: u64,
    pub worker_ids: *const u64,
    pub dp_ranks: *const u32,
    pub cached_tokens: *const u64,
    pub cache_tiers: *const WorkerSelectorCacheTiersV1,
    pub loads: *const WorkerSelectorLoadV1,
    pub capacities: *const WorkerSelectorCapacityV1,
    pub routing: *const WorkerSelectorRoutingV1,
    pub default_costs: *const f64,
    pub default_kv_overlaps: *const f64,
    pub default_decode_loads: *const u64,
}

#[repr(C)]
struct WorkerSelectorCandidateColumnsHeaderV1 {
    struct_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct WorkerSelectorInputV1 {
    pub struct_size: u32,
    pub candidate_format: u32,
    pub flags: u32,
    pub block_size: u32,
    pub selection_mode: u32,
    pub isl_tokens: u64,
    pub expected_output_tokens: u64,
    pub request_id: ByteSliceV1,
    pub session_id: ByteSliceV1,
    pub candidate_columns: *const WorkerSelectorCandidateColumnsV1,
    pub candidate_count: usize,
}

#[repr(C)]
struct WorkerSelectorInputHeaderV1 {
    struct_size: u32,
    candidate_format: u32,
}

#[repr(C)]
#[derive(Debug)]
/// Callback-owned error output storage.
///
/// The host initializes `written` to zero and ignores the storage on success. Before returning a
/// non-success status, a callback that sets `written` must initialize exactly that many bytes,
/// never exceeding `capacity`.
pub struct WorkerSelectorErrorBufferV1 {
    pub ptr: *mut u8,
    pub capacity: usize,
    pub written: usize,
}

/// Construct state for one router role.
///
/// On [`STATUS_OK`], the callback must write a non-null state that remains valid for serialized
/// `select` calls and exactly one `destroy` call. On failure, it must leave or write a null state
/// and release any partially constructed resources itself. The host calls `destroy` only after a
/// successful, non-null creation.
pub type WorkerSelectorCreateV1 = unsafe extern "C" fn(
    config: ByteSliceV1,
    router_role: u32,
    state_out: *mut *mut c_void,
    error_out: *mut WorkerSelectorErrorBufferV1,
) -> i32;

pub type WorkerSelectorSelectV1 = unsafe extern "C" fn(
    state: *mut c_void,
    input: *const WorkerSelectorInputV1,
    candidate_index_out: *mut usize,
    error_out: *mut WorkerSelectorErrorBufferV1,
) -> i32;

pub type WorkerSelectorDestroyV1 = unsafe extern "C" fn(state: *mut c_void);

/// Return the optional candidate inputs required by this configured plugin state.
///
/// The host calls this once after successful creation and caches the result. On [`STATUS_OK`], the
/// callback must initialize `candidate_inputs_out` with a valid [`CandidateInputs`] bitset.
pub type WorkerSelectorRequiredCandidateInputsV1 = unsafe extern "C" fn(
    state: *mut c_void,
    candidate_inputs_out: *mut u64,
    error_out: *mut WorkerSelectorErrorBufferV1,
) -> i32;

#[repr(C)]
#[derive(Clone, Copy)]
/// Raw ABI v1 plugin descriptor.
///
/// `select` calls for one plugin state are serialized, though separate states may be active on
/// different threads. `select` runs synchronously on the scheduler actor's request hot path and
/// must be bounded and non-blocking. Every input pointer is borrowed only for its callback.
/// Callbacks must not unwind across the ABI, and `destroy` must stop plugin threads that access
/// plugin state before returning. The safe Rust export macro contains unwind panics, but a plugin
/// built with `panic=abort` can still terminate the host process.
pub struct WorkerSelectorPluginV1 {
    pub abi_version: u32,
    pub struct_size: u32,
    pub create: Option<WorkerSelectorCreateV1>,
    pub select: Option<WorkerSelectorSelectV1>,
    pub destroy: Option<WorkerSelectorDestroyV1>,
    pub required_candidate_inputs: Option<WorkerSelectorRequiredCandidateInputsV1>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct WorkerSelectorPluginHeaderV1 {
    pub abi_version: u32,
    pub struct_size: u32,
}

pub type WorkerSelectorEntrypointV1 = unsafe extern "C" fn() -> *const WorkerSelectorPluginV1;

/// Safe borrowed view passed to a plugin implementation.
#[derive(Clone, Copy)]
pub struct SelectionInput<'a> {
    raw: &'a WorkerSelectorInputV1,
    inputs: CandidateInputs,
    worker_ids: &'a [u64],
    dp_ranks: &'a [u32],
    cached_tokens: Option<&'a [u64]>,
    cache_tiers: Option<&'a [WorkerSelectorCacheTiersV1]>,
    loads: Option<&'a [WorkerSelectorLoadV1]>,
    capacities: Option<&'a [WorkerSelectorCapacityV1]>,
    routing: Option<&'a [WorkerSelectorRoutingV1]>,
    default_costs: Option<&'a [f64]>,
    default_kv_overlaps: Option<&'a [f64]>,
    default_decode_loads: Option<&'a [u64]>,
}

impl<'a> SelectionInput<'a> {
    unsafe fn from_abi(
        raw: *const WorkerSelectorInputV1,
        required_inputs: CandidateInputs,
    ) -> Result<Self, &'static str> {
        if raw.is_null() {
            return Err("selection input is null");
        }
        if !(raw as usize).is_multiple_of(align_of::<WorkerSelectorInputV1>()) {
            return Err("selection input is misaligned");
        }
        let header = unsafe { &*raw.cast::<WorkerSelectorInputHeaderV1>() };
        if header.struct_size < size_of::<WorkerSelectorInputV1>() as u32 {
            return Err("selection input is smaller than ABI v1");
        }
        if header.candidate_format != CANDIDATE_FORMAT_COLUMNAR_V1 {
            return Err("candidate format does not match columnar ABI v1");
        }
        let raw = unsafe { &*raw };
        let columns_ptr = raw.candidate_columns;
        if columns_ptr.is_null()
            || !(columns_ptr as usize)
                .is_multiple_of(align_of::<WorkerSelectorCandidateColumnsV1>())
        {
            return Err("candidate columns are null or misaligned");
        }
        let columns_header =
            unsafe { &*columns_ptr.cast::<WorkerSelectorCandidateColumnsHeaderV1>() };
        if columns_header.struct_size < size_of::<WorkerSelectorCandidateColumnsV1>() as u32 {
            return Err("candidate columns are smaller than ABI v1");
        }
        let columns = unsafe { &*columns_ptr };
        let inputs = CandidateInputs::from_bits(columns.provided_inputs)
            .ok_or("selection input contains unknown candidate inputs")?;
        if inputs != inputs.with_identity() {
            return Err("candidate inputs require identity");
        }
        if !inputs.contains(required_inputs) {
            return Err("selection input is missing a required candidate input");
        }
        let (worker_ids, dp_ranks) = if inputs.contains(CandidateInputs::IDENTITY) {
            (
                unsafe {
                    required_column(
                        columns.worker_ids,
                        raw.candidate_count,
                        "worker-ID column is null or invalid",
                    )
                }?,
                unsafe {
                    required_column(
                        columns.dp_ranks,
                        raw.candidate_count,
                        "data-parallel-rank column is null or invalid",
                    )
                }?,
            )
        } else {
            (&[][..], &[][..])
        };
        Ok(Self {
            raw,
            inputs,
            worker_ids,
            dp_ranks,
            cached_tokens: unsafe {
                optional_column(
                    inputs.contains(CandidateInputs::CACHED_TOKENS),
                    columns.cached_tokens,
                    raw.candidate_count,
                    "cached-token column is null or invalid",
                )
            }?,
            cache_tiers: unsafe {
                optional_column(
                    inputs.contains(CandidateInputs::CACHE_TIERS),
                    columns.cache_tiers,
                    raw.candidate_count,
                    "cache-tier column is null or invalid",
                )
            }?,
            loads: unsafe {
                optional_column(
                    inputs.contains(CandidateInputs::LOAD),
                    columns.loads,
                    raw.candidate_count,
                    "load column is null or invalid",
                )
            }?,
            capacities: unsafe {
                optional_column(
                    inputs.contains(CandidateInputs::CAPACITY),
                    columns.capacities,
                    raw.candidate_count,
                    "capacity column is null or invalid",
                )
            }?,
            routing: unsafe {
                optional_column(
                    inputs.contains(CandidateInputs::ROUTING),
                    columns.routing,
                    raw.candidate_count,
                    "routing column is null or invalid",
                )
            }?,
            default_costs: unsafe {
                optional_column(
                    inputs.contains(CandidateInputs::DEFAULT_COST),
                    columns.default_costs,
                    raw.candidate_count,
                    "default-cost column is null or invalid",
                )
            }?,
            default_kv_overlaps: unsafe {
                optional_column(
                    inputs.contains(CandidateInputs::DEFAULT_KV_OVERLAP),
                    columns.default_kv_overlaps,
                    raw.candidate_count,
                    "KV-overlap column is null or invalid",
                )
            }?,
            default_decode_loads: unsafe {
                optional_column(
                    inputs.contains(CandidateInputs::DEFAULT_DECODE_LOAD),
                    columns.default_decode_loads,
                    raw.candidate_count,
                    "decode-load column is null or invalid",
                )
            }?,
        })
    }

    pub fn block_size(&self) -> u32 {
        self.raw.block_size
    }

    pub fn isl_tokens(&self) -> u64 {
        self.raw.isl_tokens
    }

    pub fn expected_output_tokens(&self) -> Option<u64> {
        (self.raw.flags & INPUT_HAS_EXPECTED_OUTPUT_TOKENS != 0)
            .then_some(self.raw.expected_output_tokens)
    }

    pub fn tracks_prefill_tokens(&self) -> bool {
        self.raw.flags & INPUT_TRACKS_PREFILL_TOKENS != 0
    }

    pub fn has_shared_cache_hits(&self) -> bool {
        self.raw.flags & INPUT_HAS_SHARED_CACHE_HITS != 0
    }

    pub fn candidate_inputs(&self) -> CandidateInputs {
        self.inputs
    }

    pub fn selection_mode(&self) -> SelectionMode {
        match self.raw.selection_mode {
            SELECTION_MODE_QUERY_ONLY => SelectionMode::QueryOnly,
            SELECTION_MODE_TRACKED => SelectionMode::Tracked,
            SELECTION_MODE_TRACKED_WITH_ADMISSION => SelectionMode::TrackedWithAdmission,
            _ => SelectionMode::Unknown,
        }
    }

    pub fn request_id(&self) -> Option<&'a str> {
        self.optional_str(INPUT_HAS_REQUEST_ID, self.raw.request_id)
    }

    pub fn session_id(&self) -> Option<&'a str> {
        self.optional_str(INPUT_HAS_SESSION_ID, self.raw.session_id)
    }

    /// Number of eligible candidates represented by every provided column.
    pub fn candidate_count(&self) -> usize {
        self.worker_ids.len()
    }

    pub fn is_empty(&self) -> bool {
        self.worker_ids.is_empty()
    }

    /// Worker IDs, aligned by index with every other candidate column.
    pub fn worker_ids(&self) -> &'a [u64] {
        self.worker_ids
    }

    /// Data-parallel ranks, aligned by index with [`Self::worker_ids`].
    pub fn dp_ranks(&self) -> &'a [u32] {
        self.dp_ranks
    }

    pub fn cached_tokens(&self) -> Option<&'a [u64]> {
        self.cached_tokens
    }

    pub fn cache_tiers(&self) -> Option<&'a [WorkerSelectorCacheTiersV1]> {
        self.cache_tiers
    }

    pub fn loads(&self) -> Option<&'a [WorkerSelectorLoadV1]> {
        self.loads
    }

    pub fn capacities(&self) -> Option<&'a [WorkerSelectorCapacityV1]> {
        self.capacities
    }

    pub fn routing(&self) -> Option<&'a [WorkerSelectorRoutingV1]> {
        self.routing
    }

    pub fn default_costs(&self) -> Option<&'a [f64]> {
        self.default_costs
    }

    pub fn default_kv_overlaps(&self) -> Option<&'a [f64]> {
        self.default_kv_overlaps
    }

    pub fn default_decode_loads(&self) -> Option<&'a [u64]> {
        self.default_decode_loads
    }

    pub fn candidate_stable_routing_id(&self, index: usize) -> Option<&'a str> {
        let value = self.routing?.get(index)?.stable_routing_id;
        // SAFETY: Candidate strings share the callback-scoped lifetime of this input view.
        let bytes = unsafe { value.as_slice() }.ok()?;
        if bytes.is_empty() {
            None
        } else {
            str::from_utf8(bytes).ok()
        }
    }

    fn optional_str(&self, flag: u32, value: ByteSliceV1) -> Option<&'a str> {
        if self.raw.flags & flag == 0 {
            return None;
        }
        // SAFETY: `from_abi` validated the containing input and the host promises UTF-8 IDs.
        let bytes = unsafe { value.as_slice() }.ok()?;
        str::from_utf8(bytes).ok()
    }
}

unsafe fn optional_column<'a, T>(
    present: bool,
    ptr: *const T,
    len: usize,
    error: &'static str,
) -> Result<Option<&'a [T]>, &'static str> {
    if !present {
        return Ok(None);
    }
    Ok(Some(unsafe { required_column(ptr, len, error) }?))
}

unsafe fn required_column<'a, T>(
    ptr: *const T,
    len: usize,
    error: &'static str,
) -> Result<&'a [T], &'static str> {
    if len == 0 {
        return Ok(&[]);
    }
    if ptr.is_null()
        || !(ptr as usize).is_multiple_of(align_of::<T>())
        || len > isize::MAX as usize / size_of::<T>().max(1)
    {
        return Err(error);
    }
    // SAFETY: The ABI contract requires the host to keep each declared column alive for the call.
    Ok(unsafe { slice::from_raw_parts(ptr, len) })
}

/// Implement this trait in a `cdylib`, then invoke [`export_worker_selector_plugin!`].
///
/// The host may move an instance to its queue actor, but serializes calls to that instance.
pub trait WorkerSelectorPlugin: Send + Sized + 'static {
    /// Create independent state for one router role.
    fn from_config(config: &[u8], router_role: RouterRole) -> Result<Self, String>;

    /// Candidate inputs required by this configured strategy.
    ///
    /// Dynamo evaluates this once at creation and materializes only these columns. Return
    /// [`CandidateInputs::IDENTITY`] for a strategy that only needs worker identity, or
    /// [`CandidateInputs::NONE`] for a strategy that always delegates to the default.
    fn required_candidate_inputs(&self) -> CandidateInputs;

    /// Select a candidate or delegate this request to Dynamo's default selector.
    /// Calls for one instance are serialized, so stateful strategies can mutate directly.
    /// This runs synchronously on the scheduler actor's request hot path and must be bounded and
    /// non-blocking.
    fn select(&mut self, input: SelectionInput<'_>) -> Result<Selection, String>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Selection {
    /// Select an index shared by the aligned candidate columns in [`SelectionInput`].
    Candidate(usize),
    /// Delegate this request to Dynamo's configured default selector.
    UseDefault,
}

#[doc(hidden)]
pub mod __private {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    struct PluginState<T> {
        plugin: T,
        required_inputs: CandidateInputs,
    }

    pub unsafe extern "C" fn create<T: WorkerSelectorPlugin>(
        config: ByteSliceV1,
        router_role: u32,
        state_out: *mut *mut c_void,
        error_out: *mut WorkerSelectorErrorBufferV1,
    ) -> i32 {
        if state_out.is_null() {
            unsafe { write_error(error_out, "state output is null") };
            return STATUS_INVALID_INPUT;
        }
        unsafe { state_out.write(std::ptr::null_mut()) };
        let config = match unsafe { config.as_slice() } {
            Ok(config) => config,
            Err(error) => {
                unsafe { write_error(error_out, error) };
                return STATUS_INVALID_INPUT;
            }
        };
        match catch_unwind(AssertUnwindSafe(|| {
            let plugin = T::from_config(config, RouterRole::from_abi(router_role))?;
            let required_inputs = plugin.required_candidate_inputs().with_identity();
            Ok::<_, String>((plugin, required_inputs))
        })) {
            Ok(Ok((plugin, required_inputs))) => {
                let state = PluginState {
                    plugin,
                    required_inputs,
                };
                unsafe { state_out.write(Box::into_raw(Box::new(state)).cast()) };
                STATUS_OK
            }
            Ok(Err(error)) => {
                unsafe { write_error(error_out, &error) };
                STATUS_ERROR
            }
            Err(_) => {
                unsafe { write_error(error_out, "plugin panicked") };
                STATUS_PANICKED
            }
        }
    }

    pub unsafe extern "C" fn required_candidate_inputs<T: WorkerSelectorPlugin>(
        state: *mut c_void,
        candidate_inputs_out: *mut u64,
        error_out: *mut WorkerSelectorErrorBufferV1,
    ) -> i32 {
        if state.is_null() || candidate_inputs_out.is_null() {
            unsafe { write_error(error_out, "state or candidate-inputs output is null") };
            return STATUS_INVALID_INPUT;
        }
        let state = unsafe { &*state.cast::<PluginState<T>>() };
        unsafe { candidate_inputs_out.write(state.required_inputs.bits()) };
        STATUS_OK
    }

    pub unsafe extern "C" fn select<T: WorkerSelectorPlugin>(
        state: *mut c_void,
        input: *const WorkerSelectorInputV1,
        candidate_index_out: *mut usize,
        error_out: *mut WorkerSelectorErrorBufferV1,
    ) -> i32 {
        if state.is_null() || candidate_index_out.is_null() {
            unsafe { write_error(error_out, "state or selection output is null") };
            return STATUS_INVALID_INPUT;
        }
        let state = unsafe { &mut *state.cast::<PluginState<T>>() };
        let input = match unsafe { SelectionInput::from_abi(input, state.required_inputs) } {
            Ok(input) => input,
            Err(error) => {
                unsafe { write_error(error_out, error) };
                return STATUS_INVALID_INPUT;
            }
        };
        match catch_unwind(AssertUnwindSafe(|| state.plugin.select(input))) {
            Ok(Ok(Selection::Candidate(index))) if index < input.candidate_count() => {
                unsafe { candidate_index_out.write(index) };
                STATUS_OK
            }
            Ok(Ok(Selection::Candidate(index))) => {
                unsafe {
                    write_error(
                        error_out,
                        &format!(
                            "candidate index {index} is out of range for {} candidates",
                            input.candidate_count()
                        ),
                    )
                };
                STATUS_ERROR
            }
            Ok(Ok(Selection::UseDefault)) => STATUS_USE_DEFAULT,
            Ok(Err(error)) => {
                unsafe { write_error(error_out, &error) };
                STATUS_ERROR
            }
            Err(_) => {
                unsafe { write_error(error_out, "plugin panicked") };
                STATUS_PANICKED
            }
        }
    }

    pub unsafe extern "C" fn destroy<T: WorkerSelectorPlugin>(state: *mut c_void) {
        if state.is_null() {
            return;
        }
        let _ = catch_unwind(AssertUnwindSafe(|| {
            drop(unsafe { Box::from_raw(state.cast::<PluginState<T>>()) });
        }));
    }

    unsafe fn write_error(error_out: *mut WorkerSelectorErrorBufferV1, message: &str) {
        let Some(error_out) = (unsafe { error_out.as_mut() }) else {
            return;
        };
        error_out.written = 0;
        if error_out.capacity == 0 || error_out.ptr.is_null() {
            return;
        }
        let bytes = message.as_bytes();
        let written = bytes.len().min(error_out.capacity);
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), error_out.ptr, written) };
        error_out.written = written;
    }
}

/// Export one [`WorkerSelectorPlugin`] implementation from a `cdylib`.
#[macro_export]
macro_rules! export_worker_selector_plugin {
    ($plugin:ty) => {
        static DYNAMO_WORKER_SELECTOR_PLUGIN_V1: $crate::WorkerSelectorPluginV1 =
            $crate::WorkerSelectorPluginV1 {
                abi_version: $crate::ABI_VERSION_V1,
                struct_size: ::std::mem::size_of::<$crate::WorkerSelectorPluginV1>() as u32,
                create: Some($crate::__private::create::<$plugin>),
                select: Some($crate::__private::select::<$plugin>),
                destroy: Some($crate::__private::destroy::<$plugin>),
                required_candidate_inputs: Some(
                    $crate::__private::required_candidate_inputs::<$plugin>,
                ),
            };

        #[unsafe(no_mangle)]
        pub extern "C" fn dynamo_worker_selector_plugin_v1(
        ) -> *const $crate::WorkerSelectorPluginV1 {
            &DYNAMO_WORKER_SELECTOR_PLUGIN_V1
        }
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    macro_rules! offsets {
        ($ty:ty; $($field:ident),+ $(,)?) => {
            [$(std::mem::offset_of!($ty, $field)),+]
        };
    }

    #[cfg(target_pointer_width = "64")]
    #[test]
    fn abi_v1_layout_is_stable_on_64_bit_targets() {
        assert_eq!(
            [
                (size_of::<CandidateInputs>(), align_of::<CandidateInputs>(),),
                (size_of::<ByteSliceV1>(), align_of::<ByteSliceV1>()),
                (
                    size_of::<WorkerSelectorCacheTiersV1>(),
                    align_of::<WorkerSelectorCacheTiersV1>(),
                ),
                (
                    size_of::<WorkerSelectorLoadV1>(),
                    align_of::<WorkerSelectorLoadV1>(),
                ),
                (
                    size_of::<WorkerSelectorCapacityV1>(),
                    align_of::<WorkerSelectorCapacityV1>(),
                ),
                (
                    size_of::<WorkerSelectorRoutingV1>(),
                    align_of::<WorkerSelectorRoutingV1>(),
                ),
                (
                    size_of::<WorkerSelectorCandidateColumnsV1>(),
                    align_of::<WorkerSelectorCandidateColumnsV1>(),
                ),
                (
                    size_of::<WorkerSelectorInputV1>(),
                    align_of::<WorkerSelectorInputV1>(),
                ),
                (
                    size_of::<WorkerSelectorInputHeaderV1>(),
                    align_of::<WorkerSelectorInputHeaderV1>(),
                ),
                (
                    size_of::<WorkerSelectorCandidateColumnsHeaderV1>(),
                    align_of::<WorkerSelectorCandidateColumnsHeaderV1>(),
                ),
                (
                    size_of::<WorkerSelectorErrorBufferV1>(),
                    align_of::<WorkerSelectorErrorBufferV1>(),
                ),
                (
                    size_of::<WorkerSelectorPluginV1>(),
                    align_of::<WorkerSelectorPluginV1>(),
                ),
                (
                    size_of::<WorkerSelectorPluginHeaderV1>(),
                    align_of::<WorkerSelectorPluginHeaderV1>(),
                ),
            ],
            [
                (8, 8),
                (16, 8),
                (40, 8),
                (24, 8),
                (16, 8),
                (24, 8),
                (96, 8),
                (88, 8),
                (8, 4),
                (4, 4),
                (24, 8),
                (40, 8),
                (8, 4),
            ]
        );
        assert_eq!(offsets!(ByteSliceV1; ptr, len), [0, 8]);
        assert_eq!(
            offsets!(WorkerSelectorCacheTiersV1;
                effective_overlap_blocks,
                device_overlap_blocks,
                host_pinned_overlap_blocks,
                disk_overlap_blocks,
                shared_cache_beyond_device_blocks,
            ),
            [0, 8, 16, 24, 32]
        );
        assert_eq!(
            offsets!(WorkerSelectorLoadV1;
                active_prefill_tokens,
                active_decode_blocks,
                additional_active_blocks,
            ),
            [0, 8, 16]
        );
        assert_eq!(
            offsets!(WorkerSelectorCapacityV1;
                total_kv_blocks,
                max_num_batched_tokens,
            ),
            [0, 8]
        );
        assert_eq!(
            offsets!(WorkerSelectorRoutingV1; stable_routing_id, preferred_taint_multiplier),
            [0, 16]
        );
        assert_eq!(
            offsets!(WorkerSelectorCandidateColumnsV1;
                struct_size,
                _reserved,
                provided_inputs,
                worker_ids,
                dp_ranks,
                cached_tokens,
                cache_tiers,
                loads,
                capacities,
                routing,
                default_costs,
                default_kv_overlaps,
                default_decode_loads,
            ),
            [0, 4, 8, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88]
        );
        assert_eq!(
            offsets!(WorkerSelectorInputV1;
                struct_size,
                candidate_format,
                flags,
                block_size,
                selection_mode,
                isl_tokens,
                expected_output_tokens,
                request_id,
                session_id,
                candidate_columns,
                candidate_count,
            ),
            [0, 4, 8, 12, 16, 24, 32, 40, 56, 72, 80]
        );
        assert_eq!(
            offsets!(WorkerSelectorInputHeaderV1; struct_size, candidate_format),
            [0, 4]
        );
        assert_eq!(
            offsets!(WorkerSelectorCandidateColumnsHeaderV1; struct_size),
            [0]
        );
        assert_eq!(
            offsets!(WorkerSelectorErrorBufferV1; ptr, capacity, written),
            [0, 8, 16]
        );
        assert_eq!(
            offsets!(WorkerSelectorPluginV1;
                abi_version,
                struct_size,
                create,
                select,
                destroy,
                required_candidate_inputs,
            ),
            [0, 4, 8, 16, 24, 32]
        );
        assert_eq!(
            offsets!(WorkerSelectorPluginHeaderV1; abi_version, struct_size),
            [0, 4]
        );
    }

    struct ShimPlugin {
        called: bool,
        out_of_range: bool,
    }

    impl WorkerSelectorPlugin for ShimPlugin {
        fn from_config(config: &[u8], router_role: RouterRole) -> Result<Self, String> {
            assert_eq!(router_role, RouterRole::Decode);
            Ok(Self {
                called: false,
                out_of_range: config == b"out-of-range",
            })
        }

        fn required_candidate_inputs(&self) -> CandidateInputs {
            CandidateInputs::CACHED_TOKENS
                | CandidateInputs::CAPACITY
                | CandidateInputs::ROUTING
                | CandidateInputs::DEFAULT_COST
                | CandidateInputs::DEFAULT_KV_OVERLAP
                | CandidateInputs::DEFAULT_DECODE_LOAD
        }

        fn select(&mut self, input: SelectionInput<'_>) -> Result<Selection, String> {
            if self.out_of_range {
                return Ok(Selection::Candidate(input.candidate_count()));
            }
            assert_eq!(
                input.candidate_inputs(),
                self.required_candidate_inputs().with_identity()
            );
            assert_eq!(input.worker_ids(), [1]);
            assert_eq!(input.dp_ranks(), [2]);
            assert_eq!(input.cached_tokens(), Some(&[17][..]));
            assert_eq!(input.cache_tiers(), None);
            assert_eq!(input.loads(), None);
            assert_eq!(input.candidate_stable_routing_id(0), Some("worker-1"));
            assert_eq!(input.default_costs(), Some(&[3.5][..]));
            assert_eq!(input.default_kv_overlaps(), Some(&[4.5][..]));
            assert_eq!(input.default_decode_loads(), Some(&[5][..]));
            let capacity = &input.capacities().expect("capacity input")[0];
            assert_eq!(capacity.total_kv_blocks(), Some(8_192));
            assert_eq!(capacity.max_num_batched_tokens(), None);
            if std::mem::replace(&mut self.called, true) {
                panic!("boom");
            }
            Ok(Selection::UseDefault)
        }
    }

    struct PanickingRequirements;

    impl WorkerSelectorPlugin for PanickingRequirements {
        fn from_config(_config: &[u8], _router_role: RouterRole) -> Result<Self, String> {
            Ok(Self)
        }

        fn required_candidate_inputs(&self) -> CandidateInputs {
            panic!("boom")
        }

        fn select(&mut self, _input: SelectionInput<'_>) -> Result<Selection, String> {
            unreachable!()
        }
    }

    #[test]
    fn export_shim_delegates_and_contains_panics() {
        let mut state = std::ptr::null_mut();
        let mut error_bytes = [0_u8; 64];
        let mut error = WorkerSelectorErrorBufferV1 {
            ptr: error_bytes.as_mut_ptr(),
            capacity: error_bytes.len(),
            written: 0,
        };
        let status = unsafe {
            __private::create::<ShimPlugin>(
                ByteSliceV1::empty(),
                ROUTER_ROLE_DECODE,
                &mut state,
                &mut error,
            )
        };
        assert_eq!(status, STATUS_OK);

        let mut required_inputs = u64::MAX;
        let status = unsafe {
            __private::required_candidate_inputs::<ShimPlugin>(
                state,
                &mut required_inputs,
                &mut error,
            )
        };
        assert_eq!(status, STATUS_OK);
        assert_eq!(
            required_inputs,
            (CandidateInputs::CACHED_TOKENS
                | CandidateInputs::CAPACITY
                | CandidateInputs::ROUTING
                | CandidateInputs::DEFAULT_COST
                | CandidateInputs::DEFAULT_KV_OVERLAP
                | CandidateInputs::DEFAULT_DECODE_LOAD)
                .with_identity()
                .bits()
        );

        let stable_routing_id = b"worker-1";
        let worker_ids = [1_u64];
        let dp_ranks = [2_u32];
        let cached_tokens = [17_u64];
        let capacities = [WorkerSelectorCapacityV1 {
            total_kv_blocks: 8_192,
            max_num_batched_tokens: CAPACITY_UNAVAILABLE,
        }];
        let routing = [WorkerSelectorRoutingV1 {
            stable_routing_id: ByteSliceV1::from_slice(stable_routing_id),
            preferred_taint_multiplier: 1.0,
        }];
        let default_costs = [3.5_f64];
        let default_kv_overlaps = [4.5_f64];
        let default_decode_loads = [5_u64];
        let columns = WorkerSelectorCandidateColumnsV1 {
            struct_size: size_of::<WorkerSelectorCandidateColumnsV1>() as u32,
            _reserved: 0,
            provided_inputs: required_inputs,
            worker_ids: worker_ids.as_ptr(),
            dp_ranks: dp_ranks.as_ptr(),
            cached_tokens: cached_tokens.as_ptr(),
            cache_tiers: std::ptr::null(),
            loads: std::ptr::null(),
            capacities: capacities.as_ptr(),
            routing: routing.as_ptr(),
            default_costs: default_costs.as_ptr(),
            default_kv_overlaps: default_kv_overlaps.as_ptr(),
            default_decode_loads: default_decode_loads.as_ptr(),
        };
        let input = WorkerSelectorInputV1 {
            struct_size: size_of::<WorkerSelectorInputV1>() as u32,
            candidate_format: CANDIDATE_FORMAT_COLUMNAR_V1,
            flags: 0,
            block_size: 16,
            selection_mode: SELECTION_MODE_QUERY_ONLY,
            isl_tokens: 1,
            expected_output_tokens: 0,
            request_id: ByteSliceV1::empty(),
            session_id: ByteSliceV1::empty(),
            candidate_columns: &columns,
            candidate_count: 1,
        };
        let mut selected = usize::MAX;

        let invalid_columns = WorkerSelectorCandidateColumnsV1 {
            cached_tokens: std::ptr::null(),
            ..columns
        };
        let invalid_input = WorkerSelectorInputV1 {
            candidate_columns: &invalid_columns,
            ..input
        };
        let status = unsafe {
            __private::select::<ShimPlugin>(state, &invalid_input, &mut selected, &mut error)
        };
        assert_eq!(status, STATUS_INVALID_INPUT);

        let invalid_input = WorkerSelectorInputV1 {
            candidate_format: 128,
            ..input
        };
        let status = unsafe {
            __private::select::<ShimPlugin>(state, &invalid_input, &mut selected, &mut error)
        };
        assert_eq!(status, STATUS_INVALID_INPUT);

        let status =
            unsafe { __private::select::<ShimPlugin>(state, &input, &mut selected, &mut error) };
        assert_eq!(status, STATUS_USE_DEFAULT);
        assert_eq!(selected, usize::MAX);

        let status =
            unsafe { __private::select::<ShimPlugin>(state, &input, &mut selected, &mut error) };
        assert_eq!(status, STATUS_PANICKED);
        assert_eq!(
            str::from_utf8(&error_bytes[..error.written]).unwrap(),
            "plugin panicked"
        );
        unsafe { __private::destroy::<ShimPlugin>(state) };

        let mut state = std::ptr::null_mut();
        let status = unsafe {
            __private::create::<ShimPlugin>(
                ByteSliceV1::from_slice(b"out-of-range"),
                ROUTER_ROLE_DECODE,
                &mut state,
                &mut error,
            )
        };
        assert_eq!(status, STATUS_OK);
        let status =
            unsafe { __private::select::<ShimPlugin>(state, &input, &mut selected, &mut error) };
        assert_eq!(status, STATUS_ERROR);
        unsafe { __private::destroy::<ShimPlugin>(state) };

        let mut state = std::ptr::null_mut();
        let status = unsafe {
            __private::create::<PanickingRequirements>(
                ByteSliceV1::empty(),
                ROUTER_ROLE_DECODE,
                &mut state,
                &mut error,
            )
        };
        assert_eq!(status, STATUS_PANICKED);
        assert!(state.is_null());
    }
}
