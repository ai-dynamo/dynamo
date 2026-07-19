// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stable C ABI and safe Rust authoring API for Dynamo worker-selector plugins.
//!
//! Plugins are trusted native code. ABI pointers are borrowed for the duration of a callback;
//! no Rust-owned allocation crosses the dynamic-library boundary.

use std::{ffi::c_void, slice, str};

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
        // SAFETY: The ABI contract requires the producer to keep this range alive for the call.
        Ok(unsafe { slice::from_raw_parts(self.ptr, self.len) })
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct WorkerSelectorCandidateV1 {
    pub worker_id: u64,
    pub dp_rank: u32,
    pub _padding: u32,
    pub cached_tokens: u64,
    pub effective_overlap_blocks: f64,
    pub device_overlap_blocks: u64,
    pub host_pinned_overlap_blocks: u64,
    pub disk_overlap_blocks: u64,
    pub shared_cache_beyond_device_blocks: u64,
    pub active_prefill_tokens: u64,
    pub active_decode_blocks: u64,
    pub additional_active_blocks: u64,
    pub stable_routing_id: ByteSliceV1,
    /// Stock routing's multiplicative preferred-taint cost adjustment. `1.0` is neutral.
    pub preferred_taint_multiplier: f64,
    /// Total KV capacity, or CAPACITY_UNAVAILABLE.
    pub total_kv_blocks: u64,
    /// Maximum batched-token capacity, or CAPACITY_UNAVAILABLE.
    pub max_num_batched_tokens: u64,
}

impl WorkerSelectorCandidateV1 {
    pub fn total_kv_blocks(&self) -> Option<u64> {
        (self.total_kv_blocks != CAPACITY_UNAVAILABLE).then_some(self.total_kv_blocks)
    }

    pub fn max_num_batched_tokens(&self) -> Option<u64> {
        (self.max_num_batched_tokens != CAPACITY_UNAVAILABLE).then_some(self.max_num_batched_tokens)
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct WorkerSelectorInputV1 {
    pub struct_size: u32,
    pub candidate_size: u32,
    pub flags: u32,
    pub block_size: u32,
    pub selection_mode: u32,
    pub isl_tokens: u64,
    pub expected_output_tokens: u64,
    pub request_id: ByteSliceV1,
    pub session_id: ByteSliceV1,
    pub candidates: *const WorkerSelectorCandidateV1,
    pub candidate_count: usize,
}

#[repr(C)]
struct WorkerSelectorInputHeaderV1 {
    struct_size: u32,
    candidate_size: u32,
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
    candidates: &'a [WorkerSelectorCandidateV1],
}

impl<'a> SelectionInput<'a> {
    unsafe fn from_abi(raw: *const WorkerSelectorInputV1) -> Result<Self, &'static str> {
        if raw.is_null() {
            return Err("selection input is null");
        }
        let header = unsafe { &*raw.cast::<WorkerSelectorInputHeaderV1>() };
        if header.struct_size < size_of::<WorkerSelectorInputV1>() as u32 {
            return Err("selection input is smaller than ABI v1");
        }
        if header.candidate_size != size_of::<WorkerSelectorCandidateV1>() as u32 {
            return Err("candidate size does not match ABI v1");
        }
        let raw = unsafe { &*raw };
        if raw.candidate_count > 0 && raw.candidates.is_null() {
            return Err("candidate array is null");
        }
        let candidates = if raw.candidate_count == 0 {
            &[]
        } else {
            // SAFETY: The host owns this array and keeps it alive for the callback.
            unsafe { slice::from_raw_parts(raw.candidates, raw.candidate_count) }
        };
        Ok(Self { raw, candidates })
    }

    pub fn block_size(self) -> u32 {
        self.raw.block_size
    }

    pub fn isl_tokens(self) -> u64 {
        self.raw.isl_tokens
    }

    pub fn expected_output_tokens(self) -> Option<u64> {
        (self.raw.flags & INPUT_HAS_EXPECTED_OUTPUT_TOKENS != 0)
            .then_some(self.raw.expected_output_tokens)
    }

    pub fn tracks_prefill_tokens(self) -> bool {
        self.raw.flags & INPUT_TRACKS_PREFILL_TOKENS != 0
    }

    pub fn has_shared_cache_hits(self) -> bool {
        self.raw.flags & INPUT_HAS_SHARED_CACHE_HITS != 0
    }

    pub fn selection_mode(self) -> SelectionMode {
        match self.raw.selection_mode {
            SELECTION_MODE_QUERY_ONLY => SelectionMode::QueryOnly,
            SELECTION_MODE_TRACKED => SelectionMode::Tracked,
            SELECTION_MODE_TRACKED_WITH_ADMISSION => SelectionMode::TrackedWithAdmission,
            _ => SelectionMode::Unknown,
        }
    }

    pub fn request_id(self) -> Option<&'a str> {
        self.optional_str(INPUT_HAS_REQUEST_ID, self.raw.request_id)
    }

    pub fn session_id(self) -> Option<&'a str> {
        self.optional_str(INPUT_HAS_SESSION_ID, self.raw.session_id)
    }

    /// Eligible candidates in unspecified order. Tie-break on stable candidate fields when needed.
    pub fn candidates(self) -> &'a [WorkerSelectorCandidateV1] {
        self.candidates
    }

    pub fn candidate_stable_routing_id(self, index: usize) -> Option<&'a str> {
        let candidate = self.candidates.get(index)?;
        // SAFETY: Candidate strings share the callback-scoped lifetime of this input view.
        let bytes = unsafe { candidate.stable_routing_id.as_slice() }.ok()?;
        if bytes.is_empty() {
            None
        } else {
            str::from_utf8(bytes).ok()
        }
    }

    fn optional_str(self, flag: u32, value: ByteSliceV1) -> Option<&'a str> {
        if self.raw.flags & flag == 0 {
            return None;
        }
        // SAFETY: `from_abi` validated the containing input and the host promises UTF-8 IDs.
        let bytes = unsafe { value.as_slice() }.ok()?;
        str::from_utf8(bytes).ok()
    }
}

/// Implement this trait in a `cdylib`, then invoke [`export_worker_selector_plugin!`].
///
/// The host may move an instance to its queue actor, but serializes calls to that instance.
pub trait WorkerSelectorPlugin: Send + Sized + 'static {
    /// Create independent state for one router role.
    fn from_config(config: &[u8], router_role: RouterRole) -> Result<Self, String>;

    /// Select a candidate or delegate this request to Dynamo's default selector.
    /// Calls for one instance are serialized, so stateful strategies can mutate directly.
    /// This runs synchronously on the scheduler actor's request hot path and must be bounded and
    /// non-blocking.
    fn select(&mut self, input: SelectionInput<'_>) -> Result<Selection, String>;
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum Selection {
    /// Select an index into [`SelectionInput::candidates`].
    Candidate(usize),
    /// Delegate this request to Dynamo's configured default selector.
    UseDefault,
}

#[doc(hidden)]
pub mod __private {
    use super::*;
    use std::panic::{AssertUnwindSafe, catch_unwind};

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
            T::from_config(config, RouterRole::from_abi(router_role))
        })) {
            Ok(Ok(plugin)) => {
                unsafe { state_out.write(Box::into_raw(Box::new(plugin)).cast()) };
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
        let input = match unsafe { SelectionInput::from_abi(input) } {
            Ok(input) => input,
            Err(error) => {
                unsafe { write_error(error_out, error) };
                return STATUS_INVALID_INPUT;
            }
        };
        let plugin = unsafe { &mut *state.cast::<T>() };
        match catch_unwind(AssertUnwindSafe(|| plugin.select(input))) {
            Ok(Ok(Selection::Candidate(index))) if index < input.candidates().len() => {
                unsafe { candidate_index_out.write(index) };
                STATUS_OK
            }
            Ok(Ok(Selection::Candidate(index))) => {
                unsafe {
                    write_error(
                        error_out,
                        &format!(
                            "candidate index {index} is out of range for {} candidates",
                            input.candidates().len()
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
            drop(unsafe { Box::from_raw(state.cast::<T>()) });
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
                (size_of::<ByteSliceV1>(), align_of::<ByteSliceV1>()),
                (
                    size_of::<WorkerSelectorCandidateV1>(),
                    align_of::<WorkerSelectorCandidateV1>(),
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
            [(16, 8), (128, 8), (88, 8), (8, 4), (24, 8), (32, 8), (8, 4)]
        );
        assert_eq!(offsets!(ByteSliceV1; ptr, len), [0, 8]);
        assert_eq!(
            offsets!(WorkerSelectorCandidateV1;
                worker_id,
                dp_rank,
                _padding,
                cached_tokens,
                effective_overlap_blocks,
                device_overlap_blocks,
                host_pinned_overlap_blocks,
                disk_overlap_blocks,
                shared_cache_beyond_device_blocks,
                active_prefill_tokens,
                active_decode_blocks,
                additional_active_blocks,
                stable_routing_id,
                preferred_taint_multiplier,
                total_kv_blocks,
                max_num_batched_tokens,
            ),
            [
                0, 8, 12, 16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 104, 112, 120
            ]
        );
        assert_eq!(
            offsets!(WorkerSelectorInputV1;
                struct_size,
                candidate_size,
                flags,
                block_size,
                selection_mode,
                isl_tokens,
                expected_output_tokens,
                request_id,
                session_id,
                candidates,
                candidate_count,
            ),
            [0, 4, 8, 12, 16, 24, 32, 40, 56, 72, 80]
        );
        assert_eq!(
            offsets!(WorkerSelectorInputHeaderV1; struct_size, candidate_size),
            [0, 4]
        );
        assert_eq!(
            offsets!(WorkerSelectorErrorBufferV1; ptr, capacity, written),
            [0, 8, 16]
        );
        assert_eq!(
            offsets!(WorkerSelectorPluginV1; abi_version, struct_size, create, select, destroy),
            [0, 4, 8, 16, 24]
        );
        assert_eq!(
            offsets!(WorkerSelectorPluginHeaderV1; abi_version, struct_size),
            [0, 4]
        );
    }

    struct ShimPlugin(bool);

    impl WorkerSelectorPlugin for ShimPlugin {
        fn from_config(_config: &[u8], router_role: RouterRole) -> Result<Self, String> {
            assert_eq!(router_role, RouterRole::Decode);
            Ok(Self(false))
        }

        fn select(&mut self, input: SelectionInput<'_>) -> Result<Selection, String> {
            let candidate = &input.candidates()[0];
            assert_eq!(input.candidate_stable_routing_id(0), Some("worker-1"));
            assert_eq!(candidate.total_kv_blocks(), Some(8_192));
            assert_eq!(candidate.max_num_batched_tokens(), None);
            if std::mem::replace(&mut self.0, true) {
                panic!("boom");
            }
            Ok(Selection::UseDefault)
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

        let stable_routing_id = b"worker-1";
        let candidate = WorkerSelectorCandidateV1 {
            worker_id: 1,
            dp_rank: 0,
            _padding: 0,
            cached_tokens: 0,
            effective_overlap_blocks: 0.0,
            device_overlap_blocks: 0,
            host_pinned_overlap_blocks: 0,
            disk_overlap_blocks: 0,
            shared_cache_beyond_device_blocks: 0,
            active_prefill_tokens: 0,
            active_decode_blocks: 0,
            additional_active_blocks: 0,
            stable_routing_id: ByteSliceV1::from_slice(stable_routing_id),
            preferred_taint_multiplier: 1.0,
            total_kv_blocks: 8_192,
            max_num_batched_tokens: CAPACITY_UNAVAILABLE,
        };
        let input = WorkerSelectorInputV1 {
            struct_size: size_of::<WorkerSelectorInputV1>() as u32,
            candidate_size: size_of::<WorkerSelectorCandidateV1>() as u32,
            flags: 0,
            block_size: 16,
            selection_mode: SELECTION_MODE_QUERY_ONLY,
            isl_tokens: 1,
            expected_output_tokens: 0,
            request_id: ByteSliceV1::empty(),
            session_id: ByteSliceV1::empty(),
            candidates: &candidate,
            candidate_count: 1,
        };
        let mut selected = usize::MAX;
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
    }
}
