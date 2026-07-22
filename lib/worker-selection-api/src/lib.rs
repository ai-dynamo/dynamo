// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Stable C ABI and safe Rust authoring API for Dynamo worker-selection plugins.
//!
//! Plugins are trusted native code. ABI pointers are borrowed for the duration of a callback;
//! no Rust-owned allocation crosses the dynamic-library boundary.

use std::{
    ffi::c_void,
    ops::{BitOr, BitOrAssign},
    slice,
};

pub const ABI_VERSION_V1: u32 = 1;
pub const ENTRYPOINT_SYMBOL_V1: &[u8] = b"dynamo_worker_selector_plugin_v1\0";

pub const STATUS_OK: i32 = 0;
pub const STATUS_INVALID_INPUT: i32 = 1;
pub const STATUS_ERROR: i32 = 2;
pub const STATUS_PANICKED: i32 = 3;
/// Delegate to the host's default selector; worker_index_out is ignored.
pub const STATUS_USE_DEFAULT: i32 = 4;

/// ABI encoding for an unavailable optional worker capacity.
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

/// Worker-column storage format used by [`WorkerSelectorInputV1`].
pub const WORKER_COLUMNS_FORMAT_V1: u32 = 1;

/// Worker inputs a plugin needs for selection.
///
/// Dynamo materializes only the requested inputs. Any non-empty requirement includes
/// [`Self::IDENTITY`] so a returned index always maps to a worker and data-parallel rank.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WorkerInputs(u64);

impl WorkerInputs {
    pub const NONE: Self = Self(0);
    pub const IDENTITY: Self = Self(1 << 0);
    pub const CACHED_TOKENS: Self = Self(1 << 1);
    pub const CACHE_TIERS: Self = Self(1 << 2);
    pub const LOAD: Self = Self(1 << 3);
    pub const CAPACITY: Self = Self(1 << 4);
    pub const ROUTING: Self = Self(1 << 5);
    /// Dynamo's complete per-worker cost before temperature sampling.
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

impl BitOr for WorkerInputs {
    type Output = Self;

    fn bitor(self, rhs: Self) -> Self::Output {
        Self(self.0 | rhs.0)
    }
}

impl BitOrAssign for WorkerInputs {
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
pub struct WorkerSelectorColumnsV1 {
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
struct WorkerSelectorColumnsHeaderV1 {
    struct_size: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct WorkerSelectorInputV1 {
    pub struct_size: u32,
    pub worker_columns_format: u32,
    pub flags: u32,
    pub block_size: u32,
    pub selection_mode: u32,
    pub isl_tokens: u64,
    pub expected_output_tokens: u64,
    pub request_id: ByteSliceV1,
    pub session_id: ByteSliceV1,
    pub worker_columns: *const WorkerSelectorColumnsV1,
    pub worker_count: usize,
}

#[repr(C)]
struct WorkerSelectorInputHeaderV1 {
    struct_size: u32,
    worker_columns_format: u32,
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
    worker_index_out: *mut usize,
    error_out: *mut WorkerSelectorErrorBufferV1,
) -> i32;

pub type WorkerSelectorDestroyV1 = unsafe extern "C" fn(state: *mut c_void);

/// Return the optional worker inputs required by this configured plugin state.
///
/// The host calls this once after successful creation and caches the result. On [`STATUS_OK`], the
/// callback must initialize `worker_inputs_out` with a valid [`WorkerInputs`] bitset.
pub type WorkerSelectorRequiredInputsV1 = unsafe extern "C" fn(
    state: *mut c_void,
    worker_inputs_out: *mut u64,
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
    pub required_worker_inputs: Option<WorkerSelectorRequiredInputsV1>,
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct WorkerSelectorPluginHeaderV1 {
    pub abi_version: u32,
    pub struct_size: u32,
}

pub type WorkerSelectorEntrypointV1 = unsafe extern "C" fn() -> *const WorkerSelectorPluginV1;

mod plugin;

#[doc(hidden)]
pub use plugin::__private;
pub use plugin::{Selection, SelectionInput, WorkerSelectorPlugin};

#[cfg(test)]
mod tests {
    use super::*;
    use std::mem::{align_of, size_of};

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
                (size_of::<WorkerInputs>(), align_of::<WorkerInputs>(),),
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
                    size_of::<WorkerSelectorColumnsV1>(),
                    align_of::<WorkerSelectorColumnsV1>(),
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
                    size_of::<WorkerSelectorColumnsHeaderV1>(),
                    align_of::<WorkerSelectorColumnsHeaderV1>(),
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
            offsets!(WorkerSelectorColumnsV1;
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
                worker_columns_format,
                flags,
                block_size,
                selection_mode,
                isl_tokens,
                expected_output_tokens,
                request_id,
                session_id,
                worker_columns,
                worker_count,
            ),
            [0, 4, 8, 12, 16, 24, 32, 40, 56, 72, 80]
        );
        assert_eq!(
            offsets!(WorkerSelectorInputHeaderV1; struct_size, worker_columns_format),
            [0, 4]
        );
        assert_eq!(offsets!(WorkerSelectorColumnsHeaderV1; struct_size), [0]);
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
                required_worker_inputs,
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

        fn required_worker_inputs(&self) -> WorkerInputs {
            WorkerInputs::CACHED_TOKENS
                | WorkerInputs::CAPACITY
                | WorkerInputs::ROUTING
                | WorkerInputs::DEFAULT_COST
                | WorkerInputs::DEFAULT_KV_OVERLAP
                | WorkerInputs::DEFAULT_DECODE_LOAD
        }

        fn select(&mut self, input: SelectionInput<'_>) -> Result<Selection, String> {
            if self.out_of_range {
                return Ok(Selection::Worker(input.worker_count()));
            }
            assert_eq!(
                input.worker_inputs(),
                self.required_worker_inputs().with_identity()
            );
            assert_eq!(input.worker_ids(), [1]);
            assert_eq!(input.dp_ranks(), [2]);
            assert_eq!(input.cached_tokens(), Some(&[17][..]));
            assert_eq!(input.cache_tiers(), None);
            assert_eq!(input.loads(), None);
            assert_eq!(input.worker_stable_routing_id(0), Some("worker-1"));
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

        fn required_worker_inputs(&self) -> WorkerInputs {
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
            __private::required_worker_inputs::<ShimPlugin>(state, &mut required_inputs, &mut error)
        };
        assert_eq!(status, STATUS_OK);
        assert_eq!(
            required_inputs,
            (WorkerInputs::CACHED_TOKENS
                | WorkerInputs::CAPACITY
                | WorkerInputs::ROUTING
                | WorkerInputs::DEFAULT_COST
                | WorkerInputs::DEFAULT_KV_OVERLAP
                | WorkerInputs::DEFAULT_DECODE_LOAD)
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
        let columns = WorkerSelectorColumnsV1 {
            struct_size: size_of::<WorkerSelectorColumnsV1>() as u32,
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
            worker_columns_format: WORKER_COLUMNS_FORMAT_V1,
            flags: 0,
            block_size: 16,
            selection_mode: SELECTION_MODE_QUERY_ONLY,
            isl_tokens: 1,
            expected_output_tokens: 0,
            request_id: ByteSliceV1::empty(),
            session_id: ByteSliceV1::empty(),
            worker_columns: &columns,
            worker_count: 1,
        };
        let mut selected = usize::MAX;

        let invalid_columns = WorkerSelectorColumnsV1 {
            cached_tokens: std::ptr::null(),
            ..columns
        };
        let invalid_input = WorkerSelectorInputV1 {
            worker_columns: &invalid_columns,
            ..input
        };
        let status = unsafe {
            __private::select::<ShimPlugin>(state, &invalid_input, &mut selected, &mut error)
        };
        assert_eq!(status, STATUS_INVALID_INPUT);

        let invalid_input = WorkerSelectorInputV1 {
            worker_columns_format: 128,
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
