// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! C-compatible ABI definitions for dynamically loaded AIPerf steppable
//! replays.
//!
//! This crate deliberately contains no loader and no simulator implementation.
//! The host and a plugin exchange only the fixed-layout records below. Memory
//! returned in [`ByteSliceV1`] is released through the function table that
//! created it; neither side frees allocations made by the other side.

use std::ffi::{c_char, c_void};

/// The first compatible major version of the Steppable plugin ABI.
pub const PLUGIN_ABI_MAJOR_V1: u32 = 1;

/// `DirectRequestV1::flags`: `output_token_ids` is present.
pub const REQUEST_FLAG_OUTPUT_TOKEN_IDS: u32 = 1 << 0;
/// `DirectRequestV1::flags`: `uuid` is caller supplied.
pub const REQUEST_FLAG_UUID: u32 = 1 << 1;
/// `DirectRequestV1::flags`: `preferred_dp_rank` is present.
pub const REQUEST_FLAG_PREFERRED_DP_RANK: u32 = 1 << 2;
/// `DirectRequestV1::flags`: `preferred_prefill_dp_rank` is present.
pub const REQUEST_FLAG_PREFERRED_PREFILL_DP_RANK: u32 = 1 << 3;
/// `DirectRequestV1::flags`: `arrival_timestamp_ms` is present.
pub const REQUEST_FLAG_ARRIVAL_TIMESTAMP: u32 = 1 << 4;
/// `DirectRequestV1::flags`: `policy_class` is present.
pub const REQUEST_FLAG_POLICY_CLASS: u32 = 1 << 5;
/// `DirectRequestV1::flags`: `replay_context` is present.
pub const REQUEST_FLAG_REPLAY_CONTEXT: u32 = 1 << 6;

/// `PluginDescriptorV1::capabilities`: compact trace requests are supported.
pub const CAPABILITY_COMPACT_REQUEST_V1: u64 = 1 << 0;
/// `PluginDescriptorV1::capabilities`: compact hash-buffer leases are
/// supported through the optional V1 tail.
pub const CAPABILITY_COMPACT_BUFFER_LEASES_V1: u64 = 1 << 1;

/// `ReplayContextV1::flags`: `session_id` is present.
pub const REPLAY_CONTEXT_FLAG_SESSION_ID: u32 = 1 << 0;
/// `ReplayContextV1::flags`: `turn_index` is present.
pub const REPLAY_CONTEXT_FLAG_TURN_INDEX: u32 = 1 << 1;
/// `ReplayContextV1::flags`: `metadata` is present.
pub const REPLAY_CONTEXT_FLAG_METADATA: u32 = 1 << 2;

/// Largest permitted `ReplayContextV1::metadata` payload in a V1 submission.
///
/// The metadata bytes remain opaque to the ABI, but a shared bound lets every
/// host and provider reject an oversized request before it can partially
/// mutate replay state. This matches Dynamo's V1 metadata boundary.
pub const MAX_REPLAY_CONTEXT_METADATA_BYTES_V1: usize = 64 * 1024;

/// `RequestFactV1::flags`: admission fields are present.
pub const REQUEST_FACT_FLAG_ADMISSION: u32 = 1 << 0;
/// `RequestFactV1::flags`: latency fields are present.
pub const REQUEST_FACT_FLAG_LATENCIES: u32 = 1 << 1;
/// `RequestFactV1::flags`: actual output length is present.
pub const REQUEST_FACT_FLAG_OUTPUT_LENGTH: u32 = 1 << 2;

/// `EngineEventV1::flags`: `token_id` is an emitted output token.
pub const ENGINE_EVENT_FLAG_TOKEN: u32 = 1 << 0;
/// `EngineEventV1::flags`: `terminal_status` is present.
pub const ENGINE_EVENT_FLAG_TERMINAL: u32 = 1 << 1;

/// `EngineEventV1::terminal_status`: the request completed successfully.
pub const TERMINAL_STATUS_COMPLETED: u32 = 1;
/// `EngineEventV1::terminal_status`: admission rejected the request.
pub const TERMINAL_STATUS_REJECTED: u32 = 2;
/// `EngineEventV1::terminal_status`: cancellation terminated the request.
pub const TERMINAL_STATUS_CANCELED: u32 = 3;
/// `EngineEventV1::terminal_status`: the backend failed the request.
pub const TERMINAL_STATUS_FAILED: u32 = 4;

/// `SlaThresholdsV1::flags`: `ttft_ms` is present.
pub const SLA_FLAG_TTFT: u32 = 1 << 0;
/// `SlaThresholdsV1::flags`: `itl_ms` is present.
pub const SLA_FLAG_ITL: u32 = 1 << 1;
/// `SlaThresholdsV1::flags`: `e2e_ms` is present.
pub const SLA_FLAG_E2E: u32 = 1 << 2;

/// A versioned C-ABI operation status.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct StatusV1(pub i32);

impl StatusV1 {
    /// The operation succeeded.
    pub const OK: Self = Self(0);
    /// The caller supplied an invalid argument or buffer.
    pub const INVALID_ARGUMENT: Self = Self(1);
    /// The operation is not supported by this plugin.
    pub const UNSUPPORTED: Self = Self(2);
    /// The plugin rejected a malformed or incompatible request.
    pub const REJECTED: Self = Self(3);
    /// The plugin encountered an internal error without unwinding across FFI.
    pub const INTERNAL: Self = Self(4);

    /// Returns whether this status represents success.
    #[must_use]
    pub const fn is_ok(self) -> bool {
        self.0 == Self::OK.0
    }
}

/// A borrowed or producer-owned byte slice.
///
/// The owner and validity duration are defined by the operation that returns
/// it. Input slices are borrowed for the call only. Output slices must be
/// released through [`PluginVTableV1::release_bytes`] unless an operation
/// explicitly documents a shorter borrowed lifetime.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ByteSliceV1 {
    /// Start of the bytes, or null only when `len` is zero.
    pub data: *const u8,
    /// Number of bytes, represented with a fixed-width scalar.
    pub len: u64,
}

impl ByteSliceV1 {
    /// An empty slice.
    pub const EMPTY: Self = Self {
        data: std::ptr::null(),
        len: 0,
    };
}

/// A borrowed sequence of token identifiers.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct U32SliceV1 {
    /// Start of the tokens, or null only when `len` is zero.
    pub data: *const u32,
    /// Number of token IDs.
    pub len: u64,
}

impl U32SliceV1 {
    /// An empty token sequence.
    pub const EMPTY: Self = Self {
        data: std::ptr::null(),
        len: 0,
    };
}

/// A nonzero host-registered immutable hash-buffer identifier.
#[repr(transparent)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HashBufferIdV1(pub u64);

impl HashBufferIdV1 {
    /// The invalid identifier, which no host may register.
    pub const INVALID: Self = Self(0);

    /// Returns whether this identifier can name a registered buffer.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.0 != Self::INVALID.0
    }
}

/// A borrowed range within a host-registered immutable hash buffer.
///
/// `len` is the number of compact hash IDs, rather than the logical prompt
/// length in tokens. Consumers validate `offset + len` against the registered
/// buffer before dereferencing it.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HashBufferRangeV1 {
    /// Nonzero identifier returned by `register_hash_buffer`.
    pub buffer_id: HashBufferIdV1,
    /// Start index in the registered `u32` buffer.
    pub offset: u64,
    /// Number of compact hash IDs in this request.
    pub len: u64,
}

impl HashBufferRangeV1 {
    /// Returns whether the range has a nonzero buffer ID and does not wrap.
    ///
    /// The plugin still validates the resulting end index against the
    /// registered buffer length before it dereferences the range.
    #[must_use]
    pub const fn is_valid(self) -> bool {
        self.buffer_id.is_valid() && self.offset.checked_add(self.len).is_some()
    }
}

/// Callback invoked by a plugin when it no longer retains a hash-buffer
/// lease.
///
/// The callback runs synchronously, must not panic, and must not re-enter the
/// plugin. It receives the opaque host context supplied at replay creation.
pub type ReleaseHashBufferFnV1 = unsafe extern "C" fn(*mut c_void, HashBufferIdV1);

/// Host callbacks used only by the compact hash-buffer lease creation tail.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct HashBufferLeaseCallbacksV1 {
    /// Size of this structure as compiled by the host.
    pub struct_size: u32,
    /// Reserved for compatible-minor callback flags.
    pub flags: u32,
    /// Opaque context returned unchanged to [`ReleaseHashBufferFnV1`].
    pub context: *mut c_void,
    /// Releases one retained hash-buffer lease.
    pub release_hash_buffer: Option<ReleaseHashBufferFnV1>,
}

impl HashBufferLeaseCallbacksV1 {
    /// Empty callbacks, which do not permit a plugin to retain a lease.
    pub const EMPTY: Self = Self {
        struct_size: std::mem::size_of::<Self>() as u32,
        flags: 0,
        context: std::ptr::null_mut(),
        release_hash_buffer: None,
    };

    /// Returns whether this complete callback record can release leases.
    #[must_use]
    pub const fn has_release_callback(self) -> bool {
        self.struct_size as usize >= std::mem::size_of::<Self>()
            && self.release_hash_buffer.is_some()
    }
}

/// A replay request identifier represented in a layout independent of `Uuid`.
pub type RequestIdV1 = [u8; 16];

/// Correlation and provenance retained by a replay request.
///
/// All strings are UTF-8 byte slices. `metadata` is an application-owned
/// opaque byte sequence: the ABI never parses it and routing must not depend
/// on its representation. It is intentionally separate from scheduling data
/// so a provider can retain provenance without defining a Rust type across the
/// boundary.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ReplayContextV1 {
    /// Size of this structure as compiled by the caller.
    pub struct_size: u32,
    /// Optional-field bits defined by the V1 context contract.
    pub flags: u32,
    /// Required authored request identity.
    pub authored_id: ByteSliceV1,
    /// Optional conversation/session identity.
    pub session_id: ByteSliceV1,
    /// Optional opaque application-owned provenance bytes, bounded by
    /// [`MAX_REPLAY_CONTEXT_METADATA_BYTES_V1`].
    pub metadata: ByteSliceV1,
    /// Optional authored turn number when the corresponding flag is set.
    pub turn_index: u64,
    /// Prompt-token-source discriminant.
    pub prompt_token_source: u32,
    /// Reserved for compatible-minor extension.
    pub reserved: u32,
}

impl ReplayContextV1 {
    /// An absent context.
    pub const EMPTY: Self = Self {
        struct_size: std::mem::size_of::<Self>() as u32,
        flags: 0,
        authored_id: ByteSliceV1::EMPTY,
        session_id: ByteSliceV1::EMPTY,
        metadata: ByteSliceV1::EMPTY,
        turn_index: 0,
        prompt_token_source: 0,
        reserved: 0,
    };
}

/// Fixed-layout input for one replay submission.
///
/// Presence of optional fields is represented by [`Self::flags`]. String and
/// metadata-bearing values are borrowed only for the duration of `submit`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DirectRequestV1 {
    /// Size of this structure as compiled by the caller.
    pub struct_size: u32,
    /// Optional-field bits defined by the V1 request contract.
    pub flags: u32,
    /// Materialized prompt token identities.
    pub tokens: U32SliceV1,
    /// Optional exact output-token plan.
    pub output_token_ids: U32SliceV1,
    /// Maximum generated output tokens when no exact plan is present.
    pub max_output_tokens: u64,
    /// Caller-selected request UUID when the UUID-present flag is set.
    pub uuid: RequestIdV1,
    /// Effective data-parallel rank.
    pub dp_rank: u32,
    /// Preferred aggregate/decode rank when the corresponding flag is set.
    pub preferred_dp_rank: u32,
    /// Preferred prefill rank when the corresponding flag is set.
    pub preferred_prefill_dp_rank: u32,
    /// Arrival timestamp when the corresponding flag is set.
    pub arrival_timestamp_ms: f64,
    /// Router priority.
    pub priority: i32,
    /// Strict router priority.
    pub strict_priority: u32,
    /// Optional scheduling policy class, present when its flag is set.
    pub policy_class: ByteSliceV1,
    /// Optional correlation/provenance retained by replay.
    pub replay_context: ReplayContextV1,
}

/// Compact request form emitted by the legacy trace compiler.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CompactRequestV1 {
    pub struct_size: u32,
    pub flags: u32,
    pub input_token_count: u64,
    pub trace_block_size: u32,
    pub reserved: u32,
    pub hash_ids: U32SliceV1,
    /// Ordinary submission metadata. Its `tokens` slice must be empty.
    pub request: DirectRequestV1,
}

/// A borrowed batch of direct submissions.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct DirectRequestSliceV1 {
    /// Start of requests, or null only when `len` is zero.
    pub data: *const DirectRequestV1,
    /// Number of requests.
    pub len: u64,
}

impl DirectRequestSliceV1 {
    /// An empty batch.
    pub const EMPTY: Self = Self {
        data: std::ptr::null(),
        len: 0,
    };
}

/// A borrowed batch of request identifiers.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RequestIdSliceV1 {
    /// Start of IDs, or null only when `len` is zero.
    pub data: *const RequestIdV1,
    /// Number of IDs.
    pub len: u64,
}

/// A host-owned writable batch of request identifiers.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RequestIdMutSliceV1 {
    /// Start of writable IDs, or null only when `len` is zero.
    pub data: *mut RequestIdV1,
    /// Number of IDs. Must equal the submitted request count.
    pub len: u64,
}

/// One event returned by a replay step.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EngineEventV1 {
    /// Request that emitted this event.
    pub request_id: RequestIdV1,
    /// Event bits defined by [`ENGINE_EVENT_FLAG_TOKEN`] and
    /// [`ENGINE_EVENT_FLAG_TERMINAL`].
    pub flags: u32,
    /// Emitted output token when the token-present bit is set.
    pub token_id: u32,
    /// Terminal-status discriminant when the terminal-present bit is set:
    /// [`TERMINAL_STATUS_COMPLETED`], [`TERMINAL_STATUS_REJECTED`],
    /// [`TERMINAL_STATUS_CANCELED`], or [`TERMINAL_STATUS_FAILED`].
    pub terminal_status: u32,
    /// Reserved for compatible-minor extension.
    pub reserved: u32,
}

/// A plugin-owned batch of replay events.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct EngineEventSliceV1 {
    /// Start of the events, or null only when `len` is zero.
    pub data: *const EngineEventV1,
    /// Number of events.
    pub len: u64,
}

/// Cached per-request measurements produced while stepping a replay.
///
/// A plugin includes a fact only when a measurement becomes available or
/// changes. Hosts retain the latest record by request identifier and therefore
/// never need a getter FFI call in their event loop.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RequestFactV1 {
    /// Request to which these measurements belong.
    pub request_id: RequestIdV1,
    /// Presence bits for optional measurement groups.
    pub flags: u32,
    /// Reserved for compatible-minor extension.
    pub reserved: u32,
    /// Input tokens served from cache at admission.
    pub reused_input_tokens: u64,
    /// Actual generated output tokens at terminal completion.
    pub output_length: u64,
    /// First scheduler admission timestamp.
    pub admission_ms: f64,
    /// Time to first token.
    pub ttft_ms: f64,
    /// Mean inter-token latency.
    pub mean_itl_ms: f64,
}

/// A plugin-owned batch of request measurements.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct RequestFactSliceV1 {
    /// Start of facts, or null only when `len` is zero.
    pub data: *const RequestFactV1,
    /// Number of records.
    pub len: u64,
}

/// A plugin-owned opaque replay instance.
#[repr(transparent)]
#[derive(Debug, Clone, Copy)]
pub struct ReplayHandleV1(pub *mut c_void);

/// Cached externally visible replay state.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ReplayStateV1 {
    /// Simulated replay time in milliseconds.
    pub now_ms: f64,
    /// Next internal event time, or NaN when idle.
    pub next_event_ms: f64,
    /// Number of submitted nonterminal requests.
    pub in_flight: u64,
    /// Nonzero when there is no outstanding work.
    pub is_idle: u8,
    /// Reserved for compatible-minor extension.
    pub reserved: [u8; 7],
}

impl ReplayStateV1 {
    /// Empty, idle state.
    pub const EMPTY: Self = Self {
        now_ms: 0.0,
        next_event_ms: f64::NAN,
        in_flight: 0,
        is_idle: 1,
        reserved: [0; 7],
    };
}

/// Goodput threshold configuration applied to a replay report epoch.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct SlaThresholdsV1 {
    /// Presence bits for threshold values.
    pub flags: u32,
    /// Reserved for compatible-minor extension.
    pub reserved: u32,
    /// Optional time-to-first-token limit.
    pub ttft_ms: f64,
    /// Optional mean inter-token-latency limit.
    pub itl_ms: f64,
    /// Optional end-to-end latency limit.
    pub e2e_ms: f64,
}

impl SlaThresholdsV1 {
    /// No configured SLA thresholds.
    pub const EMPTY: Self = Self {
        flags: 0,
        reserved: 0,
        ttft_ms: 0.0,
        itl_ms: 0.0,
        e2e_ms: 0.0,
    };
}

/// Request supplied while constructing a fully configured replay instance.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct CreateRequestV1 {
    /// Size of this structure as compiled by the caller.
    pub struct_size: u32,
    /// Reserved for flags defined by a compatible ABI minor version.
    pub flags: u32,
    /// Provider-owned, bounded creation payload.
    pub provider_payload: ByteSliceV1,
}

/// Constructs a replay and, on failure, returns a plugin-owned UTF-8
/// diagnostic through `error`. The host releases a non-empty diagnostic with
/// [`PluginVTableV1::release_bytes`].
pub type CreateFnV1 =
    unsafe extern "C" fn(CreateRequestV1, *mut ReplayHandleV1, *mut ByteSliceV1) -> StatusV1;

/// Creates a replay that may retain compact hash-buffer ranges.
///
/// This optional V1-tail operation keeps [`CreateFnV1`] byte-for-byte
/// unchanged for existing plugins. The plugin must reject callbacks without a
/// release function before it accepts any retained range.
pub type CreateWithHashBufferLeasesFnV1 = unsafe extern "C" fn(
    CreateRequestV1,
    HashBufferLeaseCallbacksV1,
    *mut ReplayHandleV1,
    *mut ByteSliceV1,
) -> StatusV1;

/// Registers an immutable host-owned hash buffer and returns a nonzero ID.
pub type RegisterHashBufferFnV1 =
    unsafe extern "C" fn(ReplayHandleV1, U32SliceV1, *mut HashBufferIdV1) -> StatusV1;

/// Submits a compact request that refers to a registered hash-buffer range.
pub type SubmitCompactHashBufferRangeFnV1 = unsafe extern "C" fn(
    ReplayHandleV1,
    CompactRequestV1,
    HashBufferRangeV1,
    *mut RequestIdV1,
) -> StatusV1;

/// One externally driven step request.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StepRequestV1 {
    /// Size of this structure as compiled by the caller.
    pub struct_size: u32,
    /// Reserved for compatible-minor flags.
    pub flags: u32,
    /// Simulation time through which the replay may advance.
    pub until_ms: f64,
}

/// Batched facts returned by a steppable replay operation.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct StepResultV1 {
    /// Size of this structure as compiled by the producer.
    pub struct_size: u32,
    /// Reserved for compatible-minor flags.
    pub flags: u32,
    /// Simulated time after the operation.
    pub end_ms: f64,
    /// The next internal event time, or NaN when there is none.
    pub next_event_ms: f64,
    /// Number of nonterminal submitted requests.
    pub in_flight: u64,
    /// Nonzero when the replay is idle.
    pub is_idle: u8,
    /// Padding reserved for ABI growth and C alignment.
    pub reserved: [u8; 7],
    /// Plugin-owned batch of engine events and request facts.
    pub events: EngineEventSliceV1,
    /// Plugin-owned cached request measurements.
    pub request_facts: RequestFactSliceV1,
}

impl StepResultV1 {
    /// Empty output initialized for an FFI call.
    pub const EMPTY: Self = Self {
        struct_size: std::mem::size_of::<Self>() as u32,
        flags: 0,
        end_ms: 0.0,
        next_event_ms: f64::NAN,
        in_flight: 0,
        is_idle: 1,
        reserved: [0; 7],
        events: EngineEventSliceV1 {
            data: std::ptr::null(),
            len: 0,
        },
        request_facts: RequestFactSliceV1 {
            data: std::ptr::null(),
            len: 0,
        },
    };
}

/// Plugin identity and ABI compatibility metadata.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct PluginDescriptorV1 {
    /// The ABI major implemented by this plugin.
    pub abi_major: u32,
    /// The ABI minor implemented by this plugin.
    pub abi_minor: u32,
    /// Size of this descriptor as compiled by the plugin.
    pub struct_size: u32,
    /// Reserved for compatible-minor flags.
    pub flags: u32,
    /// Operations supported by the plugin.
    pub capabilities: u64,
    /// NUL-terminated static UTF-8 provider identifier.
    pub provider_id: *const c_char,
    /// The immutable table used for all plugin operations.
    pub vtable: *const PluginVTableV1,
}

// Safety: descriptors are immutable ABI metadata. Plugins return pointers to
// process-lifetime static descriptors, and a host only reads their fields after
// validation while retaining the dynamic library that owns them.
unsafe impl Sync for PluginDescriptorV1 {}

impl PluginDescriptorV1 {
    /// ABI major required by this V1 descriptor.
    pub const ABI_MAJOR: u32 = PLUGIN_ABI_MAJOR_V1;
}

/// The V1 operation-table prefix present in every compatible plugin.
///
/// This remains a separate type so a host can safely validate a plugin built
/// before optional compatible-minor tails were appended.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PluginVTableV1Prefix {
    pub struct_size: u32,
    pub flags: u32,
    pub create: Option<CreateFnV1>,
    pub submit:
        Option<unsafe extern "C" fn(ReplayHandleV1, DirectRequestV1, *mut RequestIdV1) -> StatusV1>,
    pub submit_batch: Option<
        unsafe extern "C" fn(ReplayHandleV1, DirectRequestSliceV1, RequestIdMutSliceV1) -> StatusV1,
    >,
    pub cancel: Option<
        unsafe extern "C" fn(
            ReplayHandleV1,
            *const RequestIdV1,
            *mut EngineEventV1,
            *mut u8,
        ) -> StatusV1,
    >,
    pub cancel_batch: Option<
        unsafe extern "C" fn(ReplayHandleV1, RequestIdSliceV1, *mut EngineEventSliceV1) -> StatusV1,
    >,
    pub step:
        Option<unsafe extern "C" fn(ReplayHandleV1, StepRequestV1, *mut StepResultV1) -> StatusV1>,
    pub take_report:
        Option<unsafe extern "C" fn(ReplayHandleV1, f64, *mut ByteSliceV1) -> StatusV1>,
    pub release_bytes: Option<unsafe extern "C" fn(ByteSliceV1)>,
    pub release_events: Option<unsafe extern "C" fn(EngineEventSliceV1)>,
    pub release_request_facts: Option<unsafe extern "C" fn(RequestFactSliceV1)>,
    pub state: Option<unsafe extern "C" fn(ReplayHandleV1, *mut ReplayStateV1) -> StatusV1>,
    pub advance_now_ms: Option<unsafe extern "C" fn(ReplayHandleV1, f64) -> StatusV1>,
    pub set_capture_per_request: Option<unsafe extern "C" fn(ReplayHandleV1, u8) -> StatusV1>,
    pub set_sla_thresholds:
        Option<unsafe extern "C" fn(ReplayHandleV1, SlaThresholdsV1) -> StatusV1>,
    pub last_error: Option<unsafe extern "C" fn(ReplayHandleV1, *mut ByteSliceV1) -> StatusV1>,
    pub destroy: Option<unsafe extern "C" fn(ReplayHandleV1)>,
}

/// The compatible-minor V1 function-table tail for compact hash-buffer leases.
///
/// Hosts obtain this record only through
/// [`PluginVTableV1::compact_buffer_leases`], which first proves the table is
/// large enough to contain the complete tail.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct CompactBufferLeaseVTableTailV1 {
    /// Creates a replay with a host release callback for compact buffer leases.
    pub create_with_hash_buffer_leases: Option<CreateWithHashBufferLeasesFnV1>,
    /// Registers one immutable host-owned compact hash-ID buffer.
    pub register_hash_buffer: Option<RegisterHashBufferFnV1>,
    /// Submits a compact request by a registered hash-buffer range.
    pub submit_compact_hash_buffer_range: Option<SubmitCompactHashBufferRangeFnV1>,
}

/// All callable operations supplied by a Steppable plugin.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct PluginVTableV1 {
    /// Size of this table as compiled by the plugin.
    pub struct_size: u32,
    /// Reserved for compatible-minor flags.
    pub flags: u32,
    /// Creates one fully configured replay handle.
    pub create: Option<CreateFnV1>,
    /// Submits one direct request and returns its concrete request identifier.
    pub submit:
        Option<unsafe extern "C" fn(ReplayHandleV1, DirectRequestV1, *mut RequestIdV1) -> StatusV1>,
    /// Submits a batch after validating all records, writing one ID per input.
    pub submit_batch: Option<
        unsafe extern "C" fn(ReplayHandleV1, DirectRequestSliceV1, RequestIdMutSliceV1) -> StatusV1,
    >,
    /// Cancels one request and returns its terminal event when cancellation won.
    pub cancel: Option<
        unsafe extern "C" fn(
            ReplayHandleV1,
            *const RequestIdV1,
            *mut EngineEventV1,
            *mut u8,
        ) -> StatusV1,
    >,
    /// Cancels a batch and returns terminal events for cancellations that won.
    pub cancel_batch: Option<
        unsafe extern "C" fn(ReplayHandleV1, RequestIdSliceV1, *mut EngineEventSliceV1) -> StatusV1,
    >,
    /// Advances the replay and returns a batch of events and cached facts.
    pub step:
        Option<unsafe extern "C" fn(ReplayHandleV1, StepRequestV1, *mut StepResultV1) -> StatusV1>,
    /// Drains the plugin-owned replay report into an encoded result.
    pub take_report:
        Option<unsafe extern "C" fn(ReplayHandleV1, f64, *mut ByteSliceV1) -> StatusV1>,
    /// Releases an output byte slice previously returned by this table.
    pub release_bytes: Option<unsafe extern "C" fn(ByteSliceV1)>,
    /// Releases an event slice previously returned by `step`.
    pub release_events: Option<unsafe extern "C" fn(EngineEventSliceV1)>,
    /// Releases a request-fact slice previously returned by `step`.
    pub release_request_facts: Option<unsafe extern "C" fn(RequestFactSliceV1)>,
    /// Returns current state when the host needs to synchronize after create.
    pub state: Option<unsafe extern "C" fn(ReplayHandleV1, *mut ReplayStateV1) -> StatusV1>,
    /// Monotonically advances the externally controlled replay clock.
    pub advance_now_ms: Option<unsafe extern "C" fn(ReplayHandleV1, f64) -> StatusV1>,
    /// Enables or disables retained per-request causality records.
    pub set_capture_per_request: Option<unsafe extern "C" fn(ReplayHandleV1, u8) -> StatusV1>,
    /// Configures goodput classification thresholds.
    pub set_sla_thresholds:
        Option<unsafe extern "C" fn(ReplayHandleV1, SlaThresholdsV1) -> StatusV1>,
    /// Returns the last operation error as plugin-owned UTF-8 bytes.
    ///
    /// A successful call with an empty slice means no diagnostic is retained.
    pub last_error: Option<unsafe extern "C" fn(ReplayHandleV1, *mut ByteSliceV1) -> StatusV1>,
    /// Destroys one replay handle.
    pub destroy: Option<unsafe extern "C" fn(ReplayHandleV1)>,
    /// Optional compatible-minor compact trace submission operation.
    pub submit_compact: Option<
        unsafe extern "C" fn(ReplayHandleV1, CompactRequestV1, *mut RequestIdV1) -> StatusV1,
    >,
    /// Creates a replay with a host release callback for compact buffer leases.
    pub create_with_hash_buffer_leases: Option<CreateWithHashBufferLeasesFnV1>,
    /// Registers one immutable host-owned compact hash-ID buffer.
    pub register_hash_buffer: Option<RegisterHashBufferFnV1>,
    /// Submits a compact request by a registered hash-buffer range.
    pub submit_compact_hash_buffer_range: Option<SubmitCompactHashBufferRangeFnV1>,
}

impl PluginVTableV1 {
    /// Bytes a consumer must be able to read for every original V1 operation.
    pub const REQUIRED_SIZE: usize = std::mem::size_of::<PluginVTableV1Prefix>();
    /// Bytes required before a host may read the compact-submit tail.
    pub const COMPACT_SUBMIT_SIZE: usize =
        std::mem::offset_of!(Self, create_with_hash_buffer_leases);
    /// Bytes required before a host may read the compact buffer-lease tail.
    pub const COMPACT_BUFFER_LEASES_SIZE: usize = std::mem::size_of::<Self>();

    /// Returns whether a table of `struct_size` bytes advertises the compact
    /// request operation without reading beyond the table's declared extent.
    #[must_use]
    pub const fn supports_compact_submit(struct_size: u32, present: bool) -> Option<()> {
        if struct_size as usize >= Self::COMPACT_SUBMIT_SIZE && present {
            Some(())
        } else {
            None
        }
    }

    /// Returns whether a descriptor and table expose every compact
    /// hash-buffer lease operation without reading beyond the declared tail.
    #[must_use]
    pub const fn supports_compact_buffer_leases(
        struct_size: u32,
        capabilities: u64,
        create_with_hash_buffer_leases_present: bool,
        register_hash_buffer_present: bool,
        submit_compact_hash_buffer_range_present: bool,
    ) -> bool {
        struct_size as usize >= Self::COMPACT_BUFFER_LEASES_SIZE
            && capabilities & CAPABILITY_COMPACT_BUFFER_LEASES_V1 != 0
            && create_with_hash_buffer_leases_present
            && register_hash_buffer_present
            && submit_compact_hash_buffer_range_present
    }

    /// Reads the compact hash-buffer lease tail only after the descriptor
    /// capability and declared table extent prove every operation is present.
    ///
    /// # Safety
    ///
    /// `vtable` must point to a readable V1 prefix and its `struct_size` must
    /// truthfully describe the readable allocation.
    pub unsafe fn compact_buffer_leases(
        vtable: *const PluginVTableV1Prefix,
        capabilities: u64,
    ) -> Option<CompactBufferLeaseVTableTailV1> {
        // Safety: required by this function's contract.
        let prefix = unsafe { &*vtable };
        if (prefix.struct_size as usize) < Self::COMPACT_BUFFER_LEASES_SIZE {
            return None;
        }
        let tail = unsafe {
            vtable
                .cast::<u8>()
                .add(std::mem::offset_of!(Self, create_with_hash_buffer_leases))
                .cast::<CompactBufferLeaseVTableTailV1>()
                .read()
        };
        if Self::supports_compact_buffer_leases(
            prefix.struct_size,
            capabilities,
            tail.create_with_hash_buffer_leases.is_some(),
            tail.register_hash_buffer.is_some(),
            tail.submit_compact_hash_buffer_range.is_some(),
        ) {
            Some(tail)
        } else {
            None
        }
    }

    /// Reads the optional compatible-minor tail only after its declared extent
    /// proves the field is present.
    ///
    /// # Safety
    ///
    /// `vtable` must point to a readable V1 prefix and its `struct_size` must
    /// truthfully describe the readable allocation.
    pub unsafe fn compact_submit(
        vtable: *const PluginVTableV1Prefix,
    ) -> Option<unsafe extern "C" fn(ReplayHandleV1, CompactRequestV1, *mut RequestIdV1) -> StatusV1>
    {
        // Safety: required by this function's contract.
        let prefix = unsafe { &*vtable };
        if (prefix.struct_size as usize) < Self::COMPACT_SUBMIT_SIZE {
            return None;
        }
        unsafe {
            vtable
                .cast::<u8>()
                .add(std::mem::offset_of!(Self, submit_compact))
                .cast::<Option<
                    unsafe extern "C" fn(
                        ReplayHandleV1,
                        CompactRequestV1,
                        *mut RequestIdV1,
                    ) -> StatusV1,
                >>()
                .read()
        }
    }
}

/// Why a loaded V1 plugin descriptor cannot be used by a host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DescriptorValidationError {
    /// The loader received a null descriptor pointer.
    NullDescriptor,
    /// The plugin implements a different incompatible ABI major.
    IncompatibleMajor,
    /// The descriptor was compiled with fewer fields than V1 requires.
    DescriptorTooSmall,
    /// The descriptor does not identify its provider.
    MissingProviderId,
    /// The descriptor did not supply an operation table.
    MissingVTable,
    /// The operation table was compiled with fewer fields than V1 requires.
    VTableTooSmall,
    /// A required V1 operation was absent.
    MissingOperation,
}

/// Validates that a plugin descriptor has the complete V1 shape.
///
/// # Safety
///
/// `descriptor` must be null or point to readable memory containing at least
/// a [`PluginDescriptorV1`]. When the descriptor's table pointer is non-null,
/// it must point to readable memory containing at least a [`PluginVTableV1`].
pub unsafe fn validate_descriptor_v1(
    descriptor: *const PluginDescriptorV1,
) -> Result<*const PluginVTableV1Prefix, DescriptorValidationError> {
    if descriptor.is_null() {
        return Err(DescriptorValidationError::NullDescriptor);
    }
    // Safety: required by this function's contract.
    let descriptor = unsafe { &*descriptor };
    if descriptor.abi_major != PLUGIN_ABI_MAJOR_V1 {
        return Err(DescriptorValidationError::IncompatibleMajor);
    }
    if (descriptor.struct_size as usize) < std::mem::size_of::<PluginDescriptorV1>() {
        return Err(DescriptorValidationError::DescriptorTooSmall);
    }
    if descriptor.provider_id.is_null() {
        return Err(DescriptorValidationError::MissingProviderId);
    }
    if descriptor.vtable.is_null() {
        return Err(DescriptorValidationError::MissingVTable);
    }
    // Safety: the descriptor's required V1 table prefix is part of this
    // function's contract. Do not form a reference to a newer optional tail.
    let vtable = unsafe { &*descriptor.vtable.cast::<PluginVTableV1Prefix>() };
    if (vtable.struct_size as usize) < PluginVTableV1::REQUIRED_SIZE {
        return Err(DescriptorValidationError::VTableTooSmall);
    }
    if vtable.create.is_none()
        || vtable.submit.is_none()
        || vtable.submit_batch.is_none()
        || vtable.cancel.is_none()
        || vtable.cancel_batch.is_none()
        || vtable.step.is_none()
        || vtable.take_report.is_none()
        || vtable.release_bytes.is_none()
        || vtable.release_events.is_none()
        || vtable.release_request_facts.is_none()
        || vtable.state.is_none()
        || vtable.advance_now_ms.is_none()
        || vtable.set_capture_per_request.is_none()
        || vtable.set_sla_thresholds.is_none()
        || vtable.last_error.is_none()
        || vtable.destroy.is_none()
    {
        return Err(DescriptorValidationError::MissingOperation);
    }
    Ok(descriptor.vtable.cast::<PluginVTableV1Prefix>())
}

/// Fixed entry point exported by every V1 Steppable plugin.
pub type PluginEntryV1 = unsafe extern "C" fn() -> *const PluginDescriptorV1;
