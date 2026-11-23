// SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! Records what the NVTX macros actually hand to NVTX, through NVTX's static injection hook.
//!
//! The unit tests in `src/nvtx.rs` run with no tool attached, so NVTX's stubs drop every
//! call and nothing there can see a payload. This binary attaches a recorder in place of a
//! profiler and asserts, per macro, which NVTX entry point was called, with which message,
//! payload type and payload value.
//!
//! # How the recorder attaches
//!
//! NVTX initializes once per process, on the first NVTX call. It then looks for a tool in
//! this order (`nvtx3/nvtxDetail/nvtxInit.h`, `nvtxInitializeInjectionLibrary`):
//! `NVTX_INJECTION64_PATH`, a pre-injected `InitializeInjectionNvtx2Preinject`, the static
//! hook `InitializeInjectionNvtx2_fnptr`, the CUDA driver. The headers define the static
//! hook as a weak, null function pointer on every GCC/Clang non-Windows target; the
//! strong `#[no_mangle]` definition below overrides it at link time, so it is in place
//! before any NVTX call can run. NVTX then calls it with its export table, and it swaps
//! the core module's `MarkEx`, `MarkA`, `RangeStartEx`, `RangeStartA` and `RangeEnd`
//! slots for the recorders. `NVTX_INJECTION64_PATH` takes precedence, so it is removed
//! before the first NVTX call (it is set when this binary runs under `nsys`).
//!
//! It lives in its own test binary because the hook is process-wide and one-shot. The
//! few export-table types are declared here rather than taken from `nvtx-sys`'s `tools`
//! feature: a dev-dependency cannot be optional, so enabling `tools` would build
//! `nvtx-sys` (and require libclang for its bindgen step) in every `dynamo-runtime` test
//! build, including the ones with the `nvtx` feature off.

#![cfg(feature = "nvtx")]

use std::ffi::{CStr, c_char, c_int, c_uint, c_void};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::thread::ThreadId;

use dynamo_runtime::nvtx::{self, IDENTITY_STAGE, rid_key};
use dynamo_runtime::{
    dynamo_nvtx_keyed_mark, dynamo_nvtx_keyed_range, dynamo_nvtx_mark, dynamo_nvtx_range,
    dynamo_nvtx_request_mark, dynamo_nvtx_request_range,
};
use nvtx_sys::{EventAttributes, MessageType, PayloadType};

// ── NVTX export-table ABI (nvtx3/nvtxDetail/nvtxTypes.h) ─────────────────────

/// `NvtxGetExportTableFunc_t`.
type GetExportTable = unsafe extern "C" fn(export_table_id: u32) -> *const c_void;

/// `NvtxInitializeInjectionNvtxFunc_t`: returns nonzero on success.
type InitializeInjection = unsafe extern "C" fn(get_export_table: Option<GetExportTable>) -> c_int;

/// `NvtxFunctionTable`: an array of pointers to NVTX's function-pointer slots.
type FunctionTable = *mut *mut *const c_void;

/// `NvtxExportTableCallbacks`.
#[repr(C)]
struct ExportTableCallbacks {
    struct_size: usize,
    get_module_function_table: Option<
        unsafe extern "C" fn(
            module: u32,
            out_table: *mut FunctionTable,
            out_size: *mut c_uint,
        ) -> c_int,
    >,
}

/// `NVTX_ETID_CALLBACKS`.
const NVTX_ETID_CALLBACKS: u32 = 1;
/// `NVTX_CB_MODULE_CORE`.
const NVTX_CB_MODULE_CORE: u32 = 1;
/// `NvtxCallbackIdCore` entries the recorder replaces.
const NVTX_CBID_CORE_MARK_EX: usize = 1;
const NVTX_CBID_CORE_MARK_A: usize = 2;
const NVTX_CBID_CORE_RANGE_START_EX: usize = 4;
const NVTX_CBID_CORE_RANGE_START_A: usize = 5;
const NVTX_CBID_CORE_RANGE_END: usize = 7;

/// The static injection hook. NVTX's own definition is weak; this one wins at link time.
#[allow(non_upper_case_globals)]
#[unsafe(no_mangle)]
pub static InitializeInjectionNvtx2_fnptr: InitializeInjection = initialize_injection;

// ── Recorder ─────────────────────────────────────────────────────────────────

/// Which NVTX core entry point was called.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Call {
    MarkEx,
    MarkA,
    RangeStartEx,
    RangeStartA,
    RangeEnd,
}

/// One recorded NVTX call.
#[derive(Debug, Clone)]
struct Event {
    call: Call,
    thread: ThreadId,
    /// The range id returned by a start, or passed to an end.
    range_id: Option<u64>,
    message: Option<String>,
    /// `version`, `size` and `messageType` of an `*Ex` call's attributes.
    attributes: Option<(u16, u16, i32)>,
    /// `payloadType`. An `A` call has no attributes, which NVTX treats as
    /// `NVTX_PAYLOAD_UNKNOWN`, so it is recorded as that.
    payload_type: i32,
    /// `payload.ullValue`, read only when `payloadType` says it is valid.
    payload: Option<u64>,
}

static EVENTS: Mutex<Vec<Event>> = Mutex::new(Vec::new());
static HOOK_ATTACHED: AtomicBool = AtomicBool::new(false);
static NEXT_RANGE_ID: AtomicU64 = AtomicU64::new(1);

const UNSIGNED_INT64: i32 = PayloadType::NVTX_PAYLOAD_TYPE_UNSIGNED_INT64 as i32;
const UNKNOWN: i32 = PayloadType::NVTX_PAYLOAD_UNKNOWN as i32;
const ASCII: i32 = MessageType::NVTX_MESSAGE_TYPE_ASCII as i32;

fn lock<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    // A panicking test must not turn every later test into a poison error.
    mutex
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn push(event: Event) {
    lock(&EVENTS).push(event);
}

/// # Safety
/// `message` is null or a valid NUL-terminated string.
unsafe fn read_message(message: *const c_char) -> Option<String> {
    if message.is_null() {
        return None;
    }
    // SAFETY: non-null and NUL-terminated per the caller's contract.
    Some(
        unsafe { CStr::from_ptr(message) }
            .to_string_lossy()
            .into_owned(),
    )
}

/// # Safety
/// `attrs` is null or points to a valid `nvtxEventAttributes_t`.
unsafe fn ex_event(call: Call, attrs: *const EventAttributes, range_id: Option<u64>) -> Event {
    let mut event = Event {
        call,
        thread: std::thread::current().id(),
        range_id,
        message: None,
        attributes: None,
        payload_type: UNKNOWN,
        payload: None,
    };
    // SAFETY: null or valid per the caller's contract.
    let Some(attrs) = (unsafe { attrs.as_ref() }) else {
        return event;
    };
    event.attributes = Some((attrs.version, attrs.size, attrs.messageType));
    event.payload_type = attrs.payloadType;
    if attrs.payloadType == UNSIGNED_INT64 {
        // SAFETY: `payloadType` says `ullValue` is the active member.
        event.payload = Some(unsafe { attrs.payload.ullValue });
    }
    if attrs.messageType == ASCII {
        // SAFETY: `messageType` says `ascii` is the active member, a C string.
        event.message = unsafe { read_message(attrs.message.ascii) };
    }
    event
}

/// # Safety
/// `message` is null or a valid NUL-terminated string.
unsafe fn ascii_event(call: Call, message: *const c_char, range_id: Option<u64>) -> Event {
    Event {
        call,
        thread: std::thread::current().id(),
        range_id,
        // SAFETY: forwarded from NVTX's caller.
        message: unsafe { read_message(message) },
        attributes: None,
        payload_type: UNKNOWN,
        payload: None,
    }
}

unsafe extern "C" fn record_mark_ex(attrs: *const EventAttributes) {
    // SAFETY: NVTX passes the caller's attributes pointer through unchanged.
    push(unsafe { ex_event(Call::MarkEx, attrs, None) });
}

unsafe extern "C" fn record_mark_a(message: *const c_char) {
    // SAFETY: NVTX passes the caller's C string through unchanged.
    push(unsafe { ascii_event(Call::MarkA, message, None) });
}

unsafe extern "C" fn record_range_start_ex(attrs: *const EventAttributes) -> u64 {
    let id = NEXT_RANGE_ID.fetch_add(1, Ordering::Relaxed);
    // SAFETY: NVTX passes the caller's attributes pointer through unchanged.
    push(unsafe { ex_event(Call::RangeStartEx, attrs, Some(id)) });
    id
}

unsafe extern "C" fn record_range_start_a(message: *const c_char) -> u64 {
    let id = NEXT_RANGE_ID.fetch_add(1, Ordering::Relaxed);
    // SAFETY: NVTX passes the caller's C string through unchanged.
    push(unsafe { ascii_event(Call::RangeStartA, message, Some(id)) });
    id
}

unsafe extern "C" fn record_range_end(id: u64) {
    push(Event {
        call: Call::RangeEnd,
        thread: std::thread::current().id(),
        range_id: Some(id),
        message: None,
        attributes: None,
        payload_type: UNKNOWN,
        payload: None,
    });
}

/// NVTX's tool entry point: point the core module's slots at the recorders. Returning
/// 0 makes NVTX turn every call into a no-op, which the tests report as a missing hook.
unsafe extern "C" fn initialize_injection(get_export_table: Option<GetExportTable>) -> c_int {
    let Some(get_export_table) = get_export_table else {
        return 0;
    };
    // SAFETY: NVTX's own export-table accessor; it returns null for an unknown id.
    let callbacks = unsafe { get_export_table(NVTX_ETID_CALLBACKS) }.cast::<ExportTableCallbacks>();
    // SAFETY: null or NVTX's static `NvtxExportTableCallbacks`.
    let Some(callbacks) = (unsafe { callbacks.as_ref() }) else {
        return 0;
    };
    if callbacks.struct_size < std::mem::size_of::<ExportTableCallbacks>() {
        return 0;
    }
    let Some(get_module_function_table) = callbacks.get_module_function_table else {
        return 0;
    };
    let mut table: FunctionTable = std::ptr::null_mut();
    let mut size: c_uint = 0;
    // SAFETY: both out pointers are valid for writes.
    if unsafe { get_module_function_table(NVTX_CB_MODULE_CORE, &mut table, &mut size) } == 0
        || table.is_null()
    {
        return 0;
    }
    let recorders: [(usize, *const c_void); 5] = [
        (NVTX_CBID_CORE_MARK_EX, record_mark_ex as *const c_void),
        (NVTX_CBID_CORE_MARK_A, record_mark_a as *const c_void),
        (
            NVTX_CBID_CORE_RANGE_START_EX,
            record_range_start_ex as *const c_void,
        ),
        (
            NVTX_CBID_CORE_RANGE_START_A,
            record_range_start_a as *const c_void,
        ),
        (NVTX_CBID_CORE_RANGE_END, record_range_end as *const c_void),
    ];
    for (cbid, recorder) in recorders {
        if cbid >= size as usize {
            return 0;
        }
        // SAFETY: `cbid` is within the `size` slots NVTX reported for the table.
        let slot = unsafe { *table.add(cbid) };
        if slot.is_null() {
            return 0;
        }
        // SAFETY: `slot` is NVTX's function-pointer slot for `cbid`; NVTX has not
        // finished initializing, so nothing reads it concurrently.
        unsafe { *slot = recorder };
    }
    HOOK_ATTACHED.store(true, Ordering::Release);
    1
}

// ── Harness ──────────────────────────────────────────────────────────────────

/// Serializes the tests: the recorder and the `DYN_NVTX` switch are process-global.
static TEST_LOCK: Mutex<()> = Mutex::new(());

/// The request id used throughout, and its pinned key (the vector `src/nvtx.rs` pins).
/// Pinned rather than recomputed so a payload of the wrong key cannot pass.
const RID: &str = "6f1c2b0e-4a57-4c0c-9b3f-0d5e8a1b7c42";
const KEY: u64 = 1_346_822_055_457_142_067;

/// Run `f` with annotations switched on or off, and return the NVTX calls it made.
fn capture(annotations_on: bool, f: impl FnOnce()) -> Vec<Event> {
    let _serial = lock(&TEST_LOCK);

    // SAFETY: the environment is mutated only here, with TEST_LOCK held, and nothing in
    // this binary reads it on another thread meanwhile: every test runs its NVTX calls
    // (NVTX reads NVTX_INJECTION64_PATH on the first one) inside `f`, under the same
    // lock, and joins any thread it spawns before returning.
    unsafe {
        // A dynamic injection library would take precedence over the static hook.
        std::env::remove_var("NVTX_INJECTION64_PATH");
        std::env::set_var(nvtx::ENV_VAR, if annotations_on { "1" } else { "0" });
    }
    nvtx::init().expect("DYN_NVTX=1 and DYN_NVTX=0 are valid switch values");
    assert_eq!(nvtx::enabled(), annotations_on);

    lock(&EVENTS).clear();
    f();
    let events = std::mem::take(&mut *lock(&EVENTS));

    if annotations_on {
        assert!(
            HOOK_ATTACHED.load(Ordering::Acquire),
            "NVTX never called the static injection hook, so nothing could be recorded"
        );
    }
    events
}

/// Assert `event` is a payload-carrying `*Ex` call named `message` with payload `KEY`.
#[track_caller]
fn assert_keyed(event: &Event, call: Call, message: &str) {
    assert_eq!(event.call, call, "{event:?}");
    assert_eq!(event.message.as_deref(), Some(message), "{event:?}");
    assert_eq!(
        event.attributes,
        Some((
            nvtx_sys::NVTX_VERSION as u16,
            nvtx_sys::NVTX_EVENT_ATTRIBUTES_SIZE as u16,
            ASCII
        )),
        "attributes must declare this NVTX version, their full size and an ASCII message: {event:?}"
    );
    assert_eq!(
        event.payload_type, UNSIGNED_INT64,
        "payload type must be NVTX_PAYLOAD_TYPE_UNSIGNED_INT64: {event:?}"
    );
    assert_eq!(
        event.payload,
        Some(KEY),
        "payload must be rid_key(rid): {event:?}"
    );
}

/// Assert `event` is a plain `A` call named `message` with no payload.
#[track_caller]
fn assert_plain(event: &Event, call: Call, message: &str) {
    assert_eq!(event.call, call, "{event:?}");
    assert_eq!(event.message.as_deref(), Some(message), "{event:?}");
    assert_eq!(event.payload_type, UNKNOWN, "{event:?}");
    assert_eq!(event.payload, None, "{event:?}");
}

/// Assert `end` closes the range `start` opened.
#[track_caller]
fn assert_ends(start: &Event, end: &Event) {
    assert_eq!(end.call, Call::RangeEnd, "{end:?}");
    assert!(start.range_id.is_some(), "{start:?}");
    assert_eq!(end.range_id, start.range_id, "{start:?} / {end:?}");
}

fn identity_message() -> String {
    format!("{IDENTITY_STAGE} rid={RID} xrid=client-7 k=v")
}

const ATTRS: &[(&str, &str)] = &[("k", "v"), ("xrid", "client-7")];

// ── Tests ────────────────────────────────────────────────────────────────────

#[test]
fn the_pinned_key_is_rid_key() {
    assert_eq!(rid_key(RID), KEY);
}

#[test]
fn request_mark_carries_the_request_key() {
    let events = capture(true, || dynamo_nvtx_request_mark!("test.request_mark", RID));
    assert_eq!(events.len(), 1, "no attrs, so no identity mark: {events:?}");
    assert_keyed(&events[0], Call::MarkEx, "test.request_mark");
}

#[test]
fn request_mark_with_attrs_emits_the_identity_mark_first() {
    let events = capture(true, || {
        dynamo_nvtx_request_mark!("test.request_mark", RID, ATTRS)
    });
    assert_eq!(events.len(), 2, "{events:?}");
    assert_keyed(&events[0], Call::MarkEx, &identity_message());
    assert_keyed(&events[1], Call::MarkEx, "test.request_mark");
}

#[test]
fn request_range_carries_the_request_key() {
    let events = capture(true, || {
        let _range = dynamo_nvtx_request_range!("test.request_range", RID);
    });
    assert_eq!(events.len(), 2, "{events:?}");
    assert_keyed(&events[0], Call::RangeStartEx, "test.request_range");
    assert_ends(&events[0], &events[1]);
}

#[test]
fn request_range_with_attrs_emits_the_identity_mark_first() {
    let events = capture(true, || {
        let _range = dynamo_nvtx_request_range!("test.request_range", RID, ATTRS);
    });
    assert_eq!(events.len(), 3, "{events:?}");
    assert_keyed(&events[0], Call::MarkEx, &identity_message());
    assert_keyed(&events[1], Call::RangeStartEx, "test.request_range");
    assert_ends(&events[1], &events[2]);
}

#[test]
fn keyed_range_carries_the_key_only_when_given_one() {
    let events = capture(true, || {
        let key = nvtx::request_key(RID);
        assert_eq!(key, Some(KEY));
        let keyed = dynamo_nvtx_keyed_range!("test.keyed_range", key);
        let unkeyed = dynamo_nvtx_keyed_range!("test.unkeyed_range", None);
        drop(unkeyed);
        drop(keyed);
    });
    assert_eq!(events.len(), 4, "{events:?}");
    assert_keyed(&events[0], Call::RangeStartEx, "test.keyed_range");
    assert_plain(&events[1], Call::RangeStartA, "test.unkeyed_range");
    assert_ends(&events[1], &events[2]);
    assert_ends(&events[0], &events[3]);
}

#[test]
fn keyed_mark_carries_the_key_only_when_given_one() {
    let events = capture(true, || {
        dynamo_nvtx_keyed_mark!("test.keyed_mark", nvtx::request_key(RID));
        dynamo_nvtx_keyed_mark!("test.unkeyed_mark", None);
    });
    assert_eq!(events.len(), 2, "{events:?}");
    assert_keyed(&events[0], Call::MarkEx, "test.keyed_mark");
    assert_plain(&events[1], Call::MarkA, "test.unkeyed_mark");
}

#[test]
fn plain_marks_and_ranges_carry_no_payload() {
    let events = capture(true, || {
        let owned = String::from("test.plain_mark.expr");
        dynamo_nvtx_mark!("test.plain_mark");
        dynamo_nvtx_mark!("test.plain_mark.{}", 7);
        dynamo_nvtx_mark!(&owned);
        let _range = dynamo_nvtx_range!("test.plain_range.{}", 8);
    });
    assert_eq!(events.len(), 5, "{events:?}");
    assert_plain(&events[0], Call::MarkA, "test.plain_mark");
    assert_plain(&events[1], Call::MarkA, "test.plain_mark.7");
    assert_plain(&events[2], Call::MarkA, "test.plain_mark.expr");
    assert_plain(&events[3], Call::RangeStartA, "test.plain_range.8");
    assert_ends(&events[3], &events[4]);
}

/// The reason ranges are start/end pairs rather than push/pop: a guard held across an
/// `.await` may be dropped on another worker thread, and must still end its own range.
#[test]
fn a_range_dropped_on_another_thread_ends_its_own_range() {
    let events = capture(true, || {
        let range = dynamo_nvtx_request_range!("test.cross_thread", RID);
        let joined = std::thread::spawn(move || drop(range)).join();
        assert!(joined.is_ok(), "the dropping thread panicked");
    });
    assert_eq!(events.len(), 2, "{events:?}");
    assert_keyed(&events[0], Call::RangeStartEx, "test.cross_thread");
    assert_ends(&events[0], &events[1]);
    assert_ne!(
        events[0].thread, events[1].thread,
        "the range must end on the thread that dropped the guard"
    );
}

#[test]
fn nothing_reaches_nvtx_while_annotations_are_off() {
    let events = capture(false, || {
        assert_eq!(nvtx::request_key(RID), None);
        dynamo_nvtx_request_mark!("test.off", RID, ATTRS);
        let _request = dynamo_nvtx_request_range!("test.off", RID, ATTRS);
        let _keyed = dynamo_nvtx_keyed_range!("test.off", Some(KEY));
        dynamo_nvtx_keyed_mark!("test.off", Some(KEY));
        dynamo_nvtx_mark!("test.off");
        let _range = dynamo_nvtx_range!("test.off");
    });
    assert!(events.is_empty(), "{events:?}");
}
