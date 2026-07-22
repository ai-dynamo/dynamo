use super::*;
use std::{
    ffi::c_void,
    mem::{align_of, size_of},
    slice, str,
};

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
