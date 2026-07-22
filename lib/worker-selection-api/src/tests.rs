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
        __private::required_candidate_inputs::<ShimPlugin>(state, &mut required_inputs, &mut error)
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
