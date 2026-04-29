// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! `dynamo-sysprofile-demo` — generate a synthetic multi-component Perfetto
//! trace that simulates a Dynamo distributed inference pipeline.
//!
//! Produces `.pftrace.gz` files (one per component) in the output directory.
//! Open them in <https://ui.perfetto.dev> to visualize.
//!
//! Usage:
//!     dynamo-sysprofile-demo [output-dir]
//!     # default: ./sysprofile-demo-output/

use std::path::PathBuf;
use std::time::{SystemTime, UNIX_EPOCH};

use dynamo_sysprofile::perfetto;
use dynamo_sysprofile::writer::TraceWriter;

fn now_ns() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64
}

fn make_traceparent(request_idx: u32) -> String {
    format!(
        "00-{:032x}-{:016x}-01",
        request_idx as u128 + 0xdead_0000,
        request_idx as u64 + 0xbeef
    )
}

const RUN_ID: &str = "demo-bench-001";
const MS: u64 = 1_000_000; // 1ms in nanoseconds
const US: u64 = 1_000; // 1us in nanoseconds

struct Component {
    writer: TraceWriter,
}

impl Component {
    fn new(dir: &PathBuf, name: &str, host: &str, pid: u32, seq_id: u32) -> Self {
        let path = dir.join(format!("{name}.pftrace.gz"));
        let writer = TraceWriter::new(path, name, host, pid, seq_id)
            .unwrap_or_else(|e| panic!("failed to create writer for {name}: {e}"));
        Self { writer }
    }

    fn intern(&self, stage: &str) -> u64 {
        self.writer.intern_name(stage)
    }

    fn begin(&self, ts: u64, name_iid: u64, traceparent: &str) {
        let anns = vec![
            perfetto::build_debug_annotation_str("traceparent", traceparent),
            perfetto::build_debug_annotation_str("dynamo.run_id", RUN_ID),
        ];
        self.writer.write_slice_begin(ts, name_iid, &anns);
    }

    fn end(&self, ts: u64) {
        self.writer.write_slice_end(ts);
    }

    fn instant(&self, ts: u64, name_iid: u64, traceparent: &str) {
        let anns = vec![
            perfetto::build_debug_annotation_str("traceparent", traceparent),
            perfetto::build_debug_annotation_str("dynamo.run_id", RUN_ID),
        ];
        self.writer.write_instant(ts, name_iid, &anns);
    }

    fn named_track(&self, name: &str) -> u64 {
        self.writer.create_named_track(name)
    }

    fn gpu_kernel(&self, track: u64, ts: u64, dur: u64, name_iid: u64, stage: &str) {
        let anns = vec![perfetto::build_debug_annotation_str("stage", stage)];
        self.writer.write_slice_begin_on_track(track, ts, name_iid, &anns);
        self.writer.write_slice_end_on_track(track, ts + dur);
    }

    fn nccl_op(&self, track: u64, ts: u64, dur: u64, name_iid: u64, bytes: u64) {
        let anns = vec![perfetto::build_debug_annotation_str("bytes", &bytes.to_string())];
        self.writer.write_slice_begin_on_track(track, ts, name_iid, &anns);
        self.writer.write_slice_end_on_track(track, ts + dur);
    }

    fn nats_event(&self, track: u64, ts: u64, name_iid: u64, msg_id: &str) {
        let anns = vec![perfetto::build_debug_annotation_str("msg_id", msg_id)];
        self.writer.write_instant_on_track(track, ts, name_iid, &anns);
    }
}

fn main() {
    let output_dir = std::env::args()
        .nth(1)
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("sysprofile-demo-output"));

    std::fs::create_dir_all(&output_dir).expect("failed to create output directory");

    eprintln!("dynamo-sysprofile-demo: generating synthetic traces...");
    eprintln!("  output: {}", output_dir.display());
    eprintln!("  run_id: {RUN_ID}");

    let base_ts = now_ns();

    // Components across two hosts
    let frontend = Component::new(&output_dir, "frontend", "node-0", 1000, 1);
    let router = Component::new(&output_dir, "router", "node-0", 1001, 2);
    let prefill0 = Component::new(&output_dir, "engine-prefill-0", "node-0", 2000, 3);
    let prefill1 = Component::new(&output_dir, "engine-prefill-1", "node-1", 2100, 5);
    let decode = Component::new(&output_dir, "engine-decode-0", "node-0", 2001, 4);

    // Intern stage names
    let iid_recv = frontend.intern("dynamo.frontend.recv");
    let iid_preprocess = frontend.intern("dynamo.frontend.preprocess");
    let iid_route = router.intern("dynamo.router.schedule");
    let iid_kv_lookup = router.intern("dynamo.router.kv_lookup");
    let iid_metrics = router.intern("dynamo.router.metrics");
    let iid_transport_send = router.intern("dynamo.transport.send");
    let iid_transport_recv0 = prefill0.intern("dynamo.transport.recv");
    let iid_transport_recv1 = prefill1.intern("dynamo.transport.recv");
    let iid_prefill_compute0 = prefill0.intern("dynamo.prefill.compute");
    let iid_prefill_compute1 = prefill1.intern("dynamo.prefill.compute");
    let iid_kv_transfer0 = prefill0.intern("dynamo.prefill.kv_transfer");
    let iid_kv_transfer1 = prefill1.intern("dynamo.prefill.kv_transfer");
    let iid_decode_recv = decode.intern("dynamo.decode.recv");
    let iid_decode_compute = decode.intern("dynamo.decode.compute");
    let iid_decode_first = decode.intern("dynamo.decode.first_token");
    let iid_decode_send = decode.intern("dynamo.decode.detok_send");

    // GPU kernel names
    let iid_gemm = prefill0.intern("gemm_fp16_kernel");
    let iid_attn = prefill0.intern("flash_attn_fwd");
    let iid_lnorm = prefill0.intern("layer_norm_kernel");
    let iid_softmax = prefill0.intern("softmax_kernel");
    let iid_gemm1 = prefill1.intern("gemm_fp16_kernel");
    let iid_attn1 = prefill1.intern("flash_attn_fwd");
    let iid_lnorm1 = prefill1.intern("layer_norm_kernel");
    let iid_softmax1 = prefill1.intern("softmax_kernel");
    let iid_dec_gemm = decode.intern("gemm_fp16_kernel");
    let iid_dec_attn = decode.intern("flash_attn_fwd");

    // NCCL operation names
    let iid_allreduce0 = prefill0.intern("nccl_AllReduce");
    let iid_allgather0 = prefill0.intern("nccl_AllGather");
    let iid_allreduce1 = prefill1.intern("nccl_AllReduce");
    let iid_allgather1 = prefill1.intern("nccl_AllGather");

    // NATS event names
    let iid_nats_pub = router.intern("nats_publish");
    let iid_nats_sub0 = prefill0.intern("nats_subscribe");
    let iid_nats_sub1 = prefill1.intern("nats_subscribe");
    let iid_nats_sub_dec = decode.intern("nats_subscribe");

    // Create named tracks
    let gpu0_pf0 = prefill0.named_track("GPU 0 Compute");
    let gpu1_pf0 = prefill0.named_track("GPU 1 Compute");
    let gpu0_pf1 = prefill1.named_track("GPU 0 Compute");
    let gpu1_pf1 = prefill1.named_track("GPU 1 Compute");
    let gpu0_dec = decode.named_track("GPU 0 Compute");

    let nccl_pf0 = prefill0.named_track("NCCL AllReduce");
    let nccl_pf1 = prefill1.named_track("NCCL AllReduce");

    let nats_router = router.named_track("NATS Communication");
    let nats_pf0 = prefill0.named_track("NATS Communication");
    let nats_pf1 = prefill1.named_track("NATS Communication");
    let nats_dec = decode.named_track("NATS Communication");

    let num_requests = 40u32;

    eprintln!("  simulating {num_requests} requests across 5 components on 2 hosts");

    for req in 0..num_requests {
        let tp = make_traceparent(req);
        let req_base = base_ts + (req as u64) * 30 * MS;

        // Route even requests to prefill-0, odd to prefill-1
        let use_prefill1 = req % 2 == 1;
        let prefill = if use_prefill1 { &prefill1 } else { &prefill0 };
        let iid_transport_recv = if use_prefill1 { iid_transport_recv1 } else { iid_transport_recv0 };
        let iid_prefill_compute = if use_prefill1 { iid_prefill_compute1 } else { iid_prefill_compute0 };
        let iid_kv_transfer = if use_prefill1 { iid_kv_transfer1 } else { iid_kv_transfer0 };

        // Vary timing to produce diverse critical paths
        let prompt_factor = 1 + (req % 8) as u64; // 1-8x multiplier
        let jitter = (req as u64 * 137) % 500; // pseudo-random jitter in us

        // == Frontend: receive HTTP request ==
        let t0 = req_base;
        frontend.begin(t0, iid_recv, &tp);
        {
            let t1 = t0 + 100 * US;
            frontend.begin(t1, iid_preprocess, &tp);
            // Tokenization: 0.3ms - 2.5ms depending on prompt
            let preprocess_dur = 300 * US + prompt_factor * 280 * US;
            frontend.end(t1 + preprocess_dur);
        }
        // Frontend total: 1ms - 4ms
        let fe_dur = MS + prompt_factor * 400 * US + jitter * US;
        let t_fe_end = t0 + fe_dur;
        frontend.end(t_fe_end);

        // == Router: schedule ==
        let t_route_start = t_fe_end + 80 * US;
        router.begin(t_route_start, iid_route, &tp);
        {
            // KV cache lookup: 0.1ms - 0.8ms
            let t_kv = t_route_start + 30 * US;
            router.begin(t_kv, iid_kv_lookup, &tp);
            let kv_dur = 100 * US + (req as u64 % 7) * 100 * US;
            router.end(t_kv + kv_dur);

            // Metrics recording
            let t_met = t_kv + kv_dur + 20 * US;
            router.begin(t_met, iid_metrics, &tp);
            router.end(t_met + 50 * US);
        }
        // Router total: 0.5ms - 1.5ms
        let route_dur = 500 * US + (req as u64 % 10) * 100 * US;
        let t_route_end = t_route_start + route_dur;
        router.end(t_route_end);

        // == NATS: router publishes request ==
        let msg_id = format!("msg-{req:04}");
        router.nats_event(nats_router, t_route_end + 10 * US, iid_nats_pub, &msg_id);

        // == Transport: send (router side) ==
        let t_send = t_route_end + 30 * US;
        router.begin(t_send, iid_transport_send, &tp);
        let send_dur = 200 * US + jitter * US / 3;
        let t_send_end = t_send + send_dur;
        router.end(t_send_end);

        // == Transport: recv (engine side) ==
        // Cross-host latency: 50-300us
        let network_latency = 50 * US + if use_prefill1 { 200 * US } else { 30 * US };
        let t_prefill_recv = t_send_end + network_latency;
        prefill.begin(t_prefill_recv, iid_transport_recv, &tp);
        prefill.end(t_prefill_recv + 80 * US);

        // == NATS: engine subscribes to request ==
        let (nats_engine, iid_nats_sub_engine) = if use_prefill1 {
            (nats_pf1, iid_nats_sub1)
        } else {
            (nats_pf0, iid_nats_sub0)
        };
        prefill.nats_event(nats_engine, t_prefill_recv + 10 * US, iid_nats_sub_engine, &msg_id);

        // == Prefill: compute ==
        let t_compute = t_prefill_recv + 120 * US;
        prefill.begin(t_compute, iid_prefill_compute, &tp);
        // Prefill time: 3ms - 20ms depending on prompt length
        let prefill_dur = 3 * MS + prompt_factor * 2 * MS + jitter * 3 * US;
        let t_compute_end = t_compute + prefill_dur;
        prefill.end(t_compute_end);

        // == GPU kernels during prefill compute ==
        {
            let (gpu0, gpu1) = if use_prefill1 { (gpu0_pf1, gpu1_pf1) } else { (gpu0_pf0, gpu1_pf0) };
            let (k_gemm, k_attn, k_lnorm, k_softmax) = if use_prefill1 {
                (iid_gemm1, iid_attn1, iid_lnorm1, iid_softmax1)
            } else {
                (iid_gemm, iid_attn, iid_lnorm, iid_softmax)
            };
            let nccl_track = if use_prefill1 { nccl_pf1 } else { nccl_pf0 };
            let (k_ar, k_ag) = if use_prefill1 { (iid_allreduce1, iid_allgather1) } else { (iid_allreduce0, iid_allgather0) };

            // TP imbalance: prefill-1 kernels are ~10% slower
            let tp_slow = if use_prefill1 { 110 } else { 100 };

            let mut t_gpu = t_compute + 20 * US;
            let num_layers = 2 + (prompt_factor as u32 % 3);

            for layer in 0..num_layers {
                // Attention kernel on GPU 0
                let attn_dur = (400 + layer as u64 * 50 + prompt_factor * 30) * US * tp_slow / 100;
                prefill.gpu_kernel(gpu0, t_gpu, attn_dur, k_attn, "prefill");

                // Softmax on GPU 1
                let sm_dur = (100 + prompt_factor * 15) * US * tp_slow / 100;
                prefill.gpu_kernel(gpu1, t_gpu + 50 * US, sm_dur, k_softmax, "prefill");

                t_gpu += attn_dur + 30 * US;

                // GEMM on GPU 0
                let gemm_dur = (600 + prompt_factor * 80) * US * tp_slow / 100;
                prefill.gpu_kernel(gpu0, t_gpu, gemm_dur, k_gemm, "prefill");

                // LayerNorm on GPU 1
                let ln_dur = (80 + prompt_factor * 10) * US * tp_slow / 100;
                prefill.gpu_kernel(gpu1, t_gpu + 100 * US, ln_dur, k_lnorm, "prefill");

                t_gpu += gemm_dur + 20 * US;

                // NCCL AllReduce between layers
                let ar_dur = (200 + (layer as u64) * 40) * US;
                let ar_bytes = 16 * 1024 * 1024; // 16 MB
                prefill.nccl_op(nccl_track, t_gpu, ar_dur, k_ar, ar_bytes);
                t_gpu += ar_dur + 10 * US;
            }

            // Final NCCL AllGather
            let ag_dur = 300 * US;
            prefill.nccl_op(nccl_track, t_gpu, ag_dur, k_ag, 8 * 1024 * 1024);
        }

        // == Prefill: KV transfer to decode ==
        let t_transfer = t_compute_end + 60 * US;
        prefill.begin(t_transfer, iid_kv_transfer, &tp);
        // KV transfer: 1ms - 3ms
        let transfer_dur = MS + (req as u64 % 4) * 500 * US;
        let t_transfer_end = t_transfer + transfer_dur;
        prefill.end(t_transfer_end);

        // == NATS: decode subscribes ==
        let dec_msg = format!("dec-{req:04}");
        prefill.nats_event(nats_engine, t_transfer + 10 * US, iid_nats_sub_engine, &dec_msg);
        decode.nats_event(nats_dec, t_transfer_end + 30 * US, iid_nats_sub_dec, &dec_msg);

        // == Decode: recv ==
        let t_decode_recv = t_transfer_end + 100 * US;
        decode.begin(t_decode_recv, iid_decode_recv, &tp);
        decode.end(t_decode_recv + 60 * US);

        // == Decode: first token instant ==
        let t_first = t_decode_recv + 120 * US;
        decode.instant(t_first, iid_decode_first, &tp);

        // == Decode: token generation (3-10 iterations) ==
        let num_tokens = 3 + (req % 8);
        let mut t_tok = t_first;
        for tok in 0..num_tokens {
            let step_start = t_tok + 500 * US;
            decode.begin(step_start, iid_decode_compute, &tp);
            // Each decode step: 0.8ms - 1.5ms
            let step_dur = 800 * US + (tok as u64 * 70 * US) + (req as u64 % 5) * 50 * US;
            decode.end(step_start + step_dur);

            // GPU kernels during decode step
            let dec_gemm_dur = (300 + tok as u64 * 20) * US;
            decode.gpu_kernel(gpu0_dec, step_start + 30 * US, dec_gemm_dur, iid_dec_gemm, "decode");
            let dec_attn_dur = (200 + tok as u64 * 15) * US;
            decode.gpu_kernel(gpu0_dec, step_start + 30 * US + dec_gemm_dur + 10 * US, dec_attn_dur, iid_dec_attn, "decode");

            t_tok = step_start + step_dur;
        }

        // == Decode: detokenize + send ==
        let t_detok = t_tok + 40 * US;
        decode.begin(t_detok, iid_decode_send, &tp);
        decode.end(t_detok + 200 * US);
    }

    // Finish all writers
    let fe_count = frontend.writer.finish().unwrap();
    let rt_count = router.writer.finish().unwrap();
    let pf0_count = prefill0.writer.finish().unwrap();
    let pf1_count = prefill1.writer.finish().unwrap();
    let dc_count = decode.writer.finish().unwrap();

    eprintln!();
    eprintln!("  traces written:");
    eprintln!("    frontend.pftrace.gz           ({fe_count} packets)");
    eprintln!("    router.pftrace.gz             ({rt_count} packets)");
    eprintln!("    engine-prefill-0.pftrace.gz   ({pf0_count} packets)");
    eprintln!("    engine-prefill-1.pftrace.gz   ({pf1_count} packets)");
    eprintln!("    engine-decode-0.pftrace.gz    ({dc_count} packets)");
    eprintln!();
    eprintln!("  merge into report:");
    eprintln!("    dynamo-sysprofile-merge {}", output_dir.display());
    eprintln!();
    eprintln!("  or open individual traces in Perfetto UI:");
    eprintln!("    https://ui.perfetto.dev");
}
