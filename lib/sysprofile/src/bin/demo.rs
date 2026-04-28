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
        let ann = perfetto::build_debug_annotation_str("traceparent", traceparent);
        self.writer.write_slice_begin(ts, name_iid, &[ann]);
    }

    fn end(&self, ts: u64) {
        self.writer.write_slice_end(ts);
    }

    fn instant(&self, ts: u64, name_iid: u64, traceparent: &str) {
        let ann = perfetto::build_debug_annotation_str("traceparent", traceparent);
        self.writer.write_instant(ts, name_iid, &[ann]);
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

    let host = "node-0";
    let base_ts = now_ns();

    let frontend = Component::new(&output_dir, "frontend", host, 1000, 1);
    let router = Component::new(&output_dir, "router", host, 1001, 2);
    let prefill = Component::new(&output_dir, "engine-prefill-0", host, 2000, 3);
    let decode = Component::new(&output_dir, "engine-decode-0", host, 2001, 4);

    let iid_recv = frontend.intern("dynamo.frontend.recv");
    let iid_preprocess = frontend.intern("dynamo.frontend.preprocess");
    let iid_route = router.intern("dynamo.router.schedule");
    let iid_kv_lookup = router.intern("dynamo.router.kv_lookup");
    let iid_transport_send = router.intern("dynamo.transport.send");
    let iid_transport_recv = prefill.intern("dynamo.transport.recv");
    let iid_prefill_compute = prefill.intern("dynamo.prefill.compute");
    let iid_kv_transfer = prefill.intern("dynamo.prefill.kv_transfer");
    let iid_decode_recv = decode.intern("dynamo.decode.recv");
    let iid_decode_compute = decode.intern("dynamo.decode.compute");
    let iid_decode_first = decode.intern("dynamo.decode.first_token");
    let iid_decode_send = decode.intern("dynamo.decode.detok_send");

    let num_requests = 20u32;
    let ms = 1_000_000u64; // 1ms in ns

    eprintln!("  simulating {} requests across 4 components", num_requests);

    for req in 0..num_requests {
        let tp = make_traceparent(req);
        let req_base = base_ts + (req as u64) * 50 * ms;

        // -- Frontend: receive HTTP request --
        let t0 = req_base;
        frontend.begin(t0, iid_recv, &tp);
        {
            // -- Frontend: preprocess (tokenize) --
            let t1 = t0 + 200 * (ms / 1000); // 0.2ms
            frontend.begin(t1, iid_preprocess, &tp);
            let t2 = t1 + 800 * (ms / 1000) + (req as u64 * 100 * (ms / 1000)); // 0.8-2.8ms
            frontend.end(t2);
        }
        let t_recv_end = t0 + 2 * ms + (req as u64 * 150 * (ms / 1000));
        frontend.end(t_recv_end);

        // -- Router: schedule --
        let t_route_start = t_recv_end + 100 * (ms / 1000);
        router.begin(t_route_start, iid_route, &tp);
        {
            // KV lookup
            let t_kv = t_route_start + 50 * (ms / 1000);
            router.begin(t_kv, iid_kv_lookup, &tp);
            let t_kv_end = t_kv + 300 * (ms / 1000) + (req as u64 * 50 * (ms / 1000));
            router.end(t_kv_end);
        }
        let t_route_end = t_route_start + ms + (req as u64 * 80 * (ms / 1000));
        router.end(t_route_end);

        // -- Router: transport send --
        let t_send = t_route_end + 50 * (ms / 1000);
        router.begin(t_send, iid_transport_send, &tp);
        let t_send_end = t_send + 500 * (ms / 1000);
        router.end(t_send_end);

        // -- Prefill engine: transport recv --
        let t_prefill_recv = t_send_end + 200 * (ms / 1000); // network latency
        prefill.begin(t_prefill_recv, iid_transport_recv, &tp);
        let t_prefill_recv_end = t_prefill_recv + 100 * (ms / 1000);
        prefill.end(t_prefill_recv_end);

        // -- Prefill engine: compute --
        let t_compute = t_prefill_recv_end + 50 * (ms / 1000);
        prefill.begin(t_compute, iid_prefill_compute, &tp);
        // Prefill time varies with prompt length (5-25ms)
        let prefill_duration = (5 + req % 20) as u64 * ms;
        let t_compute_end = t_compute + prefill_duration;
        prefill.end(t_compute_end);

        // -- Prefill engine: KV transfer to decode --
        let t_transfer = t_compute_end + 100 * (ms / 1000);
        prefill.begin(t_transfer, iid_kv_transfer, &tp);
        let t_transfer_end = t_transfer + 2 * ms;
        prefill.end(t_transfer_end);

        // -- Decode engine: recv from prefill --
        let t_decode_recv = t_transfer_end + 150 * (ms / 1000);
        decode.begin(t_decode_recv, iid_decode_recv, &tp);
        let t_decode_recv_end = t_decode_recv + 80 * (ms / 1000);
        decode.end(t_decode_recv_end);

        // -- Decode engine: first token --
        let t_first = t_decode_recv_end + 100 * (ms / 1000);
        decode.instant(t_first, iid_decode_first, &tp);

        // -- Decode engine: generate tokens (3-8 decode iterations) --
        let num_tokens = 3 + (req % 6);
        let mut t_tok = t_first;
        for tok in 0..num_tokens {
            let t_step = t_tok + ms + (tok as u64 * 200 * (ms / 1000));
            decode.begin(t_step, iid_decode_compute, &tp);
            let decode_step_duration = ms + (req as u64 * 30 * (ms / 1000));
            let t_step_end = t_step + decode_step_duration;
            decode.end(t_step_end);
            t_tok = t_step_end;
        }

        // -- Decode engine: detokenize + send --
        let t_detok = t_tok + 50 * (ms / 1000);
        decode.begin(t_detok, iid_decode_send, &tp);
        let t_detok_end = t_detok + 300 * (ms / 1000);
        decode.end(t_detok_end);
    }

    // Finish all writers
    let fe_count = frontend.writer.finish().unwrap();
    let rt_count = router.writer.finish().unwrap();
    let pf_count = prefill.writer.finish().unwrap();
    let dc_count = decode.writer.finish().unwrap();

    eprintln!();
    eprintln!("  traces written:");
    eprintln!("    frontend.pftrace.gz        ({fe_count} packets)");
    eprintln!("    router.pftrace.gz          ({rt_count} packets)");
    eprintln!("    engine-prefill-0.pftrace.gz ({pf_count} packets)");
    eprintln!("    engine-decode-0.pftrace.gz  ({dc_count} packets)");
    eprintln!();
    eprintln!("  open in Perfetto UI:");
    eprintln!("    1. Go to https://ui.perfetto.dev");
    eprintln!("    2. Click 'Open trace file'");
    eprintln!("    3. Select one or more .pftrace.gz files from {}", output_dir.display());
    eprintln!();
    eprintln!("  tip: Perfetto supports opening multiple traces simultaneously.");
    eprintln!("       Open all 4 files to see the full cross-component timeline.");
}
