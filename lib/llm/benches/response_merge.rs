// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//! CPU cost of joining SGLang `/generate` pieces, with Qwen3-30B-A3B shapes
//! (48 layers, top-8 experts). Before timing, it prints the allocations each
//! scenario makes, so copies and extra memory show up as numbers.
//!
//! cargo bench -p dynamo-llm --bench response_merge

use std::alloc::System;
use std::hint::black_box;

use axum::response::sse::Event;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use criterion::{BatchSize, Criterion, criterion_group, criterion_main};
use dynamo_llm::protocols::response_merge::ResponseAssembler;
use dynamo_llm::protocols::sglang::merge::merger;
use serde_json::{Value, json};
use stats_alloc::{INSTRUMENTED_SYSTEM, Region, StatsAlloc};

#[global_allocator]
static GLOBAL: &StatsAlloc<System> = &INSTRUMENTED_SYSTEM;

const PROMPT: usize = 2048;
const OUTPUT: usize = 512;
const EXPERT_ROW_BYTES: usize = 48 * 8 * 4;

/// Run `work` on `input` and print what it allocated beyond `input` itself.
fn report<T, R>(name: &str, input: T, work: impl FnOnce(T) -> R) {
    let region = Region::new(GLOBAL);
    drop(black_box(work(input)));
    let stats = region.change();
    let mib = |bytes: isize| bytes as f64 / (1 << 20) as f64;
    let allocated = stats.bytes_allocated as isize + stats.bytes_reallocated.max(0);
    println!(
        "{name:<24} allocations {:>6}  reallocations {:>5}  allocated {:>8.2} MiB",
        stats.allocations,
        stats.reallocations,
        mib(allocated),
    );
}

fn experts(rows: usize, fill: u8) -> String {
    STANDARD.encode(vec![fill; rows * EXPERT_ROW_BYTES])
}

fn prompt_logprobs() -> Value {
    Value::Array((0..PROMPT).map(|id| json!([-0.25, id, null])).collect())
}

/// SGLang's incremental chunks, one token each. Aggregated chunks repeat the
/// prompt logprobs; P/D decode chunks carry none and stale expert rows.
fn chunks(aggregated: bool) -> Vec<Value> {
    (0..OUTPUT)
        .map(|i| {
            let last = i + 1 == OUTPUT;
            let mut meta = json!({
                "id": "bench", "prompt_tokens": PROMPT, "completion_tokens": i + 1,
                "cached_tokens": 0, "output_token_logprobs_length": i + 1,
                "finish_reason": if last { json!({"type": "length"}) } else { Value::Null },
                "input_token_logprobs": if aggregated { prompt_logprobs() } else { json!([]) },
                "output_token_logprobs": [[-0.5, i, null]],
                "output_top_logprobs": [[[-0.5, i, null], [-1.5, i + 1, null]]],
            });
            if last {
                meta["routed_experts"] = json!(experts(PROMPT + OUTPUT - 1, 0));
            }
            json!({"text": "tok ", "output_ids": [i], "meta_info": meta})
        })
        .collect()
}

/// A short request without logprobs or routed experts: 64 tokens.
fn short_chunks() -> Vec<Value> {
    (0..64)
        .map(|i| {
            let finish = if i == 63 {
                json!({"type": "length"})
            } else {
                Value::Null
            };
            json!({"text": "tok ", "output_ids": [i], "meta_info": {
                "id": "bench", "finish_reason": finish, "prompt_tokens": 128,
                "completion_tokens": i + 1, "cached_tokens": 0,
            }})
        })
        .collect()
}

fn prefill() -> Value {
    json!({"text": "tok ", "output_ids": [0], "meta_info": {
        "input_token_logprobs": prompt_logprobs(),
        "routed_experts": experts(PROMPT, 1),
    }})
}

fn unary(prefill: Option<Value>, chunks: Vec<Value>) -> Value {
    let mut assembler = ResponseAssembler::new(merger());
    if let Some(prefill) = prefill {
        assembler.set_prefill(prefill);
    }
    for chunk in chunks {
        assembler.push(chunk);
    }
    assembler.finish().unwrap()
}

/// Ways to turn a chunk into the SSE event the frontend writes.
type Render = Box<dyn FnMut(&Value) -> Event>;
type NamedRender = (&'static str, fn() -> Render);

fn renders() -> [NamedRender; 3] {
    [
        ("string", || {
            Box::new(|chunk| Event::default().data(chunk.to_string()))
        }),
        ("buffer", || {
            // One serialization buffer reused across chunks.
            let mut buffer = Vec::new();
            Box::new(move |chunk| {
                buffer.clear();
                serde_json::to_writer(&mut buffer, chunk).unwrap();
                Event::default().data(std::str::from_utf8(&buffer).unwrap())
            })
        }),
        ("json_data", || {
            Box::new(|chunk| Event::default().json_data(chunk).unwrap())
        }),
    ]
}

/// Each event is dropped once written, as the HTTP body would.
fn streamed(prefill: Value, chunks: Vec<Value>, mut render: Render) -> usize {
    let mut assembler = ResponseAssembler::new(merger());
    assembler.set_prefill(prefill);
    let events = chunks
        .into_iter()
        .map(|chunk| assembler.render_chunk(chunk, &mut render));
    events.fold(0, |count, event| {
        drop(black_box(event));
        count + 1
    })
}

fn streamed_without_prefill(chunks: Vec<Value>, mut render: Render) -> usize {
    let mut assembler = ResponseAssembler::new(merger());
    let events = chunks
        .into_iter()
        .map(|chunk| assembler.render_chunk(chunk, &mut render));
    events.fold(0, |count, event| {
        drop(black_box(event));
        count + 1
    })
}

/// Floor for streaming: serializing chunks that already hold prefill's values.
fn serialize_joined(chunks: &[Value], render: Render) -> usize {
    chunks.iter().map(render).fold(0, |count, event| {
        drop(black_box(event));
        count + 1
    })
}

fn joined_chunks() -> Vec<Value> {
    let mut chunks = chunks(false);
    for chunk in &mut chunks {
        chunk["meta_info"]["input_token_logprobs"] = prompt_logprobs();
    }
    chunks
}

fn bench(c: &mut Criterion) {
    report("unary aggregated", chunks(true), |chunks| {
        unary(None, chunks)
    });
    report("unary p/d", (prefill(), chunks(false)), |(p, c)| {
        unary(Some(p), c)
    });
    for (name, render) in renders() {
        let joined = (prefill(), chunks(false));
        report(&format!("stream p/d {name}"), joined, |(p, c)| {
            streamed(p, c, render())
        });
        report(&format!("stream floor {name}"), joined_chunks(), |c| {
            serialize_joined(&c, render())
        });
    }

    // Each joined scenario is paired with the work it cannot avoid: dropping
    // the pieces it was handed, or serializing the bytes it must send.
    let large = BatchSize::LargeInput;
    c.bench_function("unary aggregated", |b| {
        b.iter_batched(|| chunks(true), |chunks| unary(None, chunks), large)
    });
    c.bench_function("unary aggregated: drop input", |b| {
        b.iter_batched(|| chunks(true), drop, large)
    });
    c.bench_function("unary p/d", |b| {
        b.iter_batched(
            || (prefill(), chunks(false)),
            |(p, c)| unary(Some(p), c),
            large,
        )
    });
    c.bench_function("unary p/d: drop input", |b| {
        b.iter_batched(|| (prefill(), chunks(false)), drop, large)
    });
    // No regression for requests that use no merged fields: main rendered
    // each chunk with `to_string`; the new path reuses one buffer.
    c.bench_function("short stream: main render", |b| {
        b.iter_batched(
            short_chunks,
            |c| {
                c.into_iter().fold(0, |count, c| {
                    drop(black_box(Event::default().data(c.to_string())));
                    count + 1
                })
            },
            large,
        )
    });
    c.bench_function("short stream: assembler render", |b| {
        b.iter_batched(
            short_chunks,
            |c| streamed_without_prefill(c, (renders()[1].1)()),
            large,
        )
    });
    c.bench_function("short unary: join", |b| {
        b.iter_batched(short_chunks, |chunks| unary(None, chunks), large)
    });
    // Latency split: one push while tokens arrive, and finish after the last.
    for (name, aggregated) in [("aggregated", true), ("p/d", false)] {
        let pushed = move || {
            let mut chunks = chunks(aggregated);
            let last = chunks.pop().unwrap();
            let mut assembler = ResponseAssembler::new(merger());
            if !aggregated {
                assembler.set_prefill(prefill());
            }
            chunks.into_iter().for_each(|chunk| assembler.push(chunk));
            (assembler, last)
        };
        c.bench_function(&format!("unary {name}: push one chunk"), |b| {
            b.iter_batched(
                pushed,
                |(mut a, last)| {
                    a.push(last);
                    a
                },
                large,
            )
        });
        c.bench_function(&format!("unary {name}: push one chunk, drop only"), |b| {
            b.iter_batched(
                pushed,
                |(a, last)| {
                    drop(last);
                    a
                },
                large,
            )
        });
        c.bench_function(&format!("unary {name}: finish"), |b| {
            b.iter_batched(
                move || {
                    let (mut a, last) = pushed();
                    a.push(last);
                    a
                },
                ResponseAssembler::finish,
                large,
            )
        });
    }
    for (name, render) in renders() {
        c.bench_function(&format!("stream p/d {name}"), |b| {
            b.iter_batched(
                || (prefill(), chunks(false)),
                |(p, c)| streamed(p, c, render()),
                large,
            )
        });
        c.bench_function(&format!("stream floor {name}: serialize joined"), |b| {
            b.iter_batched_ref(joined_chunks, |c| serialize_joined(c, render()), large)
        });
    }
}

criterion_group!(benches, bench);
criterion_main!(benches);
