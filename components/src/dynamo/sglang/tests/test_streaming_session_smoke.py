# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Streaming session smoke test for Dynamo + SGLang.

Tests the full session lifecycle with realistic agentic workloads:
main agent turns interleaved with subagent sessions that open, accumulate
KV cache across multi-turn conversations, and close to free KV.

Modes:
  --mode smoke       Quick functional test (default). Validates open/close lifecycle.
  --mode kv-pressure Longer test with metrics polling. Validates KV reclamation
                     after session close via sglang:kv_physical_usage.

Example:
    # Quick smoke test
    python test_streaming_session_smoke.py --base-url http://localhost:8000

    # KV pressure test with metrics
    python test_streaming_session_smoke.py --mode kv-pressure --metrics-port 8081

Requires https://github.com/sgl-project/sglang/pull/22726 to be merged.
"""

import argparse
import time
import uuid
from typing import Any, Optional

import requests

# ---------------------------------------------------------------------------
# Inlined prompts from real agentic workloads (OpenCode traces).
# Main agent is a lead orchestrator; subagents are specialized workers.
# ---------------------------------------------------------------------------

MAIN_AGENT_SYSTEM = (
    "You are a principal GPU systems engineer building a GPU-accelerated "
    "inference engine. Coordinate subagents for specialized research tasks. "
    "Keep responses concise and technical."
)

MAIN_AGENT_TURNS = [
    (
        "Design a minimal GPU inference engine for a Qwen 0.6B-class model in Rust. "
        "List the key components needed: model loading, tokenization, CUDA integration, "
        "KV cache, attention, MLP, generation loop, and sampling. For each component, "
        "identify what can be done in Rust vs what needs CUDA kernels."
    ),
    (
        "Based on the component list, define the module structure. We need: "
        "config/, tensor/, cuda/, kernels/, model/, cache/, weights/, tokenizer/, "
        "generation/, and cli/. For each module, describe its responsibility and "
        "key data structures. Focus on memory layout decisions for GPU tensors."
    ),
    (
        "Review the subagent findings and synthesize an implementation plan. "
        "Phase 1: tensor runtime + CUDA setup. Phase 2: weight loading + embeddings. "
        "Phase 3: attention + KV cache. Phase 4: MLP + full forward pass. "
        "Phase 5: generation loop + sampling. Identify the critical path."
    ),
]

# Each subagent has a multi-turn conversation that builds up KV cache.
# Turns grow progressively longer to simulate real prefix accumulation.
SUBAGENT_SESSIONS = [
    {
        "name": "kernel-design",
        "turns": [
            (
                "Design GPU kernels needed for a Qwen 0.6B inference engine. "
                "Architecture: hidden_size=1024, num_layers=28, num_heads=16, "
                "kv_heads=8, head_dim=128, intermediate_size=3072. GQA with "
                "16 Q heads, 8 KV heads. RMSNorm pre-norm. SiLU activation. "
                "RoPE on full head_dim. Design kernels for: RMSNorm, RoPE "
                "application, softmax, SiLU gated MLP, and embedding lookup."
            ),
            (
                "For the RMSNorm kernel: implement a two-pass approach. "
                "Pass 1: compute sum of squares with parallel reduction. "
                "Pass 2: normalize each element. Handle hidden_size=1024 "
                "with one thread block per token. Use shared memory for "
                "the reduction. Show the CUDA kernel signature and launch config."
            ),
            (
                "For RoPE: implement rotary position embeddings on the Q and K "
                "tensors. Each head has dim=128, so rotate pairs of elements. "
                "Use the standard cos/sin precomputed table indexed by position. "
                "The kernel should handle both prefill (multiple positions) and "
                "decode (single position) modes efficiently."
            ),
            (
                "For the attention kernel: we have GQA with 16 Q heads and 8 KV "
                "heads. Each Q head pair shares one KV head. Implement scaled "
                "dot-product attention with causal masking. For decode, use a "
                "single-query attention kernel that reads the full KV cache. "
                "For prefill, use flash-attention-style tiling."
            ),
            (
                "Now design the SiLU gated MLP kernel. The MLP computes: "
                "output = down_proj(silu(gate_proj(x)) * up_proj(x)). "
                "The gate and up projections can be fused into one GEMM "
                "with 2*intermediate_size output, then apply silu element-wise "
                "on the first half, multiply with second half, then down_proj."
            ),
        ],
    },
    {
        "name": "kv-cache-design",
        "turns": [
            (
                "Design a KV cache for Qwen 0.6B with GQA (16 Q heads, 8 KV heads, "
                "head_dim=128). 28 decoder layers. Requirements: GPU-resident, "
                "support prefill (populate from prompt) and decode (append one token), "
                "bounded capacity, clear memory layout. Consider contiguous vs "
                "per-layer allocation and how GQA affects storage."
            ),
            (
                "Define the memory layout. For each layer, we store K and V tensors. "
                "With 8 KV heads and head_dim=128, each token needs "
                "8 * 128 * 2 (K+V) * 2 bytes (fp16) = 4096 bytes per layer. "
                "For 28 layers, that is 28 * 4096 = 114688 bytes per token. "
                "Allocate max_seq_len * per_token_bytes contiguously per layer."
            ),
            (
                "Implement the append operation for decode. Given a new K and V "
                "for position T, copy them into the cache at index T for each layer. "
                "This is a simple copy kernel: for each layer, each KV head, copy "
                "head_dim elements. Also implement a read operation that returns "
                "a view of K[:T+1] and V[:T+1] for attention computation."
            ),
            (
                "Handle the prefill case. During prefill, we process the full prompt "
                "and need to store all K/V at once. This is a bulk copy from the "
                "attention output into the cache. The key difference from decode: "
                "we write seq_len positions at once instead of 1. Use a 2D grid "
                "where blockIdx.x = position, blockIdx.y = layer * num_kv_heads."
            ),
            (
                "Add sequence length tracking. The cache needs to know the current "
                "length for each sequence to: (1) index the append position during "
                "decode, (2) set the attention mask length, (3) compute RoPE positions. "
                "Store this as a simple integer per sequence. Increment after each "
                "decode step. Set to prompt_length after prefill."
            ),
            (
                "Consider memory efficiency. With max_seq_len=4096 and 28 layers, "
                "the cache for one sequence is 4096 * 114688 = 448MB. For batch_size=1, "
                "this fits comfortably. But for larger batches or longer sequences, "
                "we would need paged attention. For the MVP, pre-allocate the full "
                "cache and document the memory bound."
            ),
        ],
    },
    {
        "name": "weight-loading",
        "turns": [
            (
                "Design a weight loading strategy for Qwen 0.6B in Rust. "
                "Weight structure: vocab_size=151936, hidden_size=1024, num_layers=28. "
                "Attention: qkv_proj (3072x1024), o_proj (1024x1024). "
                "MLP: gate_up_proj (6144x1024), down_proj (1024x3072). "
                "LayerNorm: input_layernorm, post_attention_layernorm (1024 each). "
                "Choose between safetensors direct load vs conversion step."
            ),
            (
                "Implement safetensors parsing in Rust. The format is: "
                "8 bytes header length (LE u64), JSON header with tensor metadata "
                "(name, dtype, shape, data_offsets), then raw tensor data. "
                "Parse the header, build a name->TensorInfo map, then mmap or "
                "read the data section. Use the safetensors crate if available, "
                "or implement minimal parsing for our known tensor set."
            ),
            (
                "Map checkpoint tensor names to our model structure. Qwen uses: "
                "model.embed_tokens.weight, model.layers.N.self_attn.q_proj.weight, "
                "model.layers.N.self_attn.k_proj.weight, model.layers.N.self_attn.v_proj.weight, "
                "model.layers.N.self_attn.o_proj.weight, model.layers.N.mlp.gate_proj.weight, "
                "model.layers.N.mlp.up_proj.weight, model.layers.N.mlp.down_proj.weight, "
                "model.layers.N.input_layernorm.weight, model.layers.N.post_attention_layernorm.weight."
            ),
            (
                "Handle dtype conversion. Qwen checkpoints are typically in bf16 or fp16. "
                "Our GPU kernels should operate in fp16 for L40S compatibility. If the "
                "checkpoint is bf16, convert to fp16 during loading (simple bit manipulation: "
                "bf16 and fp16 share the sign bit and exponent range for normal values). "
                "Upload converted tensors to GPU via cudaMemcpy."
            ),
        ],
    },
    {
        "name": "generation-loop",
        "turns": [
            (
                "Design the generation loop and sampling for Qwen 0.6B. "
                "Prefill: process full prompt tokens, compute logits for last position, "
                "populate KV cache. Decode: sample token, append to sequence, run forward "
                "pass with cached KV + new token, repeat. Stop on EOS (151643 for Qwen) "
                "or max_tokens."
            ),
            (
                "Implement sampling strategies. Start with greedy (argmax over logits). "
                "Then add temperature scaling: logits /= temperature before softmax. "
                "Then top-k: zero out all logits below the k-th largest, renormalize. "
                "Then top-p (nucleus): sort logits descending, compute cumulative "
                "probability, zero out tokens beyond the p threshold."
            ),
            (
                "Design the forward pass orchestration. For each decode step: "
                "(1) embed the new token, (2) for each layer: RMSNorm -> attention "
                "(with KV cache read/append) -> residual -> RMSNorm -> MLP -> residual, "
                "(3) final RMSNorm, (4) LM head projection, (5) sample. "
                "Minimize host-device synchronization: only sync for the sampled token ID."
            ),
        ],
    },
]


def _extract_reply(body: dict[str, Any]) -> str:
    """Extract assistant reply from chat completion, handling reasoning models."""
    msg = body["choices"][0]["message"]
    return msg.get("content") or msg.get("reasoning_content") or ""


def _get_model(base_url: str) -> str:
    response = requests.get(f"{base_url}/v1/models", timeout=30)
    response.raise_for_status()
    return response.json()["data"][0]["id"]


def _chat(
    base_url: str,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int = 64,
    session_id: Optional[str] = None,
    session_action: Optional[str] = None,
    priority: Optional[int] = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    nvext: dict[str, Any] = {}
    if session_id is not None:
        sc: dict[str, Any] = {"session_id": session_id}
        if session_action is not None:
            sc["action"] = session_action
            if session_action == "open":
                sc["timeout"] = 300
        nvext["session_control"] = sc
    if priority is not None:
        nvext["agent_hints"] = {"priority": priority}
    if nvext:
        payload["nvext"] = nvext

    resp = requests.post(f"{base_url}/v1/chat/completions", json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _get_metrics(metrics_url: str) -> dict[str, float]:
    try:
        text = requests.get(metrics_url, timeout=5).text
    except Exception:
        return {}
    result = {}
    for name in [
        "sglang:kv_physical_usage",
        "sglang:token_usage",
        "sglang:num_streaming_sessions",
        "sglang:streaming_session_held_tokens",
        "sglang:cache_hit_rate",
    ]:
        for line in text.splitlines():
            if line.startswith(name + "{") or line.startswith(name + " "):
                try:
                    result[name] = float(line.split()[-1])
                except ValueError:
                    pass
                break
    return result


# ---------------------------------------------------------------------------
# Smoke test: quick functional validation
# ---------------------------------------------------------------------------


def run_smoke_test(base_url: str, model: str, max_tokens: int) -> None:
    system = {"role": "system", "content": MAIN_AGENT_SYSTEM}
    history: list[dict[str, str]] = [system]

    # Main agent turn 1
    print("main agent turn 1")
    history.append({"role": "user", "content": MAIN_AGENT_TURNS[0]})
    body = _chat(base_url, model, history, max_tokens)
    assistant_reply = _extract_reply(body)
    history.append({"role": "assistant", "content": assistant_reply})
    usage = body.get("usage", {})
    print(
        f"  in={usage.get('prompt_tokens', 0)} out={usage.get('completion_tokens', 0)}"
    )

    # Subagent session 1 (3 turns)
    sub = SUBAGENT_SESSIONS[0]
    session_id = f"smoke-{sub['name']}-{uuid.uuid4().hex[:8]}"
    print(f"subagent '{sub['name']}' session={session_id}")
    sub_history: list[dict[str, str]] = []
    for idx, turn in enumerate(sub["turns"][:3]):
        action = "open" if idx == 0 else None
        sub_history.append({"role": "user", "content": turn})
        body = _chat(
            base_url,
            model,
            sub_history,
            max_tokens,
            session_id=session_id,
            session_action=action,
        )
        reply = _extract_reply(body)
        sub_history.append({"role": "assistant", "content": reply})
        usage = body.get("usage", {})
        tag = " [OPEN]" if idx == 0 else ""
        print(
            f"  turn {idx + 1}{tag}: in={usage.get('prompt_tokens', 0)} out={usage.get('completion_tokens', 0)}"
        )

    # Close subagent session
    sub_history.append(
        {"role": "user", "content": "Summarize your findings in 2 sentences."}
    )
    body = _chat(
        base_url,
        model,
        sub_history,
        max_tokens,
        session_id=session_id,
        session_action="close",
    )
    usage = body.get("usage", {})
    print(
        f"  close [CLOSE]: in={usage.get('prompt_tokens', 0)} out={usage.get('completion_tokens', 0)}"
    )

    # Main agent turn 2
    print("main agent turn 2")
    history.append({"role": "user", "content": MAIN_AGENT_TURNS[1]})
    body = _chat(base_url, model, history, max_tokens)
    usage = body.get("usage", {})
    print(
        f"  in={usage.get('prompt_tokens', 0)} out={usage.get('completion_tokens', 0)}"
    )

    print("smoke test passed")


# ---------------------------------------------------------------------------
# KV pressure test: validate KV reclamation with metrics
# ---------------------------------------------------------------------------


def run_kv_pressure_test(
    base_url: str,
    model: str,
    max_tokens: int,
    metrics_url: str,
    turn_delay: float,
    session_delay: float,
) -> None:
    system = {"role": "system", "content": MAIN_AGENT_SYSTEM}
    main_history: list[dict[str, str]] = [system]
    main_turn_idx = 0

    print(f"KV pressure test: {len(SUBAGENT_SESSIONS)} subagent sessions")
    print(f"Metrics: {metrics_url}")
    print()

    def log_metrics(label: str) -> dict[str, float]:
        m = _get_metrics(metrics_url)
        phys = m.get("sglang:kv_physical_usage", -1)
        token = m.get("sglang:token_usage", -1)
        sessions = m.get("sglang:num_streaming_sessions", 0)
        hit = m.get("sglang:cache_hit_rate", -1)
        print(
            f"  [{label}] phys={phys:.4f} token={token:.4f} sessions={sessions:.0f} hit={hit:.4f}"
        )
        return m

    # Baseline
    log_metrics("baseline")

    # Interleave main agent turns with subagent sessions
    session_peaks: list[float] = []
    session_floors: list[float] = []

    for sub_idx, sub in enumerate(SUBAGENT_SESSIONS):
        # Main agent turn before each subagent
        if main_turn_idx < len(MAIN_AGENT_TURNS):
            print(f"\nmain agent turn {main_turn_idx + 1}")
            main_history.append(
                {"role": "user", "content": MAIN_AGENT_TURNS[main_turn_idx]}
            )
            body = _chat(base_url, model, main_history, max_tokens)
            reply = _extract_reply(body)
            main_history.append({"role": "assistant", "content": reply})
            usage = body.get("usage", {})
            print(
                f"  in={usage.get('prompt_tokens', 0)} out={usage.get('completion_tokens', 0)}"
            )
            main_turn_idx += 1
            time.sleep(turn_delay)

        # Subagent session
        session_id = f"bench-{sub['name']}-{uuid.uuid4().hex[:8]}"
        print(
            f"\nsubagent '{sub['name']}' ({len(sub['turns'])} turns) session={session_id}"
        )
        sub_history: list[dict[str, str]] = []
        peak_phys = 0.0

        for idx, turn in enumerate(sub["turns"]):
            is_last = idx == len(sub["turns"]) - 1
            action = "open" if idx == 0 else ("close" if is_last else None)
            sub_history.append({"role": "user", "content": turn})

            t0 = time.monotonic()
            body = _chat(
                base_url,
                model,
                sub_history,
                max_tokens,
                session_id=session_id,
                session_action=action,
            )
            elapsed = (time.monotonic() - t0) * 1000
            reply = _extract_reply(body)
            sub_history.append({"role": "assistant", "content": reply})
            usage = body.get("usage", {})
            tag = " [OPEN]" if idx == 0 else (" [CLOSE]" if is_last else "")
            print(
                f"  turn {idx + 1}/{len(sub['turns'])}{tag}: "
                f"in={usage.get('prompt_tokens', 0)} out={usage.get('completion_tokens', 0)} "
                f"{elapsed:.0f}ms"
            )

            m = _get_metrics(metrics_url)
            phys = m.get("sglang:kv_physical_usage", 0)
            if phys > peak_phys:
                peak_phys = phys

            if not is_last:
                time.sleep(turn_delay)

        session_peaks.append(peak_phys)

        # Wait after close so metrics settle
        print(f"  --- session closed, waiting {session_delay:.0f}s ---")
        time.sleep(session_delay)
        m = log_metrics(f"post-close-{sub['name']}")
        session_floors.append(m.get("sglang:kv_physical_usage", 0))

    # Final state
    print()
    m = log_metrics("final")

    # Summary
    print("\n=== KV Pressure Summary ===")
    for i, sub in enumerate(SUBAGENT_SESSIONS):
        reclaim = (
            f"{(1 - session_floors[i] / session_peaks[i]) * 100:.0f}% reclaimed"
            if session_peaks[i] > 0
            else "n/a"
        )
        print(
            f"  {sub['name']:20s}  peak={session_peaks[i]:.4f}  "
            f"post-close={session_floors[i]:.4f}  ({reclaim})"
        )
    print(f"  {'final':20s}  phys={m.get('sglang:kv_physical_usage', -1):.4f}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Streaming session smoke/benchmark test."
    )
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--model", help="Model ID (auto-detect if omitted)")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument(
        "--mode",
        choices=["smoke", "kv-pressure"],
        default="smoke",
        help="Test mode: smoke (quick) or kv-pressure (with metrics)",
    )
    parser.add_argument("--metrics-port", type=int, default=8081)
    parser.add_argument("--turn-delay", type=float, default=1.0)
    parser.add_argument("--session-delay", type=float, default=6.0)
    args = parser.parse_args()

    model = args.model or _get_model(args.base_url)
    print(f"model: {model}")
    print(f"mode: {args.mode}")
    print()

    if args.mode == "smoke":
        run_smoke_test(args.base_url, model, args.max_tokens)
    elif args.mode == "kv-pressure":
        metrics_url = f"http://localhost:{args.metrics_port}/metrics"
        run_kv_pressure_test(
            args.base_url,
            model,
            args.max_tokens,
            metrics_url,
            args.turn_delay,
            args.session_delay,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
