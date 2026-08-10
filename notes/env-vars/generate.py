# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Regenerate notes/env-vars/dynamo-launch-env-vars.html from the Dynamo source tree.

Run from the repo root:  python3 notes/env-vars/generate.py

Three sources are combined:
  1. lib/runtime/src/config/environment_names.rs  — the canonical name registry, parsed
     with a line scanner that keeps each constant's `///` doc comment and its module path.
  2. `add_argument(..., env_var=...)` / `add_negatable_bool_argument(...)` calls in the
     ArgGroup classes under components/src/dynamo/, parsed with an AST walk. Component
     membership follows the real composition: the frontend parser installs
     FrontendArgGroup + Router + KvRouter + AicPerf; each worker installs
     DynamoRuntimeArgGroup + its own backend group.
  3. A curated table (`C` below) of variables read directly via env::var / os.environ
     that have no CLI flag. Each entry was checked against its read site.

Variables that only appear in recipes, example launch scripts, or docs without being read
by Dynamo code are deliberately excluded, as are planner, profiler, mocker,
standalone-router, omni/diffusion, and test-only variables.
"""
import ast
import collections
import html
import os
import re

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(REPO)


def extract_cli_args():
    """Walk components/src/dynamo and collect every env-var-backed CLI argument."""
    out = []
    base = "components/src/dynamo"
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = [d for d in dirnames if d not in ("__pycache__", "tests")]
        for fn in filenames:
            if not fn.endswith(".py"):
                continue
            path = os.path.join(dirpath, fn)
            try:
                tree = ast.parse(open(path, encoding="utf-8").read())
            except (OSError, SyntaxError):
                continue
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                fname = (
                    node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else node.func.id
                    if isinstance(node.func, ast.Name)
                    else None
                )
                if fname not in ("add_argument", "add_negatable_bool_argument"):
                    continue
                kw = {k.arg: k.value for k in node.keywords if k.arg}
                if "env_var" not in kw:
                    continue

                def lit(v):
                    try:
                        return ast.literal_eval(v)
                    except (ValueError, SyntaxError):
                        return ast.unparse(v)

                out.append(
                    {
                        "file": os.path.relpath(path, base),
                        "line": node.lineno,
                        "env": lit(kw["env_var"]),
                        "flag": lit(kw["flag_name"]) if "flag_name" in kw else None,
                        "default": lit(kw["default"]) if "default" in kw else None,
                        "type": ast.unparse(kw["arg_type"])
                        if "arg_type" in kw
                        else None,
                        "choices": lit(kw["choices"]) if "choices" in kw else None,
                        "help": lit(kw["help"]) if "help" in kw else None,
                    }
                )
    return out


def extract_rust_registry():
    """Parse the canonical Rust constant registry, keeping module path and doc comment."""
    path = "lib/runtime/src/config/environment_names.rs"
    out, mods, doc = [], [], []
    for i, line in enumerate(open(path, encoding="utf-8").read().splitlines()):
        s = line.strip()
        m = re.match(r"pub mod (\w+)\s*\{", s)
        if m:
            mods = mods[: (len(line) - len(line.lstrip())) // 4] + [m.group(1)]
            doc = []
            continue
        if s.startswith("///"):
            doc.append(s[3:].strip())
            continue
        m = re.match(r'pub const (\w+): &str = "([^"]+)";', s)
        if m:
            out.append(
                {
                    "const": m.group(1),
                    "env": m.group(2),
                    "mod": "::".join(mods),
                    "doc": " ".join(d for d in doc if d),
                    "line": i + 1,
                }
            )
            doc = []
            continue
        if s and not s.startswith("//"):
            doc = []
    return out


args = extract_cli_args()
rust = extract_rust_registry()

GH = "https://github.com/ai-dynamo/dynamo/blob/main/"

records = {}  # env name -> record


def add(
    env,
    scope,
    section,
    flag=None,
    default=None,
    choices=None,
    desc="",
    source="",
    note="",
):
    if not env or not isinstance(env, str) or not env.replace("_", "").isalnum():
        return
    r = records.get(env)
    if r is None:
        r = {
            "env": env,
            "scope": set(),
            "sections": {},
            "flag": flag,
            "default": default,
            "choices": choices,
            "desc": desc,
            "source": source,
            "note": note,
        }
        records[env] = r
    r["scope"] |= set(scope)
    for s in scope:
        r["sections"][s] = section
    for k, v in (
        ("flag", flag),
        ("default", default),
        ("choices", choices),
        ("desc", desc),
        ("source", source),
        ("note", note),
    ):
        if v and not r.get(k):
            r[k] = v


def add_flag(
    flag, scope, section, default=None, desc="", source="", aliases=None, choices=None
):
    """Record a CLI flag that has no environment-variable equivalent."""
    key = "flag:" + flag
    r = records.setdefault(
        key,
        {
            "env": None,
            "flag": flag,
            "aliases": aliases or [],
            "scope": set(),
            "sections": {},
            "default": default,
            "choices": choices,
            "desc": desc,
            "source": source,
        },
    )
    r["scope"] |= set(scope)
    for sc in scope:
        r["sections"][sc] = section


# ---------------------------------------------------------------- CLI-declared
FILE_SCOPE = {
    "frontend/frontend_args.py": (["frontend"], "Frontend core"),
    "common/configuration/groups/router_args.py": (
        ["frontend"],
        "Router: mode & admission",
    ),
    "common/configuration/groups/kv_router_args.py": (["frontend"], "KV router tuning"),
    "common/configuration/groups/aic_perf_args.py": (
        ["frontend"],
        "AIC performance model",
    ),
    "common/configuration/groups/runtime_args.py": (
        ["vllm", "trtllm", "sglang"],
        "Worker runtime & identity",
    ),
    "vllm/backend_args.py": (["vllm"], "vLLM engine wrapper"),
    "trtllm/backend_args.py": (["trtllm"], "TensorRT-LLM engine wrapper"),
    "sglang/backend_args.py": (["sglang"], "SGLang engine wrapper"),
    "common/configuration/groups/http_args.py": (
        ["vllm", "trtllm", "sglang"],
        "Multimodal HTTP fetch client",
    ),
}
SRC_PREFIX = "components/src/dynamo/"
for a in args:
    if a["file"] not in FILE_SCOPE:
        continue
    scope, section = FILE_SCOPE[a["file"]]
    env = a["env"]
    if not isinstance(env, str) or env.startswith("f'") or env in ("legacy_env",):
        continue
    d = a["default"]
    if isinstance(d, str) and (
        d.endswith("()") or "." in d and d[0].islower() and "(" in d
    ):
        d = f"(computed: {d})"
    add(
        env,
        scope,
        section,
        flag=a["flag"],
        default=d,
        choices=a["choices"],
        desc=a["help"] or "",
        source=SRC_PREFIX + a["file"] + "#L%d" % a["line"],
    )

# frontend_decoding_args is parameterised per backend
for be, pref in (("vllm", "VLLM"), ("trtllm", "TRTLLM"), ("sglang", "SGL")):
    add(
        f"DYN_{pref}_FRONTEND_DECODING",
        [be],
        "Multimodal",
        flag="--frontend-decoding",
        default=False,
        desc="Enable frontend decoding of multimodal images. Images are decoded in the Rust frontend and transferred to the backend via NIXL RDMA, bypassing in-engine decode.",
        source="components/src/dynamo/common/configuration/groups/frontend_decoding_args.py",
    )

# --------------------------------------------------------------- Rust registry
ALL4 = ["frontend", "vllm", "trtllm", "sglang"]
WORKERS = ["vllm", "trtllm", "sglang"]
MOD_SCOPE = {
    "logging": (ALL4, "Logging"),
    "logging::otlp": (ALL4, "OpenTelemetry export"),
    "runtime": (ALL4, "Tokio runtime"),
    "runtime::system": (ALL4, "System status server"),
    "runtime::compute": (ALL4, "Tokio runtime"),
    "runtime::canary": (ALL4, "Health checks"),
    "worker": (ALL4, "Shutdown & lifecycle"),
    "nats": (ALL4, "NATS"),
    "nats::auth": (ALL4, "NATS"),
    "nats::stream": (ALL4, "NATS"),
    "etcd": (ALL4, "etcd"),
    "etcd::auth": (ALL4, "etcd"),
    "request_plane": (ALL4, "Request / event plane"),
    "tcp_response_stream": (ALL4, "Request / event plane"),
    "event_plane": (ALL4, "Request / event plane"),
    "zmq_broker": (ALL4, "Request / event plane"),
    "discovery": (ALL4, "Discovery"),
    "model::model_express": (ALL4, "Model download"),
    "model::huggingface": (ALL4, "Model download"),
    "llm::request_trace": (ALL4, "Request tracing"),
    "llm::audit": (ALL4, "Request tracing"),
    "llm::fpm_trace": (WORKERS, "Forward-pass metric trace"),
    "llm::metrics": (["frontend"], "Metrics"),
    "router": (["frontend"], "KV router tuning"),
    "kvbm": (["vllm", "trtllm"], "KVBM (KV block manager)"),
    "kvbm::cpu_cache": (["vllm", "trtllm"], "KVBM (KV block manager)"),
    "kvbm::disk_cache": (["vllm", "trtllm"], "KVBM (KV block manager)"),
    "kvbm::object_storage": (["vllm", "trtllm"], "KVBM (KV block manager)"),
    "kvbm::transfer": (["vllm", "trtllm"], "KVBM (KV block manager)"),
    "kvbm::leader": (["vllm", "trtllm"], "KVBM (KV block manager)"),
    "kvbm::nixl": (["vllm", "trtllm"], "KVBM (KV block manager)"),
    "cuda": (["vllm", "trtllm"], "KVBM (KV block manager)"),
    "llm": (["frontend"], "HTTP service & API surface"),
}
LLM_WORKER_OVERRIDE = {  # llm:: constants that are actually worker-side
    "DYN_LORA_ENABLED": (ALL4, "LoRA"),
    "DYN_LORA_PATH": (ALL4, "LoRA"),
}
SKIP_MODS = {"mocker", "testing", "build"}
for r in rust:
    if r["mod"] in SKIP_MODS:
        continue
    if r["env"].endswith("_") or r["env"] in ("OUT_DIR",):
        continue
    if r["env"] in LLM_WORKER_OVERRIDE:
        scope, section = LLM_WORKER_OVERRIDE[r["env"]]
    elif r["mod"] in MOD_SCOPE:
        scope, section = MOD_SCOPE[r["mod"]]
    else:
        continue
    if r["env"].startswith("DYN_LORA_ALLOCATION") or r["env"] == "DYN_LORA_MCF_CONFIG":
        section = "LoRA"
    add(
        r["env"],
        scope,
        section,
        desc=r["doc"],
        source="lib/runtime/src/config/environment_names.rs#L%d" % r["line"],
    )

# ------------------------------------------------------------ curated non-CLI
C = [
    # env, scope, section, default, desc, source
    (
        "DYN_HTTP_SVC_CHAT_PATH",
        ["frontend"],
        "HTTP service & API surface",
        "/v1/chat/completions",
        "Override the route path for the chat-completions endpoint.",
        "lib/llm/src/http/service/openai.rs",
    ),
    (
        "DYN_HTTP_SVC_CMP_PATH",
        ["frontend"],
        "HTTP service & API surface",
        "/v1/completions",
        "Override the route path for the legacy completions endpoint.",
        "lib/llm/src/http/service/openai.rs",
    ),
    (
        "DYN_HTTP_SVC_EMB_PATH",
        ["frontend"],
        "HTTP service & API surface",
        "/v1/embeddings",
        "Override the route path for the embeddings endpoint.",
        "lib/llm/src/http/service/openai.rs",
    ),
    (
        "DYN_HTTP_SVC_RESPONSES_PATH",
        ["frontend"],
        "HTTP service & API surface",
        "/v1/responses",
        "Override the route path for the Responses API endpoint.",
        "lib/llm/src/http/service/openai.rs",
    ),
    (
        "DYN_HTTP_SVC_ANTHROPIC_PATH",
        ["frontend"],
        "HTTP service & API surface",
        "/v1/messages",
        "Override the route path for the Anthropic Messages endpoint (requires DYN_ENABLE_ANTHROPIC_API).",
        "lib/llm/src/http/service/anthropic.rs",
    ),
    (
        "DYN_HTTP_SVC_MODELS_PATH",
        ["frontend"],
        "HTTP service & API surface",
        "/v1/models",
        "Override the route path for the model-listing endpoint.",
        "lib/llm/src/http/service/openai.rs",
    ),
    (
        "DYN_HTTP_SVC_FILES_PATH",
        ["frontend"],
        "HTTP service & API surface",
        "/v1/files",
        "Override the route path for the files endpoint.",
        "lib/llm/src/http/service/openai.rs",
    ),
    (
        "DYN_HTTP_SVC_BATCHES_PATH",
        ["frontend"],
        "HTTP service & API surface",
        "/v1/batches",
        "Override the route path for the batches endpoint.",
        "lib/llm/src/http/service/openai.rs",
    ),
    (
        "DYN_HTTP_SVC_METRICS_PATH",
        ["frontend"],
        "HTTP service & API surface",
        "/metrics",
        "Override the route path that serves Prometheus metrics.",
        "lib/llm/src/http/service/metrics.rs",
    ),
    (
        "DYN_HTTP_SVC_HEALTH_PATH",
        ["frontend"],
        "HTTP service & API surface",
        "/health",
        "Override the route path for the frontend readiness probe.",
        "lib/llm/src/http/service/health.rs",
    ),
    (
        "DYN_HTTP_SVC_LIVE_PATH",
        ["frontend"],
        "HTTP service & API surface",
        "/live",
        "Override the route path for the frontend liveness probe.",
        "lib/llm/src/http/service/health.rs",
    ),
    (
        "DYN_HTTP_GRACEFUL_SHUTDOWN_TIMEOUT_SECS",
        ["frontend"],
        "HTTP service & API surface",
        None,
        "How long the HTTP server waits for in-flight requests to drain before forcing shutdown.",
        "lib/llm/src/http/service/service_v2.rs#L937",
    ),
    (
        "DYN_ENABLE_EXPERIMENTAL_PARSERS_V2",
        ["frontend"],
        "Preprocessing, templates & parsers",
        "false",
        "Use the v2 tool-call / reasoning parser implementation instead of the v1 aggregator path.",
        "lib/llm/src/protocols/openai/chat_completions/tool_parser_v2.rs",
    ),
    (
        "DYN_METADATA_HEADER",
        ["frontend"],
        "HTTP service & API surface",
        None,
        "Name of the HTTP request header carrying opaque per-request metadata forwarded to workers.",
        "lib/llm/src/http/service/metadata.rs#L24",
    ),
    (
        "DYN_MAX_OUTPUT_TOKENS",
        ["frontend"],
        "HTTP service & API surface",
        None,
        "Server-side ceiling applied to max_tokens / max_output_tokens on incoming requests. Also reported on the Anthropic model-info route.",
        "lib/llm/src/http/service/openai.rs#L2925",
    ),
    (
        "DYN_CONTEXT_WINDOW",
        ["frontend"],
        "HTTP service & API surface",
        None,
        "Override the advertised context window used for request validation and the Anthropic model-info route.",
        "lib/llm/src/http/service/openai.rs#L2922",
    ),
    (
        "DYN_TOKENIZER_CACHE",
        ["frontend"],
        "Preprocessing, templates & parsers",
        None,
        "Enable or disable the frontend tokenizer-result cache.",
        "lib/llm/src/model_card.rs#L1194",
    ),
    (
        "DYN_TOKENIZER_CACHE_BYTES",
        ["frontend"],
        "Preprocessing, templates & parsers",
        None,
        "Byte budget for the tokenizer-result cache.",
        "lib/llm/src/model_card.rs#L1196",
    ),
    (
        "DYN_TOKENIZER_CACHE_EXTEND",
        ["frontend"],
        "Preprocessing, templates & parsers",
        None,
        "Allow the tokenizer cache to be extended with entries seen at runtime.",
        "lib/llm/src/model_card.rs#L1199",
    ),
    (
        "DYN_METRICS_TTFT",
        ["frontend"],
        "Metrics",
        "0.001,480.0,18",
        "Histogram bucket config (min,max,count) for the time-to-first-token metric.",
        "lib/llm/src/http/service/metrics.rs#L773",
    ),
    (
        "DYN_METRICS_ITL",
        ["frontend"],
        "Metrics",
        None,
        "Histogram bucket config (min,max,count) for the inter-token-latency metric.",
        "lib/llm/src/http/service/metrics.rs",
    ),
    (
        "DYN_METRICS_REQUEST_DURATION",
        ["frontend"],
        "Metrics",
        None,
        "Histogram bucket config (min,max,count) for total request duration.",
        "lib/llm/src/http/service/metrics.rs",
    ),
    (
        "DYN_METRICS_INPUT_SEQUENCE",
        ["frontend"],
        "Metrics",
        None,
        "Histogram bucket config (min,max,count) for input sequence length.",
        "lib/llm/src/http/service/metrics.rs",
    ),
    (
        "DYN_METRICS_OUTPUT_SEQUENCE",
        ["frontend"],
        "Metrics",
        None,
        "Histogram bucket config (min,max,count) for output sequence length.",
        "lib/llm/src/http/service/metrics.rs",
    ),
    (
        "DYN_METRICS_EMBEDDING_LATENCY",
        ["frontend"],
        "Metrics",
        None,
        "Histogram bucket config (min,max,count) for embedding latency.",
        "lib/llm/src/http/service/metrics.rs",
    ),
    (
        "DYN_FRONTEND_FD_LIMIT_TARGET",
        ["frontend"],
        "Frontend core",
        None,
        "Target soft file-descriptor limit the frontend raises RLIMIT_NOFILE to at startup.",
        "components/src/dynamo/frontend/main.py#L63",
    ),
    (
        "DYN_MULTIMODAL_LOADER_CACHE_GB",
        ["frontend"],
        "Multimodal",
        None,
        "Size in GB of the frontend media-loader cache for fetched images/video.",
        "lib/llm/src/preprocessor/media/loader.rs#L392",
    ),
    (
        "DYN_VLLM_ENABLE_INFERENCE_V1_GENERATE",
        ["frontend"],
        "HTTP service & API surface",
        "false",
        "Register the vLLM-compatible /inference/v1/generate route on the frontend.",
        "lib/llm/src/http/service/service_v2.rs#L968",
    ),
    (
        "DYN_VLLM_STREAM_INTERVAL",
        ["frontend"],
        "Preprocessing, templates & parsers",
        None,
        "Token batching interval used by the vLLM pre/post processor (--dyn-chat-processor vllm).",
        "components/src/dynamo/frontend/vllm_processor.py#L896",
    ),
    (
        "DYN_SGLANG_STREAM_INTERVAL",
        ["frontend"],
        "Preprocessing, templates & parsers",
        None,
        "Token batching interval used by the SGLang pre/post processor (--dyn-chat-processor sglang).",
        "components/src/dynamo/frontend/sglang_processor.py#L781",
    ),
    (
        "DYN_VLLM_SKIP_REQUEST_VALIDATION",
        ["frontend"],
        "Preprocessing, templates & parsers",
        "1",
        "Skip request re-validation in the Python pre/post processor path.",
        "components/src/dynamo/frontend/prepost.py#L48",
    ),
    (
        "DYN_RL_PORT",
        ["frontend"],
        "Frontend core",
        None,
        "Port for the RL weight-sync router served by the frontend when RL support is enabled.",
        "lib/llm/src/http/service/service_v2.rs#L670",
    ),
    (
        "DYN_ENABLE_RL",
        ["frontend", "vllm", "sglang"],
        "RL training support",
        "false",
        "Enable RL training support (weight sync / metadata upload). On the frontend this also mounts the RL router.",
        "lib/llm/src/http/service/service_v2.rs#L1198",
    ),
    (
        "DYN_MOONCAKE_KV_EVENTS_ENDPOINT",
        ["frontend"],
        "KV router tuning",
        None,
        "Mooncake/HiCache master endpoint the router queries for shared-cache hits (paired with --shared-cache-type hicache).",
        "lib/llm/src/discovery/model_manager.rs#L1065",
    ),
    (
        "DYN_MOONCAKE_KV_EVENTS_ENDPOINT",
        ["sglang"],
        "SGLang extras (non-CLI)",
        None,
        "Mooncake/HiCache KV-events endpoint this SGLang worker registers so the router can query shared-cache hits.",
        "components/src/dynamo/sglang/register.py#L281",
    ),
    (
        "DYN_ENCODER_CUDA_TO_CPU_RATIO",
        ["frontend"],
        "KV router tuning",
        None,
        "Weighting ratio used by device-aware-weighted routing when mixing CUDA and CPU encode workers.",
        "lib/runtime/src/pipeline/network/egress/push_router.rs#L1026",
    ),
    (
        "DYN_ADMISSION_CONTROL",
        ["frontend"],
        "Router: mode & admission",
        None,
        "Legacy on/off switch for worker-busy admission control; superseded by the explicit threshold flags.",
        "components/src/dynamo/common/configuration/groups/router_args.py#L138",
    ),
    (
        "DYN_USE_KV_EVENTS",
        ["frontend"],
        "KV router tuning",
        None,
        "Legacy alias for --router-kv-events / DYN_ROUTER_USE_KV_EVENTS.",
        "lib/kv-router/src/scheduling/config.rs#L198",
    ),
    (
        "DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT",
        ["frontend"],
        "KV router tuning",
        None,
        "Deprecated: legacy overlap-score weight. Still read, warns on use; replaced by DYN_ROUTER_PREFILL_LOAD_SCALE.",
        "components/src/dynamo/common/configuration/groups/kv_router_args.py#L83",
    ),
    (
        "DYN_OVERLAP_SCORE_WEIGHT",
        ["frontend"],
        "KV router tuning",
        None,
        "Deprecated: older alias of DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT. Read only if that one is unset.",
        "components/src/dynamo/common/configuration/groups/kv_router_args.py#L83",
    ),
    # ---- shared runtime, all four
    (
        "DYN_TCP_RPC_PORT",
        ALL4,
        "Request / event plane",
        "OS-assigned",
        "Fixed TCP port for the request-plane RPC listener; unset means an OS-assigned port.",
        "lib/runtime/src/pipeline/network/manager.rs#L96",
    ),
    (
        "DYN_TCP_RPC_HOST",
        ALL4,
        "Request / event plane",
        None,
        "Bind host advertised for the request-plane TCP RPC listener.",
        "lib/runtime/src/pipeline/network/manager.rs",
    ),
    (
        "DYN_TCP_POOL_SIZE",
        ALL4,
        "Request / event plane",
        "50",
        "TCP client connection-pool size per peer.",
        "lib/runtime/src/pipeline/network/egress/tcp_client.rs#L124",
    ),
    (
        "DYN_TCP_CONNECT_TIMEOUT",
        ALL4,
        "Request / event plane",
        "3",
        "TCP connect timeout in seconds.",
        "lib/runtime/src/pipeline/network/egress/tcp_client.rs#L130",
    ),
    (
        "DYN_TCP_REQUEST_TIMEOUT",
        ALL4,
        "Request / event plane",
        "10",
        "TCP request timeout in seconds.",
        "lib/runtime/src/pipeline/network/egress/tcp_client.rs#L118",
    ),
    (
        "DYN_TCP_CHANNEL_BUFFER",
        ALL4,
        "Request / event plane",
        "100",
        "Per-stream channel buffer depth for the TCP transport.",
        "lib/runtime/src/pipeline/network/egress/tcp_client.rs#L136",
    ),
    (
        "DYN_TCP_MAX_MESSAGE_SIZE",
        ALL4,
        "Request / event plane",
        None,
        "Maximum single message size accepted on the TCP request plane.",
        "lib/runtime/src/pipeline/network.rs#L50",
    ),
    (
        "DYN_TCP_LATENCY_TRACE",
        ALL4,
        "Request / event plane",
        "false",
        "Emit per-hop latency traces for TCP request-plane calls.",
        "lib/runtime/src/pipeline/network/egress/tcp_client.rs#L82",
    ),
    (
        "DYN_ZMQ_EVENT_SUBSCRIBER_CHANNEL_CAPACITY",
        ALL4,
        "Request / event plane",
        None,
        "Channel capacity for the ZMQ event-plane dynamic subscriber.",
        "lib/runtime/src/transports/event_plane/dynamic_subscriber.rs#L60",
    ),
    (
        "DYN_HEALTH_CHECK_ENABLED",
        ALL4,
        "Health checks",
        None,
        "Enable the periodic endpoint health-check canary on the system status server.",
        "lib/runtime/src/system_status_server.rs",
    ),
    (
        "DYN_HEALTH_CHECK_REQUEST_TIMEOUT",
        ALL4,
        "Health checks",
        None,
        "Timeout for a single health-check canary request.",
        "lib/runtime/src/config.rs#L176",
    ),
    (
        "DYN_SYSTEM_USE_ENDPOINT_HEALTH_STATUS",
        ALL4,
        "Health checks",
        None,
        "List of endpoints whose health status gates the system /health response.",
        "lib/runtime/src/config.rs",
    ),
    (
        "DYN_GRACEFUL_SHUTDOWN_GRACE_PERIOD_SECS",
        ALL4,
        "Shutdown & lifecycle",
        None,
        "Grace period before shutdown begins, so load balancers can drain traffic first.",
        "lib/backend-common/src/worker.rs#L45",
    ),
    (
        "DYN_MEMORY_DISABLE_NUMA",
        ALL4,
        "Memory",
        "false",
        "Disable NUMA-aware host memory allocation.",
        "lib/memory/src/numa/mod.rs#L49",
    ),
    (
        "DYN_ENABLE_RUST_NVTX",
        ALL4,
        "Profiling",
        "false",
        "Emit NVTX ranges from the Rust runtime for Nsight Systems capture.",
        "lib/runtime/src/nvtx.rs#L46",
    ),
    (
        "DYN_NVTX",
        ALL4,
        "Profiling",
        "0",
        "Emit NVTX ranges from the Python layer.",
        "components/src/dynamo/common/utils/nvtx_utils.py#L38",
    ),
    # ---- worker shared (all three backends)
    (
        "DYN_STABLE_ROUTING_ID",
        WORKERS,
        "Worker runtime & identity",
        None,
        "Pin this worker's routing identity across restarts instead of generating a fresh instance id.",
        "lib/llm/src/local_model/runtime_config.rs#L522",
    ),
    (
        "DYN_SELF_HOST_METADATA",
        WORKERS,
        "Worker runtime & identity",
        None,
        "Host the model card / metadata from this worker rather than publishing it through discovery.",
        "lib/llm/src/local_model.rs#L41",
    ),
    (
        "DYN_PREFILL_DRAIN_TIMEOUT_S",
        WORKERS,
        "Shutdown & lifecycle",
        "30",
        "Budget for draining in-flight prefill work during graceful shutdown.",
        "lib/backend-common/src/worker.rs#L50",
    ),
    (
        "DYN_ENGINE_HEALTH_CHECK_INTERVAL",
        WORKERS,
        "Health checks",
        None,
        "Interval between engine liveness probes run by the engine monitor.",
        "components/src/dynamo/common/engine_monitor.py#L20",
    ),
    (
        "DYN_ENGINE_HEALTH_CHECK_TIMEOUT",
        WORKERS,
        "Health checks",
        None,
        "Timeout for one engine liveness probe.",
        "components/src/dynamo/common/engine_monitor.py#L21",
    ),
    (
        "DYN_ENGINE_HEALTH_SHUTDOWN_TIMEOUT",
        WORKERS,
        "Health checks",
        None,
        "How long the monitor waits after an unhealthy engine before shutting the worker down.",
        "components/src/dynamo/common/engine_monitor.py#L22",
    ),
    (
        "DYN_TOPOLOGY_ENABLED",
        WORKERS,
        "Topology & KV transfer",
        "false",
        "Read cluster topology domain files so KV transfer can be constrained to a domain.",
        "components/src/dynamo/common/utils/topology.py#L32",
    ),
    (
        "DYN_TOPOLOGY_MOUNT_PATH",
        WORKERS,
        "Topology & KV transfer",
        None,
        "Directory containing the topology domain files.",
        "components/src/dynamo/common/utils/topology.py#L33",
    ),
    (
        "DYN_KV_TRANSFER_DOMAIN",
        WORKERS,
        "Topology & KV transfer",
        None,
        "Topology domain this worker advertises for KV transfer affinity.",
        "components/src/dynamo/common/utils/topology.py#L34",
    ),
    (
        "DYN_KV_TRANSFER_ENFORCEMENT",
        WORKERS,
        "Topology & KV transfer",
        None,
        "KV transfer enforcement mode: strict (reject cross-domain) or preferred (penalise).",
        "components/src/dynamo/common/utils/topology.py#L35",
    ),
    (
        "DYN_KV_TRANSFER_PREFERRED_WEIGHT",
        WORKERS,
        "Topology & KV transfer",
        None,
        "Penalty weight applied to cross-domain candidates in preferred enforcement mode.",
        "components/src/dynamo/common/utils/topology.py#L36",
    ),
    (
        "DYN_MM_IMAGE_CACHE_SIZE",
        WORKERS,
        "Multimodal",
        "8",
        "LRU entry count for the worker-side decoded-image cache.",
        "components/src/dynamo/common/multimodal/image_loader.py#L60",
    ),
    (
        "DYN_MM_VIDEO_NUM_FRAMES",
        WORKERS,
        "Multimodal",
        "32",
        "Default number of frames sampled from a video input.",
        "components/src/dynamo/common/multimodal/video_loader.py#L77",
    ),
    (
        "DYN_MM_LOCAL_PATH",
        WORKERS,
        "Multimodal",
        None,
        "Directory allowlisted for local (file://) media inputs. Unset means local paths are rejected.",
        "components/src/dynamo/common/http/url_validator.py#L99",
    ),
    (
        "DYN_MM_ALLOW_INTERNAL",
        ALL4,
        "Multimodal",
        "false",
        "Allow media URLs that resolve to private/internal addresses. Off by default as SSRF protection.",
        "lib/llm/src/preprocessor/media/loader.rs#L140",
    ),
    (
        "PYTHONHASHSEED",
        WORKERS,
        "Worker runtime & identity",
        "0",
        "Set by the workers before engine start so prefix hashes are stable across processes.",
        "components/src/dynamo/vllm/main.py",
    ),
    # ---- vLLM specific extras
    (
        "DYN_SPLIT_ENCODE",
        ["vllm"],
        "vLLM extras (non-CLI)",
        "1",
        "Split encode work out of the prefill worker in the multimodal prefill path.",
        "components/src/dynamo/vllm/multimodal_utils/prefill_worker_utils.py#L34",
    ),
    (
        "DYN_GMS_SCRATCH_KV_ENABLED",
        ["vllm"],
        "vLLM extras (non-CLI)",
        None,
        "Enable GPU Memory Service scratch-KV patches in the vLLM worker (set automatically in headless mode).",
        "lib/gpu_memory_service/common/utils.py#L18",
    ),
    (
        "DYN_LORA_HOTSWAP_ENABLED",
        ["vllm"],
        "LoRA",
        None,
        "Allow LoRA adapters to be hot-swapped on a running vLLM engine.",
        "components/src/dynamo/common/lora/manager.py",
    ),
    (
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING",
        ["vllm"],
        "LoRA",
        None,
        "vLLM's own switch that must be on for runtime LoRA updates. Set by the Dynamo worker when LoRA is enabled.",
        "components/src/dynamo/vllm/main.py",
    ),
    (
        "VLLM_LORA_MODULES_LOADING_TIMEOUT",
        ["vllm"],
        "LoRA",
        None,
        "Timeout vLLM applies when loading LoRA modules.",
        "components/src/dynamo/vllm/main.py",
    ),
    (
        "VLLM_NIXL_SIDE_CHANNEL_HOST",
        ["vllm"],
        "vLLM extras (non-CLI)",
        None,
        "Host vLLM's NIXL connector advertises for its KV side channel. Dynamo sets it from the worker's resolved IP.",
        "components/src/dynamo/vllm/args.py",
    ),
    (
        "VLLM_WORKER_MULTIPROC_METHOD",
        ["vllm"],
        "vLLM extras (non-CLI)",
        None,
        "Multiprocessing start method for vLLM workers.",
        "components/src/dynamo/vllm/main.py",
    ),
    (
        "VLLM_LOG_STATS_INTERVAL",
        ["vllm"],
        "vLLM extras (non-CLI)",
        None,
        "Interval at which vLLM logs engine stats; Dynamo aligns it with its own metric polling.",
        "components/src/dynamo/vllm/main.py",
    ),
    (
        "VLLM_NO_USAGE_STATS",
        ["vllm"],
        "vLLM extras (non-CLI)",
        None,
        "Disable vLLM usage telemetry.",
        "components/src/dynamo/vllm/main.py",
    ),
    (
        "VLLM_CONFIGURE_LOGGING",
        ["vllm"],
        "vLLM extras (non-CLI)",
        None,
        "Let Dynamo own logging configuration instead of vLLM.",
        "components/src/dynamo/vllm/main.py",
    ),
    (
        "PROMETHEUS_MULTIPROC_DIR",
        ["vllm"],
        "vLLM extras (non-CLI)",
        None,
        "Directory for multiprocess Prometheus collectors when vLLM runs multiple engine processes.",
        "components/src/dynamo/vllm/main.py",
    ),
    (
        "DYN_RUNTIME_ENABLED_KVBM",
        ["vllm", "trtllm"],
        "KVBM (KV block manager)",
        None,
        "Gate that turns the KVBM connector on inside the engine process.",
        "lib/bindings/kvbm/python/kvbm/utils.py#L61",
    ),
    # ---- TRT-LLM specific extras
    (
        "DYN_TRTLLM_PUBLISH_EVENTS_AND_METRICS",
        ["trtllm"],
        "TensorRT-LLM extras (non-CLI)",
        None,
        "Deprecated alias for --publish-kv-events / DYN_TRTLLM_PUBLISH_KV_EVENTS.",
        "components/src/dynamo/trtllm/args.py#L103",
    ),
    (
        "DYN_TRTLLM_SERVER_DISABLE_GC",
        ["trtllm"],
        "TensorRT-LLM extras (non-CLI)",
        "false",
        "Disable Python cyclic GC in the TensorRT-LLM worker to remove GC pauses from the hot path.",
        "components/src/dynamo/trtllm/main.py#L99",
    ),
    (
        "DYN_KVBM_TRTLLM_ZMQ_PORT",
        ["trtllm"],
        "KVBM (KV block manager)",
        None,
        "Override the ZMQ port used by the TensorRT-LLM KVBM event consolidator.",
        "lib/bindings/kvbm/python/kvbm/trtllm_integration/consolidator_config.py#L104",
    ),
    (
        "DYN_KVBM_KV_EVENTS_ENABLE_CONSOLIDATOR",
        ["vllm", "trtllm"],
        "KVBM (KV block manager)",
        None,
        "Enable the KVBM KV-event consolidator that merges engine and KVBM events.",
        "lib/bindings/kvbm/python/kvbm/trtllm_integration/consolidator_config.py#L50",
    ),
    (
        "DYN_ENABLE_TEST_LOGITS_PROCESSOR",
        ["trtllm"],
        "TensorRT-LLM extras (non-CLI)",
        "0",
        "Smoke-test hook that installs a dummy logits processor. Test-only.",
        "components/src/dynamo/common/backend/engine.py#L457",
    ),
    # ---- SGLang specific extras
    (
        "DYN_FORWARDPASS_METRIC_PORT",
        ["sglang"],
        "SGLang extras (non-CLI)",
        None,
        "ZMQ port SGLang publishes forward-pass metrics on; used instead of the default system port wiring.",
        "components/src/dynamo/sglang/args.py#L93",
    ),
    (
        "DYN_SGL_ALLOW_TOP_LOGPROBS",
        ["sglang"],
        "SGLang extras (non-CLI)",
        "0",
        "Override the guard that blocks top-logprobs requests on SGLang, where they are known to be unreliable.",
        "components/src/dynamo/common/backend/logprobs.py#L237",
    ),
    (
        "DYN_SKIP_SGLANG_LOG_FORMATTING",
        ["sglang"],
        "SGLang extras (non-CLI)",
        "false",
        "Leave SGLang's own log formatting in place instead of reformatting it into the Dynamo log format.",
        "lib/bindings/python/src/dynamo/runtime/logging.py#L162",
    ),
    (
        "SGLANG_BLOCK_NONZERO_RANK_CHILDREN",
        ["sglang"],
        "SGLang extras (non-CLI)",
        None,
        "SGLang switch controlling whether non-zero-rank child processes are blocked; set by the Dynamo worker.",
        "components/src/dynamo/sglang/main.py",
    ),
]
for env, scope, section, default, desc, source in C:
    add(env, scope, section, default=default, desc=desc, source=source)

# ----------------------------------------------- CLI flags with no env var
CLI_ONLY = "CLI-only flags (no environment variable)"
add_flag(
    "--version",
    ALL4,
    CLI_ONLY,
    desc="Print the component version and exit.",
    source="components/src/dynamo/frontend/frontend_args.py#L175",
)
add_flag(
    "--tool-call-parser",
    ["frontend"],
    CLI_ONLY,
    desc="SGLang-native pass-through flag. Accepted only with --dyn-chat-processor sglang, and forwarded to the SGLang pre/post processor. The Dynamo-native equivalent is --dyn-tool-call-parser / DYN_TOOL_CALL_PARSER on the worker.",
    source="components/src/dynamo/frontend/main.py#L308",
)
add_flag(
    "--reasoning-parser",
    ["frontend"],
    CLI_ONLY,
    desc="SGLang-native pass-through flag. Accepted only with --dyn-chat-processor sglang. The Dynamo-native equivalent is --dyn-reasoning-parser / DYN_REASONING_PARSER on the worker.",
    source="components/src/dynamo/frontend/main.py#L309",
)
add_flag(
    "--chat-template",
    ["frontend"],
    CLI_ONLY,
    desc="SGLang-native pass-through flag. Accepted only with --dyn-chat-processor sglang. The Dynamo-native equivalent is --custom-jinja-template / DYN_CUSTOM_JINJA_TEMPLATE on the worker.",
    source="components/src/dynamo/frontend/main.py#L310",
)
add_flag(
    "--admission-control",
    ["frontend"],
    CLI_ONLY,
    choices=["token-capacity", "none"],
    desc="Deprecated and ignored: accepted so existing launch commands keep starting, but sets nothing. Hidden from --help. Use the explicit --active-decode-blocks-threshold / --active-prefill-tokens-threshold flags instead.",
    source="components/src/dynamo/common/configuration/groups/router_args.py#L148",
)
add_flag(
    "--router-kv-overlap-score-weight",
    ["frontend"],
    CLI_ONLY,
    aliases=["--kv-overlap-score-weight"],
    desc="Deprecated: legacy overlap-score weight, hidden from --help and warned on use. Superseded by --router-prefill-load-scale. Its env vars DYN_ROUTER_KV_OVERLAP_SCORE_WEIGHT and DYN_OVERLAP_SCORE_WEIGHT are still read.",
    source="components/src/dynamo/common/configuration/groups/kv_router_args.py#L201",
)

# -i / --interactive reads its env var through env_or_default() rather than the
# add_argument() helper, so the AST pass above does not see it.
add(
    "DYN_INTERACTIVE",
    ["frontend"],
    "Frontend core",
    flag="--interactive",
    default=False,
    desc="Run an interactive text chat in the terminal instead of serving HTTP.",
    source="components/src/dynamo/frontend/frontend_args.py#L182",
)
records["DYN_INTERACTIVE"]["aliases"] = ["-i"]
records["DYN_ADMISSION_CONTROL"]["flag"] = "--admission-control"

# ---------------------------------------------------------------- emit
for r in records.values():
    r["scope"] = sorted(r["scope"], key=lambda s: ALL4.index(s))

NOTES = {
    "frontend": "Beyond the flags below, the frontend forwards unrecognised arguments to the "
    "selected chat processor: with <code>--dyn-chat-processor vllm</code> it accepts vLLM's own "
    "<code>FrontendArgs</code> and <code>AsyncEngineArgs</code> flags, and with "
    "<code>--dyn-chat-processor sglang</code> it accepts the three SGLang-native flags listed under "
    "CLI-only flags. Those pass-through flags belong to the engine, not to Dynamo, and have no "
    "Dynamo environment variable.",
    "vllm": "The vLLM worker parses Dynamo's own flags first and forwards everything it does not "
    "recognise to vLLM's <code>AsyncEngineArgs</code> parser (<code>--tensor-parallel-size</code>, "
    "<code>--gpu-memory-utilization</code>, <code>--kv-transfer-config</code>, and the rest). Those "
    "flags are vLLM's, change with the vLLM version, and have no Dynamo environment variable - though "
    "many have a <code>VLLM_*</code> variable of their own.",
    "trtllm": "TensorRT-LLM engine settings that Dynamo does not wrap as a flag are supplied through "
    "the <code>--extra-engine-args</code> YAML file or the <code>--override-engine-args</code> "
    "dictionary string rather than as individual CLI flags, so they have no Dynamo environment variable.",
    "sglang": "The SGLang worker parses Dynamo's own flags first and forwards everything it does not "
    "recognise to SGLang's <code>ServerArgs</code> parser (<code>--tp-size</code>, "
    "<code>--mem-fraction-static</code>, <code>--enable-trace</code>, and the rest). Those flags are "
    "SGLang's, change with the SGLang version, and have no Dynamo environment variable.",
}


TABS = [
    ("frontend", "Frontend", "python -m dynamo.frontend"),
    ("vllm", "vLLM worker", "python -m dynamo.vllm"),
    ("trtllm", "TensorRT-LLM worker", "python -m dynamo.trtllm"),
    ("sglang", "SGLang worker", "python -m dynamo.sglang"),
]

SECTION_ORDER = [
    "Frontend core",
    "HTTP service & API surface",
    "Preprocessing, templates & parsers",
    "Router: mode & admission",
    "KV router tuning",
    "AIC performance model",
    "Worker runtime & identity",
    "vLLM engine wrapper",
    "TensorRT-LLM engine wrapper",
    "SGLang engine wrapper",
    "vLLM extras (non-CLI)",
    "TensorRT-LLM extras (non-CLI)",
    "SGLang extras (non-CLI)",
    "KVBM (KV block manager)",
    "Forward-pass metric trace",
    "RL training support",
    "Multimodal",
    "Multimodal HTTP fetch client",
    "LoRA",
    "Topology & KV transfer",
    "Discovery",
    "Request / event plane",
    "NATS",
    "etcd",
    "Model download",
    "Logging",
    "OpenTelemetry export",
    "Request tracing",
    "Metrics",
    "System status server",
    "Health checks",
    "Shutdown & lifecycle",
    "Tokio runtime",
    "Memory",
    "Profiling",
]


def sec_key(s):
    return (SECTION_ORDER.index(s) if s in SECTION_ORDER else 999, s)


def esc(x):
    return html.escape("" if x is None else str(x))


def fmt_default(d):
    if d is None:
        return '<span class="none">—</span>'
    if d is True:
        return "<code>true</code>"
    if d is False:
        return "<code>false</code>"
    if d == "":
        return '<code>""</code>'
    if isinstance(d, list):
        return "<code>%s</code>" % esc(", ".join(map(str, d)) or "[]")
    return "<code>%s</code>" % esc(d)


# ---------------------------------------------------------------- perf tagging
# Whether a setting can move a performance number. Every entry starts as
# "unexamined". Proposed verdicts live in perf-classification.md and are shown for
# review in perf-classification-review.html; they are only copied in here once a
# human has signed them off. Key on the variable name, or on the flag for a
# CLI-only row (e.g. "--version").
PERF_DEFAULT = "unexamined"
PERF = {}
PERF_VALUES = ("impact", "no impact", "unexamined")
PERF_SLUG = {"impact": "impact", "no impact": "noimpact", "unexamined": "unexamined"}


def perf_of(r):
    return PERF.get(r["env"] or r["flag"], PERF_DEFAULT)


SCOPE_LABEL = {"frontend": "FE", "vllm": "vLLM", "trtllm": "TRT", "sglang": "SGL"}


def row(r, tab):
    scopes = "".join(
        f'<span class="chip s-{s}{"" if s in r["scope"] else " off"}">{SCOPE_LABEL[s]}</span>'
        for s in ALL4
    )
    if r["env"] is None:
        name_cell = (
            f'<code class="env cli">{esc(r["flag"])}</code>'
            '<span class="chip cli-only">CLI only</span>'
        )
    else:
        name_cell = f'<code class="env">{esc(r["env"])}</code>'
    flags = ([r["flag"]] if r.get("flag") else []) + list(r.get("aliases") or [])
    # A CLI-only row already shows its primary flag in the name column; the flag
    # column then carries only the aliases, if any.
    shown = flags[1:] if r["env"] is None else flags
    flag = " ".join(f'<code class="flag">{esc(f)}</code>' for f in shown)
    ch = ""
    if r.get("choices") and isinstance(r["choices"], list):
        ch = (
            '<div class="choices">choices: '
            + " ".join(f"<code>{esc(c)}</code>" for c in r["choices"])
            + "</div>"
        )
    src = r.get("source") or ""
    srclink = (
        f'<a class="src" href="{GH}{esc(src)}" target="_blank" rel="noopener">{esc(src.split("#")[0].split("/")[-1])}</a>'
        if src
        else ""
    )
    shared = " shared" if len(r["scope"]) > 1 else ""
    perf = perf_of(r)
    perf_chip = f'<span class="chip perf p-{PERF_SLUG[perf]}">{esc(perf)}</span>'
    return f"""<tr class="r{shared}" data-perf="{PERF_SLUG[perf]}" data-text="{esc(((r['env'] or '')+' '+' '.join(flags)+' '+(r.get('desc') or ''))).lower()}">
  <td class="c-name">{name_cell}{scopes}{perf_chip}</td>
  <td class="c-flag">{flag}{ch}</td>
  <td class="c-def">{fmt_default(r.get('default'))}</td>
  <td class="c-desc">{esc(r.get('desc') or '')}{('<div class="srcwrap">'+srclink+'</div>') if srclink else ''}</td>
</tr>"""


def build_tab(tab):
    rows = [r for r in records.values() if tab in r["scope"]]
    by_sec = collections.defaultdict(list)
    for r in rows:
        by_sec[r["sections"].get(tab, "Other")].append(r)
    out = []
    for sec in sorted(by_sec, key=sec_key):
        items = sorted(by_sec[sec], key=lambda r: r["env"] or r["flag"])
        out.append(
            f'<section class="sec"><h3>{esc(sec)} <span class="count">{len(items)}</span></h3>'
            f"<table><thead><tr><th>Variable</th><th>CLI flag</th><th>Default</th><th>Description</th></tr></thead><tbody>"
            + "\n".join(row(r, tab) for r in items)
            + "</tbody></table></section>"
        )
    return "\n".join(out), len(rows)


# shared tab: vars in >1 scope
def build_shared():
    groups = [
        (
            "Shared by all four (frontend + vLLM + TensorRT-LLM + SGLang)",
            lambda s: len(s) == 4,
        ),
        (
            "Shared by all three workers (vLLM + TensorRT-LLM + SGLang)",
            lambda s: set(s) == set(WORKERS),
        ),
        ("Shared by two components", lambda s: len(s) == 2),
    ]
    out = []
    for title, pred in groups:
        items = sorted(
            [r for r in records.values() if pred(r["scope"])],
            key=lambda r: r["env"] or r["flag"],
        )
        if not items:
            continue
        by_sec = collections.defaultdict(list)
        for r in items:
            by_sec[sorted(r["sections"].values(), key=sec_key)[0]].append(r)
        inner = []
        for sec in sorted(by_sec, key=sec_key):
            inner.append(
                f'<h4>{esc(sec)} <span class="count">{len(by_sec[sec])}</span></h4>'
                "<table><thead><tr><th>Variable</th><th>CLI flag</th><th>Default</th><th>Description</th></tr></thead><tbody>"
                + "\n".join(
                    row(r, r["scope"][0])
                    for r in sorted(by_sec[sec], key=lambda x: x["env"] or x["flag"])
                )
                + "</tbody></table>"
            )
        out.append(
            f'<section class="sec shared-group"><h3>{esc(title)} <span class="count">{len(items)}</span></h3>'
            + "\n".join(inner)
            + "</section>"
        )
    return "\n".join(out), len([r for r in records.values() if len(r["scope"]) > 1])


panels, navs = [], []
for key, label, cmd in TABS:
    body, n = build_tab(key)
    navs.append(
        f'<button class="tab" data-tab="{key}">{esc(label)}<span class="n">{n}</span></button>'
    )
    note = (
        f'<p class="note"><b>Engine pass-through.</b> {NOTES[key]}</p>'
        if key in NOTES
        else ""
    )
    panels.append(
        f'<div class="panel" id="p-{key}"><p class="cmd"><code>{esc(cmd)}</code></p>{note}{body}</div>'
    )
sbody, sn = build_shared()
navs.append(
    f'<button class="tab" data-tab="shared">Shared<span class="n">{sn}</span></button>'
)
panels.append(
    f'<div class="panel" id="p-shared"><p class="cmd">Variables and flags read by more than one component. Each also appears in its own tab above.</p>{sbody}</div>'
)

CSS = """
:root{--bg:#f7f8fa;--card:#fff;--ink:#12151a;--muted:#5d6672;--line:#e3e7ed;--accent:#3d7dd8;--code:#f2f4f7;
--fe:#3d7dd8;--vllm:#2f9e6e;--trt:#c26a1f;--sgl:#8355c9;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif}
header{background:var(--card);border-bottom:1px solid var(--line);padding:26px 32px 0}
.wrap{max-width:1400px;margin:0 auto}
h1{margin:0 0 6px;font-size:24px;letter-spacing:-.01em}
.sub{color:var(--muted);margin:0 0 18px;max-width:900px}
.tools{display:flex;gap:12px;align-items:center;margin-bottom:14px;flex-wrap:wrap}
input[type=search]{flex:1;min-width:240px;max-width:420px;padding:9px 12px;border:1px solid var(--line);border-radius:8px;font-size:14px;background:var(--card)}
input[type=search]:focus{outline:2px solid rgba(61,125,216,.25);border-color:var(--accent)}
label.tog{display:flex;gap:6px;align-items:center;color:var(--muted);font-size:13px;cursor:pointer;user-select:none}
nav{display:flex;gap:4px;flex-wrap:wrap}
.tab{appearance:none;background:transparent;border:1px solid transparent;border-bottom:none;padding:10px 16px;border-radius:8px 8px 0 0;
font:600 14px/1 inherit;color:var(--muted);cursor:pointer;display:flex;gap:8px;align-items:center}
.tab:hover{color:var(--ink)}
.tab.on{background:var(--bg);border-color:var(--line);color:var(--ink);margin-bottom:-1px}
.tab .n{background:var(--code);color:var(--muted);border-radius:20px;padding:2px 8px;font-size:11px;font-weight:600}
.tab.on .n{background:var(--accent);color:#fff}
main{max-width:1400px;margin:0 auto;padding:24px 32px 80px}
.panel{display:none}.panel.on{display:block}
.cmd{color:var(--muted);font-size:13px;margin:0 0 12px}
.note{background:#fff8ec;border:1px solid #f0e0c2;border-radius:10px;padding:12px 16px;margin:0 0 20px;font-size:13px;color:#5a4a2e}
.note b{color:#4a3a1e}
.sec{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:6px 20px 18px;margin-bottom:20px}
.sec h3{font-size:15px;margin:16px 0 10px;display:flex;gap:10px;align-items:center;letter-spacing:.01em}
.sec h4{font-size:13px;color:var(--muted);margin:20px 0 8px;text-transform:uppercase;letter-spacing:.06em;display:flex;gap:8px;align-items:center}
.count{background:var(--code);color:var(--muted);border-radius:20px;padding:1px 8px;font-size:11px;font-weight:600}
table{width:100%;border-collapse:collapse}
thead th{text-align:left;font-size:11px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);
padding:8px 10px;border-bottom:1px solid var(--line);font-weight:600}
td{padding:11px 10px;border-bottom:1px solid var(--line);vertical-align:top}
tr:last-child td{border-bottom:none}
tr.r:hover{background:#fafbfd}
.c-name{width:23%}.c-flag{width:19%}.c-def{width:11%}
code{font:12.5px/1.5 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace;background:var(--code);padding:1.5px 5px;border-radius:4px}
code.env{background:transparent;padding:0;font-weight:600;color:var(--ink);word-break:break-all}
code.flag{color:#1f5fae;background:#eef4fc}
code.env.cli{color:#1f5fae}
.chip.cli-only{background:#6b7480;letter-spacing:.02em}
.chip.perf{text-transform:lowercase;letter-spacing:.02em}
.p-impact{background:#c0392b}.p-noimpact{background:#2f9e6e}.p-unexamined{background:#98a2ae}
select{padding:6px 8px;border:1px solid var(--line);border-radius:7px;background:var(--card);font:inherit;font-size:13px;color:var(--ink);cursor:pointer}
select:focus{outline:2px solid rgba(61,125,216,.25);border-color:var(--accent)}
.none{color:#a8b0bb}
.chip{display:inline-block;margin:5px 3px 0 0;font:600 9.5px/1.4 ui-monospace,monospace;padding:2px 5px;border-radius:4px;color:#fff}
.s-frontend{background:var(--fe)}.s-vllm{background:var(--vllm)}.s-trtllm{background:var(--trt)}.s-sglang{background:var(--sgl)}
.chip.off{background:#eceff3;color:#c2c8d0}
.c-desc{color:#2b3038;font-size:13.5px}
.choices{margin-top:5px;font-size:11px;color:var(--muted);line-height:1.9}
.choices code{font-size:11px}
.srcwrap{margin-top:5px}
a.src{font:11px ui-monospace,monospace;color:var(--muted);text-decoration:none;border-bottom:1px dotted var(--line)}
a.src:hover{color:var(--accent)}
.empty{padding:28px;text-align:center;color:var(--muted)}
footer{max-width:1400px;margin:0 auto;padding:0 32px 60px;color:var(--muted);font-size:12.5px}
"""

JS = """
const tabs=[...document.querySelectorAll('.tab')],panels=[...document.querySelectorAll('.panel')];
function show(k){tabs.forEach(t=>t.classList.toggle('on',t.dataset.tab===k));
panels.forEach(p=>p.classList.toggle('on',p.id==='p-'+k));location.hash=k;filter();}
tabs.forEach(t=>t.onclick=()=>show(t.dataset.tab));
const q=document.getElementById('q'),so=document.getElementById('sharedonly'),pf=document.getElementById('perf');
function filter(){const s=q.value.trim().toLowerCase(),only=so.checked,pv=pf.value;
document.querySelectorAll('.panel.on .sec').forEach(sec=>{let vis=0;
 sec.querySelectorAll('tr.r').forEach(r=>{const ok=(!s||r.dataset.text.includes(s))&&(!only||r.classList.contains('shared'))&&(!pv||r.dataset.perf===pv);
  r.style.display=ok?'':'none';if(ok)vis++;});
 sec.querySelectorAll('table').forEach(t=>{const any=[...t.querySelectorAll('tr.r')].some(r=>r.style.display!=='none');
  t.style.display=any?'':'none';const h=t.previousElementSibling;if(h&&h.tagName==='H4')h.style.display=any?'':'none';});
 sec.style.display=vis?'':'none';});}
q.oninput=filter;so.onchange=filter;pf.onchange=filter;
show((location.hash||'#frontend').slice(1));
document.addEventListener('click',e=>{const c=e.target.closest('code.env');if(!c)return;
navigator.clipboard.writeText(c.textContent);const o=c.style.background;c.style.background='#d8ecd8';setTimeout(()=>c.style.background=o,320);});
"""

n_env = len([r for r in records.values() if r["env"]])
n_cli = len(records) - n_env
total = len(records)
doc = f"""<!doctype html>
<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Dynamo launch environment variables</title>
<style>{CSS}</style></head>
<body>
<header><div class="wrap">
<h1>Dynamo launch environment variables</h1>
<p class="sub">Environment variables read at process start by the Dynamo frontend and the vLLM, TensorRT-LLM, and SGLang workers.
Every variable is read by the process that hosts it, so it must be set on that process. Variables backed by a CLI flag are
overridden by the flag when both are given. The few Dynamo flags with no environment variable are listed too, marked
<span class="chip cli-only">CLI only</span>. Every entry also carries a performance-impact tag — whether changing it can move a
performance number — which starts at <span class="chip perf p-unexamined">unexamined</span> until the setting has been reviewed.
Click a name to copy it.</p>
<div class="tools">
  <input type="search" id="q" placeholder="Filter by name, flag, or description…" autocomplete="off">
  <label class="tog" for="perf">Performance impact
    <select id="perf">
      <option value="">all</option>
      <option value="impact">impact</option>
      <option value="noimpact">no impact</option>
      <option value="unexamined">unexamined</option>
    </select>
  </label>
  <label class="tog"><input type="checkbox" id="sharedonly"> only variables shared across components</label>
  <span style="color:var(--muted);font-size:13px">{n_env} variables · {n_cli} CLI-only flags</span>
</div>
<nav>{''.join(navs)}</nav>
</div></header>
<main>{''.join(panels)}</main>
<footer>Derived from <code>lib/runtime/src/config/environment_names.rs</code>, the <code>ArgGroup</code> definitions under
<code>components/src/dynamo/*/backend_args.py</code> and <code>components/src/dynamo/common/configuration/groups/</code>, and direct
<code>env::var</code> / <code>os.environ</code> reads in <code>lib/</code> and <code>components/</code>. CLI-only flags come from
the same AST pass, collecting <code>add_argument</code> calls that declare no <code>env_var</code>. Variables that appear only in
example launch scripts, recipes, or docs (and are not read by Dynamo code) are excluded, as are planner, profiler, mocker, router-standalone,
omni/diffusion, and test-only variables.</footer>
<script>{JS}</script></body></html>"""

if globals().get("__name__") == "__main__":
    open("notes/env-vars/dynamo-launch-env-vars.html", "w").write(doc)
    print("wrote notes/env-vars/dynamo-launch-env-vars.html", total, "vars")
    for key, label, _ in TABS:
        print(" ", label, len([r for r in records.values() if key in r["scope"]]))
