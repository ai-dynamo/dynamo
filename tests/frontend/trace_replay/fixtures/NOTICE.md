# DeepSeek V4.1 BFCL replay fixtures

These five fixtures derive from the Berkeley Function Calling Leaderboard
(BFCL), part of [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard),
revision `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`. The upstream
[Apache 2.0 license](LICENSE.BFCL) is retained here. Each record links its
specific pinned source file through `source.context.ref` and uses
`source.kind: "benchmark"`.

The captures used `deepseek-ai/DeepSeek-V4.1-Flash`, revision
`2cba9e42aa026125f3ed06c6d98c1db82f7ca027`, on 2026-11-10. The independent
vLLM OpenAI server was version `0.30.1rc1.dev493+gac9126e58`, installed in the
public image
`nvcr.io/nvidia/ai-dynamo/vllm-runtime:1.6.0-deepseek-v4.1-flash-dev.2-post.1`,
digest `sha256:a16bc24d474919b5305a9d53a306ae27e35f0c21fcf54430a2047be39cc963b4`.
The Dynamo frontend was outside the capture request path. These are genuine
V4.1 outputs; V4 response text or arguments were not reused.

No model weights are included or needed for replay. The manifest pins the
three native assets, `config.json`, `tokenizer.json`, and
`tokenizer_config.json`, by SHA-256. This checkpoint does not ship
`generation_config.json`; no replacement file is synthesized. No Jinja
override is used by the native DeepSeek formatter.

The request builder selected the first BFCL turn, added the documented
language hint, normalized tool names, and converted BFCL schema types to JSON
Schema. Per-case transformations and original tool-name mappings remain in
`reference.capture.bfcl`. Export removed only capture-specific
`return_token_ids` and the captured `stream` selector. The runner chooses
unary or SSE transport. Captured thinking settings are preserved.

| BFCL ID | Captured thinking | Prompt IDs | Output IDs |
|---|---|---:|---:|
| `simple_python_245` | disabled | 386 | 50 |
| `multiple_2` | enabled | 466 | 56 |
| `parallel_82` | enabled | 467 | 159 |
| `parallel_multiple_48` | disabled | 397 | 85 |
| `irrelevance_28` | disabled | 339 | 9 |

Recorded prompt IDs, output IDs, and source JSONL record bytes survive
unchanged. All five completed output arrays include terminal EOS token 1.
Their worker finish is `stop`, with null stop reason; the frontend must derive
API `tool_calls` for the four tool-bearing cases. The worker does not inject
the parser-derived API finish reason. Chunk schedules are synthetic
rechunking of recorded IDs and make no claim about original streaming timing.

The five response expectations were independently reviewed using the pinned
checkpoint's
[official encoding implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/2cba9e42aa026125f3ed06c6d98c1db82f7ca027/encoding/encoding.py),
SHA-256 `502bdaec8a3fd88ebc24c4721a7038fbe42f2063c664638127056107920035c1`,
with `tokenizers==0.21.4`. Its encoder reproduces every selected prompt ID
array exactly. Parsing and rendering each complete output at its conversation
continuation index roundtrips every decoded byte, including EOS. The request's
reasoning effort is preserved, while its prompt-only effort prefix stays
outside the completion. No candidate Dynamo output establishes these goldens.

The official formatter inserts two structural newlines before a DSML call
block. Those newlines are excluded from visible content. The captured peer
parser retains them in the three content-bearing tool-call cases; its original
parsed messages remain in `reference.message` for diagnosis. Empty logical
content/reasoning channels are projected to nullable OpenAI channels. Tool
arguments remain typed JSON values, including V4.1's explicit `unit: "km/h"`
velocity arguments, and tool order is significant.

The self-contained `prompt-audit.json.gz` receipt contains exactly the five
selected request/prompt identities. Its `requests_sha256` explicitly identifies
the full upstream capture request set of 9,392 requests, not a five-request
document. The manifest records the selected receipt hash, full source audit
hash, immutable source commit, and source manifest hash. Approved model,
tokenizer, vendor encoder, request-byte, prompt-ID, and output-ID identities
bind each reviewed response to its original evidence. All five original JSONL
lines are copied verbatim; only public-safe manifest metadata is selected.

Additional traces without independent response expectations remain
**unjudged** at the output checkpoint until review. Captured peer responses
remain diagnostic; captured engine responses alone do not establish output
correctness.
