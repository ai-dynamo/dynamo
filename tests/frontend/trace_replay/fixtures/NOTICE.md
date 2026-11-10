# DeepSeek V4 BFCL replay fixtures

These five fixtures derive from the Berkeley Function Calling Leaderboard
(BFCL), part of [ShishirPatil/gorilla](https://github.com/ShishirPatil/gorilla/tree/6ea57973c7a6097fd7c5915698c54c17c5b1b6c8/berkeley-function-call-leaderboard),
revision `6ea57973c7a6097fd7c5915698c54c17c5b1b6c8`. The upstream
[Apache 2.0 license](LICENSE.BFCL) is retained here. Each record links to its
specific pinned source file in `source.context.ref`.

The captures used `deepseek-ai/DeepSeek-V4-Flash-0731`, revision
`7872f01b1d1fe23eabc4c98b48bffcef5a386062`, served by stock vLLM 0.31.0.
They are DeepSeek V4 captures, not V4.1. No model weights are included.
The manifest pins the four native tokenizer/config assets by SHA-256.
The unrelated Jinja override present in the capture bundle is excluded because
the native DeepSeek formatter does not use it.

The request builder selected the first BFCL turn, added the documented
language hint, normalized tool names, and converted BFCL schema types to JSON
Schema. Per-case transformations and original tool-name mappings remain in
`reference.capture.bfcl`. Export removes only the capture-only
`return_token_ids` field and the captured `stream` transport selector. The
runner selects unary or SSE transport. Recorded prompt IDs and output IDs are
retained in full; raw capture and original request-byte hashes are retained for
audit. EOS token 1 remains in completed output IDs. A captured API
`tool_calls` finish reason becomes worker `stop` only when EOS proves engine
termination; the frontend must derive the API finish reason itself. Worker
chunk schedules are synthetic rechunking, not captured streaming timing.

| BFCL ID | Captured thinking | Prompt IDs | Output IDs |
|---|---|---:|---:|
| `simple_python_245` | disabled | 395 | 48 |
| `multiple_2` | enabled | 451 | 68 |
| `parallel_82` | enabled | 452 | 143 |
| `parallel_multiple_48` | disabled | 406 | 97 |
| `irrelevance_28` | disabled | 348 | 18 |

The five response expectations were independently reviewed using DeepSeek's
[official encoding implementation](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731/blob/7872f01b1d1fe23eabc4c98b48bffcef5a386062/encoding/encoding_dsv4.py),
at the same model revision, SHA-256
`abc0d26120250dda0ae077dc64aa28836026e61e970854aaeb792445e6a0dde6`.
With `tokenizers==0.21.4`, its prompt renderer and the pinned tokenizer reproduce
all 9,392 source-capture prompt ID arrays exactly, including these five. Tools
are attached to a system message (a synthetic empty one when absent) and the
captured thinking settings are preserved.
Parsing and rendering each complete output roundtrips every decoded byte,
including EOS. This check uses no candidate Dynamo output.

The official formatter inserts two newlines before a DSML tool-call block.
Those newlines are structural and are excluded from expected visible content.
The captured vLLM parser retains them in two cases; its original parsed message
is preserved separately in `reference.message` for diagnosis. Empty logical
content/reasoning channels are projected to nullable OpenAI wire channels.
Tool arguments are typed JSON values and tool order is significant.

Additional traces without independent response expectations remain
**unjudged** at the output checkpoint until review. Captured peer responses
remain diagnostic; captured engine responses alone do not establish output
correctness.
