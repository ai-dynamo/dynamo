<!--
SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Hosted web search gateway

This example adds opt-in `web_search` execution in front of Dynamo's
`/v1/responses` endpoint. Dynamo generates a search query, the gateway calls an
operator-configured search service, and Dynamo uses the returned snippets to
answer. The gateway returns `web_search_call` items and URL citation annotations.

This is the first implementation of
[DEP #15403](https://github.com/ai-dynamo/dynamo/issues/15403), extending
[hosted tools issue #12859](https://github.com/ai-dynamo/dynamo/issues/12859).
It does not enable hosted tools on the native Dynamo endpoint. Clients must use
the gateway endpoint. No GPU worker or frontend-crates change is required.

## Run

Requirements: Python 3.11+, an existing Dynamo Responses endpoint with a
function-capable model, and an HTTP search service implementing the contract below.
A commercial search API typically needs a small adapter for this contract.

From the Dynamo repository root:

```bash
python3 -m venv /tmp/dynamo-search-venv
source /tmp/dynamo-search-venv/bin/activate
python -m pip install -r examples/deployments/hosted_web_search/requirements.txt
python examples/deployments/hosted_web_search/gateway.py \
  --dynamo-url http://127.0.0.1:8000 \
  --search-url http://127.0.0.1:9000/search
```

The gateway listens on `127.0.0.1:8001`. Set `DYNAMO_API_KEY` and `SEARCH_API_KEY`
if the respective services require bearer credentials. Client authorization is
not forwarded to either service. The example has no client authentication or TLS;
keep it on loopback or behind an authenticated ingress before exposing it.
Only `POST /v1/responses` is served; model discovery stays on Dynamo.

```bash
curl http://127.0.0.1:8001/v1/responses \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "<served-model>",
    "input": "Where is the frontend integrated in Dynamo? Cite your sources.",
    "tools": [{"type": "web_search"}],
    "tool_choice": {"type": "web_search"},
    "include": ["web_search_call.action.sources"],
    "max_output_tokens": 2048
  }'
```

Use `tool_choice: "auto"` to let the model decide whether to search. Explicit
`{"type":"web_search"}` requires search on the first turn and restricts subsequent
turns to search or text. This explicit choice is a gateway extension that some
OpenAI SDK versions do not include in their choice type; `"required"` with only
the search tool also forces an initial search.

## Search service contract

The configured endpoint receives a POST body:

```json
{"query":"Dynamo frontend integration","max_results":5}
```

It must return HTTP 200 and a JSON object with at most `max_results` entries:

```json
{
  "results": [
    {
      "url": "https://github.com/ai-dynamo/dynamo",
      "title": "Dynamo",
      "snippet": "The repository contains Python frontend components and Rust serving code."
    }
  ]
}
```

An empty result list is valid. URLs must use HTTP or HTTPS and contain no embedded
credentials. The gateway does not fetch result URLs or follow provider redirects.
Provider response bodies are never copied into error messages. Snippets enter the
model context as tool output; the private tool description tells the model to
treat them as untrusted data and cite their numbered markers.

## Supported behavior and limits

- Accept one `{"type":"web_search"}` definition without search options. Other
  hosted tools, search filters, location options, `allowed_tools`, stored responses,
  background execution, and conversation references are rejected before inference.
- Preserve function and namespace definitions. Execute only the gateway's private
  search function; return client function calls for the client to execute. If a
  model selects both kinds in one turn, perform the searches and return the client
  calls without another inference turn. Named function choices and `none` cannot
  trigger a hosted search. `required` requires a tool on the first completed turn.
- Use stateless requests. Replay messages and ordinary function call/result items
  explicitly. Hosted output items cannot be replayed as input; callers continuing
  a mixed-tool conversation must retain relevant search evidence in message text.
- Charge output tokens across all model turns to one `max_output_tokens` budget.
  Report accumulated inference usage, including intermediate turns. Search-provider
  billing is outside this usage object.
- Annotate only `[N]` markers that reference returned sources. Citation spans use
  Unicode character offsets and cover the marker. Citations identify supplied
  evidence; the gateway does not verify that the answer is supported by it.
- `stream: true` returns Responses SSE with increasing sequence numbers, search
  progress, output items, and a terminal event. Each model turn is buffered;
  text does not stream token by token. A failure after headers emits
  `response.failed`; JSON failures use an HTTP error status.
- Disconnects cancel outstanding HTTP work. Defaults are a 120-second total
  deadline, three searches, five results per search, 4,096 output tokens, a 1 MiB
  client/model body limit, and a 64 KiB provider body limit. There are no retries.
  Set lower per-request limits with `max_tool_calls` and `max_output_tokens`;
  operator limits are fields on `Config` when embedding `create_app`.

## Request/response evidence

The HTTP tests capture a request and response in pytest's temporary directory.
Pass `-o tmp_path_retention_policy=all` to retain successful captures.
For the same hosted-tool request, the native adapter reports HTTP 400:

```json
{
  "message": "Failed to convert responses request: Unsupported Responses tools type 'web_search': the Chat Completions adapter supports only function tools and none, auto, required, named function, or function-only allowed_tools choices",
  "type": "Bad Request",
  "code": 400
}
```

The gateway test sends this request (also tested with streaming enabled):

```json
{
  "model": "test-model",
  "input": "Where is the frontend integrated in Dynamo?",
  "tools": [{"type": "web_search"}],
  "include": ["web_search_call.action.sources"],
  "max_output_tokens": 20,
  "stream": false
}
```

The captured HTTP 200 response contains the following fields. Request-local IDs
and unrelated metadata are omitted here. Inference and search are scripted in
this deterministic test; the result is not evidence of a real internet search.

```json
{
  "status": "completed",
  "output": [
    {
      "type": "web_search_call",
      "status": "completed",
      "action": {
        "type": "search",
        "query": "Dynamo frontend integration",
        "sources": [{"type": "url", "url": "https://github.com/ai-dynamo/dynamo"}]
      }
    },
    {
      "type": "message",
      "role": "assistant",
      "content": [{
        "type": "output_text",
        "text": "🦀 Dynamo uses Rust. [1]",
        "annotations": [{
          "type": "url_citation",
          "url": "https://github.com/ai-dynamo/dynamo",
          "title": "Dynamo",
          "start_index": 20,
          "end_index": 23
        }]
      }]
    }
  ],
  "usage": {"input_tokens": 20, "output_tokens": 12, "total_tokens": 32}
}
```

The tests verify the private function result is replayed into a second inference
request, its token budget falls from 20 to 15, and no private function call reaches
the client. They also cover mixed tools, provider errors, deadlines, cancellation,
and partial failure after a completed search.

With the repository's test dependencies installed:

```bash
DYN_TEST_OUTPUT_PATH=/tmp/dynamo-search-tests \
  python -m pytest tests/frontend/test_hosted_web_search_gateway.py -q
```
