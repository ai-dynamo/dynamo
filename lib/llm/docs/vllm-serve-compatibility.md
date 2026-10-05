<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# vLLM serve compatibility reference

This is a development reference for the implementation of the
[frontend compatibility design](https://github.com/ai-dynamo/dynamo/blob/da45b41777ec507974d3a6a41053a9e908e530d3/lib/llm/docs/frontend-protocol-framework-compatibility.md),
not a compatibility promise for an already published Dynamo release. Chat
completions is the primary endpoint. Completions is covered for compatibility.
The implementation is still in progress; fields and combinations not established
by evidence below remain **Unverified**.

## Review-branch evidence boundary

The admission implementation is committed at
`73051721b695e7b9b0770ac51a8ed71e11fb34ef` (B5), based on integration commit
`df905cc155ef929e9787b3d84fe676b0531d402e`. Its tested tree is
`ef5af0ba16934e9ead9b5012f7014fdc044ac631`, with source-diff SHA256
`c5f5d96d654e37d6d64587110be90c9584f6a2a71d3a3d92155b7499fdc7f269`.
Run `20261003T040432Z-df905cc155ef-fd2fb9` passed 1,387 Rust executions and
310 Python executions, including four actual native HTTP comparisons covering
both chat processors and default/full-vocabulary engine limits. Those four
parametrized tests contain the bounded request matrices; they are not four
universal server-compatibility claims. The rebuilt runtime wheel SHA256 is
`d6af4a1635ea5306f5b95b273c7840dd1c3e8de14d6c289e45577c2334cca13d`.

The catalog/schema change is a separate dependent branch (B6). Its validation
must establish metadata wiring independently; B5 has no public catalog route.
Earlier dirty-source digests retained below are historical implementation
checkpoints, not validation of the extracted B6 branch. The standalone native
prompt-count probe and historical same-release control likewise retain their
original scope rather than becoming new B5 executions.

## How to read a compatibility claim

A claim applies to an endpoint, frontend processor, frontend and worker revisions,
vLLM version, streaming mode, and deployment configuration. Inventory membership
only means Dynamo recognizes a field as meaningful; it does not establish support.
An internal capability advertisement permits reviewed lowering, not every native
server field or every deployment topology.

Use these statuses when reporting a difference:

- **Compatible:** the specified behavior matches the pinned native server and has
  conformance evidence for the stated configuration.
- **Upstream drift:** a previously verified behavior changed upstream and Dynamo
  alignment is pending. Name both upstream revisions and the affected Dynamo state.
- **Dynamo gap:** Dynamo does not match its target, without evidence that a new
  upstream change caused the mismatch. Historical frontend losses belong here too.
- **Intentional divergence:** Dynamo deliberately differs for a documented reason.
  An accidental mismatch or an unexplained failing test is not intentional.
- **Unverified:** evidence is missing or cannot establish the full combination.

The currently recorded GPU comparisons use aggregated serving, TCP request
transport, file discovery, Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca`, eager execution, a 256-token model limit,
one scheduled sequence at a time, and 256 MiB KV cache. They do not establish
concurrent-batch, disaggregated, multimodal, CPU/XPU, or other-transport parity.

## Tested current-server behavior

The recorded current native target is vLLM 0.30.0, source
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`. Both Rust and Python vLLM chat processors
were compared; completions still uses the Rust pipeline under either chat setting.

| Surface | Status and exact tested scope | Limitations |
|---|---|---|
| `allowed_token_ids` | Compatible for unary `null`, empty, and `[0]` cases; `[0]` produces constrained output | Not proof for arbitrary token lists or models |
| `prompt_logprobs` | Compatible for tested `null`, `0`, `1` and guarded unary `-1` cases; unary payload is at chat response root or per completion choice | Positive and full-vocabulary streaming counts are rejected; unary `-1` requires the explicit worker wire capability and a sufficient engine limit described below |
| Chat `logprobs` / `top_logprobs` / `logprob_token_ids` | Compatible for the tested omitted/null/zero/one count and null/empty/`[0]` selection matrix | Preserve omitted versus explicit null; do not extrapolate to full-vocabulary or every model's tokenization |
| Completion generated logprobs | Compatible for tested counts `0`/`1`, token selections, streaming/unary, token-ID rendering, offsets and lengths | Floating probabilities use the test's documented numeric tolerance |
| Multiple choices | Compatible for tested `n=1`/`n=2` logprob projection and token-ID rendering | `n=2` uses legal deterministic sampling; not a concurrent-batch claim |
| `bad_words_token_ids` | Unverified at the full native HTTP boundary | Reviewed transport/adapter tests exist; those do not prove public-server parity |
| Legacy `nvext.prompt_logprobs` projection | Unverified as a complete native comparison | Dynamo-specific output is distinct from native root/per-choice placement |

The executable cases and comparison rules live in
[`test_vllm_native_protocol_http.py`](../../../tests/frontend/test_vllm_native_protocol_http.py).
The retained 176 comparisons were run on base Dynamo
`da45b41777ec507974d3a6a41053a9e908e530d3` with dirty-source digest
`8cce9b097d0fde6e20736f06d21ab18346955f6372b8a24933acb82d8a83a51d`.
That result predates the later pre-stream capability check. The same 176
comparisons were subsequently rerun successfully with that check and explicit
factory identity at source-state digest
`715768cdffd59c31af615be11764247fe974165bdc3d53c3300843ea3effac59`,
using a fresh binding. Captured pre-stream errors identify Python vLLM chat as
`processor=Vllm` and completions under that same frontend as `processor=Rust`.
See the [admission guide](vllm-extension-admission.md) for scope and limits;
these historical cases do not verify every field or topology.

The same matrix also passed after moving chat-only generation controls onto the
chat request, at source-state digest
`b81825853c461a8cfa66c2e9bedc514b0886dbf40781a2547b4f8e5d598ffce3`.
That rerun is regression evidence for the table above, not native conformance
evidence for chat-template controls, which are not exercised by this matrix.

Other known limitations include reasoning/tool-buffered output paths without
native runtime coverage. Generated and prompt probability boundary rules are
covered separately below. vLLM 0.29.0 has a pinned declaration inventory and inspected admission
rules, but the 0.30.0 GPU results do not establish 0.29.0 runtime compatibility.

### Generated probability values

Native vLLM floors sampled-token and alternative-token log probabilities at
`-9999.0` in its chat response builder. Dynamo applies that same floor in its
Rust chat projection, consistent with its completion and Python chat projections.
This shared Rust renderer also affects other backends, but the native comparison
described here verifies only vLLM.

The current vLLM worker replaces negative infinity before JSON transport and
preserves finite internal values. The public generated-token projection then
applies the floor; it must not be applied to prompt probabilities. NaN and
positive infinity fail with safe diagnostics as a deliberate malformed-output
policy, not a claim of native error parity.

Boundary tests compare with the installed native chat helper using text and
token-ID display, zero and nonzero alternative counts, JSON round trips, and
engine-object nonmutation checks. A combined Rust aggregation test preserves
prompt `-10000` while flooring generated `-10000` to `-9999` on both endpoints.
Injected boundary values do not prove that a model emits every tested value,
nor do they establish malformed-output handling for old workers or other
backends. See the
[projection guide](logprob-response-projection.md)
for validation scope.

### Prompt probability values

The current vLLM worker converts prompt negative infinity to `-9999.0` before
JSON transport, matching the pinned native
[`clamp_prompt_logprobs`](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/entrypoints/generate/base/serving.py#L376-L388)
helper. Finite values below `-9999.0` are not floored. Rank, decoded text, null
positions, and the original engine objects are preserved. This normalization
feeds both native-shaped prompt output and the legacy `nvext` projection; it
does not add prompt data to public SSE chunks.

NaN and positive infinity fail as malformed engine output instead of becoming
invented tiny probabilities. This is a deliberate fail-closed policy, not a
claim that native vLLM has the same malformed-output error behavior. Released
workers that have already replaced infinity with their old `-1e30` sentinel
cannot be repaired unambiguously by a new frontend: that number may also be a
legitimate finite logprob. No lossy frontend rewrite is attempted.

Direct installed-vLLM helper comparisons cover the normalization boundary; Rust
projection tests cover the resulting finite values. This is not a new native
HTTP/model test demonstrating generation of an infinite prompt logprob. See the
[projection guide](logprob-response-projection.md)
for scope and evidence.

### Native full-vocabulary prompt logprobs

`test_native_vllm_prompt_count_contract` establishes the native vLLM 0.30.0
baseline independently of Dynamo. Both chat completions and completions were
exercised with `prompt_logprobs` set to `null`, `-2`, `-1`, `0`, and `1`, in
streaming and non-streaming mode, under two engine limits:

| Request | Native server with `--max-logprobs 20` | Native server with `--max-logprobs -1` |
|---|---|---|
| Non-streaming `prompt_logprobs=-1` | HTTP 400: configured engine limit | HTTP 200: full vocabulary per scored prompt token |
| Streaming `prompt_logprobs=-1` or `1` | HTTP 400 before SSE | HTTP 400 before SSE |
| `prompt_logprobs=-2` | HTTP 400 | HTTP 400 |
| Non-streaming `prompt_logprobs=0` or `1` | HTTP 200 with prompt payload | HTTP 200 with prompt payload |
| Streaming `prompt_logprobs=0` | HTTP 200/SSE without prompt payload | HTTP 200/SSE without prompt payload |

The positive control used the pinned Qwen3-0.6B model and a two-token prompt
(a constant server-side template for chat) to bound response size. It verified
every one of the model's 151,936 vocabulary entries at the scored position,
the initial null position, finite log probabilities, token decoding, and native
root/per-choice placement. This is not ordinary chat-template conformance or
Dynamo full-vocabulary support. The request/engine validators are in the pinned
[chat protocol](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/entrypoints/openai/chat_completion/protocol.py),
[completion protocol](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/entrypoints/openai/completion/protocol.py),
and [sampling parameters](https://github.com/vllm-project/vllm/blob/ced6857afa0ea7b2e3f0846a62e1394e90f15607/vllm/sampling_params.py).

The draft Dynamo implementation now accepts signed public `prompt_logprobs=-1`
for non-streaming requests through the Rust and Python vLLM processors when
the worker explicitly advertises the required wire support and a sufficient
engine limit. Configure the worker with `--max-logprobs -1` to allow the full
vocabulary. Streaming `-1` and positive counts are rejected before SSE; zero
retains the native streaming behavior above.

A separate Dynamo/native comparison uses the model's ordinary chat template,
not the constant template of the native-only probe. Both processors passed all
20 prompt-count cases across the two endpoints with an unrestricted engine
limit, including every vocabulary entry at every scored prompt position,
decoded text, ranks, probabilities, and root/per-choice placement. The ordinary
engine-limit matrix also passed 96 cases per processor. Together these are
232 Dynamo/native comparisons at source-state digest
`c1140e2841410d6011010627fc5664763835e27599d809a6d1bb7fb929ddb89f`
on base `da45b41777ec507974d3a6a41053a9e908e530d3`.

The public signed count is lowered into the unchanged unsigned internal wire
format. A missing capability, insufficient engine limit, or unsupported frontend
pipeline rejects full-vocabulary requests. The selected worker is checked again
at dispatch; another worker's capability cannot authorize it. Inspected 1.4/1.5
workers lack the sentinel conversion and cannot be authorized by their older
generic generation marker. Pairwise current/1.4/1.5 admission behavior is
documented below. Simultaneous mixed-worker and disaggregated full-vocabulary
HTTP behavior remains Unverified. These aggregated 0.30 results do not establish
those paths or native 0.29 behavior.

## Rolling-upgrade behavior

Two published Dynamo revisions are used as immutable release boundaries:

- Dynamo 1.4.0: `03014943323e78feb5bd672ef08b72caea0918ac`, vLLM 0.26.0.
- Dynamo 1.5.0: `b83b1d9304ebfc624709ac46db32b1b6f1ff1615`, vLLM 0.28.0.

Actual HTTP tests use each release's complete Python source and published runtime
wheel. The new worker's version does not upgrade the old frontend's request parser.

| Frontend → worker | Current evidence | Client implication |
|---|---|---|
| Current Rust or Python → 1.4 | Compatible for the tested baseline and allowed-token requests when an exact legacy declaration exists | Wrong-model or absent declarations reject the extension with field-specific HTTP 400 before SSE; ordinary requests still work |
| 1.4 Rust → current | Compatible for eight baseline/allowed-token streaming/unary cases across both endpoints | Rust uses a validated passthrough path even though these fields are absent from `CommonExt` |
| 1.4 Python chat → current | Dynamo gap: `allowed_token_ids=[0]` returns HTTP 200 with ordinary text, not constrained output | Do not rely on this directive through the released Python chat frontend; use the verified Rust path or upgrade the frontend |
| Current Rust or Python → 1.5 | Compatible for 48 tested baseline/allowed-token requests across exact, wrong-model, and absent legacy declarations | The released worker advertises the reviewed generic generation capability, so these requests do not require an exact 1.4-style declaration |
| 1.5 Rust → current | Compatible for eight baseline/allowed-token streaming/unary cases across both endpoints | This is tested HTTP behavior, not a conclusion drawn from adapter fixtures |
| 1.5 Python chat → current | Dynamo gap: `allowed_token_ids=[0]` returns HTTP 200 with unconstrained output in streaming and unary modes | Use the verified Rust path or upgrade the frontend; a same-release 1.5 control has not been run |

The Dynamo 1.4 Python frontend-to-1.4 worker control reproduced the same loss
against native vLLM 0.26.0: `allowed_token_ids=[0]` returns unconstrained chat
output with HTTP 200. This establishes a historical 1.4 frontend gap, not a
regression introduced by the new worker. Source inspection shows a relevant
loss boundary: the old chat request marks
`unsupported_fields` as `skip_serializing`, and the Python processor's outgoing
sampling map omits these extension directives. The current implementation preserves
reviewed directives across that boundary. A current worker cannot recover a field
that was removed before dispatch.

For a rolling deployment that relies on this directive, upgrade the frontend
first, using the exact legacy declaration for 1.4 workers, or use the verified
released Rust frontend path. Upgrading only the worker cannot fix the old Python
parser. This guidance concerns the tested directive, not all protocol features.

The current-versus-1.4 tests use source-state digest
`4b1e2771a914ef9a90a530b4770ed70763b21faa5105efc1b810df24f8497347` for the new-frontend
cases. The 1.4 Python reverse case uses digest
`d12927a0fa9bee6a0d332f047565f704273303e780825a4ac800f2e00e06fc43`.
These are narrow compatibility claims, not completion of the full N-2 requirement.
The 1.5 matrix uses the latter digest too. Its new-frontend direction passed both
processor cases; the reverse direction passed Rust and failed Python chat. Raw
responses for every case were captured before comparison, including successful
constrained completions through the Rust pipeline under the Python chat setting.

### Full-vocabulary counts across release boundaries

With `--max-logprobs -1` on the worker, the pairwise current/1.4 and current/1.5
HTTP tests passed both upgrade directions and both frontend processor settings.
They exercised chat and completion, streaming and unary, with `prompt_logprobs`
omitted, explicitly null, or `-1`.

| Boundary | `prompt_logprobs=-1` | Omitted or null control |
|---|---|---|
| Current frontend → 1.4/1.5 worker | HTTP 400/JSON before streaming: required full-vocabulary wire capability is absent | HTTP 200 with generated text matching the corresponding native server |
| 1.4/1.5 frontend → current worker | HTTP 400/JSON before streaming: the released unsigned request parser cannot accept `-1` | HTTP 200 with generated text matching the corresponding native server |

An exact legacy declaration does not grant full-vocabulary wire support. The
new-frontend direction also passed with wrong-model and absent declarations.
Native positive controls under the same unrestricted engine limit returned the
full vocabulary for unary `-1`, so these Dynamo rejections are not explained by
an engine limit. Enabling this feature requires compatible frontend and worker
versions; upgrading only one side does not suffice.

At source-state digest
`d1ca1533a43a4d05a757aae51bb000121b7fb0d441d7f90ffed59b1a68d56d50`,
the eight deployment cases captured 192 Dynamo responses (128 ordinary successes
and 64 expected rejections) plus 96 native controls. This is safe pairwise
feature rejection, not full-vocabulary output support on old releases, finite
prompt-count conformance, or proof for concurrent mixed-worker cohorts,
replacement, or disaggregated deployments.

## Extension placement, admission, and errors

Send reviewed native parameters at their native top-level request locations.
The internal `extra_args.backend_extensions` envelope is a frontend/worker transport
detail, not a public request namespace. Do not use `nvext.extra_fields` as a bag
of arbitrary backend request parameters; it selects Dynamo response extensions.

The chat-template controls `add_generation_prompt` and `continue_final_message`
are top-level **chat-only** parameters. The implementation declares them directly
on the chat request, and its generated OpenAPI request schema exposes them for
`/v1/chat/completions`, not `/v1/completions`. This ownership change does not add a
new namespace or change their existing chat defaults: omitted
`add_generation_prompt` means true, and `continue_final_message=true` requires
explicit `add_generation_prompt=false`.

Completions rejects either key even when its value is `null` or `false`; the
unsupported-field ignore switch cannot bypass this rule. This is Dynamo's
fail-closed endpoint policy. Native-vLLM parity for misplaced fields and the full
chat-template value matrix remains Unverified; do not infer it from schema
visibility or Rust rendering regressions.

The current token pipeline rejects unsupported sampling directives before creating
a stream. It checks the committed worker-set contract, and routing checks the
selected worker's live capability again before dispatch. An operator's exact
legacy declaration cannot override a malformed or restricted new advertisement.
See [legacy declaration configuration](vllm-protocol-maintenance.md#explicit-legacy-identification-boundary).

Public errors must identify the rejected field without including its value.
Frontend-owned Python extension errors use explicit `InvalidArgument`; generic
HTTP exceptions and backend diagnostics remain sanitized. A failure discovered
after streaming has genuinely begun cannot retroactively change HTTP status.
Pre-dispatch capability validation must not be deferred to that stage.

Chat/completion handler errors now use the native-vLLM-shaped outer envelope:
`{"error": {"message": "...", "type": "BadRequestError", "param": null, "code": 400}}`.
This applies to these shared routes regardless of backend, including configured
custom paths. Clients reading flat `message`, `type`, or `code` fields must
migrate to `error.message`, `error.type`, and `error.code`. Released Dynamo 1.4
and 1.5 frontends retain their flat errors. The envelope change itself does not
alter worker-wire errors; the optional public parameter channel below does.
Responses, Anthropic, and unrelated APIs retain their own representations.

This closes the outer-envelope mismatch for handler-returned errors, not the
entire native error contract. Global middleware/extractor failures and errors
emitted after streaming begins are not universally adapted. Dynamo still maps
HTTP 422 to 400, whereas native vLLM can return 422. Native status/type/parameter
parity across every category remains unverified. Generated OpenAPI request and
handler-error schemas follow the registered chat/completion handler at both
default and custom paths. This mapping does not fill gaps in the underlying
base-request schema or describe every global middleware error.

Frontend-local compatibility rejections can include an optional `error.details`
object. It contains `schema_version: 1`, the
`field`, a `kind` (`invalid_value`, `unsafe_combination`, or `unsupported_field`),
the `stage` (`request_validation`, `admission`, or `backend_capability`), the `profile`, and
reviewed `alternatives`. The profile identifies the endpoint, target revision,
processor, transport, and deployment mode; unknown facts remain explicit.
An invalid signed count is rejected before pipeline selection, so its stage is
`request_validation` and its profile is null. This still identifies
`error.param = "prompt_logprobs"`; it does not claim a target profile was selected.

The annotation also covers unhandled fields recognized by the generated native
inventory, invalid token-array shapes for `allowed_token_ids`,
`bad_words_token_ids`, and `logprob_token_ids`, and nonempty `logprob_token_ids`
without output logprobs. These are `request_validation` failures with a null
profile. For several unhandled native fields, `field` identifies the first
alphabetically; the safe message lists all recognized rejected fields.
Sampling capability failures at profile admission identify the requested public
field with stage `backend_capability` and the selected profile. If a malformed
capability blocks every requested sampling field, the first alphabetically is
reported. Neither case exposes the capability's raw contents.

These details do not include request values, model names, or worker diagnostics.
They supplement the existing safe message and HTTP status rather than changing
them. They are not yet present for every validation error: JSON deserialization,
other typed-field validators, unknown-only keys, aggregate extension-envelope
bounds/conflicts, post-admission routing checks, Python processor errors, and
worker-originated errors do not yet carry this annotation.
The route adapter sets `error.param` to the annotated field or an explicitly
public semantic-error parameter; otherwise it is null. It never infers a field
from arbitrary diagnostic text.
This annotation is local to the frontend and adds no N-2 worker error variant.

The vLLM error adapter now preserves a `VLLMValidationError.parameter` when it
names a declared top-level chat/completion request field and the status is 400.
It uses the explicit `HttpError(..., param=...)` API. The field identity crosses
the Python/Rust error boundary, including callable invocation, generator polling,
and preflight of errors containing a legacy JSON diagnostic. Generic duck-typed
HTTP exceptions cannot opt in merely by carrying a `param` attribute. Messages
remain sanitized (`Invalid request` for this public-parameter channel); native
message wording is not reproduced. Unknown/nested parameter paths remain
unverified and are not exposed.

The wire representation adds an optional `parameter` inside the existing public
`message` variant. Missing or malformed optional metadata does not prevent the
current reader from decoding the error. A synthetic legacy-shaped reader test
checks unknown-field tolerance; actual N-2 deployments with this new metadata
have not been tested. This does not establish Harmony support, 422 parity, or
post-header SSE error parity. See the
[public error guide](completion-error-contract.md).

Fresh-wheel native HTTP comparisons verify the nested error shape, code/type,
and `param: "prompt_logprobs"` for invalid negative counts and unsupported
streaming counts on both processors and endpoints. Eight actual current/1.4
and current/1.5 deployment cases verify the version-specific error envelopes
and successful omitted/null controls. See the
[review-branch evidence boundary](#review-branch-evidence-boundary)
for the tested source state and remaining gaps. These checks do not establish
parity for every error category.

For operational visibility, `/metrics` exposes the bounded
`dynamo_frontend_protocol_decisions_total` counter for the implemented validation,
admission, and Rust prompt-data decoding boundaries. It distinguishes rejection,
migration ignores, admission, legacy-envelope use, and malformed prompt data. See the
[metric meanings and coverage limits](vllm-protocol-maintenance.md#compatibility-decision-metrics).
These are decision counts, not unique request counts or evidence of native parity.

## Inspecting registered compatibility rules

To inspect rules attached to the currently registered pipelines, use Dynamo's
`GET /v1/models/{model}/compatibility` subresource. It reports partial admission
descriptors and sampling-extension transport policy, not a model-wide support
promise. Multiple cohorts remain separate, and engines without a registered
descriptor are explicitly unknown. See the
[catalog contract and limitations](vllm-protocol-maintenance.md#registered-pipeline-catalog).
Use `/v1/models/{model}/ready` for readiness; a catalog entry is not a routing
reservation or proof of native-server parity.

Historically, the catalog was checked against an actual aggregated vLLM 0.30.0 worker
through discovery, using fresh Dynamo bindings at source-state digest
`71e68dba1472a1b0a2367caa55a3fa1b2c24b132dc67e5abe48a92b06634428f`
on base commit `da45b41777ec507974d3a6a41053a9e908e530d3`. Both chat processors
were tested: the catalog reports Rust or vLLM for chat as configured, while
completions remains Rust. It reports the pinned admission-rule revision, the three
sampling fields' v1-plus-legacy transport policy, and an unversioned preprocessed
RPC representation. It does not claim that extension-envelope v1 versions the
whole RPC message. The same run passed the 176 native response comparisons
described above and retained the served OpenAPI document for each processor.

These checks establish the catalog's metadata wiring for the tested aggregated
deployment, not native HTTP parity for every listed sampling field. In particular,
the `bad_words_token_ids` and full `nvext` limitations above remain. Catalog
coverage for actual mixed-version or disaggregated deployments remains Unverified.

## When a framework version changes

Follow the [version-bump procedure](vllm-protocol-maintenance.md#framework-version-bump-procedure):
update immutable pins, regenerate inventory, inspect the semantic diff, classify
each change, run affected native comparisons, and update this reference with the
exact tested Dynamo state. A static schema diff is an investigation candidate,
not proof of a runtime incompatibility. The authored drift workflow is preserved
on `codex/vllm-compatibility-deferred`, not installed by these review branches.
It still needs separately authorized GitHub execution and a fully classified
checked-in example.
