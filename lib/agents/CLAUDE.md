# dynamo-agents

## Scope

`dynamo-agents` owns agent-facing identity and best-effort telemetry primitives. Keep it small and dependency-light.

Current contract:
- `workflow_type_id`: reusable agent/profile class label.
- `workflow_id`: structured workflow or stage grouping.
- `program_id`: schedulable reason/tool trajectory.
- `parent_program_id`: subagent lineage.
- Exact per-LLM-call correlation uses Dynamo's HTTP `x-request-id` header. Harnesses should set it to the same value recorded as `llm_call_id`. This is separate from `agent_context` and from Dynamo's internal `request_id`.

## Design Decisions

- This crate owns the agent context schema and normalized agent-specific trace bus/JSONL sink.
- Harness tool events enter Dynamo through the event plane and are relayed onto the same normalized trace bus as Dynamo-owned LLM request events. Harness JSONL is only a local/debug sink, not the architectural path.
- `dynamo-llm` is only an integration layer: parse `nvext.agent_context`, carry it through preprocessing, and emit request-end metrics.
- Keep crate-specific adapters out of `dynamo-agents`. For example, `RequestTracker -> AgentRequestMetrics` belongs in `dynamo-llm` because `RequestTracker` is an LLM crate type. `dynamo-agents` should expose neutral schemas and emit APIs, not depend on request trackers, routers, backends, or other owning crates.
- The trace sink is best-effort telemetry, not durable audit. It may share patterns with audit, but do not couple it to audit internals until a generic telemetry crate is justified.
- Use `tokio_util::sync::CancellationToken` for shutdown integration. Callers with a Dynamo runtime should pass `Runtime::child_token()` or `DistributedRuntime::child_token()`.
- Keep invariant trace fields strongly typed with serde-renamed enums instead of allocating strings for every record.

## Roadmap

- Near term: passive request/tool traces, profile extraction, and simple workflow/program correlation.
- Next: workflow profiles that can drive scheduling hints without changing request behavior.
- Follow-up telemetry: add routing decision details once Dynamo has a stable source of truth for candidate workers, selected worker, routing policy, and routing cost/breakdown. Keep this out of v1 unless it can be sourced without coupling `dynamo-agents` to router internals.
- Follow-up telemetry: add exact tool-output token estimates if Dynamo exposes a stable tokenizer endpoint or shared tokenizer API. The current local HTTP launch did not expose `/tokenize` or `/v1/tokenize`; avoid adding model-specific tokenizers to harness integrations just for v1 telemetry.
- Later: program lifecycle APIs for admission, pause/resume at tool boundaries, KV prewarm, and cache demotion.

## Guardrails

- Do not add backend-specific scheduling or cache-control logic here.
- Do not make this crate depend on `dynamo-llm` or runtime internals.
- Do not export helpers that accept foreign crate types. If another crate needs to publish traces, it should map its local state into `dynamo_agents::trace` types at that crate boundary.
- Prefer additive schema changes; preserve JSONL compatibility for existing fields.
