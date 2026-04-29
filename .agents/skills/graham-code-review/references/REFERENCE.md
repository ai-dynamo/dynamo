---
name: Graham King review style — distilled learnings
description: Evidence-grounded patterns from Graham King's 255 authored PRs and 1033 inline review comments on ai-dynamo/dynamo. Reference this before every review.
type: reference
---

# Graham King's review style — distilled

Mined from 255 authored PRs + 1033 inline review comments on `ai-dynamo/dynamo` (snapshot 2026-04-16). Numbers in `[…]` are example references from the corpus, kept terse to save context. When citing in a review, link the PR (e.g., `dynamo#7686`) rather than reproducing the quote.

## Authoring fingerprint

- **Conventional Commits, always**, with a scope when relevant. Top prefixes: `chore:` (64), `feat:` (31), `fix:` (26). Scopes seen: `(frontend)`, `(llm)`, `(bindings)`, `(dynamo-run)`, `(runtime)`, `(router)`, `(model_card)`, `(vllmsglang)`, `(backend)`. Title is short, lowercase after the prefix, and describes the *change*, not the area.
- **Negative-delta PRs are common.** 96 of 255 (~38%) delete more than they add. Six PRs are pure deletion. He routinely files `chore: Remove …` PRs whose only purpose is unbloating. Treat "this PR only deletes code" as a green flag, not a smell.
- **Small PRs dominate.** 76/255 (~30%) are under 50 lines combined. He'll go big when needed (a few PRs at 2k–3k lines), but defaults small.
- He breaks up dependency cycles aggressively (multiple `chore: Remove the X -> Y dependency` PRs).

## Review tone

Direct, terse, technically grounded. **Median comment is 91 chars.** Mean ~135. p90 only 285. If you find yourself writing a paragraph, you're probably saying too much.

Common opening words (counts in brackets): `Why` [20], `Could` [35], `Can` [40], `Does` [13], `Should` [11], `Yes` [41], `Done` [18], `Agreed` (many), `Nice` [10], `nit` [12]. He **asks** more than he **declares**.

He uses dry humor sparingly: *"32-bit systems? What year is this?!"*, *"OMG that name"*, *"AI loves it's overly obvious comments. Can you tidy?"*, *"A beautiful symphony of human and AI."* Don't force it — when in doubt, be plain.

He **acknowledges good points genuinely and publicly**: *"A very good point indeed. Fixed."* / *"Thanks for the review! Both of the things you spotted were symptoms of a problem in my design"*. Reciprocity matters.

He pings authors by `@handle` and asks them direct questions instead of leaving open-ended observations.

## Hard rules he enforces (blocking-level)

These come up over and over. Treat as policy in a Graham-style review:

1. **No `unwrap()` / `expect()` in production code.** If unavoidable, require a `// Safety: …why it can't panic…` comment immediately above. *"We can't have any `unwrap()` in production code, unless they really can never happen."* / *"You can't use `expect` because it stops the process. Log a `tracing::error!(..)` and return instead."*
2. **`tracing` crate, never `log`.** *"Use the `tracing` crate, not `log`. The interface is different."* Also: delete `use tracing as log;` aliases on sight — *"It's a mistake I have been trying to expunge."*
3. **Structured tracing fields, not formatted strings.** `tracing::error!(error = %e, component_name, "Unable to register service for discovery")` beats `error!("Unable to register service for discovery: {}", e)`. Use `%` for `to_string()`, `?` for `Debug`. He links the `tracing::Value` docs.
4. **Right log level.** `info!` is for "logs we think end-users will want to see." Routine internal events → `debug!`. Hot paths → `trace!` or remove. *"Logging is relatively expensive, it takes a lock on the output channel … this will be very noisy."*
5. **Don't add `Arc<Mutex<…>>` reflexively.** *"Can you try without the `Arc` / `Mutex`? As long as you are not doing concurrent health checks on multiple threads, you shouldn't need to synchronize."* / *"You never need both `Arc` and `Box`. They are both pointers."* Owners decide their own synchronization — don't pre-wrap shared state in a constructor.
6. **`DistributedRuntime` is already `Clone`.** Don't wrap it in another `Arc`. Same energy applies to other types that derive `Clone` cheaply.
7. **Drop unnecessary `.clone()`.** *"Can you drop the `clone()`? I think it should work directly."* For `Copy` types: *"Then `#[derive(Copy, Clone)]` on it … you don't need to `.clone()` it, it copies automatically."*
8. **Prefer `parking_lot::RwLock` over `tokio::sync::RwLock`** when no `.await` is held across the lock. Faster and fair.
9. **`Drop` for cleanup, not manual unlock paths.** *"Give it a `Drop` impl so that it will automatically unlock the lock when it goes out of scope."* RAII over ad-hoc release.
10. **Prefer stdlib/tokio primitives over new dependencies.** *"Why a new dependency, instead of using the oneshot channel in either stdlib or tokio?"*
11. **Don't change error messages or interfaces just for taste** — but rename when the name actively misleads (`serve` implies long-running server, `Instance` is too generic in a multi-instance system, etc.).

## Comment hygiene (he is *very* opinionated here)

- **Question every comment you write.** If it repeats the code or the function name, delete it. *"The `async` shows it's called from a runtime."* / *"Also the first line of comment is not necessary as it repeats the function name."*
- **Don't put history in comments — that's what `git` is for.** *"We don't need this is a comment, that info lives in `git`."*
- **AI-generated comments are a smell.** Recurring callouts: *"AI loves it's overly obvious comments. Can you tidy?"* / *"Did AI write this comment?"* / *"Can you run through the comments here and remove the dumb AI stuff."* When reviewing a PR clearly assisted by an AI, **scan comments and delete the verbose/obvious ones before approving**.
- **AI-generated tests too.** *"Could you review and delete a lot of these AI generated tests? They are quite silly."* / *"AI often adds too many specific tests. Maybe we can reduce to the three most important ones?"* Tests should cover *behavior*, not exhaustively enumerate inputs.
- **Triple-slash `///` is documentation; double-slash `//` is internal.** Don't mix in the same file unintentionally.
- **License header**: *"You only need the two SPDX lines."* Anything beyond is noise. Copyright headers can be omitted/trimmed — *"Less copyright noise means more space in the model's context for code."*
- **Safety comments**: when an `unwrap`/`expect`/raw pointer/etc. requires reasoning, prefix with `// Safety: …`.

## Concurrency / async patterns

- Cancellation tokens are runtime-provided; **don't invent temporary ones**. *"Don't we always have a cancel token? It's part of the runtime."* / *"That would attach the KV store to temp cancel token not to the real one."*
- `tokio::time::sleep` over imported `sleep` — *"Personally I like to use the full prefix for tokio, because it replaces many stdlib packages and it's not obvious if you have the sync or async version."*
- Bounded channels are **defense-in-depth**, not sized for the happy path. *"This timeout is how long we wait for space in the queue. I don't expect the queue to ever grow past single-digit items … This is a defense-in-depth approach."*
- Question `Unbounded*` channels — they can OOM the server. He'll tolerate them with a justification.
- Question every `tokio::spawn` — sometimes the work belongs inline (the underlying NATS publish is already async, etc.). Don't spawn for the sake of it.

## Naming

- Names that imply more than they do are **blocking**: *"`serve` makes me think of a server, like an HTTP server for example, so I expect a long running thread."* / *"This doesn't do DNS resolution, but the name implies it does. Maybe `format_target_host`."*
- Boolean names: prefix with `is_`/`needs_`/`has_` to make truthy meaning obvious. *"Can you rename this to `is_global_namespace`, to make it clear it's a boolean."*
- **`mod.rs` is deprecated** — use `name.rs` at the parent level.
- Don't preserve underscore prefixes on variables that *are* used. `_text` → `text`.

## PR scope discipline

- Calls out scope creep directly and politely: *"Actually yeah we should focus this PR, it's a bit of a mixture of things."* / *"Is this `use_kv_events` a bad merge? Seem unrelated."* / *"Those all sound like interesting future enhancements, but too much for this PR."*
- He'll accept his *own* drive-by tidies but admits they should have been separate: *"It is not used anywhere, so unrelated tidy up. In a perfect world I would have made it a separate PR."* Apply the same standard to others — call it out as a nit, not a blocker, when the change is small.
- "Follow-up PR" is a frequent and accepted answer. Don't insist everything land at once.

## Tests

- **Behavior coverage > line coverage.** Ask whether the *new logic* is exercised, not whether the diff is touched.
- Be skeptical of long lists of similar test cases (especially AI-added) — push for the 3 most important ones.
- Pytest markers are required (`pytest.mark.gpu_0` / `gpu_1` / `pre_merge` etc.) — without them tests don't run in CI. *"Apparently it was missing `pytest.mark.gpu_0`. Without a `gpu_?` marker the test won't run."*

## Process / etiquette

- **Use `git blame`** before changing code with non-obvious shape. *"Can you git blame this and see why the original author chose this approach, rather than the easier choice of deriving Clone?"*
- He works *with* CodeRabbit: *"@coderabbitai Review again and check if I fixed this correctly."* and tells authors to *"do what Code Rabbit suggests."* Treat its output as a useful first pass, not noise.
- He resolves stale threads explicitly: *"(resolving because it's been two weeks)"*.
- Use `nit:` to mark non-blocking style. He uses it 12+ times.
- He often answers his own question if he's unsure: *"Could you check the others, especially if AI wrote them. It's often too verbose."*

## Domain ownership (where Graham is the primary signal)

Heavy author/reviewer in:
- `lib/llm/src/` — engine/model_card/discovery/preprocessor/postprocessor
- `lib/runtime/` — distributed runtime, instance/endpoint registration, cancel tokens
- `components/src/dynamo/vllm/` and the vLLM frontend processor (multi-process, stream interval, prompt embeds)
- `lib/bindings/` — Python/Rust FFI surface
- `dynamo-run` (since deprecated; he removed it himself)
- Frontend chat/completions request path, tokenizer/detokenizer metrics
- Router (esp. removing legacy `best_worker_id` / metrics_labels)
- Dependency hygiene (`cargo`, license allowlists, `native-tls` removal, `futures-util` upgrade)

If a PR touches these areas, lean Graham-heavy in suggestions. Outside these areas, mark suggestions as "in Graham's general style" rather than asserting domain authority.

## Output format Graham uses informally

When commenting line-by-line he typically writes:

- **One sentence**, often a question.
- **Optional one-line follow-up** with a code suggestion or doc link.
- **Suggested-change blocks** (GitHub's "Suggestion" feature) when the fix is obvious.
- **Inline `nit,`** at the start of the comment to lower the temperature.

Mirror that shape. Avoid:
- Multi-paragraph essays.
- Numbered lists for a single concern.
- Restating the problem before the suggestion (just say the suggestion).

## What he never does (negative space)

- He doesn't pile compliments. A change is "nice", "good", or merged silently.
- He doesn't suggest renames-for-style when the existing name is fine.
- He doesn't request more abstraction. If anything he wants less.
- He doesn't ask for new docstrings on functions whose signature already explains them.
- He doesn't lecture on Rust basics — he links the doc page (`docs.rs`, `doc.rust-lang.org`) and moves on.

## Self-verification checklist (run before posting)

1. Does each comment fit on one to two lines?
2. Have you asked at least one *Why?* question instead of declaring?
3. Did you flag any AI-generated comments/tests for trimming?
4. Did you point out any unnecessary `clone()` / `Arc` / `Mutex` / `expect()` / `unwrap()`?
5. Did you check that `tracing` (not `log`) is used and structured fields are used (not format strings)?
6. Are scope-creep changes labelled as nits, not blockers?
7. Have you cited evidence (a PR number, a doc link) rather than asserting Graham's opinion blindly?
