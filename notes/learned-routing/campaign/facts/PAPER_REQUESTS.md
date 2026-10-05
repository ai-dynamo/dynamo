# Paper requests (operator)

## R1 (2026-10-03 ~06:10 PDT): background section built around routing primitives

**Operator, verbatim:** "i guess the narrative would be especially in the literature review section or
something or background section give us like the routing primitives and / or the routing api abstraction
or what not, and then to like basically explain the dynamo router default and the other pedagagoical
routing cost functions and how the fit under it, in that section you can also highlight the cavaets
(major) that you want that we uncovered during this camapgin"

**Plan:** a "Background: routing primitives and the Dynamo worker-selection API" section, placed before
the model section.

1. **The abstraction:**
   - the candidate table, with eligibility and pins;
   - filters, scorers and pickers, as `WorkerFilter`, `WorkerScorer` and `WorkerPicker`;
   - the declared inputs: WorkerInputs `CACHE`, `LOAD`, `PREFILL_TIME`, `OCCUPANCY` and `PREFERRED_TAINT`;
   - the `WorkerSelectionContext` request facts: prompt tokens, blocks, prefix hashes, session context,
     policy class, worker capacity;
   - the YAML policy catalog;
   - what the host owns and what a policy owns.
2. **The Dynamo default cost function**, written in that vocabulary: overlap credit, prefill load scale,
   decode blocks, temperature softmax, and seeded tie-breaking.
3. **Each pedagogical cost function mapped onto the abstraction,** with a table of inputs, scorer or
   picker, set-dependence, and parameters: two-tier, lmetric, ramjet, dualmap, chwbl, llm-d
   precise-prefix, llm-d optimized-baseline (throughput vs modeled), sticky-session hard and bounded,
   thunderagent. Then the learned-choice model as a generalization: a utility over declared inputs, with
   set context.
4. **Major caveats uncovered in this campaign**, each with evidence and its status (fixed, follow-up, or
   inherent). Compile them from STATE.md, DEVIATIONS.md, UPSTREAM_FOLLOWUPS.md and the audits.
   Candidates:
   - replay's perfectly fresh router state;
   - `expected_output_tokens` equal to the true OSL;
   - `prefix_hashes` random when assume_kv_reuse is false;
   - synthesized closed-loop session IDs reaching the router (aisimulate#378);
   - toolagent being relabeled Mooncake;
   - unseeded default tie-breaks;
   - open-loop agentic load cliffs;
   - E0/E2E knife-edge and drain-window scoring pitfalls;
   - 3 s-quantized arrival bursts;
   - no AgentX play fitting 32K;
   - the per-request temperature override intentionally ignored by the default plugin;
   - the policy-file snapshot race (fixed in #15450).

**Timing:** run this as a restructuring pass after backbone workflow `<workflow-run>` completes.

## R2 (2026-10-03 ~10:30 PDT): backload the paper

The operator stopped the paper-backbone workflow (`<workflow-run>`) because of too many permission
prompts. Do the paper at the very END of the campaign, in one batched pass that also applies R1, the
results, and the `/audit` publication pass. Partial drafts under `notes/learned-routing/paper/` (if any)
are a starting point. Pre-install nothing new; the TeX packages are already installed.

- **R3 (operator, 2026-10-05 ~08:20 PDT):** author list is "Rudy Pei". Replace the author `\pending` marker; add no affiliation or email unless the operator asks.
