# Cross-impl parity test suite

Shared test infrastructure for diffing parser / preprocess / postprocess
behavior across Dynamo, vLLM, and SGLang. Today only the parser stage
is populated (`parser/`); other stages slot in as siblings as they land.

## Layout

```
tests/parity/
├── README.md                       (this file)
├── conftest.py                     ← session-scoped fixtures (server boots, etc.)
├── common.py                       ← ParseResult, canonical-JSON diff, decode_arguments
└── parser/
    ├── fixtures/                   ← static YAML, generated from Dynamo as oracle
    │   └── <family>/PARSER.batch.yaml         (and per-top-level-case files like PARSER.batch.8.yaml; see Fixture file schema)
    ├── capture_parser_outputs.py     ← drift-check (default) or merge any impl's output into `expected.{dynamo,vllm,sglang}`
    ├── generate_parity_chart.py    ← print the parity-status chart (run on demand; not checked in)
    │
    ├── dynamo.py                   ← M2 in-process wrapper (PyO3 binding)
    ├── vllm.py                     ← M2 in-process wrapper (ToolParserManager)
    ├── sglang.py                   ← M2 in-process wrapper (per-module detectors)
    ├── test_parity_parser.py       ← M2 harness (parser-class parity)
    │
    ├── server.py                   ← M3 subprocess boot helper
    ├── client.py                   ← M3 HTTP client (vllm + sglang)
    └── test_parity_e2e.py          ← M3 harness (server-stack parity over HTTP)
```

## Three methods (M1, M2, M3) — what each one really means

Three increasingly-realistic ways to drive the same parser logic.
They differ in **what's substituted vs what's real**, which decides
which bug class each method can catch:

```
                                                  ┌─ engine ──────────┐
                                                  │                   │
   client ─request→ chat-template ─→ tokenize ─→ engine ─→ detokenize ─→ text ─→ PARSER ─→ tool_calls JSON ─→ client
              ↑                                                              ↑
              └── M1 / M3 exercise this (real)                               └── M2 starts here (skips everything left)
                  M2 skips it (substituted by direct call)
```

### Method 1 — fallback-path test *(upcoming, next step)*

**What's run:** `python -m dynamo.frontend --dyn-chat-processor <vllm|sglang>`. Dynamo's frontend chat processor delegates tool parsing to upstream's Python parser instead of Dynamo's Rust parser. Reference path is the default `python -m dynamo.frontend` (Dynamo's Rust parser).

**What it surfaces:** gaps in Dynamo's Rust parser that the fallback masks; bugs in Dynamo's frontend code that wraps upstream parsers.

**Status:** not yet implemented. Lives in its own harness file when added (M1 tests *Dynamo's wrapping* of upstream parsers, not parity *between* impls).

### Method 2 — parser-class test (this PR's primary harness)

**What's run:** in-process Python imports, all three impls in one process — no HTTP, no model, no tokenizer, no chat-template materialization:

```python
# Dynamo Rust side (via PyO3 binding)
from dynamo._core import parse_tool_call
result = await parse_tool_call("kimi_k2", text, tools_json)

# vLLM side (native Python class)
from vllm.tool_parsers import ToolParserManager
parser = ToolParserManager.get_tool_parser("kimi_k2")(tokenizer=stub)
info = parser.extract_tool_calls(text, request)

# SGLang side (native Python class)
from sglang.srt.function_call.kimik2_detector import KimiK2Detector
result = KimiK2Detector().detect_and_parse(text, tools)
```

**What it surfaces:** parser-logic divergences between Dynamo's Rust parser class and upstream's Python parser classes. The bug class isolated from everything else in the request lifecycle.

**File:** `tests/parity/parser/test_parity_parser.py`.

### Method 3 — end-to-end HTTP test *(upcoming, next step — sibling PR #9189)*

**What's run:** real upstream serving binaries; constrained decoding forces them to emit the fixture text:

```bash
vllm serve <model> --load-format dummy \
  --enable-auto-tool-choice --tool-call-parser kimi_k2 --port 8001 &
python -m sglang.launch_server --model-path <model> --load-format dummy \
  --tool-call-parser kimi_k2 --port 8002 &
```

Both servers receive identical chat-completion requests with `structured_outputs.regex` (vLLM) / `regex` (SGLang) forcing the assistant turn to be the fixture's `model_text` byte-for-byte; the harness captures `tool_calls` JSON from each response.

**What it surfaces:** server-stack divergences between vLLM's and SGLang's HTTP pipelines — request preprocessing, tokenizer round-trip, streaming chunk boundaries, response shaping — that class-level testing (M2) can't see.

**File:** `tests/parity/parser/test_parity_e2e.py` (lands in #9189).

### Comparison

| | M1 (upcoming) | M2 (this PR) | M3 (upcoming, #9189) |
|---|---|---|---|
| **What's tested** | Dynamo frontend wrapping upstream parsers | parser **class**, in isolation | parser inside its **server** (full HTTP stack) |
| **Invocation** | `python -m dynamo.frontend` subprocess | in-process Python imports | HTTP over `/v1/chat/completions` |
| **Real engine?** | yes | no | yes (with `--load-format dummy`) |
| **Real tokenizer?** | yes | no | yes |
| **Real chat template?** | yes | no | yes |
| **Real HTTP?** | yes | no | yes |
| **Cost** | (TBD) | ~5 s for 1350 tests (693 pass / 545 skip / 112 xfail with sglang skipped locally) | ~60 s for 30 tests (server boot dominates) |
| **GPU** | yes | none | yes (`--load-format dummy` still allocates ~2.5 GiB) |
| **CI markers** | (TBD) | `unit, pre_merge, gpu_0` | `e2e, pre_merge, gpu_1` |

All three methods (when implemented) share the same fixtures,
`ParseResult` shape, and the per-case `expected.{dynamo,vllm,sglang}` schema in YAML.
They're stacked diagnostics:

- M2 says: *"the parser class disagrees"*
- M3 says: *"the server stack also disagrees"* (or, more usefully,
  *"disagrees only at the server stack — parser class agrees"*,
  which localizes the bug to chat-template / tokenizer /
  response shaping)
- M1 says: *"Dynamo's wrapper layer disagrees with the upstream
  parser it's wrapping"*

## Current parity status (M2, batch mode)

Each row is one **parser** (one family of input wire format). Each row
also names the **model(s)** that parser is wired up for — many parsers
serve more than one model (e.g. `mistral` covers the Mistral series,
`hermes` covers Hermes-3 + various Llama variants). Columns are the
sub-case IDs from [`PARSER_CASES.md`](../../lib/parsers/PARSER_CASES.md):
top-level buckets (`1`, `3`, `9`, `10`) have no sub-cases and use a
single column; the other buckets (`2`, `4`, `5`, `6`, `7`, `8`) split
into `.a`–`.d` sub-cases per the per-bucket axes documented in the
PR description.

Cell values show how each engine's recorded `expected.<impl>` block relates to Dynamo (the oracle). **Convention:** a divergent peer block carries a `reason:` field iff the divergence is *intentional* (documented contract difference, vendor behavior, etc.). No `reason:` = **research-needed** — we observed the divergence but haven't classified it yet.

- `=` — both engines match Dynamo (peer block is an anchor ref `*d_<case>` to dynamo's).
- `V` — vLLM diverges, **intentional** (engine block has `reason:` field). Rendered the same color as = in the HTML chart since the divergence is accounted for.
- `V?` — vLLM diverges, **research-needed** (engine block has no `reason:` yet; we observed the divergence but haven't classified it).
- `V!` — vLLM is expected to crash; `expected.vllm.error: <substring>` records the matching error.
- `S`, `S?`, `S!` — same as V/V?/V! for SGLang.
- `VS`, `VS?`, `V?S`, `V!S`, `VS!`, `V?S?`, `V!S!`, … — combinations (both engines diverge with any mix of intentional/research-needed/error).
- `n/a` — **not applicable**: engine marked `unavailable` (no parser registered for that family), OR the sub-case shape doesn't apply to this grammar (e.g. attribute-encoded DSML families have no `4.b` because there's no embedded JSON to malform).
- `—` — **missing fixture coverage**: no fixture entry exists for that family/case yet. If the case is intentionally not applicable, add an explicit chart-only n/a stub with `description:` and `reason:` so the chart can explain it.

19 parsers total — split into the **Top-N models** we prioritize and
**Others** wired into the harness for completeness. Both sections sorted
alphabetically within themselves.

The chart isn't checked in — it would drift behind the YAML every time a
case is added or a peer block flips. Generate it on demand and save it
somewhere you can browse:

```bash
# Markdown — paste into a PR description or browse in any editor.
python3 tests/parity/parser/generate_parity_chart.py > PARITY.md

# HTML — clickable cells link to the source fixture YAML; hover over any
# non-= cell to see the case description and the divergence reason.
python3 tests/parity/parser/generate_parity_chart.py --html > PARITY.html
```

Run from the repo root so the HTML's relative `<a href=...>` links to
fixture YAMLs resolve when you open `PARITY.html` in a browser. Both
`PARITY.md` and `PARITY.html` are for local viewing only — don't check
them in; the generator is the contract. Rows are the 19 parsers (Top-N
first, then Others, alphabetical within each). Columns are the sub-case
IDs from [`PARSER_CASES.md`](../../lib/parsers/PARSER_CASES.md). The
generator reads every `fixtures/<family>/PARSER.*.yaml` and emits one
cell per `(family, sub-case)` using the legend above.

**Example output** (illustrative — cell values are made up, **not** a
snapshot of current fixtures; run the script for the real chart):

```text
| model       | parser     | 1 | 2.a | 2.b | 2.c | 2.d | 3 | ... | 9 | 10 |
|---|---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **Top-N models** |   |   |   |   |   |   |   |   |   |   |
| Kimi K2.6   | kimi_k2    | = | =   | =   | VS  | =   | = | ... | = | =  |
| GLM 5.1     | glm47      | = | =   | n/a | V   | =   | = | ... | = | =  |
| gpt-oss     | harmony †  | S | S   | n/a | S?  | S?  | = | ... | = | S  |
```

Read a row left-to-right: `=` = both engines match Dynamo, `V` / `S` =
that engine diverges with a documented reason (rendered same color as =
in the HTML chart), `V?` / `S?` = divergence not yet classified
(research-needed), `n/a` = case doesn't apply or peer is unavailable.

### Footnotes

† vLLM has no engine peer (or returns `UNAVAILABLE` at runtime, e.g.
  `harmony#vllm` requires token IDs not text). Cells show SGLang status
  only when SGLang is wired; otherwise the row is fully `n/a`.
§ SGLang `v0.5.10.post1` (the version pinned in `container/context.yaml`)
  has no peer detector for this family. Cells show vLLM status only when
  vLLM is wired; otherwise the row is fully `n/a`. **This may change with
  future SGLang releases** — re-audit on each version bump.

`nemotron_deci` and `nemotron_nano` carry both daggers (`†§`) — neither
upstream has a peer parser, so the rows are fully `n/a`. They live under
**Others** for completeness; Dynamo-only self-parity could be added in a
follow-up but yields no cross-impl signal.

### Coverage gaps — missing upstream peer parsers

Snapshot pinned to **SGLang `v0.5.10.post1`** (per `container/context.yaml`)
and the vLLM version installed in the parity dev image. **This may change
with future releases** — re-audit on every SGLang / vLLM bump.

**SGLang `v0.5.10.post1` has no detector for 4 families that DO have a vLLM peer (`§` in matrix):**
`deepseek_v4`, `gemma4`, `jamba`, `phi4`. The harness skips the SGLang side
of these rows at runtime (`UNAVAILABLE: SGLang has no detector for
family='<name>'`); cells carry vLLM-only signal.

**`llama3_json` is also `§` in the matrix but for a different reason** —
SGLang does ship a Llama 3 detector ([sglang tool-parser docs](https://sgl-project.github.io/advanced_features/tool_parser.html));
we just haven't wired it in `_FAMILY_TO_SGLANG_DETECTOR` yet. TODO follow-up
to add it and regenerate the `llama3_json` SGLang fixtures.

**Neither upstream has a peer parser for 2 families (`†§` in matrix):**
`nemotron_deci`, `nemotron_nano`. Rows are fully `n/a`.

Wrapper enumeration: `tests/parity/parser/sglang.py::_FAMILY_TO_SGLANG_DETECTOR`
(missing keys = missing detectors). Adding an SGLang detector for any
`§`-marked family above (either upstream-shipped in a newer SGLang release,
or by us standing up a Dynamo-side stub) would let the harness surface
`S`/`VS` divergences on those rows; today they carry only `V`/`✓`
(vLLM-side).

**Hot columns:**
- `PARSER.batch.4` (malformed JSON) and `PARSER.batch.5` (missing
  end-token) — recovery is impl-defined per `PARSER_CASES.md`;
  divergences are documented rather than asserted-against.
- `PARSER.batch.8` (interleaved normal text) — vLLM and SGLang both
  drop the trailing text after the wrapper across XML-style families.
- `harmony/sglang` — 8 of 10 cases divergent because SGLang's
  `GptOssDetector` requires the strict
  `<|start|>assistant<|channel|>commentary` envelope where Dynamo
  accepts bare commentary.

What's **not** covered yet: `PARSER.fmt.*`, `PARSER.xml.*`,
`PARSER.harmony.*`, `PARSER.stream.*` — those case classes still
live in Rust unit tests only and would be added to the YAML corpus
in follow-up PRs.

## Run the parity harness

One-liner from inside a Dynamo devcontainer:

```bash
PYTHONPATH=lib/bindings/python/src python3 -m pytest tests/parity/parser/test_parity_parser.py
```

`PYTHONPATH=lib/bindings/python/src` is required — that's where the
PyO3 binding (`dynamo._core`) lives. Without it every Dynamo case
skips. vLLM / SGLang packages must also be installed for their cases
to run (otherwise they skip cleanly).

**Common filters** — pick what to run:

```bash
# One family across every bucket
... -k "kimi_k2"

# One sub-case across every family + every impl
... -k "PARSER.batch.8.b"

# Exactly one (family, sub-case, impl)
... "tests/parity/parser/test_parity_parser.py::test_parity[kimi_k2/PARSER.batch.8.b#vllm]"
```

**From the host** (outside the container), wrap the command in
`docker exec <container> bash -c 'cd /workspace && …'`. Find the
container with `docker ps --format '{{.Names}}' | grep vsc-dynamo2 | head -1`.

**Baseline:** on `main` post-Variant-A, expect roughly **1,150 passed /
275 skipped / ~5 s** wall clock across 450 cases × 3 impls (Dynamo +
vLLM + SGLang). There is no `xfail` count — divergent peers are
recorded with concrete `expected.<impl>` blocks and assert positively
against them. Failure modes are **assertion mismatches** when an engine
drifts away from its recorded block; perfect parity is when every cell
is `=` (all three engines produce byte-identical output on every
fixture).

## Resolving divergences (8 steps)

Each non-`=` cell in the generated chart is a divergent `expected.<impl>` block
in the family's YAML fixture (concrete `calls` + `normal_text`, or
`{error: <substring>}`, or `{unavailable: <reason>}`). Cells that
match Dynamo are stored as anchor refs `*d_<case>` to the dynamo
block. Goal: drive the non-`=` count toward zero by fixing whichever
side is wrong. Worked example: `kimi_k2 / PARSER.batch.8.b → V`
(vLLM drops trailing text after the wrapper).

### 1. Pick a cell

Run `generate_parity_chart.py`; pick any `V` / `S` / `VS` cell from the
output.

### 2. Look up the side-by-side diff in the YAML

Open `tests/parity/parser/fixtures/kimi_k2/PARSER.batch.8.yaml` and
find the `PARSER.batch.8.b:` block. The `expected:` block carries all
three engines' actual recorded output, so the diff is right there —
**no harness run needed to read it.**

```yaml
PARSER.batch.8.b:
  description: Narration after tool call only
  ref: originated from https://github.com/vllm-project/vllm/.../test_kimi_k2_tool_parser.py#L435
  model_text: |-
    <|tool_calls_section_begin|>...
  expected:
    dynamo: &d_8_b
      calls: [...]
      normal_text: "Let me know if you need more."   # Dynamo
    vllm:
      calls: [...]
      normal_text: ''                                 # vLLM (drops trailing)
      reason: "drops trailing normal_text after tool-call wrapper end"
    sglang: *d_8_b                                    # SGLang matches Dynamo
```

Read the three blocks side-by-side: dynamo block is the oracle, peer
blocks show that engine's concrete output (or `*d_<case>` if it
matched). The `reason:` line (when present) classifies the divergence
as intentional; absence = research-needed.

If `ref: originated from <url>` is present, the URL points at the
upstream vLLM/SGLang test the shape was derived from (useful context).

### 3. Reproduce locally

```bash
PYTHONPATH=lib/bindings/python/src python3 -m pytest \
  "tests/parity/parser/test_parity_parser.py::test_parity[kimi_k2/PARSER.batch.8.b#vllm]" \
  -v --tb=short
```

Impl tag is `#vllm` or `#sglang`.

### 4. Pick an alignment target

Parity isn't about declaring a winner. It's about picking what
Dynamo should align *to* for this (family, case) so the fix has a
clear oracle. Apply the priorities below in order: the first matching
rule wins, and the chosen target is recorded in
`reason:` (or `spec_ref:`) so future readers see the trail.

**Alignment-target priority (highest wins):**

0. **Upstream spec, if one exists.** Chat template, tokenizer
   config, model-card section, vendor docs. Spec is the unambiguous
   oracle; if both vendors disagree with it, both are upstream bugs
   to file. Record `spec_ref:` in the fixture (step 8). Rare —
   most families have no written spec; skip to 1.
1. **The non-leaking vendor.** Tool-calling tags like `<｜tool_call｜>`,
   `<tool_call>`, `[TOOL_CALLS]`, etc. must never reach the
   `message.content` shown to an end user. If one vendor leaks and
   the other doesn't, align to the non-leaking one.
2. **The impacted customer's vendor.** If a prod deployment is broken
   because of a divergence, align to that customer's vendor *now* to
   unblock. Record incident + vendor chosen in
   `reason:` (e.g.
   `"aligned to SGLang per Acme prod report 2026-05-12; vLLM variant leaks <|tool_call|> tag into content"`).
   Don't re-litigate in the same PR — ship the unblock and follow
   up separately.
3. **Vendor consensus.** When vLLM and SGLang agree, that shared
   shape is the target. When they disagree and none of the above
   applies, pick the one that better satisfies priority 1 (no leak)
   and record why in `reason:`.

**Once the target is picked, the action depends on where Dynamo stands:**

- **Spec is the target, a vendor diverges** → file an upstream bug
  at `vllm-project/vllm` or `sgl-project/sglang`, link from `reason:`.
- **A vendor is the target, Dynamo diverges** → fix Dynamo (step 5).
- **A vendor is the target, Dynamo already matches, the *other*
  vendor diverges intentionally** → leave the divergence entry,
  file an upstream bug at the disagreeing vendor.
- **Two customers need different behavior on the same case** → file a
  follow-up for a parser-side runtime parameter or environment variable,
  then link that follow-up from `reason:`.
- **Genuinely ambiguous** (no spec, both vendors leak, no incident)
  → discuss before touching code.

> **Sad reality:** sometimes the priority chain is unreconcilable
> across customers — one prod deployment needs vLLM-style output,
> another needs SGLang-style. We may eventually need a parser-side
> param or env (e.g. `DYN_PARSER_<family>_MODE=vllm|sglang`) to flip
> behavior per deployment. Track these cases in `reason:` so the
> switch can be added later from a known set.

### 5. Fix the parser

Edit `lib/parsers/src/tool_calling/<family>/...rs`. Rebuild the PyO3
binding so `capture_parser_outputs --impl dynamo --merge --overwrite-if-exists`
runs against your change:

```bash
cd lib/bindings/python && maturin develop --uv && cd -
```

See [the build guide](../../docs/getting-started/building-from-source.md)
for prerequisites.

### 6. Refresh `expected.dynamo` + re-capture engine outputs

Step 1 — refresh Dynamo's oracle output (your fix changed it):

```bash
PYTHONPATH=lib/bindings/python/src python3 \
  -m tests.parity.parser.capture_parser_outputs \
  --impl dynamo --merge --overwrite-if-exists
```

Step 2 — re-capture vLLM and SGLang outputs against the now-current
fixtures so anchor refs (`*d_<case>`) flip on/off correctly. Run in
each engine's container:

```bash
# vllm container — drift check (default, read-only)
python3 -m tests.parity.parser.capture_parser_outputs --impl vllm
# sglang container — drift check (default, read-only)
python3 -m tests.parity.parser.capture_parser_outputs --impl sglang
```

The default (no flag) drift-check reports which cases have engine
output differing from the recorded fixture. Once you've sanity-checked
the drift, re-run with `--merge --overwrite-if-exists` to write the new
engine outputs into the YAMLs (matching engines become anchor refs;
divergent ones become concrete blocks).

### 7. Re-run pytest

```bash
PYTHONPATH=lib/bindings/python/src python3 -m pytest \
  tests/parity/parser/test_parity_parser.py -q --tb=no
```

The test asserts each engine's actual output equals its recorded
`expected.<impl>` block. If an engine drifts (e.g. you fixed Dynamo
but didn't re-capture peer outputs), the test fails with a diff
pointing at the YAML edit needed:

```text
FAILED tests/parity/parser/test_parity_parser.py::test_parity[kimi_k2/PARSER.batch.8.b#vllm]
       expected: {'calls': [...], 'normal_text': ''}
       got:      {'calls': [...], 'normal_text': 'Let me know if you need more.'}
```

### 8. Collapse the peer block to an anchor ref + add spec_ref + commit

When an engine now matches Dynamo, the concrete peer block becomes
an anchor reference (`*d_<case>`) to dynamo's block. The Step 6
`capture_parser_outputs.py` re-run with merge does this
automatically; if you're editing by hand, replace the concrete block
verbatim:

```yaml
expected:
  dynamo: &d_8_b
    calls: [...]
    normal_text: '...'
  vllm: *d_8_b        # ← was a concrete `vllm:` block; now ref to dynamo
  sglang: *d_8_b      # (same, if SGLang also now matches)
```

Add a `spec_ref:` field directly to the case in its fixture YAML so
the paper trail survives — point at whatever made V/S right (spec
section, model card, GH issue, upstream PR, or the team-decision doc):

```yaml
PARSER.batch.8.b:
  description: Narration after tool call only
  ref: originated from https://github.com/vllm-project/vllm/.../test_kimi_k2_tool_parser.py#L435
  spec_ref: https://platform.moonshot.ai/docs/tool-call-spec#L42  # or GH issue / PR url
  model_text: |-
    ...
  tools:
  - ...
```

Then regenerate the chart so the cell flips:

```bash
python3 tests/parity/parser/generate_parity_chart.py > PARITY.md
```

Pytest should now be fully green; the chart cell flips to `=`.

```bash
git add lib/parsers/ tests/parity/parser/
git commit -s -m "fix(parser): align <family> with <impl> on PARSER.batch.N.x"
git push
```

## When *not* to fix — permanent divergences

The set of concrete divergent blocks should shrink, not grow. But a
few classes stay recorded forever (intentional, by design):

- **`PARSER.batch.4` (malformed args) and `PARSER.batch.5`
  (missing end-token)** — impl-defined recovery per
  `PARSER_CASES.md`. Each parser picks its own behavior (drop,
  recover, fall back to string, error). Parity not expected.
- **Schema-driven coercion vs parser-layer preservation** —
  e.g. `{"celsius": "20"}` against `celsius: integer`. Some
  parsers coerce at the parser layer, others preserve raw and
  defer coercion downstream. Both defensible.

Everything else is a candidate to fix. Permanent divergences should
carry a `reason:` field that classifies them as intentional (so the
chart shows `V`/`S`, not `V?`/`S?`).

## Fixture file schema

Two file layouts coexist per family. The loader merges them:

```
<family>/PARSER.batch.yaml          ← legacy flat: holds top-level cases (1, 2, ..., 10)
<family>/PARSER.batch.<n>.yaml      ← per-top-level-case: holds sub-cases <n>.a, <n>.b, ...
```

The per-case file is only created when a top-level case grows sub-cases.
Once any sub-case `PARSER.batch.<n>.<sub>` is introduced, the bare
`PARSER.batch.<n>` key migrates out of the flat file into the per-case
file. Case-ID uniqueness across the two files is the merge invariant.

Both file shapes use the same Variant A schema (per-case `expected:`
keyed by impl, peers as anchor refs when matching dynamo, concrete
blocks when diverging):

```yaml
family: kimi_k2
mode: batch
cases:
  PARSER.batch.1:
    description: Single tool call (happy path)
    model_text: |-
      <|tool_calls_section_begin|>...
    tools:
    - name: ...
      parameters: {...}
    expected:
      dynamo: &d_1
        calls:
        - name: ...
          arguments: {...}
        normal_text: ''
      vllm: *d_1           # vLLM matches Dynamo
      sglang: *d_1         # SGLang matches Dynamo
  PARSER.batch.8.a:        # sub-case keys also valid
    description: Narration before tool call only
    ref: originated from https://github.com/vllm-project/vllm/blob/<sha>/tests/tool_parsers/test_<family>_tool_parser.py#L<line>
    model_text: |-
      ...
    expected:
      dynamo: &d_8_a
        calls: [...]
        normal_text: 'I will check the weather.'
      vllm:                # vLLM diverges — concrete output recorded
        calls: [...]
        normal_text: 'I will check the weather. '
        reason: "preserves trailing space; Dynamo trims it"
      sglang:              # SGLang has no peer parser for this family
        unavailable: "SGLang has no detector for family='kimi_k2'"
```

**Peer block forms** (one of):
- **`*d_<case>`** anchor ref — engine matches Dynamo's output.
- **`{calls, normal_text}`** — engine produces a different concrete
  output. Optional `reason: <string>` classifies as intentional;
  absence = research-needed.
- **`{error: <substring>}`** — engine is expected to crash. Test
  passes if the engine's actual error contains the substring.
- **`{unavailable: <msg>}`** — no parser registered for this family.
  Test skips with the message.

The `ref` field is required on per-sub-case files
(`PARSER.<mode>.<n>.yaml`) and takes one of two forms:

- **`ref: originated from <url>`** — there's an upstream test exercising
  this same shape on this same family. The fixture's `model_text` may
  be freshly authored (templated narration, consistent function/args
  across families) rather than copied verbatim, but the shape is
  directly traceable to the linked upstream test. The URL names the
  impl: `vllm-project/vllm` → vLLM, `sgl-project/sglang` → SGLang.
- **`ref: dynamo`** — authored fresh in this repo, no upstream peer.
  Most sub-case taxonomy fillers (`.b` post-only, `.d` between-calls)
  land here because vLLM/SGLang don't test those shapes.

Every sub-case carries one of these two states; there's no "no
provenance" state. The legacy flat `PARSER.<mode>.yaml` (cases without
sub-cases) does NOT carry `ref` — those entries predate the convention.

#### Legacy: `TODO(research)` comments

Some older fixtures (pre-Variant-A) still carry inline YAML comments
of the form `# TODO(research): <impl> diverges — <reason>` below the
case body. Those were the side-by-side diff under the previous schema
where peer divergences lived in a Python registry, not in YAML. With
Variant A's `expected.<impl>` blocks, the comments are redundant — the
authoritative per-engine output is in the YAML data itself.

The `embed_divergence_comments.py` helper still exists for re-embedding
those comments after a regen pass, but new divergences should record
their reasoning in the `reason:` field of the peer block, not in a
comment.

Case keys are the full IDs from
[`lib/parsers/PARSER_CASES.md`](../../lib/parsers/PARSER_CASES.md)
(`PARSER.batch.1` … `PARSER.batch.10`, plus sub-cases like
`PARSER.batch.8.a`). They match the pytest
parametrize IDs directly, so a single `grep PARSER.batch.8.a` finds the
case across docs, fixtures, and Rust source comments.

`model_text` uses YAML's literal block scalar (`|-`) so multi-line
wire formats (XML-style families, harmony) read as the actual text
the model would emit, not a `\n`-escaped one-liner. UTF-8 with
`allow_unicode=True`, so DeepSeek special tokens (`｜` U+FF5C, `▁`
U+2581) appear as literal characters rather than escape sequences.

A side-effect of letting pyyaml choose the best block-scalar header
for every value: `normal_text` occasionally emits as the
unusual-looking `|2+`. Decoding this in three pieces:

- `|` — literal block scalar (preserve newlines as written).
- `2` — explicit indent indicator: "content is indented 2 spaces."
  pyyaml normally auto-detects indent from the first content line;
  it falls back to the explicit form when the body is blank-only
  and there's nothing to detect from.
- `+` — "keep" chomping: preserve all trailing newlines (`-` strips
  them, no indicator clips to one).

So `|2+` followed by one blank line decodes to `"\n"` — a single
newline. It shows up where the divergence between Dynamo and an
upstream impl is literally one inter-wrapper newline character (e.g.
back-to-back `</tool_calls>` / `<tool_calls>` fence pairs where
Dynamo treats the `\n` between them as `normal_text` and the impl
treats it as part of the wrapper). pyyaml round-trips the value to
its canonical form; the header looks cryptic but it's just `"\n"`.

## Why families' YAMLs look so similar (and why that's the point)

Open any two family files side-by-side and the case shells look
nearly identical: same `description` strings, same `tools` schemas,
same case keys `"PARSER.batch.1"`–`"PARSER.batch.10"`. **That's by
design** — `PARSER.batch.N` is the same logical scenario across every
family (run `generate_parity_chart.py` for the full list).

So a reviewer can grep `PARSER.batch.4` across all 10 families and
immediately see how each parser handles the same scenario. The
repetition *is* the diff: it's what makes per-case cross-family
comparison trivial.

### What changes per family

**1. `model_text`** — every family has its own wire format.
`case 1` ("single happy-path call") encoded by each:

| family | model_text (truncated) |
|---|---|
| `kimi_k2` | `<\|tool_calls_section_begin\|><\|tool_call_begin\|>functions.get_weather:0…` |
| `qwen3_coder` | `<tool_call>\n<function=get_weather>\n<parameter=location>\nNYC\n</parameter>…` |
| `glm47` | `<tool_call>get_weather<arg_key>location</arg_key><arg_value>NYC</arg_value>…` |
| `deepseek_v3_1` | `<｜tool▁calls▁begin｜><｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{…}…` |
| `harmony` | `<\|channel\|>commentary to=functions.get_weather <\|constrain\|>json<\|message\|>…` |
| `minimax_m2` | `<minimax:tool_call>\n<invoke name="get_weather">\n<parameter name="location">…` |
| `nemotron_deci` | `<TOOLCALL>[{"name": "get_weather", "arguments": {"location": "NYC"}}]</TOOLCALL>` |

**2. `expected`** — sometimes also differs, when Dynamo's per-family
parser quirks make the same logical scenario produce different
parsed output. Example: `case 4` (malformed input) — the
malformations themselves vary (each is malformed in a way that's
natural for that family's wire format), and Dynamo recovers them
differently:

```
kimi_k2/batch.4    expected.calls[0].arguments = "{\"location\":\"NYC\""
                   ↑ truncated raw string — Dynamo's kimi_k2 parser
                     surfaces the malformed bytes verbatim

qwen3_coder/batch.4 expected.calls[0].arguments = {"location": "NYC"}
                   ↑ recovered into a proper dict — Dynamo's
                     qwen3_coder parser is lenient with missing tags
```

Both are valid per-family Dynamo contracts. Cross-impl divergences
(vLLM and SGLang doing something *different* from Dynamo on the
same case) are visible by reading the per-case `expected.{dynamo,vllm,sglang}` triple in the YAML fixture — anchor refs (`*d_<case>`) mark agreement, concrete blocks mark divergence
as `xfail` entries with a one-sentence reason.

**3. `tools`** — sometimes minor parameter-name differences
(e.g., `city` vs `location`, `unit` field present or not), carried
over from the original Rust unit tests that seeded each family's
fixtures.

If you're *adding* a new case, mirror the case shape across all
applicable families — the harness counts on case N meaning the
same thing everywhere.

## Regenerating fixtures

Run from the repo root inside a container with `dynamo._core` built
(M2's PyO3 binding). The same `capture_parser_outputs` script handles
all three impls — pass `--impl dynamo|vllm|sglang`:

```bash
# Default: drift-check (read-only). Reports cases where the recorded
# `expected.<impl>` disagrees with what the parser now produces.
python3 -m tests.parity.parser.capture_parser_outputs --impl dynamo

# Non-destructive populate: fill `expected.<impl>` only for cases that
# don't have it yet (newly authored cases).
python3 -m tests.parity.parser.capture_parser_outputs --impl dynamo --merge

# Refresh: re-run the parser for every case on disk and overwrite the
# recorded block. Use this only when parser behavior intentionally changed.
python3 -m tests.parity.parser.capture_parser_outputs \
  --impl dynamo --merge --overwrite-if-exists
```

(The `-m` invocation is required — running the script directly puts
`tests/parity/parser/` on `sys.path`, which makes the local
`dynamo.py` wrapper shadow the real `dynamo` package.)

After regenerating, run `git diff tests/parity/parser/fixtures/` to
review the change before staging. Any peer blocks (`expected.vllm`,
`expected.sglang`) attached to a case are preserved across
`--overwrite-if-exists` runs, so refreshing `expected.dynamo` won't
accidentally drop someone else's captured engine output.

## Future stages (sibling directories)

```
tests/parity/
├── parser/         (today)
├── postprocess/    (future) — parser output → OpenAI wire response
└── preprocess/     (future) — request preprocessing, chat-template
```

Each stage has its own fixtures, wrappers, and test file but
reuses the shared `common.py` (`ParseResult`-style shape) and
`conftest.py` (session-scoped server boots) at this level. Out of
scope today; see `lib/parsers/PARSER_CASES.md`,
`components/src/dynamo/frontend/tests/FRONTEND_CASES.md`, and
`lib/parsers/PIPELINE_CASES.md` for the surrounding taxonomy that
will guide which stages are worth adding when.

## Eventual goal: YAML fixtures as the single source of truth

Today there's overlap between this harness's fixtures and the
hand-written Rust unit tests under `lib/parsers/src/tool_calling/*`
— for the ~190 black-box "given input X, parser returns Y" tests,
the same input + expected appears in both places (the M2 fixtures
were originally extracted from those Rust tests by hand).

The intended end state is **one set of fixtures, multiple thin
harnesses**, in subsequent PRs:

```
tests/parity/parser/fixtures/<family>/PARSER.batch.yaml
        │
        ├── Python harness (M2 / M3) — already reads it
        └── Rust harness (future)    — would read it too,
                                       at cargo-test speed,
                                       no Python required
```

What that buys:

- **No duplicated test data.** Adding a case in YAML immediately
  covers Dynamo (Rust harness), Dynamo-via-PyO3 (M2), and
  vLLM/SGLang servers (M3). Today, adding a Rust test means
  hand-mirroring the case into M2's YAML fixtures if you want
  cross-impl coverage.
- **Rust devs keep their fast feedback loop.** `cargo test`
  still finishes in ~0.5 s; no Python build needed.
- **Each impl is tested in its native language.** Closer to
  production semantics than going through PyO3 just to assert
  a Rust contract.

What stays in Rust-only tests after the migration:

- White-box tests on internal helpers (`detect_tool_call_start_*`,
  `find_tool_call_end_position_*`, regex-fallback paths). These
  test parser-internal state, not parity, and aren't exposed
  via PyO3. ~120 of the ~498 Rust tests fall here.
- Tokenizer / config / panic-class tests. Single-impl by nature.

Effort sketch (separate PRs after M2 + M3 land):

- **PR-X:** Rust harness that reads `PARSER.batch.yaml`, dispatches
  to `try_tool_call_parse_<family>(...)`, asserts on `expected`.
  ~1-2 days.
- **PR-Y:** Mechanical migration — delete the ~70 hand-written
  black-box Rust tests now redundant with the shared fixtures.
  ~6 hours.
- **PR-Z:** Same shape extended to format-conditional and
  customer-incident regressions (~150 more cases). ~2-3 days.
- (deferred) Streaming variant. Needs new `PARSER.stream.*`
  schema + streaming PyO3 + streaming Rust harness. Roughly
  the size of the original M3 work.

Until then, M2 and the Rust suite both exist; for ~100 cases they
test the same Dynamo contract through different surfaces. M2's
real value-add is the cross-impl half (vLLM and SGLang).

## Adding a new parser family

1. Add the family name to Dynamo's parser registry (Rust side).
2. Author a fixture YAML at
   `tests/parity/parser/fixtures/<family>/PARSER.batch.yaml` covering
   the `PARSER.batch.<n>` cases that apply (mirror an existing
   family's case shape — see `fixtures/kimi_k2/PARSER.batch.yaml`).
   Each case minimally needs `description`, `ref`, `model_text`,
   and `tools`. Author by hand, copy-edit from an upstream test,
   or have an AI fill in cases against `PARSER_CASES.md`.
3. Run `python3 -m tests.parity.parser.capture_parser_outputs --impl dynamo --merge`
   to fill in each case's `expected.dynamo` by running Dynamo against
   `model_text` + `tools`. Cases that already have `expected.dynamo`
   are left alone.
4. Add the family's vLLM and SGLang dispatch entries to
   `_FAMILY_TO_VLLM_KEY` (`vllm.py`) and
   `_FAMILY_TO_SGLANG_DETECTOR` (`sglang.py`).
5. Populate vLLM and SGLang outputs for the new family by running
   `python3 -m tests.parity.parser.capture_parser_outputs --impl <vllm|sglang> --merge`
   inside each engine's container. The merge writes anchor refs to
   dynamo for matching engines, concrete `{calls, normal_text}` for
   divergent ones, and `{unavailable: <reason>}` when the wrapper marks
   the case as such. Cases where the engine raised (any error other
   than UNAVAILABLE) are skipped — hand-record those as
   `{error: <substring>}` so the test asserts on a stable signature
   rather than the full volatile message. Add a `reason:` field to
   intentional divergences so they show as `V`/`S` not `V?`/`S?` in the
   chart.
6. Regenerate the chart: `python3 tests/parity/parser/generate_parity_chart.py > PARITY.md`.
