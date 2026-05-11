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
    ├── regenerate_fixtures.py      ← (re-)build fixtures by running Dynamo's parser
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
| **Cost** | (TBD) | ~3 s for 570 tests | ~60 s for 30 tests (server boot dominates) |
| **GPU** | yes | none | yes (`--load-format dummy` still allocates ~2.5 GiB) |
| **CI markers** | (TBD) | `unit, pre_merge, gpu_0` | `e2e, pre_merge, gpu_1` |

All three methods (when implemented) share the same fixtures,
`ParseResult` shape, and `KNOWN_DIVERGENCES` registry pattern.
They're stacked diagnostics:

- M2 says: *"the parser class disagrees"*
- M3 says: *"the server stack also disagrees"* (or, more usefully,
  *"disagrees only at the server stack — parser class agrees"*,
  which localizes the bug to chat-template / tokenizer /
  response shaping)
- M1 says: *"Dynamo's wrapper layer disagrees with the upstream
  parser it's wrapping"*

## Current parity status (M2, batch mode)

Each row is a parser family; each column `bN` is
[`PARSER.batch.N`](../../lib/parsers/PARSER_CASES.md):

```
b1   =  PARSER.batch.1   "single happy-path call"
b2   =  PARSER.batch.2   "multiple calls"
b3   =  PARSER.batch.3   "no tool call (plain text)"
b4   =  PARSER.batch.4   "malformed JSON args"
b5   =  PARSER.batch.5   "missing end-token recovery"
b6   =  PARSER.batch.6   "empty args (no-arg call)"
b7   =  PARSER.batch.7   "complex args (nested JSON / arrays)"
b8   =  PARSER.batch.8   "interleaved normal text"
b9   =  PARSER.batch.9   "empty input"
b10  =  PARSER.batch.10  "duplicate calls (same name twice)"
```

Cell values show divergence from Dynamo (the oracle):

- `✓` — both vLLM and SGLang match Dynamo's expected output.
- `V` — vLLM diverges (xfailed via `KNOWN_DIVERGENCES`).
- `S` — SGLang diverges.
- `VS` — both diverge.
- `n/a` — no peer parser registered for that family.

19 parser families total — split into the **Top-N models** we prioritize
(rows tagged with the model name in parens) and **Others** that aren't
top-N today but are wired into the harness for completeness. Both sections
sorted alphabetically within themselves.

| family                          |  b1 |  b2 |  b3 |  b4 |  b5 |  b6 |  b7 |  b8 |  b9 | b10 |
|---------------------------------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Top-N models**                |     |     |     |     |     |     |     |     |     |     |
| deepseek_v4 § (DeepSeek V4)     | ✓   | ✓   | ✓   | ✓   | V   | ✓   | V   | ✓   | ✓   | ✓   |
| gemma4 § (Gemma 4)              | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | V   | ✓   | ✓   |
| glm47 (GLM 5.1)                 | ✓   | ✓   | ✓   | ✓   | VS  | ✓   | ✓   | V   | ✓   | ✓   |
| harmony † (gpt-oss)             | S   | S   | ✓   | S   | S   | S   | S   | S   | ✓   | S   |
| kimi_k2 (Kimi K2.6)             | ✓   | ✓   | ✓   | S   | ✓   | ✓   | ✓   | VS  | ✓   | ✓   |
| minimax_m2 (MiniMax 2.7)        | ✓   | ✓   | ✓   | V   | VS  | ✓   | ✓   | ✓   | ✓   | ✓   |
| nemotron_deci †§ (Nemotron)     | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| qwen3_coder (Qwen 3.5)          | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   |
| **Others**                      |     |     |     |     |     |     |     |     |     |     |
| deepseek_v3                     | ✓   | ✓   | ✓   | V   | VS  | ✓   | ✓   | S   | ✓   | ✓   |
| deepseek_v3_1                   | ✓   | ✓   | ✓   | V   | VS  | ✓   | ✓   | S   | ✓   | ✓   |
| deepseek_v3_2                   | ✓   | ✓   | ✓   | ✓   | V   | ✓   | V   | S   | ✓   | ✓   |
| hermes                          | ✓   | ✓   | ✓   | VS  | ✓   | ✓   | ✓   | V   | ✓   | ✓   |
| jamba §                         | ✓   | ✓   | ✓   | ✓   | V   | ✓   | ✓   | V   | ✓   | ✓   |
| llama3_json §                   | ✓   | V   | ✓   | V   | V   | ✓   | ✓   | ✓   | ✓   | V   |
| mistral                         | S   | S   | ✓   | VS  | S   | S   | S   | VS  | ✓   | S   |
| nemotron_nano †§                | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a | n/a |
| phi4 §                          | ✓   | ✓   | ✓   | ✓   | V   | ✓   | V   | V   | ✓   | ✓   |
| pythonic                        | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | ✓   | VS  | ✓   | ✓   |
| qwen25 †                        | S   | S   | ✓   | S   | S   | S   | S   | S   | ✓   | S   |

### Footnotes

† vLLM has no peer parser (or returns `UNAVAILABLE` at runtime, e.g.
  `harmony#vllm` requires token IDs not text). Cells show SGLang status
  only when SGLang is wired; otherwise the row is fully `n/a`.
§ SGLang has no peer detector for this family. Cells show vLLM status
  only when vLLM is wired; otherwise the row is fully `n/a`.

`nemotron_deci` and `nemotron_nano` carry both daggers (`†§`) — neither
upstream has a peer parser, so the rows are fully `n/a`. Listed here for
completeness; Dynamo-only self-parity could be added in a follow-up but
yields no cross-impl signal.

Tally (excluding `dynamo` since it's the oracle): 380 cells against
the YAML expected — 203 parity, 67 divergence (xfailed), 110 n/a.

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

Both file shapes use the same schema:

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
      calls:
      - name: ...
        arguments: {...}
      normal_text: ''
  PARSER.batch.8.a:                    # sub-case keys also valid
    description: Narration before tool call only
    ref: https://github.com/vllm-project/vllm/blob/<sha>/tests/tool_parsers/test_<family>_tool_parser.py#L<line>
    model_text: |-
      ...
```

The `ref` field is required on per-sub-case files
(`PARSER.<mode>.<n>.yaml`) and takes one of three forms, distinguishing
how strongly the fixture is tied to upstream:

- **`ref: inspired-by <url>`** — there's an upstream test exercising
  this same shape on this same family, but the fixture's `model_text`
  is freshly authored (templated narration, consistent function/args
  across families) rather than copied verbatim. The URL points back at
  the upstream test for traceability. Most cases that have an upstream
  analogue land here.
- **`ref: ported-from <url>`** — the fixture's `model_text` is a
  verbatim (or minimally-adapted) copy of the upstream test's input.
  Rare; reserve for when the wire format is intricate enough that
  byte-equivalence matters and we want the upstream link to be a
  literal source-of-truth.
- **`ref: dynamo`** — authored fresh in this repo, no upstream peer.
  Most sub-case taxonomy fillers (`.b` post-only, `.d` between-calls)
  land here because vLLM/SGLang don't test those shapes.

The URL form (`inspired-by` / `ported-from`) names the impl in the URL
itself: `vllm-project/vllm` → vLLM, `sgl-project/sglang` → SGLang.

Every sub-case carries one of these three states; there's no "no
provenance" state. The legacy flat `PARSER.<mode>.yaml` (cases without
sub-cases) does NOT carry `ref` — those entries predate the convention.

#### Embedded divergence comments

Each per-sub-case fixture for which there's a registered cross-impl
divergence (`KNOWN_DIVERGENCES` in `test_parity_parser.py`) carries a
YAML comment block right after the case's `expected:` data showing what
the diverging impl actually produces, marked `TODO(research)`:

```yaml
PARSER.batch.8.b:
  ...
  expected:
    calls: [...]
    normal_text: "Let me know if you need more."     # Dynamo's output
  # TODO(research): vllm diverges — drops trailing normal_text...
  # vllm produces:
  #   calls: [{'name': 'get_weather', 'arguments': {'location': 'Dallas'}}]
  #   normal_text: None
```

That way the divergence is visible at the fixture level without running
the harness, and each one carries an explicit "decide whether to switch"
prompt. Comments are written by `embed_divergence_comments.py` (run
after the regenerator, since `yaml.dump` strips comments on rewrite):

```bash
PYTHONPATH=lib/bindings/python/src python3 -m tests.parity.parser.regenerate_fixtures --overwrite-if-exists
PYTHONPATH=lib/bindings/python/src python3 -m tests.parity.parser.embed_divergence_comments
```

Case keys are the full IDs from
[`lib/parsers/PARSER_CASES.md`](../../lib/parsers/PARSER_CASES.md)
(`PARSER.batch.1` … `PARSER.batch.10`, plus sub-cases like
`PARSER.batch.8.a`). They match the `KNOWN_DIVERGENCES` keys and pytest
parametrize IDs directly, so a single `grep PARSER.batch.8.a` finds the
case across docs, fixtures, and Rust source comments.

`model_text` uses YAML's literal block scalar (`|-`) so multi-line
wire formats (XML-style families, harmony) read as the actual text
the model would emit, not a `\n`-escaped one-liner. UTF-8 with
`allow_unicode=True`, so DeepSeek special tokens (`｜` U+FF5C, `▁`
U+2581) appear as literal characters rather than escape sequences.

## Why families' YAMLs look so similar (and why that's the point)

Open any two family files side-by-side and the case shells look
nearly identical: same `description` strings, same `tools` schemas,
same case keys `"PARSER.batch.1"`–`"PARSER.batch.10"`. **That's by
design** — `PARSER.batch.N` is the same logical scenario across every
family (full list in the [Current parity status](#current-parity-status-m2-batch-mode)
section above).

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
same case) are tracked in each test's `KNOWN_DIVERGENCES` registry
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
(M2's PyO3 binding):

```bash
# Default: non-destructive — new cases written, existing left alone.
python3 -m tests.parity.parser.regenerate_fixtures

# Refresh: re-run Dynamo for every case in INPUTS, overwrite on disk.
# Use this only when Dynamo's parser behavior intentionally changed.
python3 -m tests.parity.parser.regenerate_fixtures --overwrite-if-exists
```

(The `-m` invocation is required — running the script directly puts
`tests/parity/parser/` on `sys.path`, which makes the local
`dynamo.py` wrapper shadow the real `dynamo` package.)

After regenerating, run `git diff tests/parity/parser/fixtures/` to
review the change before staging. Cases on disk that aren't in
`INPUTS` today are always preserved, regardless of flag, so editing
your `INPUTS` section can't accidentally delete other contributors'
cases.

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
  hand-mirroring the case into M2's `INPUTS` if you want
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
2. Run M2's existing tests — the new family's `dynamo` wrapper
   tests will fail because no fixtures exist yet.
3. Add a section to `INPUTS` in `regenerate_fixtures.py` for every
   `(family, "PARSER.batch.<n>")` you want to cover (mirror the
   case shape from an existing family).
4. Run the regenerator to materialize `<family>/PARSER.batch.yaml`.
5. Add the family's vLLM and SGLang dispatch entries to
   `_FAMILY_TO_VLLM_KEY` (`vllm.py`) and
   `_FAMILY_TO_SGLANG_DETECTOR` (`sglang.py`).
6. Run pytest. Any cross-impl divergences surface as failures —
   classify each, add a one-sentence reason to `KNOWN_DIVERGENCES`,
   and the test goes to xfail.
