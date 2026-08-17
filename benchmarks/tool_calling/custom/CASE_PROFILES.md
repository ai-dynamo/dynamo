# Declarative case profiles

Model-specific diagnostic matrices live in one place:

```text
configs/case_profiles/<profile>.json
```

The probe discovers these files automatically. A profile contains:

- `tools`: reusable OpenAI function definitions keyed by a short local name;
- `request_presets`: reusable model controls such as thinking mode and effort;
- `defaults`: assertions shared by every case, such as reserved markers that
  must never appear in visible content or reasoning;
- `cases`: prompts, selected tool references, request overrides, and expected
  response behavior.

`kimi_k3.json` is the reference implementation. Its 76 scenarios produce 152
records under the standard `nonstream,stream` harness modes.

The automated verifier defines applicability separately in
`../e2e_verifier/profiles.json`. Its `generic_cases` are the 25-case comparable
baseline for every model. Its `model_specific_cases` mapping appends only the
cases assigned to the resolved profile. A `customer_` case can live in either
group because that prefix records where the regression came from, not which
models must run it.

## Add a model profile

1. Copy `configs/case_profiles/kimi_k3.json` to a new snake-case profile name.
2. Change the top-level `profile`, description, shared markers, request
   presets, and tool definitions for the model.
3. Keep the generic baseline model-neutral. Put model-family-only request shapes
   or expectations under that profile in `model_specific_cases`.
4. Add automatic model-name detection in `model_case_profile()` if needed.
   Explicit `--case-profile <name>` works without code changes because the CLI
   discovers profile files.
5. Add profile tests that assert the intended case count and exercise any new
   validation behavior.

Inspect a profile without an endpoint:

```bash
python3 tool_calling_probe.py --case-profile kimi_k3 --list-cases
```

Run only selected cases:

```bash
python3 tool_calling_probe.py \
  --base-url http://localhost:8000/v1 \
  --no-auth \
  --model moonshotai/Kimi-K3 \
  --case-profile kimi_k3 \
  --cases k3_tool_core_required_single_low,k3_reasoning_multiply_17_19 \
  --modes nonstream,stream
```

## Case fields

The loader resolves `tools` entries from the top-level tool registry and
deep-merges `request_preset` with per-case `request_overrides`. Common
expectations include:

- `expected_finish_reasons`
- `expect_no_tool_calls`, `min_tool_calls`, and `exact_tool_calls`
- `expected_tool_names` and exact `expected_tool_calls`
- `expected_content`, `content_pattern`, and `expected_json`
- `expect_reasoning`
- `forbidden_output_fragments`
- `scripted_followup` for deterministic tool-result turns
- `regression_prs` for customer-case provenance

Tool schemas and expected arguments remain ordinary JSON, so profiles require
no Python changes for routine model adaptations.
