# Tool Calling Test Cases

This file explains every test case defined in `tool_calling_probe.py` and
what is required for that case to pass.

Each case is run independently for each selected mode. The normal report runs
both `nonstream` and `stream`, so a model can pass one mode and fail the other.

## Global Pass Rules

Every case must satisfy these checks unless the case says otherwise:

- The HTTP request succeeds and the response can be parsed.
- The response `finish_reason` is one of the values expected by the case.
- Tool calls, when expected, have `type=function`, a non-empty `id`, a function
  name, unique tool-call IDs, and JSON object arguments.
- Tool-call arguments satisfy the declared JSON schema when schema validation is
  enabled.
- Expected tool names, tool-call counts, argument fragments, and content
  fragments are present.
- Forbidden context or parser fragments do not appear in tool arguments,
  assistant content, or reasoning where the case forbids them.
- Raw tool-call marker strings must not leak into assistant content or
  reasoning. Examples include `<|tool_call`, `<tool_call>`, `<function=`,
  `<arg_key>`, and DSML markers.
- Warnings are recorded but do not fail a case. Errors fail the case.

Common failure kinds include `unexpected_finish_reason`, `too_few_tool_calls`,
`wrong_tool_call_count`, `missing_expected_tool`, `invalid_arguments_json`,
`schema_validation`, `missing_expected_argument_fragment`,
`context_leak_to_tool_arguments`, and `tool_marker_leaked_to_reasoning`.

## Profiles

Custom cases have two applicability groups. The 25 `generic` cases form the
fixed automated baseline. Every model runs those cases in nonstreaming and
streaming mode for 50 comparable records. The verifier then appends configured
`model_specific` cases only for the resolved model profile. Kimi K2 currently
adds two customer multi-turn cases, for 27 cases and 54 total records.

The `customer_` prefix is independent of those groups: it records regression
provenance, so a customer case may be generic or model-specific. Other
model-specific profiles also expose parser stress cases for focused diagnostic
runs:

- `deepseek_v4` adds DeepSeek DSML cases.
- `gemma4` adds Gemma parser cases.
- `qwen3_coder` adds Qwen XML parser cases.
- `glm5` and `glm47` add GLM XML parser cases.
- `minimax_m2` adds MiniMax M2 parser cases.

MiniMax uses the generic cases plus MiniMax M2 parser stress cases.

## Customer regression provenance

The qualification catalog includes every concrete customer tool-calling
request shape found in merged Dynamo and `frontend-crates` fixes from
2026-05-16 through 2026-08-16. Their IDs start with `customer_`, and the
corresponding `Case.regression_prs` values are the machine-checked source of
truth.

Several PRs map to one case because they fixed the same OpenAI request contract
at different layers. For example, required tool choice with a conflicting
plain-text prompt covers reasoning routing, backend constraints, and native Kimi
structural tags without duplicating the request.

| Customer case | Applicability | Source fixes |
|---|---|---|
| `customer_codex_items_schema_tool_call` | Generic | [Dynamo #9778](https://github.com/ai-dynamo/dynamo/pull/9778) |
| `customer_required_forces_weather` | Generic | [#9804](https://github.com/ai-dynamo/dynamo/pull/9804), [#10030](https://github.com/ai-dynamo/dynamo/pull/10030), [#11205](https://github.com/ai-dynamo/dynamo/pull/11205), [#11554](https://github.com/ai-dynamo/dynamo/pull/11554), [#12684](https://github.com/ai-dynamo/dynamo/pull/12684), [frontend-crates #188](https://github.com/ai-dynamo/frontend-crates/pull/188) |
| `customer_required_forces_weather_thinking_disabled` | Generic | [#11554](https://github.com/ai-dynamo/dynamo/pull/11554), [#12684](https://github.com/ai-dynamo/dynamo/pull/12684), [frontend-crates #188](https://github.com/ai-dynamo/frontend-crates/pull/188) |
| `customer_named_calculator_choice` | Generic | [#9804](https://github.com/ai-dynamo/dynamo/pull/9804), [#10030](https://github.com/ai-dynamo/dynamo/pull/10030), [#11205](https://github.com/ai-dynamo/dynamo/pull/11205), [#11554](https://github.com/ai-dynamo/dynamo/pull/11554), [#12684](https://github.com/ai-dynamo/dynamo/pull/12684) |
| `customer_named_calculator_choice_thinking_disabled` | Generic | [#11554](https://github.com/ai-dynamo/dynamo/pull/11554), [#12684](https://github.com/ai-dynamo/dynamo/pull/12684) |
| `customer_required_marker_isolation` | Generic | [#11045](https://github.com/ai-dynamo/dynamo/pull/11045), [frontend-crates #133](https://github.com/ai-dynamo/frontend-crates/pull/133), [frontend-crates #152](https://github.com/ai-dynamo/frontend-crates/pull/152) |
| `customer_truncated_tool_markup_hidden` | Generic | [#9864](https://github.com/ai-dynamo/dynamo/pull/9864) |
| `customer_kimi_consume_prior_tool_result` | Kimi K2-specific | [#11653](https://github.com/ai-dynamo/dynamo/pull/11653) |
| `customer_kimi_parallel_weather_final_answer` | Kimi K2-specific | [#11653](https://github.com/ai-dynamo/dynamo/pull/11653) |

`customer_calculate_sum_auto` is an older retained customer regression without
a PR in this audit window. Responses-only fixes and deployment/request-default
features are outside this Chat Completions case set; they require their own
endpoint or request-contract matrix instead of a misleading duplicate here.

## Generic Cases

### `auto_single_weather`

Checks basic `tool_choice=auto` weather calling.

To pass:

- Finish with `tool_calls`.
- Emit at least one `get_weather` call.
- Arguments are schema-valid for `get_weather`.
- Decoded arguments or content contain `San Francisco`.

### `customer_calculate_sum_auto`

Customer regression for the `Compute 1+1!` request with the `calculate_sum`
tool and `tool_choice=auto`.

To pass:

- Finish with `tool_calls`.
- Emit exactly one `calculate_sum` call.
- Arguments are schema-valid for `calculate_sum`.
- Decoded arguments contain `a=1` and `b=1`.

### `auto_parallel_weather_two_cities`

Checks parallel independent calls to the same weather tool.

To pass:

- Finish with `tool_calls`.
- Emit at least two tool calls.
- Emit at least two `get_weather` calls.
- Decoded arguments or content contain both `San Francisco` and `New York`.

### `auto_echo_context_probe`

Checks that a normal auto-selected diagnostic tool receives only the requested
public marker and does not absorb hidden schema or system context.

To pass:

- Emit an `echo_context` call containing `PUBLIC_ECHO_MARKER_AUTO_93FD`.
- Tool arguments and assistant content do not contain the echo schema or system
  sentinels.

### `auto_parallel_weather_with_echo_probe`

Checks parallel calls across two tools while keeping their arguments isolated.

To pass:

- Emit two `get_weather` calls and one `echo_context` call.
- Preserve both city names and `PUBLIC_ECHO_MARKER_PARALLEL_D18A` globally.
- Do not copy either city name or `get_weather` into `echo_context` arguments.
- Do not expose the echo schema or system sentinels.

### `customer_codex_items_schema_tool_call`

Reproduces the Codex/GPT-OSS schema collision where a JSON Schema property is
literally named `items`.

To pass:

- Finish with `tool_calls`.
- Emit exactly one `collect_items` call.
- Preserve `items=["alpha", "beta"]` as a schema-valid array.

### `customer_required_marker_isolation`

Checks a required diagnostic call for raw native marker leakage and a missing
streaming `finish_reason`.

To pass:

- Finish with `tool_calls`.
- Emit at least one `echo_context` call.
- Decoded arguments or content contain `PUBLIC_ECHO_MARKER_REQUIRED_4C21`.
- Tool arguments and assistant content do not contain the echo schema or system
  sentinels.

### `customer_truncated_tool_markup_hidden`

Uses a deliberately small output budget and a long required argument to drive
the EOF/truncation path from the production Nemotron customer regression.

To pass:

- Finish with either `length` or `tool_calls`.
- Never expose raw XML/native tool markup in content or reasoning.
- If a complete tool call is recovered, its arguments must still be valid JSON.

### `customer_required_forces_weather`

Checks that `tool_choice=required` forces a tool call even when the prompt asks
for plain text.

To pass:

- Finish with `tool_calls`.
- Emit at least one `get_weather` call.
- Arguments are valid JSON and satisfy the `get_weather` schema, including the
  required `location` field.

### `customer_required_forces_weather_thinking_disabled`

Same as `customer_required_forces_weather`, but with the request-level thinking
flag disabled.

To pass:

- Finish with `tool_calls`.
- Emit at least one schema-valid `get_weather` call.
- The disabled-thinking request override must not prevent valid tool calling.

### `customer_named_calculator_choice`

Checks a named tool choice forcing the calculator tool.

To pass:

- Finish with `tool_calls`.
- Emit at least one `calculate` call.
- Arguments are schema-valid for `calculate`.
- Decoded arguments or content contain `937` and `18`.

### `customer_named_calculator_choice_thinking_disabled`

Same as `customer_named_calculator_choice`, but with the request-level thinking
flag disabled.

To pass:

- Finish with `tool_calls`.
- Emit at least one schema-valid `calculate` call.
- Decoded arguments or content contain `937` and `18`.

### `auto_multi_distinct_tools`

Checks that auto tool choice can choose multiple useful tools from a mixed tool
set.

To pass:

- Finish with `tool_calls`.
- Emit at least two tool calls.
- Use at least two distinct tool names from the provided tool set.
- Every emitted call must have valid JSON object arguments and pass schema
  validation for its tool.

### `none_suppresses_weather`

Checks that `tool_choice=none` suppresses a tempting weather call.

To pass:

- Finish with `stop`.
- Emit no tool calls.
- Produce non-empty assistant content.
- Schema validation is disabled because no tool call should be present.

### `named_array_arguments`

Checks named tool calling with array arguments.

To pass:

- Finish with `tool_calls`.
- Emit at least one `send_email` call.
- Arguments are schema-valid for `send_email`.
- Decoded arguments or content contain `alice@example.com` and
  `bob@example.com`.

### `named_nested_object_arguments`

Checks named tool calling with nested objects and arrays.

To pass:

- Finish with `tool_calls`.
- Emit at least one `create_calendar_event` call.
- Arguments are schema-valid for `create_calendar_event`.
- Decoded arguments or content contain `Design Review` and
  `alex@example.com`.

### `plain_no_tools`

Checks a normal chat response when no tools are provided.

To pass:

- Finish with `stop`.
- Emit no tool calls.
- Produce non-empty assistant content.
- Schema validation is disabled because no tools are present.

### `e2e_search_then_crawl_final_answer`

Checks a multi-step tool loop where search should lead to crawl before the
final answer.

To pass:

- Each turn finishes with either `tool_calls` or `stop`.
- Tool-call turns have valid tool-call shape and schema-valid arguments.
- Across the loop, at least one `search_web` call and one `crawl_page` call are
  executed.
- The loop ends with a final non-empty assistant answer.
- Final content contains both `streaming` and `multi-step`.

### `named_no_argument_server_time`

Checks a named tool whose strict schema accepts an empty argument object.

To pass:

- Emit exactly one `get_server_time` call.
- Emit a schema-valid empty argument object.

### `required_mixed_scalar_arguments`

Checks required tool calling with string, integer, and boolean fields.

To pass:

- Emit exactly one `book_flight` call.
- Preserve destination `Lisbon`, passenger count `2`, and `first_class=false`.

### `named_strict_nested_pipeline`

Checks a named tool with strict nested objects, arrays, integers, and booleans.

To pass:

- Emit exactly one `configure_pipeline` call.
- Produce schema-valid settings containing flags `fast` and `safe` and timeout
  `2500`.

### `named_literal_escaped_unicode`

Checks exact string preservation across quotes, a backslash, and Unicode.

To pass:

- Emit exactly one `record_literal` call.
- Preserve label `portable-literal` and the complete requested literal string.

### `required_parallel_same_tool`

Checks two required parallel calls to the same function.

To pass:

- Emit exactly two schema-valid `get_weather` calls.
- Keep `Madrid` and `Seoul` in distinct call arguments.

### `auto_irrelevant_no_call`

Checks that `tool_choice=auto` avoids an irrelevant available weather tool.

To pass:

- Finish with `stop` and emit no tool calls.
- Produce non-empty assistant content.

### `named_weather_enum_argument`

Checks a named tool call with a required string and enum field.

To pass:

- Emit exactly one `get_weather` call.
- Preserve location `Reykjavik` and unit `celsius`.

## Kimi K2 Model-Specific Cases

### `customer_kimi_consume_prior_tool_result`

Checks that Kimi K2 consumes an existing tool result instead of calling the
tool again.

To pass:

- Finish with `stop`.
- Emit no new tool calls.
- Produce non-empty assistant content.
- Content contains at least one of `15` or `cloud`.
- Schema validation is disabled because no new tool call should be present.

### `customer_kimi_parallel_weather_final_answer`

Checks the real Kimi K2 multi-turn client loop: the model requests weather
tools, the probe executes mock tools, and the model produces a final answer.

To pass:

- Each turn finishes with either `tool_calls` or `stop`.
- Tool-call turns have valid tool-call shape and schema-valid arguments.
- Across the loop, at least two `get_weather` calls are executed.
- The loop ends with a final non-empty assistant answer, not another pending
  tool-call request.
- Final content contains both `San Francisco` and `New York`.

## DeepSeek V4 Cases

### `deepseek_dsml_no_arg_named_tool`

Checks a named no-argument tool call in the DeepSeek DSML profile.

To pass:

- Finish with `tool_calls`.
- Emit exactly one `get_server_time` call.
- Arguments are valid JSON object arguments for a no-argument tool, normally
  `{}`.

### `deepseek_dsml_integer_argument`

Checks typed integer extraction in DSML-style tool calling.

To pass:

- Finish with `tool_calls`.
- Emit at least one `set_temperature` call.
- Arguments are schema-valid, including integer `celsius`.
- Decoded arguments or content contain `lab-a` and `20`.

### `deepseek_dsml_nested_arguments`

Checks nested object and array extraction in DSML-style tool calling.

To pass:

- Finish with `tool_calls`.
- Emit at least one `configure_pipeline` call.
- Arguments are schema-valid for `configure_pipeline`.
- Decoded arguments or content contain `alpha`, `beta`, and `2500`.

### `deepseek_dsml_marker_in_argument`

Checks that DSML-looking delimiter text can appear inside a string argument
without breaking parsing or leaking into content.

To pass:

- Finish with `tool_calls`.
- Emit at least one `record_literal` call.
- Arguments are schema-valid for `record_literal`.
- Decoded arguments or content contain `dsml-boundary`, `alpha`,
  `</｜DSML｜parameter>`, `<｜DSML｜invoke`, and `shadow`.
- Assistant content does not contain `</｜DSML｜parameter>` or
  `<｜DSML｜invoke`.

### `deepseek_dsml_orphan_marker_context_isolation`

Checks that DSML parser markers in system context do not become tool output.

To pass:

- Finish with `tool_calls`.
- Emit at least one `get_weather` call.
- Decoded arguments or content contain `Tokyo`.
- Tool arguments and assistant content do not contain the DSML context
  sentinel, the orphan DSML end sentinel, `<｜DSML｜`, or `</｜DSML｜`.

## Gemma Cases

### `gemma_no_arg_named_tool`

Checks a named no-argument tool call in the Gemma profile.

To pass:

- Finish with `tool_calls`.
- Emit exactly one `get_server_time` call.
- Arguments are valid JSON object arguments for a no-argument tool, normally
  `{}`.

### `gemma_scalar_arguments`

Checks scalar argument types in the Gemma profile.

To pass:

- Finish with `tool_calls`.
- Emit at least one `book_flight` call.
- Arguments are schema-valid, including string `destination`, integer
  `passengers`, and boolean `first_class`.
- Decoded arguments or content contain `Paris`, `2`, and `true`.

### `gemma_nested_arguments`

Checks nested object and array arguments in the Gemma profile.

To pass:

- Finish with `tool_calls`.
- Emit at least one `configure_pipeline` call.
- Arguments are schema-valid for `configure_pipeline`.
- Decoded arguments or content contain `alpha`, `beta`, and `2500`.

### `gemma_delimiter_string_argument`

Checks that Gemma delimiter-looking text can remain inside one string argument.

To pass:

- Finish with `tool_calls`.
- Emit at least one `run_query` call.
- Arguments are schema-valid for `run_query`.
- Decoded arguments or content contain `SELECT a,b:and{brace}`, `brace }`,
  `bracket ]`, `<|tool_call>`, `<tool_call|>`, and `<|"|>`.
- Assistant content does not contain `<|tool_call>`, `<tool_call|>`, or
  `<|"|>`.

### `gemma_same_name_parallel_weather`

Checks two independent calls to the same tool in the Gemma profile.

To pass:

- Finish with `tool_calls`.
- Emit at least two `get_weather` calls.
- Decoded arguments or content contain `Boston` and `New York`.

### `gemma_marker_context_isolation`

Checks that raw Gemma parser markers in system context do not become tool
output.

To pass:

- Finish with `tool_calls`.
- Emit at least one `get_weather` call.
- Decoded arguments or content contain `Tokyo`.
- Tool arguments and assistant content do not contain the Gemma context
  sentinel, `<|tool_call>`, `<tool_call|>`, `<|"|>`, `shadow_tool`, or
  `Sydney`.

## Qwen Cases

### `qwen_no_arg_named_tool`

Checks a named no-argument tool call in the Qwen XML profile.

To pass:

- Finish with `tool_calls`.
- Emit exactly one `get_server_time` call.
- Arguments are valid JSON object arguments for a no-argument tool, normally
  `{}`.

### `qwen_scalar_arguments`

Checks scalar argument types in the Qwen XML profile.

To pass:

- Finish with `tool_calls`.
- Emit at least one `book_flight` call.
- Arguments are schema-valid, including string `destination`, integer
  `passengers`, and boolean `first_class`.
- Decoded arguments or content contain `Paris`, `2`, and `true`.

### `qwen_nested_arguments`

Checks nested object and array arguments in the Qwen XML profile.

To pass:

- Finish with `tool_calls`.
- Emit at least one `configure_pipeline` call.
- Arguments are schema-valid for `configure_pipeline`.
- Decoded arguments or content contain `alpha`, `beta`, and `2500`.

### `qwen_xml_delimiter_string_argument`

Checks that XML-looking delimiter text can remain inside one string argument.

To pass:

- Finish with `tool_calls`.
- Emit at least one `record_literal` call.
- Arguments are schema-valid for `record_literal`.
- Decoded arguments or content contain `qwen-xml`, `alpha`, `</parameter>`,
  `<function=shadow_tool>`, `<tool_call>`, and `delta`.
- Assistant content does not contain `</parameter>`,
  `<function=shadow_tool>`, or `<tool_call>`.

### `qwen_same_name_parallel_weather`

Checks two independent calls to the same tool in the Qwen XML profile.

To pass:

- Finish with `tool_calls`.
- Emit at least two `get_weather` calls.
- Decoded arguments or content contain `Boston` and `New York`.

### `qwen_marker_context_isolation`

Checks that raw Qwen XML parser markers in system context do not become tool
output.

To pass:

- Finish with `tool_calls`.
- Emit at least one `get_weather` call.
- Decoded arguments or content contain `Tokyo`.
- Tool arguments and assistant content do not contain the Qwen context
  sentinel, `<tool_call>`, `</tool_call>`, `<function=`, `</function>`,
  `<parameter=`, `</parameter>`, `shadow_tool`, or `Sydney`.

## GLM Cases

### `glm_no_arg_named_tool`

Checks a named no-argument tool call in the GLM XML profile.

To pass:

- Finish with `tool_calls`.
- Emit exactly one `get_server_time` call.
- Arguments are valid JSON object arguments for a no-argument tool, normally
  `{}`.

### `glm_scalar_arguments`

Checks scalar argument types in the GLM XML profile.

To pass:

- Finish with `tool_calls`.
- Emit at least one `book_flight` call.
- Arguments are schema-valid, including string `destination`, integer
  `passengers`, and boolean `first_class`.
- Decoded arguments or content contain `Paris`, `2`, and `true`.

### `glm_nested_arguments`

Checks nested object and array arguments in the GLM XML profile.

To pass:

- Finish with `tool_calls`.
- Emit at least one `configure_pipeline` call.
- Arguments are schema-valid for `configure_pipeline`.
- Decoded arguments or content contain `alpha`, `beta`, and `2500`.

### `glm_xml_delimiter_string_argument`

Checks that GLM XML-looking delimiter text can remain inside one string
argument.

To pass:

- Finish with `tool_calls`.
- Emit at least one `record_literal` call.
- Arguments are schema-valid for `record_literal`.
- Decoded arguments or content contain `glm-xml`, `alpha`, `<arg_key>`,
  `<arg_value>`, `<tool_call>`, and `delta`.
- Assistant content does not contain `<arg_key>`, `<arg_value>`, or
  `<tool_call>`.

### `glm_same_name_parallel_weather`

Checks two independent calls to the same tool in the GLM XML profile.

To pass:

- Finish with `tool_calls`.
- Emit at least two `get_weather` calls.
- Decoded arguments or content contain `Boston` and `New York`.
- Assistant content does not contain `<tool_call>`, `</tool_call>`,
  `<arg_key>`, `</arg_key>`, `<arg_value>`, `</arg_value>`, or
  `get_weatherlocation`.

### `glm_marker_context_isolation`

Checks that raw GLM XML parser markers in system context do not become tool
output.

To pass:

- Finish with `tool_calls`.
- Emit at least one `get_weather` call.
- Decoded arguments or content contain `Tokyo`.
- Tool arguments and assistant content do not contain the GLM context sentinel,
  `<tool_call>`, `</tool_call>`, `<arg_key>`, `</arg_key>`, `<arg_value>`,
  `</arg_value>`, `shadow_tool`, or `Sydney`.

## MiniMax M2 Cases

### `minimax_m2_no_arg_named_tool`

Checks a named no-argument tool call in the MiniMax M2 profile.

To pass:

- Finish with `tool_calls`.
- Emit exactly one `get_server_time` call.
- Arguments are valid JSON object arguments for a no-argument tool, normally
  `{}`.

### `minimax_m2_required_scalar_arguments`

Checks scalar argument types with `tool_choice=required` in the MiniMax M2
profile.

To pass:

- Finish with `tool_calls`.
- Emit at least one `book_flight` call.
- Arguments are schema-valid, including string `destination`, integer
  `passengers`, and boolean `first_class`.
- Decoded arguments or content contain `Paris`, `2`, and `true`.

### `minimax_m2_named_array_arguments`

Checks array arguments in the MiniMax M2 profile.

To pass:

- Finish with `tool_calls`.
- Emit at least one `send_email` call.
- Arguments are schema-valid for `send_email`.
- Decoded arguments or content contain `Parser Check`, `maya@example.com`,
  `noah@example.com`, and `li@example.com`.

### `minimax_m2_named_nested_arguments`

Checks nested object and array arguments in the MiniMax M2 profile.

To pass:

- Finish with `tool_calls`.
- Emit at least one `configure_pipeline` call.
- Arguments are schema-valid for `configure_pipeline`.
- Decoded arguments or content contain `alpha`, `beta`, `2500`, and `true`.

### `minimax_m2_marker_in_argument`

Checks that MiniMax-looking delimiter text can remain inside one string
argument.

To pass:

- Finish with `tool_calls`.
- Emit at least one `record_literal` call.
- Arguments are schema-valid for `record_literal`.
- Decoded arguments or content contain `minimax-xml`, `alpha`,
  `<minimax:tool_call>`, `<minimax:invoke>`, `</minimax:tool_call>`, and
  `delta`.
- Assistant content does not contain `<minimax:tool_call>`,
  `<minimax:invoke>`, or `</minimax:tool_call>`.
