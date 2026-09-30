# Chat Templates

Vendored Jinja chat templates for models whose HuggingFace stock template does
not emit the markers required by Dynamo's tool-call / reasoning parsers.

Pass any of these at serve time via:

```bash
--custom-jinja-template examples/chat_templates/<file>.jinja
```

## Templates

| File | Model | Pair with |
|---|---|---|
| `gemma4_tool.jinja` | Google Gemma 4 thinking models | `--dyn-tool-call-parser gemma4 --dyn-reasoning-parser gemma4` |

## `gemma4_tool.jinja`

Emits the custom Gemma 4 grammar:

- Tool calls: `<|tool_call>call:name{key:<|"|>value<|"|>}<tool_call|>`
- Reasoning: `<|channel>thought\n...<channel|>`
- Tool definitions: nested `declaration:fn{description:<|"|>...<|"|>,parameters:{...}}` blocks

This template is a verbatim copy of the upstream vLLM project's
`examples/tool_chat_template_gemma4.jinja` (Apache-2.0). It is required when
the HF chat template for Gemma 4 lacks the `<|"|>`-delimited tool-definition
encoding that the parsers expect.
