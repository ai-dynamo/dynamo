# Tool Call Parser Patterns Reference

Quick reference for common tool call patterns found in LLM chat templates.

## Pattern Categories

### 1. JSON with Special Tokens

#### Bracket Markers (Mistral-style)
```
[TOOL_CALLS] [{"name": "get_weather", "arguments": {"location": "NYC"}}]
```
- Models: Mistral, Mixtral
- Parser: `base_json_parser` with bracket config
- Keys: `name`, `arguments`

#### XML-Style Tags (Hermes-style)
```xml
<tool_call>
{"name": "get_weather", "arguments": {"location": "NYC"}}
</tool_call>
```
- Models: Hermes-2, Jamba
- Parser: `base_json_parser` with XML-style markers
- Keys: `name`, `arguments`

#### Single Token Prefix (Llama-style)
```
<|python_tag|>[{"name": "get_weather", "arguments": {"location": "NYC"}}]
```
- Models: Llama 3.1, Llama 3.2
- Parser: `base_json_parser` with single start token
- Keys: `name`, `arguments`

### 2. XML-Based

#### Qwen3 Coder Style
```xml
<tool_call>
<function=get_weather>
<parameter=location>NYC</parameter>
</function>
</tool_call>
```
- Models: Qwen3-Coder, Nemotron-Nano
- Parser: `xml/parser.rs`
- Attribute-based names and parameters

### 3. Nested Special Tokens

#### DeepSeek V3
```
<｜tool▁call▁begin｜>function<｜tool▁sep｜>get_weather
```json
{"location": "NYC"}
```
<｜tool▁call▁end｜>
```
- Models: DeepSeek-V3
- Parser: `deepseek_v3_parser.rs`
- Multiline with markdown code blocks

#### DeepSeek V3.1
```
<｜tool▁call▁begin｜>get_weather<｜tool▁sep｜>{"location": "NYC"}<｜tool▁call▁end｜>
```
- Models: DeepSeek-V3.1
- Parser: `deepseek_v3_1_parser.rs`
- Inline JSON

### 4. DSML (DeepSeek V3.2)
```xml
<｜DSML｜function_calls>
<｜DSML｜invoke name="get_weather">
<｜DSML｜parameter name="location" string="true">NYC</｜DSML｜parameter>
</｜DSML｜invoke>
</｜DSML｜function_calls>
```
- Models: DeepSeek-V3.2
- Parser: `dsml/parser.rs`
- Explicit parameter types

### 5. Pythonic
```python
[get_weather(location="NYC"), get_time(timezone="EST")]
```
- Models: Custom/Experimental
- Parser: `pythonic/pythonic_parser.rs`
- Python function call syntax

### 6. Harmony
```
<|channel|>commentary to=functions.get_weather
<|constrain|>json
<|message|>{"location": "NYC"}
```
- Models: GPT-OSS
- Parser: `harmony/harmony_parser.rs`
- OpenAI Harmony protocol

## Quick Identification Guide

1. **Look for `tojson` filter** → JSON format
2. **Look for `<function=` or `<parameter=`** → XML format
3. **Look for `<｜DSML｜`** → DSML format
4. **Look for `function(arg=val)`** → Pythonic format
5. **Look for `<|channel|>commentary`** → Harmony format
6. **Check start/end markers** → Match to config preset

## Configuration Keys

For JSON formats, check these keys in the template:
- Function name: Usually `name` or `function`
- Arguments: Usually `arguments` or `parameters`
- Structure: Array `[{...}]` or single object `{...}`

## Matching Logic

1. **Exact match** → Use existing config preset
2. **Similar markers** → Create new config with same parser
3. **New format** → Generate new parser implementation
