# Tool Parser Generator Skill

A Claude Code skill for adding tool calling support to dynamo by analyzing HuggingFace model chat templates.

## Overview

This skill provides a systematic workflow for:
1. Fetching chat templates from HuggingFace models
2. Analyzing tool call patterns and formats
3. Matching against existing dynamo parsers
4. Generating new parser implementations when needed
5. Creating appropriate tests and integration code

## Key Features

- **LLM-Driven**: Leverages Claude's code analysis and generation capabilities
- **Minimal Changes**: Prefers configuration over new code when possible
- **Reference-Aware**: Compares with sglang and vLLM implementations
- **Test Generation**: Automatically creates comprehensive tests
- **Well-Documented**: Includes examples and integration guides

## Usage

Simply ask Claude to add tool calling support for a model:

```
Add tool calling support for Qwen/Qwen2.5-72B-Instruct
```

Claude will:
1. Fetch the model's tokenizer config from HuggingFace
2. Extract and analyze the chat template
3. Compare with existing dynamo parsers
4. Either configure an existing parser or generate a new one
5. Create tests and integration instructions

## Structure

```
tool-parser-generator/
├── SKILL.md                       # Main skill documentation with workflow
├── LICENSE.txt                    # Apache 2.0 license
├── README.md                      # This file
└── references/
    ├── parser-patterns.md         # Quick reference for common patterns
    └── integration-guide.md       # Step-by-step integration instructions
```

## Workflow Phases

1. **Fetch & Extract**: Get chat template from HuggingFace Hub
2. **Analyze**: Identify markers, format type, and structure
3. **Compare**: Match against existing dynamo parsers
4. **Generate/Configure**: Create parser or configuration
5. **Test**: Generate comprehensive test cases
6. **Integrate**: Add to dynamo codebase with proper registration

## Philosophy

- **Prefer Existing**: Most models (>80%) can use existing parsers
- **Minimal Code**: Configuration over implementation when possible
- **Well-Tested**: Every parser needs comprehensive tests
- **Reference-Driven**: Learn from sglang and vLLM implementations

## References

- **Dynamo Parsers**: `/lib/parsers/src/tool_calling/`
- **sglang**: https://github.com/sgl-project/sglang/tree/main/python/sglang/srt/function_call
- **vLLM**: https://github.com/vllm-project/vllm/tree/main/vllm/tool_parsers
- **HuggingFace**: https://huggingface.co/docs/transformers/chat_templating

## Example

See `SKILL.md` for a complete walkthrough of adding support for Qwen/Qwen2.5-72B-Instruct.

## License

Apache 2.0 - See LICENSE.txt for details.
