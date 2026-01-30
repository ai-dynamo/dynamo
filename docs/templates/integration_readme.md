---
orphan: true
---

# <Integration> Integration

<!-- 2-3 sentence overview of this external integration -->

## Version Compatibility

| Dynamo | <Integration> | Notes |
|--------|---------------|-------|
| 0.9.x | 1.2.x | Recommended |
| 0.8.x | 1.1.x | |

## Backend Support

| Backend | Status | Notes |
|---------|--------|-------|
| vLLM | ✅ | |
| SGLang | 🚧 | |
| TensorRT-LLM | ❌ | |

## Quick Start

```bash
# Installation
pip install <integration>

# Usage with Dynamo
python -m dynamo.<backend> --<integration>-enabled
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--<integration>-arg` | `value` | Description |

## Guides

- \[<Integration> Setup\](<integration>_setup.md) - Installation and configuration
- \[<Integration> with vLLM\](<integration>_vllm.md) - vLLM-specific usage

## External Resources

- [<Integration> Documentation](https://...)
- [<Integration> GitHub](https://github.com/...)
