---
name: debug-session
description: Start a debugging session with worklog file
user-invocable: true
disable-model-invocation: true
---

# Start Debug Session

Create a structured debugging session for an issue in the Dynamo ecosystem.

## Step 1: Get the Bug Report

Ask the user how they want to provide the bug:

**Option A: Linear ticket**
- User provides ticket ID (e.g., "DYN-123")
- Fetch via Linear MCP tools
- Extract: title, description, reproduction steps

**Option B: GitHub issue**
- User provides issue URL
- Fetch via `gh issue view <url>`
- Extract: title, description, reproduction steps

**Option C: Paste**
- Ask user to paste the bug report directly
- Parse out the key details

## Step 2: Discover Environment

Gather environment information:

!`nvidia-smi --query-gpu=name,count --format=csv,noheader 2>/dev/null || echo "No GPU detected"`

!`uname -a`

!`which python && python --version`

This tells you:
- GPU type and count (L40s, H100s, etc.)
- OS/platform
- Python environment

**Note**: The user's `~/.claude/CLAUDE.md` may have more details about their dev environment (paths, aliases, preferences). Check there for additional context.

## Step 3: Create Worklog

Create a worklog file to track the investigation:

- Filename: `<issue-slug>.md` in current directory
- Template:

```markdown
# Debug: [Issue Title]

**Date**: [today's date]
**Source**: [Linear ticket / GitHub issue / user report]
**Status**: investigating
**Environment**: [GPU type/count from nvidia-smi]

## Problem
[Description of the issue]

## Reproduction Steps
1. [Step to reproduce]
2. ...

## Expected vs Actual
- **Expected**:
- **Actual**:

## Investigation Log

### [timestamp]
[Notes on what you tried/found]

## Root Cause
[Fill in when found]

## Fix
[Fill in when implemented]
```

## Step 4: Set Up Testing

### Build Commands

Rebuild Dynamo after making changes:
```bash
cd lib/bindings/python && maturin develop --uv && cd ../../.. && uv pip install -e .
```

If a framework change is required (sglang, vllm, trtllm), check the user's `~/.claude/CLAUDE.md` for rebuild instructions specific to that framework.

### Running Examples

Examples are located at: `/home/ubuntu/dynamo/examples/backends/`

Available backends:
- `sglang/launch/` - SGLang backend examples
- `vllm/launch/` - vLLM backend examples
- `trtllm/launch/` - TensorRT-LLM backend examples

Based on the bug report, determine which backend is relevant:
- If unclear, **ask the user** which backend/example to run
- Run the example in the background
- Wait for model to be ready

### Verifying the Model is Up

```bash
curl localhost:8000/v1/models
```

### Testing with a Request

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "<model-name-from-above>",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 50
  }'
```

## Step 5: Begin Investigation

### Dynamo Infrastructure Debugging

**KV cache and routing issues:**
- Check KV event logs in `lib/llm/src/block_manager/kv_consolidator/tracker.rs`
- Look at block manager state and consolidation behavior
- Inspect routing decisions in the KV-aware router

**ZMQ / networking issues:**
- Check ZMQ socket configuration and endpoint bindings
- Look for connection timeouts or message drops
- Verify nats/etcd connectivity for service discovery

**Multi-node / disaggregated issues:**
- Check prefill/decode worker assignment
- Verify DGD (disaggregated) status reporting
- Inspect inter-node communication via `nvidia-smi` on each node
- Check NCCL and GPU direct RDMA status

**Process inspection:**
- `ps aux | grep dynamo` - check running processes
- `nvidia-smi` - GPU utilization and memory
- `ss -tlnp | grep 8000` - check port bindings
- `journalctl -u dynamo` - systemd logs if applicable

### General Debugging Workflow

1. **Reproduce first** - verify you can trigger the bug before attempting fixes
2. **Document as you go** - update the worklog with findings
3. **Minimal changes** - fix the bug, do not refactor surrounding code
4. **Verify the fix** - confirm the reproduction case now passes

Performance-critical code - avoid unnecessary abstractions or comments.
