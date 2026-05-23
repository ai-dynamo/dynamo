---
name: disagg-teardown
description: Cleanly kill all running KVBM disagg processes — vllm api_server, EngineCore subprocesses, kvbm_hub — and report GPU + memory state.
---

# Skill: KVBM Disagg Teardown

Stop everything `/disagg-bringup` started, leaving a clean GPU and no
orphaned processes.

## When to use

- Before re-running `/disagg-bringup` after a code change.
- When `pkill -f vllm` left zombies because the api_server was killed
  before its EngineCore children could shut down.
- Whenever `nvidia-smi` shows lingering `VLLM::EngineCore` processes.

## Workflow

```bash
# 1. SIGTERM the api_server processes — they should signal their EngineCore children.
pkill -f "vllm.entrypoints.openai" 2>/dev/null
pkill -f "kvbm_hub" 2>/dev/null
sleep 3

# 2. Hard-kill any EngineCore that didn't reap. Look them up in nvidia-smi
#    rather than ps because the api_server kill may have already removed
#    the pgrep-able command line.
ENGINE_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | grep -v '^$')
if [ -n "$ENGINE_PIDS" ]; then
    echo "$ENGINE_PIDS" | xargs -r kill -9 2>/dev/null
    sleep 2
fi

# 3. Hard-kill any leftover kvbm_hub
pkill -9 -f "kvbm_hub" 2>/dev/null
sleep 1

# 4. Report
echo "=== nvidia-smi compute apps ==="
nvidia-smi --query-compute-apps=pid,process_name --format=csv 2>&1
echo "=== still alive (vllm/kvbm_hub) ==="
ps aux | grep -E "vllm|kvbm_hub" | grep -v grep | awk '{print $2, $11, $12}'
echo "=== free ==="
free -h | head -3
```

## Notes

- **Defunct processes** (`<defunct>`) are zombies waiting for their parent
  to reap. They release no resources and disappear when the parent (often
  the Claude Code shell) exits. Don't attempt to `kill -9 <zombie_pid>`.
- **GB10 unified memory**: `nvidia-smi` reports `Memory-Usage: Not Supported`,
  so freeing GPU is observed via `free -h` showing system RAM recovery and
  via `nvidia-smi --query-compute-apps` going empty.
- This skill **does not** touch the canonical venv at
  `/home/ryan/.venvs/dynamo-kvbm` — that's persistent.

## See also

- `/disagg-bringup` — what this tears down
- `/disagg-hub-curl` — verifying the hub is gone
