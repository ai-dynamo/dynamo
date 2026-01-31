# MCP compute-session Bug Analysis

**Date:** 2026-02-06
**Source location:** `/Users/mkosec/work/tools/compute-session/`
**MCP config:** `~/.claude/mcp.json` -> `node /Users/mkosec/work/tools/compute-session/index.js`
**No git repo** -- standalone tool, not version controlled.

---

## Architecture Overview

The MCP compute-session server is a Node.js MCP server (using `@modelcontextprotocol/sdk` v1.12.1) that manages GPU and build sessions. Key files:

| File | Purpose |
|------|---------|
| `index.js` | MCP server entry point, tool definitions, request routing |
| `lib/session.js` | `startSession()`, `stopSession()`, `validateSession()` -- core lifecycle |
| `lib/ssh.js` | `sshExec()` -- runs commands on remote hosts via `execSync()` |
| `lib/presets.js` | GPU_PRESETS and BUILD_PRESETS -- hardcoded cluster/partition mappings |
| `lib/config.js` | Loads YAML cluster configs from `configs/*.yaml` |
| `lib/executor.js` | `executeCommand()` -- sends commands into tmux sessions with marker-based output capture |
| `lib/state.js` | Persists session state to `~/.compute-sessions/state/sessions.json` |
| `configs/computelab.yaml` | Computelab cluster config |

### Session Start Flow (GPU)

1. `sshPreflight()` -- verifies SSH connectivity to login node
2. Build `salloc --no-shell --time=T --partition=P --gpus-per-node=N --job-name=cs-NAME` command
3. `sshExec(loginNode, sallocCmd)` -- runs salloc via SSH
4. Parse `combinedOutput` (stdout + stderr) for "Granted job allocation NNNNN"
5. Parse node name from "Nodes XXXXX are ready for job" or fall back to `squeue -j JID`
6. SSH to compute node, create tmux session
7. Save state, optionally launch container via srun

---

## Bug 1: salloc Succeeds but MCP Reports Failure

### Root Cause: `sshExec()` mishandles salloc's exit code and output streams

**File:** `lib/ssh.js`, lines in `sshExec()` function.

The core issue is in how `sshExec()` handles `execSync()`:

```javascript
try {
  const result = execSync(fullCmd, {
    timeout,
    encoding: 'utf8',
    stdio: ['pipe', 'pipe', 'pipe'],
    shell: true,
  });
  stdout = result || '';
} catch (e) {
  stdout = e.stdout || '';
  stderr = e.stderr || '';
  exitCode = e.status || 1;
}
```

**Problem:** `salloc` writes its output to **stderr**, not stdout. The key messages like:
```
salloc: Granted job allocation 12345
salloc: Nodes ipp1-0787 are ready for job
```
are all written to stderr.

When `salloc` completes, it may return a **non-zero exit code** or the SSH command may report a non-zero exit because salloc's `--no-shell` mode behavior varies. In Node.js `execSync()`, any non-zero exit code throws an exception. In the catch block:
- `e.stdout` is empty (salloc doesn't write to stdout)
- `e.stderr` contains the actual "Granted job allocation" message
- `e.status` may be non-zero

Then in `session.js` the parsing code does:

```javascript
const combinedOutput = (sallocResult.stdout + '\n' + sallocResult.stderr).trim();
const jobMatch = combinedOutput.match(/Granted job allocation (\d+)/);
```

This **should** work because it combines stdout and stderr. But there's a subtle issue: **when `salloc --no-shell` succeeds, it may not actually exit immediately**. The `salloc` command with `--no-shell` on some Slurm configurations:

1. Prints "Granted job allocation" to stderr
2. Then **blocks** waiting, or exits with code 0 only after the allocation is released

If `salloc --no-shell` blocks (which is the documented behavior -- it holds the allocation until you cancel it or the time expires), then `execSync()` will **time out** after 60 seconds. When it times out:
- Node.js throws an error with `e.killed = true`
- `e.stdout` is whatever was captured (likely empty)
- `e.stderr` may be **empty or partial** because the process was killed mid-stream
- `e.status` is null (killed, not exited)

This is the most likely root cause: **`salloc --no-shell` blocks, `execSync` kills it after the timeout, and the captured stderr is empty or incomplete**, so the regex match fails.

**Supporting evidence from the bug report:**
- "salloc output: [empty or minimal]" -- consistent with timeout-killed process
- Jobs DO get allocated (they exist in squeue) -- salloc DID succeed
- Orphaned jobs pile up -- the allocation was granted but the MCP doesn't know

### Additional Factor: SSH `BatchMode=yes`

The `sshExec()` function adds `-o BatchMode=yes` to all SSH connections. This disables interactive prompts, which is correct. However, it also means that if the SSH connection is established but salloc blocks, the entire `execSync` call blocks until the 60-second timeout.

### Fix Proposal

**Option A (Recommended): Use `execFile`/`spawn` with streaming + early exit on match**

Replace `execSync` in `sshExec()` with `spawnSync` and set up a custom timeout that monitors stderr for the "Granted job allocation" pattern. Once the pattern is matched, kill the salloc process (the allocation persists because `--no-shell` decouples it):

```javascript
// In ssh.js - new function specifically for salloc
export function sshExecSalloc(host, sallocCmd, opts = {}) {
  const timeout = opts.timeout || 60000;
  const sshArgs = buildSshArgs({ host, ...opts });

  const result = spawnSync('ssh', [...sshArgs, sallocCmd], {
    timeout,
    encoding: 'utf8',
    stdio: ['pipe', 'pipe', 'pipe'],
  });

  // spawnSync captures ALL output even on timeout/kill
  return {
    stdout: (result.stdout || '').trim(),
    stderr: (result.stderr || '').trim(),
    exitCode: result.status ?? (result.signal ? 0 : 1),  // treat signal-killed as OK for salloc
  };
}
```

**Key difference:** `spawnSync` (without `shell: true`) captures output more reliably before killing, and we can treat a signal-killed salloc as success if we got the "Granted" message.

**Option B (Simpler): Background salloc + poll squeue**

Instead of parsing salloc output at all:

```javascript
// Run salloc in background, then poll squeue for the job
sshExec(loginNode, `salloc --no-shell --time=${time} ... &`, sshOpts);
// Wait a few seconds
sleepSync(5000);
// Query squeue for our job
const squeueResult = sshExec(loginNode, `squeue -u $(whoami) -n cs-${sessionName} -h -o "%i %N %T"`, sshOpts);
```

**Option C (Minimal fix): Increase timeout and use spawnSync instead of shell: true**

The `shell: true` option in `execSync` adds another layer of process management that can interfere with output capture on kill. Switching to `spawnSync` without shell may fix the output capture issue:

```javascript
import { spawnSync } from 'child_process';

export function sshExec(host, command, opts = {}) {
  const timeout = opts.timeout || 30000;
  const sshArgs = buildSshArgs({ host, ...opts });
  sshArgs.push(command);

  const result = spawnSync('ssh', sshArgs, {
    timeout,
    encoding: 'utf8',
    stdio: ['pipe', 'pipe', 'pipe'],
  });

  return {
    stdout: (result.stdout || '').trim(),
    stderr: (result.stderr || '').trim(),
    exitCode: result.status ?? 1,
  };
}
```

Also, in `session.js`, after the salloc call, accept a killed-by-timeout result if the combined output contains the grant message:

```javascript
const sallocResult = sshExec(loginNode, sallocCmd, { ...sshOpts, timeout: 90000 });
const combinedOutput = (sallocResult.stdout + '\n' + sallocResult.stderr).trim();

const jobMatch = combinedOutput.match(/Granted job allocation (\d+)/);
if (!jobMatch) {
  // Also check if salloc might have succeeded but we lost the output
  // by querying squeue directly
  const squeueResult = sshExec(loginNode,
    `squeue -u $(whoami) -n cs-${sessionName} -h -o "%i %N"`, sshOpts);
  if (squeueResult.stdout.trim()) {
    const parts = squeueResult.stdout.trim().split(/\s+/);
    jobId = parts[0];
    node = parts[1];
  } else {
    throw new Error(`Failed to allocate SLURM job.\nsalloc output: ${combinedOutput}`);
  }
}
```

---

## Bug 2: Wrong Partition Names in Presets

### Root Cause: Hardcoded partition names don't match computelab's actual Slurm config

**File:** `lib/presets.js`

The presets have these partition values for computelab:

| Preset | Partition in Code | Real Partition on Computelab |
|--------|------------------|------------------------------|
| `a100` | `mlperf-training` | `a100-80gb-pcie@cr+mp/h12sswnt/1gpu-16cpu-128gb` (or similar) |
| `cl-h100` | `all` | `h100-80gb-hbm3@ts6/mg62g4100/1gpu-32cpu-256gb` (or similar) |
| `cl-b200` | `b200@ts-a01p-1000W/umbriel-b200@ts5/8gpu-224cpu-2048gb` | This one looks correct |
| `cpu-only` | `all` | Unknown -- needs `sinfo` to find CPU-only partition |

The `computelab.yaml` config sets `default_partition: batch` which is also likely wrong for computelab-sc-01. These partition names look like they come from a different Slurm cluster.

### Fix

Run `sinfo -o "%P"` on computelab-sc-01 to get the actual partition list, then update `presets.js`:

```javascript
'a100': {
  type: 'gpu',
  cluster: 'computelab',
  description: '4x A100 on Computelab (ARM64)',
  gpus: 4,
  partition: 'a100-80gb-pcie@cr+mp/h12sswnt/1gpu-16cpu-128gb',  // FIXME: verify exact name
  constraint: 'aarch64',  // NOTE: see Bug 3 about constraint conflicts
},
'cl-h100': {
  type: 'gpu',
  cluster: 'computelab',
  description: '8x H100 on Computelab',
  gpus: 8,
  partition: 'h100-80gb-hbm3@ts6/mg62g4100/1gpu-32cpu-256gb',  // FIXME: verify exact name
},
```

Also update `computelab.yaml`:
```yaml
slurm:
  default_partition: null  # No sensible default for computelab -- must specify in preset
```

---

## Bug 3: Partition Override + Preset Conflicts

### Root Cause: `--constraint` flag from preset conflicts with partition-encoded constraints

**File:** `lib/session.js`, in `startGpuSession()`:

```javascript
if (preset.constraint) {
  sallocCmd += ` --constraint=${preset.constraint}`;
}
```

The `a100` preset has `constraint: 'aarch64'`. When you override the partition but the constraint is still applied, you get:

```
salloc --no-shell --partition=<user-override> --gpus-per-node=4 --constraint=aarch64
```

On computelab, the partition names like `a100-80gb-pcie@cr+mp/h12sswnt/1gpu-16cpu-128gb` already encode the hardware type. Adding `--constraint=aarch64` on top of that causes "Invalid feature specification" because:
1. The partition already constrains to A100 nodes
2. The `aarch64` feature may not be a valid Slurm feature on this cluster
3. The partition's node set may not have the `aarch64` feature tag

### Fix

Either:
1. Remove the `constraint` field from presets that target computelab (since partition names already encode the hardware)
2. Or add logic to skip constraint when partition is overridden:

```javascript
// Only apply constraint if using the preset's own partition (not an override)
if (preset.constraint && !opts.partition) {
  sallocCmd += ` --constraint=${preset.constraint}`;
}
```

---

## Bug 4: Preset is Required (No Custom Mode)

### Root Cause: `getPreset()` returns null for undefined, and `startSession()` throws immediately

**File:** `lib/session.js`:

```javascript
export function startSession(opts) {
  const preset = getPreset(opts.preset);
  if (!preset) {
    throw new Error(`Unknown preset "${opts.preset}". Use compute_session_presets to see available presets.`);
  }
  // ...
}
```

When no preset is passed, `opts.preset` is `undefined`, so `getPreset(undefined)` returns `null`, and the error message says `Unknown preset "undefined"`.

### Fix

Add a custom/raw mode that allows specifying cluster + partition + gpus without a preset:

```javascript
export function startSession(opts) {
  let preset;

  if (opts.preset) {
    preset = getPreset(opts.preset);
    if (!preset) {
      throw new Error(`Unknown preset "${opts.preset}".`);
    }
  } else {
    // Custom mode: require cluster + partition
    if (!opts.cluster) {
      throw new Error('Either "preset" or "cluster" is required.');
    }
    preset = {
      type: 'gpu',
      cluster: opts.cluster,
      gpus: opts.gpus || 1,
      partition: opts.partition || null,
      description: 'custom',
    };
  }
  // ...
}
```

---

## Bug 5 (Bonus): `execSync` with `shell: true` Quoting Issues

**File:** `lib/ssh.js`, in `sshExec()`:

```javascript
const fullCmd = ['ssh', ...sshArgs, command].map(arg => {
  if (arg.includes(' ') && !arg.startsWith('-o')) {
    return `"${arg}"`;
  }
  return arg;
}).join(' ');

const result = execSync(fullCmd, {
  // ...
  shell: true,
});
```

Using `shell: true` with manually-quoted strings is fragile. The command goes through the local shell, which interprets quotes, dollar signs, backticks, etc. If a partition name or other argument contains shell metacharacters (like `@`, `(`, `)`), the local shell may misinterpret them.

Computelab partition names contain `@` and `/` characters: `b200@ts-a01p-1000W/umbriel-b200@ts5/8gpu-224cpu-2048gb`. While `@` and `/` are generally safe in shell, this pattern is still risky.

### Fix

Use `execFileSync` or `spawnSync` without `shell: true`:

```javascript
import { spawnSync } from 'child_process';

export function sshExec(host, command, opts = {}) {
  const sshArgs = buildSshArgs({ host, ...opts });
  sshArgs.push(command);  // command is passed as a single arg to SSH, not through local shell

  const result = spawnSync('ssh', sshArgs, {
    timeout: opts.timeout || 30000,
    encoding: 'utf8',
    stdio: ['pipe', 'pipe', 'pipe'],
  });

  return {
    stdout: (result.stdout || '').trim(),
    stderr: (result.stderr || '').trim(),
    exitCode: result.status ?? 1,
  };
}
```

This avoids local shell interpretation entirely. SSH receives the command as a single argument and runs it through the remote shell, which is the correct behavior.

---

## Summary of Fixes by Priority

### P0 -- Fixes the orphaned-jobs problem

1. **Replace `execSync` with `spawnSync`** in `sshExec()` to properly capture stderr on timeout/kill. This is the root cause of Bug 1.
2. **Add squeue fallback** in `startGpuSession()` -- if salloc output parsing fails, query squeue for the job by name before giving up.
3. **Increase salloc timeout** from 60s to 120s to give salloc more time on busy clusters.

### P1 -- Fixes non-working presets

4. **Update partition names** in `presets.js` for computelab presets (`a100`, `cl-h100`, `cpu-only`).
5. **Skip `--constraint` when partition is overridden** to avoid conflicts (Bug 3).
6. **Update `computelab.yaml`** default_partition to something valid or null.

### P2 -- Usability improvements

7. **Add custom/presetless mode** to `startSession()` (Bug 4).
8. **Remove `shell: true`** from `sshExec()` to avoid quoting issues (Bug 5).

---

## Testing the Fix

To verify the salloc parsing fix without burning GPU time:

```bash
# Simulate what sshExec does:
node -e "
const { spawnSync } = require('child_process');
const r = spawnSync('ssh', [
  '-o', 'BatchMode=yes',
  '-o', 'ConnectTimeout=10',
  '-o', 'StrictHostKeyChecking=no',
  'computelab-sc-01',
  'salloc --no-shell --time=0:05:00 --partition=PARTITION_NAME --gpus-per-node=1 --job-name=test-parse'
], { timeout: 30000, encoding: 'utf8' });
console.log('status:', r.status);
console.log('signal:', r.signal);
console.log('stdout:', JSON.stringify(r.stdout));
console.log('stderr:', JSON.stringify(r.stderr));
"
```

Check whether `r.stderr` contains "Granted job allocation" when using `spawnSync` vs `execSync + shell: true`.

After verifying, cancel the test job: `ssh computelab-sc-01 scancel <jobid>`
