---
name: disagg-trace
description: Render a self-contained trace.html visualizing the kvbm_audit timeline from a disagg experiment dir. Three lanes (Decode | Hub | Prefill), one row per audit event, sidebar filters by request_id.
---

# Skill: KVBM Disagg Trace Viewer

Renders an HTML timeline from a `disagg-bringup` / `disagg-smoke` experiment dir. Reads `hub.log`, `prefill.log`, `decode.log` from the directory and emits `trace.html` in the same dir.

## Skill assets

| File | Purpose |
|---|---|
| `cd-trace.py` | Parses `kvbm_audit` events from the three logs, builds a 3-lane timeline, emits self-contained HTML. |

## Usage

```bash
SKILL=/home/ryan/repos/dynamo/.claude/skills/disagg-trace
python3 "$SKILL/cd-trace.py" /tmp/kvbm-experiments/<ts>-<label>/
```

Open the resulting `trace.html` in a browser. Click any `request_id` in the sidebar to filter the timeline to that request.

## See also

- `/disagg-smoke` — the smoke harness that auto-invokes this renderer at the end of each run.
