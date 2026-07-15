# fig-3 — ModelExpress Cold-Start Architecture

Dynamo Dark remake of the ModelExpress (MX) cold-start reference diagram.

## Figure

| File | Description |
|---|---|
| `gen_fig_3_modelexpress.py` | Generator — computed-geometry two-plane architecture diagram (Plotly shapes + traces). |
| `images/fig-3-modelexpress-coldstart.png` | 2x PNG render (1600 × 1000 canvas). |
| `images/fig-3-modelexpress-coldstart.svg` | Vector render (for Fern / Confluence embedding). |
| `design_tokens.yaml` | Canonical Dynamo Dark tokens (copied from the skill; do not fork). |
| `plotly_dynamo.py` | Canonical Plotly template helper (copied from the skill; do not fork). |
| `build.sh` | Reproducible one-shot build (bootstraps venv, renders, lints). |

## Type and takeaway

- **Type:** architecture / data-flow **diagram** (not a chart). Per the skill, a
  diagram is rethought into a clean Dynamo Dark composition rather than pixel-copied.
- **Takeaway:** *ModelExpress splits a lightweight control plane (metadata
  coordination) from a high-bandwidth data plane, so a newly-scheduled engine
  cold-starts by pulling weights over the fastest available GPU-direct path.*

## Structure reproduced from the reference

- **Control plane** (top band): `Inference Engine (Source)` and
  `Inference Engine (New)`, each hosting an `MX Client`; a central `MX Server`;
  and a `Metadata Store` (Redis / K8s backend). Thin grey wires carry the
  recessive control-plane links (store ↔ server ↔ clients).
- **Data plane** (bottom band): `Object Storage (Remote)` and
  `File Storage (Local / Network)` feed model weights **into the New engine** over
  three converging GPU-direct paths — **GPUDirect RDMA** (peer engine),
  **ModelStreamer** (object store), and **GPUDirect Storage / GDS** (file store).

## Dynamo Dark rethink

- Pure `#000000` ground; two recessive plane bands (`#1a1a1a`, `#3a3a3a` hairline,
  `layer="below"`) express the plane split structurally instead of via tinted
  backgrounds.
- **Green** (`#76b900`) is the single selective accent — reserved for the
  ModelExpress software components (MX Server + MX Clients), the subject.
- **cpu_blue** marks the control-plane Metadata Store; **fluorite** carries the
  data-plane weight-transfer flows. (Emerald was avoided next to NV green per
  DESIGN.md; the three roles are three distinct hues.)
- All connectors are orthogonal (right angles only); every coordinate is computed
  from named constants; the three data paths land on the New engine's bottom edge
  with arrowheads on exact edges.
- Display / hero title treatment (Helvetica Neue Light, title case) + muted
  em-dash subtitle.

## Prerequisites

`python3` (3.13) with `plotly`, `kaleido`, `pyyaml`. `build.sh` installs these into
a local `.venv` (via `uv` if available, else `python3 -m venv` + `pip`). kaleido
1.x renders through a headless Chromium it manages itself.

## Reproduction

```bash
./build.sh
# or, with an existing environment:
python3 gen_fig_3_modelexpress.py
```

## Lint

Lint the figure sources (not the whole dir, so the linter does not recurse into
`.venv`):

```bash
python3 ../../../docs/skills/blog-figures/tools/lint_figures.py \
  gen_fig_3_modelexpress.py plotly_dynamo.py --tokens design_tokens.yaml --score
```

Result: **0 errors, 0 warnings, 100.0 / 100 measured (SHIP-READY)**.
