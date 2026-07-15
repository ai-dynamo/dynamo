# fig-1 — ModelExpress Cold-Start Sequence

Dynamo Dark remake of the ModelExpress cold-start reference (a Mermaid-style
sequence diagram). Rethought — not a pixel copy — into a clean, two-plane
sequence diagram: the **metadata plane** (grey, routed through the ModelExpress
server) versus the **data plane** (green, peer-to-peer RDMA that bypasses the
server entirely).

## Figure Inventory

| File | Description | Title treatment |
|---|---|---|
| `gen_fig_1_coldstart_sequence.py` | Sequence diagram: source registers → target discovers → RDMA weight transfer, GPU to GPU | Display (Helvetica Neue Light) |
| `images/fig-1-modelexpress-coldstart.png` | Rendered PNG (1600 px wide hero, `scale=2`) | — |

## Single Takeaway

During a ModelExpress cold start, weight bytes move directly GPU-to-GPU over
RDMA between replicas while the server only ever exchanges metadata — every
metadata message lands on the server; the green weight transfer passes clean
through it.

## Structure Reproduced (from the reference)

Three lifelines: **Source replica** (already serving) · **ModelExpress server**
(metadata only) · **New replica** (target). Nine messages across three phases:

1. **Register** — `Load + post-process weights, register GPU memory with NIXL`
   (self) · `PublishMetadata(identity) → mx_source_id` · loop
   `UpdateStatus(READY)`
2. **Discover** — `ListSources(mx_source_id, READY) · own rank` →
   `candidate source workers` · `GetMetadata(worker)` →
   `tensor descriptors + NIXL connection info`
3. **Transfer** — `RDMA read of weights · GPU to GPU` (the single green accent,
   bypassing the server) · `PublishMetadata() · target becomes a new source`

## Design Notes

- **Two accents only.** `cpu_blue` marks the metadata-only server (its card
  border + dashed control spine); `dynamo_green` marks the single peer-to-peer
  data path. Everything else is neutral grey.
- **Bypass, drawn structurally.** Metadata messages terminate on the server
  lifeline with a contact node; the green RDMA edge crosses the server spine
  with no node — the visual argument that the bytes never touch the server.
- All labels are protocol/API terms from the reference. No invented numbers.

## Prerequisites

```bash
python3 -m venv .venv
./.venv/bin/python -m pip install plotly kaleido pyyaml
```

`design_tokens.yaml` and `plotly_dynamo.py` are copied verbatim from the
canonical `docs/skills/blog-figures/examples/` (do not fork the values).

## Reproduce

```bash
./build.sh          # render + lint
# or just render:
./.venv/bin/python gen_fig_1_coldstart_sequence.py
```

## Lint

```bash
../../../docs/skills/blog-figures/tools/lint_figures.py --score \
  --tokens ./design_tokens.yaml .
```
