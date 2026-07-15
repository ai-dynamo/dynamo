# fig-2 — ModelExpress RL Training Loop

Dynamo Dark remake of the ModelExpress / cold-start reinforcement-learning
loop figure.

| File | Description |
|---|---|
| `gen_fig_2_rl_loop.py` | Generator: 2x2 clockwise RL loop (Rollout → Reward → Trainer → Weight refit) with ModelExpress as the single green accent. Display/hero title treatment. |
| `build.sh` | One-shot rebuild. Renders `images/fig-2-rl-loop.{png,svg}`. |
| `design_tokens.yaml` | Canonical Dynamo Dark tokens (copied, not forked). |
| `plotly_dynamo.py` | Canonical Plotly template helper (copied, not forked). |
| `images/fig-2-rl-loop.png` | Rendered figure (1600×900, scale=2). |
| `images/fig-2-rl-loop.svg` | Vector render (for Fern / Confluence embedding). |

## Reproduce

```bash
cd drafts/modelexpress-coldstart/fig-2
./build.sh                      # uses .venv if present
```

Prerequisites (already installed in the local `.venv`): `plotly`, `kaleido`,
`pyyaml`.

## Design notes

- **Type:** diagram (4-node cyclic process loop), rethought — not a pixel copy.
- **Layout:** computed 2x2 grid so every connector is a straight horizontal or
  vertical segment landing on an exact box edge (no diagonals, no eyeballed
  endpoints).
- **Single accent:** the source highlighted Weight refit (ModelExpress) in
  blue; Dynamo Dark reserves green for the one hero item, so ModelExpress gets
  the green 1.5 px border and the green `[4] updated weights` edge that closes
  the loop. The other three stages are neutral (`#1a1a1a` fill, `#3a3a3a`
  hairline) with medium-grey forward edges.
- **Labels:** faithful to the source (step numbers `[1]`–`[4]`, edge payloads,
  stage sub-lines), with backend names alphabetized per house convention
  (SGLang / TRT-LLM / vLLM).

## Lint

```bash
python docs/skills/blog-figures/tools/lint_figures.py \
  drafts/modelexpress-coldstart/fig-2/ --score
```
