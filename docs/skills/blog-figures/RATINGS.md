# Dynamo Dark Figure Rating (0–100)

A figure's quality is scored 0–100 across five dimensions. Some dimensions
are **measured** mechanically from the generator source + `design_tokens.yaml`
by [`tools/lint_figures.py`](tools/lint_figures.py); others require **eyes on
the rendered pixels** and stay as a manual LLM-critique checklist. The two
are kept strictly separate — the script never fabricates a number for a
dimension it cannot actually measure.

**Ship gate:** the **measured** score must be **≥ 85** *and* every **judged**
item must pass. A figure that scores 100 mechanically can still be cut for a
weak takeaway or overloaded palette — that is the judged half's job.

Run it:

```bash
python3 tools/lint_figures.py --score            # human report
python3 tools/lint_figures.py --score --json     # machine-readable
```

## The Five Dimensions

| Dimension | Weight | How it's assessed | What it checks |
|---|---|---|---|
| **Palette Compliance** | 30 | Measured | Every color literal in the source is a `design_tokens.yaml` value (or an annotated `# lint-allow-hex` override). Raw off-palette hex fails. |
| **Typography Compliance** | 25 | Measured | Every font family is in the one Dynamo Dark set (Helvetica Neue / Helvetica / Arial + Roboto Mono). Geist, Inter, a display serif, or Comic Sans fail; weights sit in {300,400,500,600,700}. |
| **Contrast** | 25 | Measured | The token text-on-surface pairs the aesthetic uses meet WCAG AA (≥ 4.5:1), including the in-bar label-on-green pair. |
| **Data-Ink / Label Coverage** | 10 | **Proxy measured + judged** | *Measured proxy:* each generator emits a title + axis/annotation/label structure. *Judged:* whether every mark earns its place, redundant ink is erased, and the takeaway is directly labeled. |
| **Variety** | 10 | **Measured (light) + judged** | *Measured:* no single figure hard-codes more than four distinct accent colors. *Judged:* a chart uses ≤ 2 semantic accents; a diagram may use more role colors — context the script can't infer. |

The **measured overall** is the weighted sum of the mechanical signals
(Palette 30, Typography 25, Contrast 25, Variety 10, Label-structure 10).

## Measured vs Judged (be honest about which is which)

**Measured** (computed by the linter, deterministic):

- Palette compliance % — hexes that are tokens ÷ total hexes.
- Typography compliance % — allowed font families ÷ total families.
- Contrast pass-rate — token text/surface pairs meeting AA.
- Variety — distinct accent colors per figure (light signal).
- Label structure — generators that emit a title + labels (a *proxy* for
  data-ink coverage, not the real thing).

**Judged** (walk these against the rendered PNG — the linter prints them but
scores none):

- **Data-ink discipline** — every mark carries information; no chartjunk,
  no redundant encoding, no decorative gridlines.
- **Single-accent semantics** — green marks the one "winning" item; role
  colors are consistent across the blog.
- **Composition & geometry** — arrows land on card edges, columns share an
  anchor, legends are centered, title is top-left.
- **Before/after honesty** — a BEFORE panel shows the real pre-state, not a
  dimmed AFTER.
- **Title carries the takeaway** — the title is a declarative statement, not
  a category name; hero titles use the Helvetica-set display treatment,
  compact charts use the Arial ALL-CAPS treatment.

## Score-to-Fix Guide

| Low dimension | Fix |
|---|---|
| Palette Compliance | Replace raw hex with a `design_tokens.yaml` token; if it is a deliberate per-script override (e.g. brightening amethyst on black), annotate the line with `# lint-allow-hex` and a reason. |
| Typography Compliance | Drop any non-Dynamo-Dark font. Use the token sans (`Arial, Helvetica, sans-serif`, with Helvetica Neue leading for display titles) and mono (`Roboto Mono, ...`). No second family. |
| Contrast | Move low-contrast text onto the black ground, or switch in-bar labels to `#000000` on green fills / `#ffffff` on dark fills. |
| Data-Ink / Label Coverage | Add a title and direct labels; erase redundant ink, gridlines, and chart borders. Then re-judge the rendered figure. |
| Variety | If a chart carries > 2 semantic accents, drop back to greys + one green. Diagrams may keep role colors — that is a judged exception. |

## Relationship to the Non-Negotiables

The seven non-negotiables in [`SKILL.md`](SKILL.md) are the pass/fail floor;
this rubric is the graded scorecard on top of them. The linter mechanizes the
color, typography, and contrast floor so the render-and-critique loop can spend
its attention on the judged dimensions that actually need a human (or an LLM)
looking at the picture.
