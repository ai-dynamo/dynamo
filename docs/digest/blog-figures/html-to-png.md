# HTML to PNG Pathway

The fifth pathway. Use when the figure is typography-heavy and a vector engine would fight you.

This is the only pathway with no in-corpus example, so this file gives you a working recipe end-to-end.

## When This Pathway Wins

- **Code blocks with callouts.** A code snippet rendered as a figure with arrows, highlights, and side annotations.
- **Comparison cards.** Side-by-side cards comparing two approaches, each with its own typography hierarchy.
- **Hero images with hierarchical type.** Big title, smaller subtitle, three callouts arranged in a grid.
- **"Card stack" mocks.** Visual representations of nested or layered metadata where CSS grid/flexbox is the most expressive layout engine.
- **Tables that need rich typography.** Fixed-width comparisons where columns mix headers, code, and prose.

## When It Loses

- Anything with precise vector geometry (gates, custom dividers, layered shapes). Use hand-crafted SVG.
- Anything data-bound (charts, distributions). Use Python + Plotly.
- Architecture diagrams. Use D2 or dynamo-svg.

## `figure.html` Template

Single-file HTML. CSS variables mirror the canonical [`design_tokens.yaml`](../flash-indexer/tools/design_tokens.yaml). Fixed canvas size for deterministic PNG export.

```html
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<title>Figure</title>
<style>
  :root {
    /* From design_tokens.yaml -- mirror, do not invent */
    --bg-primary: #000000;
    --bg-surface: #1a1a1a;
    --bg-surface-alt: #2a2a2a;
    --border-frame: #76b900;
    --border-subtle: #3a3a3a;
    --text-primary: #ffffff;
    --text-secondary: #cdcdcd;
    --text-muted: #767676;
    --accent-green: #76b900;
    --accent-cpu-blue: #0071c5;
    --accent-fluorite: #fac200;
    --accent-emerald: #008564;
    --accent-amethyst: #5d1682;
    --accent-coral: #b04040;

    --font-sans: Arial, Helvetica, sans-serif;
    --font-mono: 'Roboto Mono', 'SF Mono', Menlo, Consolas, monospace;
  }

  html, body {
    margin: 0;
    padding: 0;
    background: var(--bg-primary);
    color: var(--text-primary);
    font-family: var(--font-sans);
    font-size: 12px;
    line-height: 1.4;
  }

  /* The .figure element is what Playwright screenshots. */
  .figure {
    width: 1200px;       /* Fixed canvas width for determinism */
    padding: 32px;
    background: var(--bg-primary);
    box-sizing: border-box;
  }

  h1 {
    font-size: 18px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin: 0 0 24px 0;
    color: var(--text-primary);
  }

  h2 {
    font-size: 14px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin: 0 0 8px 0;
    color: var(--text-secondary);
  }

  code, .mono {
    font-family: var(--font-mono);
    font-size: 11px;
    color: var(--text-primary);
  }

  /* Cards example -- replace with whatever the figure needs */
  .grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
  }

  .card {
    background: var(--bg-surface);
    border: 1px solid var(--border-subtle);
    border-radius: 0;        /* Iron rule */
    padding: 16px;
    opacity: 1;              /* Surfaces are not muted; only accent fills are */
  }

  .card.accent {
    border-color: var(--accent-green);
  }

  .card .label {
    font-size: 10px;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 8px;
  }

  .card .value {
    font-family: var(--font-mono);
    font-size: 24px;
    color: var(--text-primary);
  }

  .card.accent .value {
    color: var(--accent-green);
  }
</style>
</head>
<body>
  <div class="figure">
    <h1>Two Approaches at a Glance</h1>
    <div class="grid">
      <div class="card">
        <div class="label">Naive Nested Map</div>
        <div class="value mono">4M ops/s</div>
        <p>Single-threaded, single-writer-single-reader. Simple, correct, slow.</p>
      </div>
      <div class="card accent">
        <div class="label">Concurrent Positional Indexer</div>
        <div class="value mono">170M ops/s</div>
        <p>Multi-writer multi-reader with positional jump search. 40x faster.</p>
      </div>
    </div>
  </div>
</body>
</html>
```

The `.figure` element gets screenshotted. Anything outside it is ignored. Set its width once and lay everything out within.

## Playwright Recipe

The recommended path. Full CSS support, deterministic, runs locally and in CI.

Install: `pip install playwright && playwright install chromium`.

```python
from pathlib import Path
from playwright.sync_api import sync_playwright

HTML = Path(__file__).parent / "fig-1-two-approaches.html"
PNG  = Path(__file__).parent.parent / "images" / "fig-1-two-approaches.png"

with sync_playwright() as p:
    browser = p.chromium.launch()
    page = browser.new_page(
        viewport={"width": 1200, "height": 800},
        device_scale_factor=2,        # 2x retina render
    )
    page.goto(f"file://{HTML.resolve()}")
    page.locator(".figure").screenshot(path=str(PNG), omit_background=False)
    browser.close()
```

Six lines of work. Wrap it in a `gen_*.py` script next to the HTML, drop the call into `tools/build.sh`.

**Notes:**

- `device_scale_factor=2` gives a 2x render (matches `rsvg-convert -z 2` for SVG sources).
- `omit_background=False` keeps the dark background opaque. Never `True`.
- `page.locator(".figure").screenshot(...)` crops to the figure element, ignoring viewport whitespace.
- The viewport `height` should be ≥ the rendered figure height; Playwright auto-extends if needed when locator-screenshotting.

## Satori Alternative

When you need batch generation or Playwright dependencies are not available, [Satori](https://github.com/vercel/satori) renders JSX → SVG with a strict CSS subset.

Pros:

- Faster (no Chromium boot).
- Deterministic output.
- No browser dependencies.

Cons:

- Strict CSS subset: no flexbox-grid, no filters, no `position: absolute` for many use cases.
- Requires Node.js or a Satori binary.
- Output is SVG; convert to PNG with `rsvg-convert -z 2` or `sharp`.

Recipe:

```javascript
// gen_fig.js -- Node.js
import satori from 'satori';
import { Resvg } from '@resvg/resvg-js';
import fs from 'fs';

const svg = await satori(
  {
    type: 'div',
    props: {
      style: {
        display: 'flex', flexDirection: 'column',
        background: '#000000', color: '#ffffff',
        fontFamily: 'Arial', padding: 32, width: 1200,
      },
      children: [
        { type: 'h1', props: { style: { fontSize: 18, letterSpacing: '0.08em', textTransform: 'uppercase' }, children: 'Two Approaches' } },
        // ...
      ],
    },
  },
  {
    width: 1200, height: 800,
    fonts: [{ name: 'Arial', data: fs.readFileSync('Arial.ttf'), weight: 400, style: 'normal' }],
  },
);

const png = new Resvg(svg, { fitTo: { mode: 'zoom', value: 2 } }).render().asPng();
fs.writeFileSync('../images/fig-1.png', png);
```

Use Satori for hero cards and stat blocks where the layout is simple. Use Playwright when the layout needs full CSS (grid templates, complex filters, fonts via `@font-face`).

## Decision: Playwright vs Satori

- **Default to Playwright** for design fidelity. The full CSS support means the HTML you author at `figure.html` renders the same way everywhere.
- **Use Satori** when generating dozens of figures in a batch (e.g., per-customer cards) or when CI cannot install Chromium.
- **Never mix** within one blog. Pick one and use it consistently across all HTML→PNG figures in that blog.

## Output Convention

- HTML source lives in `tools/`: `tools/fig-N-name.html`.
- PNG output lives in `images/`: `images/fig-N-name.png`.
- The HTML *is* the source. There is no SVG intermediate (unlike D2 or hand-crafted SVG).
- If you also need an SVG (for vector embedding in Fern), use Satori → SVG → save to `images/`.

## In the build.sh

Add one line per HTML→PNG figure:

```bash
echo "==> Figure N (cards)..."
python3 gen_fig_n_cards.py
```

Where `gen_fig_n_cards.py` contains the Playwright snippet above.

## Self-Check (HTML→PNG-Specific)

Beyond the seven non-negotiables in [SKILL.md](SKILL.md), confirm:

- [ ] CSS variables match `design_tokens.yaml`. No raw hex in CSS outside the `:root` block.
- [ ] `border-radius: 0` on every styled element. (Default browser rounding on inputs/buttons does not apply because there are none.)
- [ ] Two font families maximum (`var(--font-sans)`, `var(--font-mono)`).
- [ ] Title uses `text-transform: uppercase` with `letter-spacing: 0.08em`.
- [ ] Body has `background: var(--bg-primary)` (not transparent, not `inherit`).
- [ ] Figure width is fixed in pixels (not `100%`, not `vw`). Determinism.
