<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Handoff: interactive enum-value chips (copy-on-click) — deferred

**Status:** researched, built, browser-verified in isolation, then **reverted**. The shipped state
uses static CSS-styled `<Badge>` chips (see [What shipped instead](#what-shipped-instead)). This doc
captures the interactive-component approach so it can be re-applied once a **Fern preview deploy** is
available to verify in-page rendering.

**Why deferred:** the interactive version needs a custom React component, and `fern docs dev` (local
preview) **cannot resolve `@/components/*` imports** — so the component can't be verified rendering
inside a real Fern page locally. Rather than ship an unverified client component, we kept the static
badges, which are close enough.

---

## Context: what these chips are

On the observability reference pages, string-typed fields (env vars, metric labels, operator labels)
list their accepted values. There are **11 such sites** across three pages:

- `docs/reference/observability/environment-variables.mdx` — 4 (`NIXL_TELEMETRY_ENABLE`,
  `VLLM_LOGGING_LEVEL`, `TLLM_LOG_LEVEL`, `DYN_SYSTEM_STARTING_HEALTH_STATUS`)
- `docs/reference/observability/metric-labels.mdx` — 5 (`stage`, `phase`, `worker_type`, `status`,
  `event_type`)
- `docs/reference/observability/operator-metrics.mdx` — 2 (`error_type`, `operation`)

We render them as `type="string"` (accurate to Go/K8s, where these are strings with an enum
constraint) plus an "Allowed values:" row of chips — deliberately **not** as a `'a' | 'b'` union in
the `type=` field.

---

## The goal: match Fern's own Schema enum chips

Fern's Schema component (the OpenAPI-backed one, e.g. the `model` field in their docs) renders enum
values as **grey chips that copy the value on click** and flip a tooltip to **"Copied"**, with a hover
effect. We wanted the same behavior.

**Why we can't just use Fern's Schema component:** `<Schema type="…">` only resolves types from a
registered **OpenAPI/API definition**, and only types **referenced by endpoints**. This docs project
registers **no API** (`fern/docs.yml` has no `api:` nav item, no `apis/` dir, no `generators.yml`). The
CRDs *do* ship real `openAPIV3Schema` with enum constraints, but wiring that into Fern means synthetic
fake-endpoint specs + hiding phantom nav pages — a separate, large spike. And it doesn't apply to env
vars / metric labels at all (those aren't types).

**Fern's actual Chip implementation** (extracted from their compiled bundle,
`app.buildwithfern.com/_next/static/chunks/…`, module `502604`):

```js
Chip = ({name, description, lang = "en"}) => {
  const {copyToClipboard, wasJustCopied} = useCopyToClipboard(name);
  const size = useContext(ChipSizeContext);
  return (
    <FernTooltip open={wasJustCopied || (description && undefined)}
                 content={wasJustCopied ? t(lang).buttons.copied : description}>
      <Badge onClick={() => copyToClipboard?.()} size={size}>{name}</Badge>
    </FernTooltip>
  );
};
```

Key facts learned from their compiled CSS/JS:
- The chip is the **same `.fern-docs-badge`** we use in MDX, just `size="sm"` with the **default
  subtle variant** (no color). Subtle = `background: var(--grayscale-a3)`, `color: var(--grayscale-a11)`,
  `border-radius: var(--radius-1)`, `font-size: var(--text-xs)`, `height: 1.25rem`.
- The "Allowed values:" label is `className="shrink-0 text-sm"` in `--grayscale-a11`.
- `<Badge>` gains an `.interactive` class (hover transition) and becomes a `<button>` when `onClick`
  is present.
- **MDX `<ParamField>` default** renders as a bare `<span>Defaults to {value}</span>` — the value is a
  bare text node, no `<code>`, no class. Only the **Schema** renderer wraps it as
  `<span class="fern-api-property-default">Defaults to <code>{value}</code></span>` (the green
  `Defaults to 3000` look). So a code-styled default is **unreachable** from MDX `<ParamField>`.

---

## The approach we built (and reverted): `EnumValues.tsx`

A custom **`"use client"`** React component (the repo's first client component) that replicates Fern's
Chip: an "Allowed values:" label + chips; each chip copies its value on click, shows a "Copied" state
for 1.5s, and has a hover effect. Styles ship **inside the component** as a `<style>` block (not in
`fern/main.css`) because the shared NVIDIA global theme **replaces the docs.yml `css:` at publish** —
the same reason `fern/components/RecipeStyles.tsx` injects its CSS that way.

Custom components are registered via `fern/docs.yml`:
```yaml
experimental:
  mdx-components:
    - ./components
```

### Full component source (`fern/components/EnumValues.tsx`)

```tsx
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * <EnumValues> — an "Allowed values:" row of copy-on-click chips, used on the
 * observability reference pages to enumerate the accepted values of a string
 * env var / metric label / operator field.
 */
"use client";

import { useCallback, useState } from "react";

const ENUM_CSS = `
.dyn-enum {
  display: inline-flex;
  flex-wrap: wrap;
  align-items: baseline;
  gap: 0.5rem;
}
.dyn-enum-label {
  flex-shrink: 0;
  color: var(--grayscale-a11, #6b7280);
  font-size: var(--text-sm, 0.875rem);
}
.dyn-enum-chip {
  display: inline-flex;
  align-items: center;
  height: 1.25rem;
  padding: 0 0.375rem;
  border: none;
  border-radius: var(--radius-1, 0.25rem);
  background-color: var(--grayscale-a3, rgba(0,0,0,0.06));
  color: var(--grayscale-a11, #6b7280);
  font-family: var(--font-code, ui-monospace, SFMono-Regular, Menlo, monospace);
  font-size: var(--text-xs, 0.75rem);
  font-weight: 500;
  line-height: 1rem;
  cursor: pointer;
  transition: background-color 0.15s ease-in-out, color 0.15s ease-in-out;
}
.dyn-enum-chip:hover {
  background-color: var(--grayscale-a4, rgba(0,0,0,0.09));
  color: var(--grayscale-a12, #111827);
}
.dyn-enum-chip.copied {
  background-color: var(--grayscale-a4, rgba(0,0,0,0.09));
  color: var(--grayscale-a12, #111827);
}
`;

function EnumChip({ value }: { value: string }) {
  const [copied, setCopied] = useState(false);

  const onClick = useCallback(() => {
    // navigator.clipboard needs a secure context; fall back to execCommand.
    const done = () => {
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1500);
    };
    try {
      if (navigator?.clipboard?.writeText) {
        navigator.clipboard.writeText(value).then(done).catch(done);
      } else {
        const ta = document.createElement("textarea");
        ta.value = value;
        ta.style.position = "fixed";
        ta.style.opacity = "0";
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
        done();
      }
    } catch {
      done();
    }
  }, [value]);

  return (
    <button
      type="button"
      className={copied ? "dyn-enum-chip copied" : "dyn-enum-chip"}
      onClick={onClick}
      title={copied ? "Copied" : `Copy "${value}"`}
      aria-label={copied ? "Copied" : `Copy ${value}`}
    >
      {value}
    </button>
  );
}

export const EnumValues = ({
  values,
  label = "Allowed values:",
}: {
  values: string[];
  label?: string;
}) => (
  <span className="dyn-enum">
    <style dangerouslySetInnerHTML={{ __html: ENUM_CSS }} />
    <span className="dyn-enum-label">{label}</span>
    {values.map((v) => (
      <EnumChip key={v} value={v} />
    ))}
  </span>
);
```

### How it's used in `.mdx`

Add the import once, below the frontmatter (ambient use renders "Unsupported JSX tag"):
```mdx
import { EnumValues } from "@/components/EnumValues";
```
Then at each site, inside the `<ParamField>` body:
```mdx
<EnumValues values={["y", "n"]} />
```

The 11 value lists:
| File | Field | values |
|---|---|---|
| environment-variables | `NIXL_TELEMETRY_ENABLE` | `["y", "n"]` |
| environment-variables | `VLLM_LOGGING_LEVEL` | `["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]` |
| environment-variables | `TLLM_LOG_LEVEL` | `["TRACE", "DEBUG", "INFO", "WARNING", "ERROR", "INTERNAL_ERROR"]` |
| environment-variables | `DYN_SYSTEM_STARTING_HEALTH_STATUS` | `["ready", "notready"]` |
| metric-labels | `stage` | `["preprocess", "route", "dispatch"]` |
| metric-labels | `phase` | `["prefill", "decode", "aggregated"]` (empty `""` explained in prose) |
| metric-labels | `worker_type` | `["prefill", "decode"]` |
| metric-labels | `status` | `["ok", "parent_block_not_found", "block_not_found", "invalid_block"]` |
| metric-labels | `event_type` | `["stored", "removed", "cleared"]` |
| operator-metrics | `error_type` | `["not_found", "already_exists", "conflict", "validation", "bad_request", "unauthorized", "forbidden", "timeout", "server_timeout", "unavailable", "rate_limited", "internal"]` |
| operator-metrics | `operation` | `["CREATE", "UPDATE", "DELETE"]` |

---

## Verification we did

- **Component logic: PASSED in a real browser.** A Playwright run against the component's runtime
  behavior (mounted in a bare page) confirmed: renders as `<button>` chips; hover changes background
  (`grayscale-a3 → a4`); **click writes the value to the clipboard** (verified via
  `navigator.clipboard.readText()`); the `.copied` class + "Copied" title toggle for 1.5s.
- **`fern check`: 0 errors** with the component wired in (validates MDX + import syntax).
- **In-Fern-page rendering: NOT verifiable locally.** `fern docs dev` returns
  `Could not resolve "@/components/EnumValues"`. Confirmed this is **not our bug**: the same error
  appears for the existing, prod-working `RecipeStyles` import. Inspecting the **live prod HTML**
  (docs.nvidia.com/dynamo) shows `RecipeStyles` compiles and renders there — so `@/components`
  resolves at **publish**, not in local preview.

**The one open risk:** whether `EnumValues` renders correctly inside a real Fern page can only be
confirmed on a **publish / preview-deploy**. The import pattern is identical to the prod-proven
`RecipeStyles`, so the expectation is that it works — but it is unverified.

---

## How to re-apply (once a preview deploy is available)

1. Recreate `fern/components/EnumValues.tsx` from the source above.
2. In each of the 3 `.mdx` files, add below the frontmatter:
   `import { EnumValues } from "@/components/EnumValues";`
3. Replace each static `<span className="enum-values">…</span>` block with
   `<EnumValues values={[…]} />` using the table above.
4. Remove the `.enum-values` / `.enum-label` CSS block from `fern/main.css` (the component is
   self-contained).
5. `fern check` (expect 0 errors). Then **push a Fern preview deploy** and verify on the live preview:
   chips render, click copies, "Copied" appears, hover works. Playwright can drive the preview URL with
   clipboard permissions granted.
6. If the preview shows the chips rendering, commit. If it shows `Could not resolve` or an unstyled
   fallback, keep the static-badge version.

---

## What shipped instead (the static fallback)

Static `<Badge>` chips styled via `fern/main.css`, scoped to a `.enum-values` wrapper so the pill
`<Badge>` used elsewhere is unaffected. Markup at each site:

```mdx
<span className="enum-values"><span className="enum-label">Allowed values:</span> <Badge intent="note" minimal>y</Badge> <Badge intent="note" minimal>n</Badge></span>
```

CSS block in `fern/main.css` (right after the `.fern-docs-badge` rule):

```css
/* Enum "Allowed values" chips: match Fern's Schema renderer, which builds each
   chip as <Badge size="sm"> with the default subtle variant. We can't invoke
   that renderer (it needs an OpenAPI-backed <Schema>), so we restyle the MDX
   <Badge> to the same tokens Fern uses: label in shrink-0 text-sm grayscale-a11,
   chips as .subtle.small (grayscale-a3 fill, grayscale-a11 text, radius-1). The
   forced background/color neutralize whatever intent the MDX <Badge> requires.
   Scoped to .enum-values so the pill <Badge> used elsewhere is unaffected. */
.enum-values {
    display: inline-flex;
    flex-wrap: wrap;
    align-items: baseline;
    gap: 0.5rem;
}
.enum-values .enum-label {
    flex-shrink: 0;
    color: var(--grayscale-a11);
    font-size: var(--text-sm);
}
.enum-values .fern-docs-badge {
    border-radius: var(--radius-1) !important;
    height: 1.25rem;
    padding: 0 0.375rem;
    font-size: var(--text-xs);
    font-weight: 500;
    background-color: var(--grayscale-a3) !important;
    color: var(--grayscale-a11) !important;
    text-transform: none;
}
```

**Known limitation of the fallback:** because the shared NVIDIA theme may strip the docs.yml `css:` at
publish, these `.enum-values` rules could be dropped in prod, leaving the chips as default pill
`<Badge>`s (still functional, just the blue pill look). If that happens in prod and the look matters,
that is itself a reason to switch to the `EnumValues` component (its styles survive publish).

**Difference vs. the interactive version:** the static chips do **not** copy on click and have no
hover/"Copied" behavior — they're display-only. Everything else (label, grey subtle look) matches.
