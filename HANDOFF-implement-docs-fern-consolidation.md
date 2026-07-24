# Handoff: Consolidate Fern docs into `docs/fern/` on `dynamo-full-docs-refactor`

## Goal
Move the Fern project root from `fern/` (repo root) into `docs/fern/`, and move all
page content from `docs/` into that same folder, so the fern root and the content
live together. This fixes custom components (`RecipeStyles`, `TerminalDemo`, etc.):
their `@/components/...` imports resolve relative to the fern root, and today the
`.mdx` pages sit in `docs/` — *outside* that root — so the imports fail.

## Verified outcome (tested in a throwaway worktree)
- `fern check` → **0 errors** in the new layout.
- Dev server (`fern docs dev`, run **from `docs/fern/`**) boots clean; recipe pages
  render `RecipeStyles` (49 `<style>` blocks), home page renders `TerminalDemo`.
- **Native incremental hot reload works** on both `.md` and `.mdx` edits
  (~500 ms reloads), because content now lives inside the fern root.
  → The old `fern/watch.sh` / `fern/dev.sh` workarounds become unnecessary and can
    be retired (they existed only because `docs/` was outside the fern root).

## Target layout
```
docs/
  fern/                 <- fern project root (has fern.config.json, docs.yml, index.yml)
    fern.config.json
    docs.yml
    index.yml
    welcome.mdx
    main.css
    components/         <- CustomFooter.tsx, RecipeStyles.tsx, TerminalDemo.tsx + doc subdirs
    assets/
    getting-started/
    kubernetes/
    recipes/
    ... (all former docs/ content) ...
```
`docs/` ends up containing only `fern/`. `examples/` stays at repo root (unchanged).

---

## Implementation steps

Run from repo root: `/home/jonathan/go/src/dynamo-full-docs-refactor`
Work on branch `dynamo-full-docs-refactor` (or a child branch off it).

> Prefer `git mv` where possible so history is preserved and the moves show up
> cleanly in `git status`. The rsync approach below is what the experiment used;
> a `git mv`-based variant is noted after.

### 1. Move the fern project files into `docs/fern/`
```bash
mkdir -p docs/fern
git mv fern/fern.config.json docs/fern/fern.config.json
git mv fern/docs.yml          docs/fern/docs.yml
git mv fern/welcome.mdx       docs/fern/welcome.mdx
git mv fern/main.css          docs/fern/main.css
git mv fern/components        docs/fern/components
git mv fern/convert_callouts.py docs/fern/convert_callouts.py
git mv fern/.gitignore        docs/fern/.gitignore
git mv fern/hero-demo         docs/fern/hero-demo   # if present
# Retire the reload workarounds (no longer needed):
git rm fern/watch.sh fern/dev.sh 2>/dev/null || true
rmdir fern 2>/dev/null || true
```

### 2. Move all page content from `docs/` into `docs/fern/`
Everything currently directly under `docs/` (except the new `fern/` dir and
`index.yml`) moves into `docs/fern/`:
```bash
for item in docs/*; do
  base=$(basename "$item")
  [ "$base" = "fern" ] && continue
  [ "$base" = "index.yml" ] && continue
  git mv "$item" "docs/fern/$base"
done
git mv docs/index.yml docs/fern/index.yml
```
Note: `docs/components/` (doc pages) merges into `docs/fern/components/` alongside
the `.tsx` files. In the experiment these coexisted with **no name collisions** —
verify `git mv` doesn't complain; if a path already exists, move its contents in.

### 3. Fix path references (pages are now one level deeper)

**a. `docs/fern/docs.yml`** — it points at content via `../docs/...`:
```bash
sed -i 's|\.\./docs/index.yml|./index.yml|g; s|\.\./docs/assets/|./assets/|g' docs/fern/docs.yml
```
This covers: `products[*].path` (→ `./index.yml`), the fonts under
`typography`, `logo.dark`/`logo.light`, and `favicon` (all `../docs/assets/` → `./assets/`).

**b. `docs/fern/index.yml`** — the Home tab points at `../fern/welcome.mdx`:
```bash
sed -i 's|\.\./fern/welcome.mdx|./welcome.mdx|g' docs/fern/index.yml
```

**c. `docs/fern/welcome.mdx`** — hero cast asset ref:
```bash
sed -i 's|src="\.\./docs/assets/|src="./assets/|g' docs/fern/welcome.mdx
```

**d. `docs/fern/templates/{sglang,trtllm,vllm}.mdx`** — `<Code src>` embeds reach
repo-root `examples/`. They were `../../examples/` (from `docs/templates/`), now need
one more `../` (from `docs/fern/templates/`):
```bash
sed -i 's|src="\.\./\.\./examples/|src="../../../examples/|g' docs/fern/templates/*.mdx
```

### 4. Validate
```bash
cd docs/fern
fern check          # expect: 0 errors (warnings about "page removed" + one
                    # contrast a11y warning are pre-existing / benign)
```
If any `Path ... does not exist` errors remain, they'll almost certainly be more
`../` relative refs that got one level deeper — grep the reported path and add a `../`.

### 5. Run the dev server (optional local check)
**Always run from `docs/fern/`.** Use a non-default port — other fern servers may be
running on 3001–3004:
```bash
cd docs/fern
fern docs dev --port 3123
# ready at http://localhost:3123 ; edit any .md/.mdx and confirm live reload
```

---

## CI changes required (`.github/workflows/fern-docs.yml`) — DO THIS

The workflow syncs source → the `docs-website` branch's root-level `fern/` layout.
It currently assumes content in `source-checkout/docs/` and fern config in
`source-checkout/fern/`. After this move, **all of that is under
`source-checkout/docs/fern/`**. Update the "Sync dev content" step and friends:

- Content rsync source: `source-checkout/docs/`  →  `source-checkout/docs/fern/`
  (and add `--exclude` for `fern.config.json`, `docs.yml`, `index.yml`,
  `components`, `main.css`, `welcome.mdx`, `convert_callouts.py`, `.gitignore`
  so only *pages* land in `fern/pages-dev/`).
- `index.yml` copy: `source-checkout/docs/index.yml` → `source-checkout/docs/fern/index.yml`.
- `fern.config.json`: `source-checkout/fern/...` → `source-checkout/docs/fern/...`.
- `components/`, `main.css`, `convert_callouts.py`, `README.md`, `.gitignore`,
  `digest/` copies: repoint each `source-checkout/fern/...` and
  `source-checkout/docs/...` prefix to `source-checkout/docs/fern/...`.
- The "Update docs.yml preserving products" step does
  `sed 's|\.\./docs/assets/|./assets/|g'` on the source `docs.yml`. Since step 3a
  above already rewrites those to `./assets/`, that sed becomes a harmless no-op —
  but the `cp ../../source-checkout/fern/docs.yml docs.yml` line must become
  `cp ../../source-checkout/docs/fern/docs.yml docs.yml`.

Because the published `docs-website` layout (`fern/pages-dev/`, `fern/versions/dev.yml`)
is produced by the workflow, it does **not** change — only the *source paths* the
workflow reads from change. Diff a dry run against a `docs-website` checkout before
merging.

## Sanity checklist before commit
- [ ] `docs/` contains only `fern/`.
- [ ] `docs/fern/fern.config.json`, `docs.yml`, `index.yml` all present.
- [ ] `cd docs/fern && fern check` → 0 errors.
- [ ] No leftover `../docs/` or `../fern/` refs: `grep -rn '\.\./docs/\|\.\./fern/' docs/fern/docs.yml docs/fern/index.yml`
- [ ] `fern/` (repo root) is gone; `watch.sh`/`dev.sh` removed.
- [ ] CI workflow source paths updated.
- [ ] `git status` shows moves as renames, not delete+add (confirms `git mv` worked).

## Reference
A fully working version of this layout is in the throwaway worktree:
`/home/jonathan/go/src/dynamo/.claude/worktrees/fern-consolidation-test/docs/fern`
(branch `worktree-fern-consolidation-test`). `fern check` passes there; diff against
it if anything is unclear.
