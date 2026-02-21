---
name: pretty-pdf
description: Convert Markdown files to pretty PDFs using pandoc with the Eisvogel template, XeLaTeX, and mermaid diagram rendering. Use when the user asks to generate a PDF, export to PDF, pandoc a markdown file, or make a pretty document from markdown.
---

# Pretty PDF from Markdown

Convert `.md` files to polished PDFs with rendered mermaid diagrams, syntax-highlighted code, a dark title page, and a table of contents.

## Prerequisites

- `pandoc` (3.x+)
- `xelatex` (via MacTeX or TeX Live)
- `mmdc` (`@mermaid-js/mermaid-cli` via npm)
- Eisvogel template installed at `~/.pandoc/templates/eisvogel.latex`

## Mermaid Lua Filter

Create a `mermaid.lua` file alongside the markdown source:

```lua
local counter = 0
local img_dir = "mermaid_images"

local function ensure_dir()
    os.execute("mkdir -p " .. img_dir)
end

function CodeBlock(block)
    if block.classes[1] ~= "mermaid" then
        return nil
    end

    ensure_dir()
    counter = counter + 1

    local infile = img_dir .. "/diagram_" .. counter .. ".mmd"
    local outfile = img_dir .. "/diagram_" .. counter .. ".png"

    local f = io.open(infile, "w")
    f:write(block.text)
    f:close()

    local cmd = string.format(
        "mmdc -i %s -o %s -b white -w 1200 -s 2 2>&1",
        infile, outfile
    )
    local handle = io.popen(cmd)
    local result = handle:read("*a")
    handle:close()

    local img = io.open(outfile, "rb")
    if not img then
        io.stderr:write("mermaid render failed: " .. result .. "\n")
        return nil
    end
    img:close()

    return pandoc.Para({
        pandoc.Image({}, outfile, ""),
    })
end
```

## Build Command

Run from the same directory as the markdown file and `mermaid.lua`:

```bash
pandoc INPUT.md \
  -o OUTPUT.pdf \
  --template eisvogel \
  --pdf-engine=xelatex \
  --lua-filter=mermaid.lua \
  --syntax-highlighting=tango \
  -V colorlinks=true \
  -V linkcolor=NavyBlue \
  -V urlcolor=NavyBlue \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  -V titlepage=true \
  -V titlepage-color=1a1a2e \
  -V titlepage-text-color=e0e0e0 \
  -V titlepage-rule-color=e94560 \
  -V titlepage-rule-height=4 \
  -V toc=true \
  -V toc-own-page=true \
  --metadata title="TITLE" \
  --metadata subtitle="SUBTITLE" \
  --metadata date="$(date +%Y-%m-%d)"
```

## Customization

| Variable | Default | Purpose |
|----------|---------|---------|
| `titlepage-color` | `1a1a2e` | Title page background (hex, no `#`) |
| `titlepage-text-color` | `e0e0e0` | Title page text color |
| `titlepage-rule-color` | `e94560` | Accent rule color |
| `titlepage-rule-height` | `4` | Accent rule thickness (pt) |
| `fontsize` | `11pt` | Body text size |
| `geometry:margin` | `1in` | Page margins |

To skip the title page, remove `-V titlepage=true`. To skip TOC, remove `-V toc=true` and `-V toc-own-page=true`.

## Workflow

1. If `mermaid.lua` doesn't exist next to the source file, create it using the filter above.
2. Replace `INPUT.md`, `OUTPUT.pdf`, `TITLE`, and `SUBTITLE` in the build command.
3. Run the command. The `mermaid_images/` directory is a build artifact and can be gitignored.
4. Open the PDF with `open OUTPUT.pdf` on macOS.

## Notes

- If the markdown has no mermaid blocks, the lua filter is harmless (no images generated).
- For dark mermaid diagram backgrounds, change `-b white` to `-b transparent` in the lua filter.
- The `--syntax-highlighting` flag replaces the deprecated `--highlight-style` in pandoc 3.x.
