---
orphan: true
---

# NVIDIA Dynamo Documentation

This directory contains the documentation source files for NVIDIA Dynamo, built with [Docusaurus](https://docusaurus.io/).

## Quick Start

```bash
# Navigate to the docs directory
cd docs

# Install dependencies
npm install

# Start development server (with hot reload)
npm run start
# Opens http://localhost:3000 in your browser

# Build for production
npm run build

# Serve production build locally
npm run serve
```

## Documentation Commands

| Command | Description |
|---------|-------------|
| `npm run start` | Start dev server with hot reload |
| `npm run build` | Build production static site to `build/` |
| `npm run serve` | Serve production build locally |
| `npm run clear` | Clear Docusaurus cache |
| `npm run docusaurus docs:version X.Y.Z` | Create a new version snapshot |

## Directory Structure

```
docs/
├── docusaurus.config.ts         # Main Docusaurus configuration
├── sidebars.ts                  # Navigation structure
├── package.json                 # Dependencies
├── versions.json                # Version manifest
├── tsconfig.json                # TypeScript config
├── docs/                        # Current version content
├── versioned_docs/              # Released versions (created via docs:version)
├── versioned_sidebars/          # Sidebars for each version
├── src/
│   └── css/custom.css           # NVIDIA theme
├── static/img/                  # Static images
├── build/                       # Generated output (gitignored)
├── agents/                      # Content source (linked in docs/)
├── backends/
├── kubernetes/
└── ...
```

## Versioning

The documentation supports multiple versions matching Dynamo releases.

### Creating a New Version

When releasing a new version of Dynamo:

```bash
cd docs
npm run docusaurus docs:version X.Y.Z
```

This will:
1. Copy `docs/` to `versioned_docs/version-X.Y.Z/`
2. Copy `sidebars.ts` to `versioned_sidebars/`
3. Add the version to `versions.json`

### Version Configuration

After creating versions, update `docusaurus.config.ts` to configure version labels and paths:

```typescript
docs: {
  lastVersion: 'X.Y.Z',  // Set the latest stable version
  versions: {
    current: { label: 'dev (next)', path: 'dev', banner: 'unreleased' },
    'X.Y.Z': { label: 'X.Y.Z (latest)', path: '', banner: 'none' },
  },
}
```

## Writing Documentation

### File Format

Documentation is written in Markdown with [MDX](https://mdxjs.com/) support.

### Frontmatter

Each document should have frontmatter:

```markdown
---
title: "Page Title"
sidebar_position: 1
---

# Page Title

Content here...
```

### Admonitions

Use Docusaurus admonitions for callouts:

```markdown
:::note
This is a note.
:::

:::tip
This is a tip.
:::

:::warning
This is a warning.
:::

:::danger
This is a danger notice.
:::
```

### Code Blocks

```markdown
```python title="example.py"
def hello():
    print("Hello, Dynamo!")
```
```

### Internal Links

Link to other docs using relative paths:

```markdown
See the [Backend Guide](./backends/vllm/README.md) for more details.
```

## Search

The documentation includes local search powered by `@easyops-cn/docusaurus-search-local`. Use `Ctrl+K` to open search.

## Theme

The site uses the Docusaurus Classic theme with custom NVIDIA branding:
- Primary color: NVIDIA Green (#76b900)
- Dark navbar and footer
- Custom logo and favicon

```toml
[dependency-groups]
docs = [
    "sphinx>=8.1",
    "nvidia-sphinx-theme>=0.0.8",
    # ... other doc dependencies
]
```

## Troubleshooting

### Build Warnings

The build process treats warnings as errors. Common issues:

- **Missing toctree entries**: Documents must be referenced in a table of contents
- **Non-consecutive headers**: Don't skip header levels (e.g., H1 → H3)
- **Broken links**: Ensure all internal and external links are valid

### Missing Dependencies

If you encounter import errors, ensure the docs dependencies are installed:

```bash
uv pip install --python .venv-docs --group docs
```

## Viewing the Documentation

After building, open `docs/build/html/index.html` in your, or use Python's built-in HTTP server:

```bash
cd docs/build/html
python -m http.server 8000
# Then visit http://localhost:8000 in your browser
```
