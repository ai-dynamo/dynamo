# Vendored SPDX license texts

Canonical license texts, one `<SPDX-ID>.txt` per identifier, sourced verbatim from
[spdx/license-list-data](https://github.com/spdx/license-list-data) (`text/<ID>.txt`,
exceptions from `text/<EXCEPTION>.txt`).

These are the license-text source for ecosystems whose components ship no LICENSE
file in the image's `licenses` build stage — chiefly **rust** (cargo-cyclonedx records
only the SPDX expression) and **go** (cyclonedx-gomod likewise), and as a fallback for
**python** / **native** components that bundle no license file. `generators/common.py`
`spdx_license_text()` looks up each ID referenced by a component's SPDX expression and
emits the concatenated text; `render_notices()` prepends a "no license file was
distributed with this package" disclaimer so canonical text is never mistaken for the
dependency's own file.

The set mirrors the SPDX identifiers on the `[licenses].allow` list in
`../policy/licenses.toml` — i.e. every license that can legally appear in a shipped
image (anything else fails the policy gate). To refresh after editing the allow-list,
re-fetch the corresponding `text/<ID>.txt` files from the SPDX data repo.
