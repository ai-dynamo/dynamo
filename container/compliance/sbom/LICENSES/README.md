# In-Container License Corpus

This directory holds verbatim license texts for every SPDX ID referenced by the
generated `ATTRIBUTIONS-container-*.csv` files. Each file is named after the
SPDX identifier it represents (e.g., `Apache-2.0.txt`).

The companion `LICENSE-MANIFEST.csv` maps each SPDX ID to the component that
contributed its canonical text and to the path within this directory. OSRB
reviewers should consult the manifest to see which component was the source
of each license file.

This directory and its contents are produced by
`container/compliance/sbom/render_attributions.py`.
