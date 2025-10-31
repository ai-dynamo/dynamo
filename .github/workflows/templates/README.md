# Workflow Templates

This directory contains reusable templates and utilities for GitHub Actions workflows.

## Files

### akamai-eccu-flush.xslt

XSLT template for generating Akamai ECCU (Edge Content Control Utility) XML requests.

**Purpose**: Generates XML for cache invalidation requests to Akamai CDN.

**Usage**:
```bash
xsltproc --stringparam target-path "path/to/flush" \
  akamai-eccu-flush.xslt akamai-eccu-flush.xslt > eccu-request.xml
```

**Used by**: `.github/workflows/publish-s3.yml` for flushing CDN cache after documentation deployment.

The template creates a hierarchical XML structure with nested `match:recursive-dirs` elements representing the directory path to invalidate in the Akamai cache.
