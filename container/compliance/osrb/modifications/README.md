# OSRB modifications

When dynamo applies a patch on top of a vendored upstream source (e.g., a CVE
backport to a pinned-old-version crate, or a small fix to a third-party
package we can't upgrade yet), the patch lives here.

## Layout

```
container/compliance/osrb/modifications/
  <ecosystem>/
    <package>-<version>/
      <descriptive-name>.patch    # the actual diff
      README.md                   # required: why we patch, upstream tracking,
                                  # why we can't bump to the upstream-fixed version
```

## Why this exists

OSRB's compliance review asks specifically: "do you modify any of the
third-party code you ship?" If we patch and don't disclose, we lose the
mere-aggregation defense for those components and have to treat them as
derivative works (which can change license obligations).

Each entry here is required for:
- License-compliance review (some licenses require modifications to be
  marked and the modified source to be available; this directory is the
  source-of-truth for "marked")
- Security review (CVE backport tracking; OSRB cross-references upstream
  CVE databases against our patches)
- Engineering hygiene (every patch needs a path back to upstream)

## Today

This directory is empty. dynamo doesn't currently patch any vendored
upstream sources. If that changes, follow the layout above.

The OSRB packager (`container/compliance/osrb/package.py`) walks this
directory and includes every patch + README in the submission bundle
under `modifications/`.
