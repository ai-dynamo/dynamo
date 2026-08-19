---
key: security
name: Security
applies_to: "**/*.rs, **/*.py, **/*.go, **/Dockerfile, **/deploy/**, **/*.yaml, **/*.yml"
---
Hardcoded secrets, credentials, or tokens; secrets or PII written to logs;
new or changed endpoints missing authentication/authorization; unvalidated
external input reaching a shell, SQL, path, or deserialization sink; `unsafe`
Rust without a justifying `SAFETY:` comment; command execution built from
untrusted strings; TLS or certificate verification disabled; overly broad
container capabilities, `runAsUser: 0`, `privileged: true`, or host mounts;
world-writable file modes; dependencies pinned to a mutable tag where an
integrity-checked pin is expected.
