# Troubleshooting Known Issues

Pointer to the stable issue patterns and signatures.

**Per-surface signatures (this skill's primary catalog):**
[references/symptom-signatures.md](symptom-signatures.md)

**Cross-skill stable patterns (shared with dynamo-deploy):**
[`dynamo-deploy/references/known-issues.md`](../../dynamo-deploy/references/known-issues.md) — sourced from `citations.md`-.

**Per-release-line bugs:** the active QA tracker. For 1.2.0:
`https://linear.app/nvidia/view/dynamo-v120-qa-bugs-ca92649b0b87`. The
skill body's `version` field names the release line; pull the matching
tracker view when refreshing this section per `SKILL_AUTHORING.md` §11
.

The day-2 signature library in `symptom-signatures.md` should be
audited against the live tracker on each release bump. Resolved bugs
fall off the signature list; new RC patterns get added.
