# Agent Instructions

Before starting any work on this repo, always read these documents first:

1. `PLANS.md`
2. `docs/design-docs/kvbm-g4-nvme-raid-plan.md`

Those two files are the primary handoff and design context for the current
`mf/kvbm-g4-v2` work. Do not start coding before reading both.

## Read Order

Use this order when picking up work:

1. `PLANS.md`
2. `docs/design-docs/kvbm-g4-nvme-raid-plan.md`
3. any files explicitly listed in the top section of `PLANS.md`

## Working Rules

- Treat `PLANS.md` as the active execution log and handoff document.
- Update `PLANS.md` as you make progress.
- At each stopping point, record what was completed, what remains, and the
  exact next file or command.
- Reuse existing disk allocation and transfer utilities where possible; do not
  introduce a parallel disk-write stack for G4 unless `PLANS.md` clearly calls
  for it.
- If you make commits in this repo, use `--signoff`.

## Current Focus

The current task is the first-pass G4 worker/storage-agent path for KVBM. The
expected first implementation slice is described in `PLANS.md` and the design
details live in `docs/design-docs/kvbm-g4-nvme-raid-plan.md`.
