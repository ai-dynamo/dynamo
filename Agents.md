# ExecPlans

When working on complex features or major refactors, use the execution plan in ./PLANS.md from design through implementation.

When implementing an ExecPlan:
- Do not ask the user for next steps.
- Proceed to the next milestone until the plan is fully complete.
- Keep PLANS.md updated as a living document.
- Compact context continuously by writing decisions, progress, open questions, and next actions into PLANS.md.
- At every stopping point, update:
  - what was completed
  - what remains
  - exact next command or file to touch
- Resolve ambiguities autonomously in the most reasonable way.
- Run tests after meaningful changes.
- Commit progress frequently in small, reversible commits. commits need to run with --signoff in this repo.
- Only stop when the entire plan is done, or when blocked by an external dependency that cannot be completed from this machine.


/workspace/model-performance/michaelfeil1209/mfdynamo/docs/design-docs/kvbm-trtllm-integration.md is the main integration document, and your task description.
