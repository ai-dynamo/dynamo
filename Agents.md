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
UPdate: trt-llm is now installed. Also rust and maturin. We can build the python dynamo repo, and use system trt-llm starting. Know, once trt-llm is messed up, its not easy to fix. better not to mess up.
We isntalled 1.3.0rc9. This is the version i try to target. 
We added a lot of mockeypatch material, now its time to bring it into reality. We want to use kvbm with the trt-llm version we are using. We dont want to modify trt-llm itself, and plug the kv manager of kvbm staright in to get all cache.
