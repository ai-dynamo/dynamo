## Description: <br>
Start or patch Dynamo router modes and run router endpoint smoke checks. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner: NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Developers and infrastructure engineers use this skill to configure Dynamo routing modes (round-robin, KV-aware, least-loaded, device-aware) and verify endpoint health via smoke tests during cluster bring-up. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Router Modes Reference](references/router-modes.md) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agent: <br>
Claude (Anthropic) running the skill harness end-to-end, supplemented by the NV-ACES Tier 1 / NV-BASE 2.12.0 static scorer for schema, license, quality, security, and PII validation. <br>

## Evaluation Tasks: <br>
Six prompts in `evals/evals.json`, split between three positive cases the skill should trigger and act on, and three negative cases it should defer to the correct sibling skill. <br>
- Positive: `enable-kv-routing`, `compare-roundrobin-vs-kv`, `router-health-check`. <br>
- Negative: `neg-deploy-recipe` (defers to `dynamo-recipe-runner`), `neg-router-500s-workers-missing` (defers to `dynamo-troubleshoot`), `neg-validate-nixl` (defers to `dynamo-interconnect-check`). <br>

## Evaluation Metrics: <br>
- Trigger-routing accuracy: does the prompt invoke this skill on positive cases and correctly defer on negatives? <br>
- NV-ACES 4-dimension static score: correctness (35% weight), discoverability (25%), reliability (25%), efficiency (15%). <br>

## Evaluation Results: <br>
- Trigger-routing: 6/6 cases routed correctly (3 positive + 3 negative); part of the 24/24 across the four bring-up skills attested on PR #9782. <br>
- NV-ACES Tier 1 / NV-BASE 2.12.0: 83.2 / 100 (Grade B). Dimension breakdown: correctness 70.0, discoverability 90.0, reliability 85.0, efficiency 100.0. <br>

## Skill Version(s): <br>
1.2.0 (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
