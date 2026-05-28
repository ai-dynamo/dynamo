## Description: <br>
Select, validate, patch, and deploy existing NVIDIA Dynamo Kubernetes recipes. Use for model/backend/GPU/deployment-mode recipe bring-up; use router-starter for router-only mode work and troubleshoot for broken deployments. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner: NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and infrastructure engineers use this skill to select, configure, and deploy NVIDIA Dynamo inference recipes on Kubernetes clusters, minimizing manual manifest editing and deployment steps. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Kubernetes Recipe Workflow](references/k8s-recipe-workflow.md) <br>
- [Dynamo Recipes](https://github.com/ai-dynamo/dynamo/tree/main/recipes) <br>
- [Dynamo Documentation](https://docs.nvidia.com/dynamo/) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Configuration instructions, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Evaluation Agent: <br>
Claude (Anthropic) running the skill harness end-to-end, supplemented by the NV-ACES Tier 1 / NV-BASE 2.12.0 static scorer for schema, license, quality, security, and PII validation. <br>

## Evaluation Tasks: <br>
Six prompts in `evals/evals.json`, split between three positive cases the skill should trigger and act on, and three negative cases it should defer to the correct sibling skill. <br>
- Positive: `deploy-qwen-vllm-disagg`, `list-sglang-recipes`, `bring-up-nemotron-end-to-end`. <br>
- Negative: `neg-switch-router-mode` (defers to `dynamo-router-starter`), `neg-pods-crashlooping` (defers to `dynamo-troubleshoot`), `neg-author-cuda-kernel` (out of scope for the bring-up skill set). <br>

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
