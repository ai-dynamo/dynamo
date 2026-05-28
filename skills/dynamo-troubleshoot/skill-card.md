## Description: <br>
Diagnose failed or unhealthy Dynamo deployments. Use when pods, model-cache jobs, PVCs, workers, frontend/router health, endpoints, or benchmark jobs fail; use recipe-runner/router-starter before this for normal bring-up. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner: NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and platform engineers use this skill to diagnose and resolve failures in Dynamo inference deployments on Kubernetes, covering pod scheduling, model-cache provisioning, operator reconciliation, and endpoint health. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Failure Decision Tree](references/failure-decision-tree.md) <br>
- [Dynamo Documentation](https://docs.nvidia.com/dynamo/) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Analysis, Configuration instructions] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [None] <br>

## Skill Version(s): <br>
1.2.0 (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
