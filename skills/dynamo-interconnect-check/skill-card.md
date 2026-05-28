## Description: <br>
Validate that a Dynamo deployment's NIXL/UCX/NCCL interconnect is ready for disaggregated serving over RDMA/NVLink. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner: NVIDIA <br>

### License/Terms of Use: <br>
Apache 2.0 <br>
## Use Case: <br>
Developers and infrastructure engineers use this skill to confirm that the NIXL/UCX/NCCL transport fabric is correctly configured for disaggregated serving after deploying a Dynamo recipe, before trusting benchmark numbers or diagnosing slow disagg performance. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: Review before execution as proposals could introduce incorrect or misleading guidance into skills. <br>
Mitigation: Review and scan skill before deployment. <br>

## Reference(s): <br>
- [Interconnect Env Vars & IB Capability Checklist](references/interconnect-env-vars.md) <br>
- [Dynamo Documentation](https://docs.nvidia.com/dynamo/) <br>


## Skill Output: <br>
**Output Type(s):** [Shell commands, Analysis] <br>
**Output Format:** [Markdown with inline bash code blocks] <br>
**Output Parameters:** [1D] <br>
**Other Properties Related to Output:** [Structured ok/warn/fail/skipped verdicts per check] <br>

## Skill Version(s): <br>
1.2.0 (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
