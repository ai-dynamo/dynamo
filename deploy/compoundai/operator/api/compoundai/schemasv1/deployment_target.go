package schemasv1

import "github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/modelschemas"

type DeploymentTargetTypeSchema struct {
	Type modelschemas.DeploymentTargetType `json:"type" enum:"stable,canary"`
}

type DeploymentTargetSchema struct {
	ResourceSchema
	DeploymentTargetTypeSchema
	Creator     *UserSchema                               `json:"creator"`
	Bento       *BentoFullSchema                          `json:"bento"`
	CanaryRules *modelschemas.DeploymentTargetCanaryRules `json:"canary_rules"`
	Config      *modelschemas.DeploymentTargetConfig      `json:"config"`
}

type DeploymentTargetListSchema struct {
	BaseListSchema
	Items []*DeploymentTargetSchema `json:"items"`
}

type CreateDeploymentTargetSchema struct {
	DeploymentTargetTypeSchema
	BentoRepository string                                    `json:"bento_repository"`
	Bento           string                                    `json:"bento"`
	CanaryRules     *modelschemas.DeploymentTargetCanaryRules `json:"canary_rules"`
	Config          *modelschemas.DeploymentTargetConfig      `json:"config"`
}
