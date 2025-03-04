package schemasv1

import "github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/modelschemas"

type BentoRepositorySchema struct {
	ResourceSchema
	Creator      *UserSchema         `json:"creator"`
	Organization *OrganizationSchema `json:"organization"`
	LatestBento  *BentoSchema        `json:"latest_bento"`
	NBentos      uint                `json:"n_bentos"`
	NDeployments uint                `json:"n_deployments"`
	LatestBentos []*BentoSchema      `json:"latest_bentos"`
	Description  string              `json:"description"`
}

type BentoRepositoryListSchema struct {
	BaseListSchema
	Items []*BentoRepositorySchema `json:"items"`
}

type BentoRepositoryWithLatestDeploymentsSchema struct {
	BentoRepositorySchema
	LatestDeployments []*DeploymentSchema `json:"latest_deployments"`
}

type BentoRepositoryWithLatestDeploymentsListSchema struct {
	BaseListSchema
	Items []*BentoRepositoryWithLatestDeploymentsSchema `json:"items"`
}

type CreateBentoRepositorySchema struct {
	Name        string                        `json:"name"`
	Description string                        `json:"description"`
	Labels      modelschemas.LabelItemsSchema `json:"labels"`
}

type UpdateBentoRepositorySchema struct {
	Description *string                        `json:"description"`
	Labels      *modelschemas.LabelItemsSchema `json:"labels,omitempty"`
}
