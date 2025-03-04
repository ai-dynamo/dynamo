package schemasv1

import "github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/modelschemas"

type ModelRepositorySchema struct {
	ResourceSchema
	Creator      *UserSchema         `json:"creator"`
	Organization *OrganizationSchema `json:"organization"`
	LatestModel  *ModelSchema        `json:"latest_model"`
	Description  string              `json:"description"`
}

type ModelRepositoryListSchema struct {
	BaseListSchema
	Items []*ModelRepositorySchema `json:"items"`
}

type CreateModelRepositorySchema struct {
	Name        string                        `json:"name"`
	Description string                        `json:"description"`
	Labels      modelschemas.LabelItemsSchema `json:"labels"`
}

type UpdateModelRepositorySchema struct {
	Description *string                        `json:"description"`
	Labels      *modelschemas.LabelItemsSchema `json:"labels,omitempty"`
}
