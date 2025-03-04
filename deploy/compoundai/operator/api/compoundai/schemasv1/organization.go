package schemasv1

import "github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/modelschemas"

type OrganizationSchema struct {
	ResourceSchema
	Creator     *UserSchema `json:"creator"`
	Description string      `json:"description"`
}

type OrganizationFullSchema struct {
	OrganizationSchema
	Config **modelschemas.OrganizationConfigSchema `json:"config"`
}

type OrganizationListSchema struct {
	BaseListSchema
	Items []*OrganizationSchema `json:"items"`
}

type UpdateOrganizationSchema struct {
	Description *string                                 `json:"description"`
	Config      **modelschemas.OrganizationConfigSchema `json:"config"`
}

type CreateOrganizationSchema struct {
	Name        string                                 `json:"name"`
	Description string                                 `json:"description"`
	Config      *modelschemas.OrganizationConfigSchema `json:"config"`
}
