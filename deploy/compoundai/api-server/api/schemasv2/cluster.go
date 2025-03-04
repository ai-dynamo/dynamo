package schemasv2

import "github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/schemas"

type ClusterSchema struct {
	schemas.ResourceSchema
	Description      string              `json:"description"`
	OrganizationName string              `json:"organization_name"`
	Creator          *schemas.UserSchema `json:"creator"`
	IsFirst          *bool               `json:"is_first,omitempty"`
}
