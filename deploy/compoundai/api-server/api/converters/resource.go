package converters

import (
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/models"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/schemas"
)

func ToResourceSchema(resource *models.Resource, resourceType schemas.ResourceType) *schemas.ResourceSchema {
	return &schemas.ResourceSchema{
		BaseSchema:   ToBaseSchema(resource.BaseModel),
		Name:         resource.Name,
		ResourceType: resourceType,
		Labels:       []schemas.LabelItemSchema{},
	}
}
