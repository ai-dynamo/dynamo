package schemasv1

import "github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/modelschemas"

type IResourceSchema interface {
	GetType() modelschemas.ResourceType
	GetName() string
}

type ResourceSchema struct {
	BaseSchema
	Name         string                        `json:"name"`
	ResourceType modelschemas.ResourceType     `json:"resource_type" enum:"user,organization,cluster,bento_repository,bento,deployment,deployment_revision,model_repository,model,api_token"`
	Labels       modelschemas.LabelItemsSchema `json:"labels"`
}

func (r ResourceSchema) GetType() modelschemas.ResourceType {
	return r.ResourceType
}

func (r ResourceSchema) GetName() string {
	return r.Name
}

func (s *ResourceSchema) TypeName() string {
	return string(s.ResourceType)
}
