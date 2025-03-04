package schemas

type IResourceSchema interface {
	GetType() ResourceType
	GetName() string
}

type ResourceSchema struct {
	BaseSchema
	Name         string            `json:"name"`
	Labels       []LabelItemSchema `json:"labels"`
	ResourceType ResourceType      `json:"resource_type" enum:"user,organization,cluster,compound_nim,compound_nim_version,deployment,deployment_revision,model_repository,model,api_token"`
}

func (r ResourceSchema) GetType() ResourceType {
	return r.ResourceType
}

func (r ResourceSchema) GetName() string {
	return r.Name
}

func (s *ResourceSchema) TypeName() string {
	return string(s.ResourceType)
}

type ResourceItem struct {
	CPU    string            `json:"cpu,omitempty"`
	Memory string            `json:"memory,omitempty"`
	GPU    string            `json:"gpu,omitempty"`
	Custom map[string]string `json:"custom,omitempty"`
}

type Resources struct {
	Requests *ResourceItem `json:"requests,omitempty"`
	Limits   *ResourceItem `json:"limits,omitempty"`
}
