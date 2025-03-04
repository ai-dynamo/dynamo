package modelschemas

type ResourceType string

const (
	ResourceTypeUser               ResourceType = "user"
	ResourceTypeOrganization       ResourceType = "organization"
	ResourceTypeCluster            ResourceType = "cluster"
	ResourceTypeBentoRepository    ResourceType = "bento_repository"
	ResourceTypeBento              ResourceType = "bento"
	ResourceTypeDeployment         ResourceType = "deployment"
	ResourceTypeDeploymentRevision ResourceType = "deployment_revision"
	ResourceTypeTerminalRecord     ResourceType = "terminal_record"
	ResourceTypeModelRepository    ResourceType = "model_repository"
	ResourceTypeModel              ResourceType = "model"
	ResourceTypeLabel              ResourceType = "label"
	ResourceTypeApiToken           ResourceType = "api_token"
	ResourceTypeYataiComponent     ResourceType = "yatai_component"
)

func (type_ ResourceType) Ptr() *ResourceType {
	return &type_
}
