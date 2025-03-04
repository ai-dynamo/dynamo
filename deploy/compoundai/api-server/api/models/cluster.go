package models

type Cluster struct {
	Resource
	CreatorAssociate
	OrganizationAssociate

	Description string `json:"description"`
	KubeConfig  string `json:"kube_config"`
}
