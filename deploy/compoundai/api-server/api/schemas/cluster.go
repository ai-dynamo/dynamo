package schemas

type ClusterSchema struct {
	ResourceSchema
	Creator     *UserSchema `json:"creator"`
	Description string      `json:"description"`
}

type ClusterListSchema struct {
	BaseListSchema
	Items []*ClusterSchema `json:"items"`
}

type ClusterFullSchema struct {
	ClusterSchema
	Organization *OrganizationSchema `json:"organization"`
	KubeConfig   *string             `json:"kube_config"`
}

type UpdateClusterSchema struct {
	Description *string `json:"description"`
	KubeConfig  *string `json:"kube_config"`
}

type CreateClusterSchema struct {
	Description string `json:"description"`
	KubeConfig  string `json:"kube_config"`
	Name        string `json:"name"`
}

type GetClusterSchema struct {
	ClusterName string `uri:"clusterName" binding:"required"`
}
