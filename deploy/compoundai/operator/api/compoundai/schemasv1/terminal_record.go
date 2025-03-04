package schemasv1

type TerminalRecordSchema struct {
	ResourceSchema
	Creator       *UserSchema         `json:"creator"`
	Organization  *OrganizationSchema `json:"organization"`
	Cluster       *ClusterSchema      `json:"cluster"`
	Deployment    *DeploymentSchema   `json:"deployment"`
	Resource      *ResourceSchema     `json:"resource"`
	PodName       string              `json:"pod_name"`
	ContainerName string              `json:"container_name"`
}

type TerminalRecordListSchema struct {
	BaseListSchema
	Items []*TerminalRecordSchema `json:"items"`
}
