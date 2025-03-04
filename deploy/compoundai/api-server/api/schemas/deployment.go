package schemas

type DeploymentSchema struct {
	ResourceSchema
	Creator        *UserSchema               `json:"creator"`
	Cluster        *ClusterFullSchema        `json:"cluster"`
	Status         DeploymentStatus          `json:"status" enum:"unknown,non-deployed,running,unhealthy,failed,deploying"`
	URLs           []string                  `json:"urls"`
	LatestRevision *DeploymentRevisionSchema `json:"latest_revision"`
	KubeNamespace  string                    `json:"kube_namespace"`
}

type DeploymentListSchema struct {
	BaseListSchema
	Items []*DeploymentSchema `json:"items"`
}

type UpdateDeploymentSchema struct {
	Targets     []*CreateDeploymentTargetSchema `json:"targets"`
	Description *string                         `json:"description,omitempty"`
	DoNotDeploy bool                            `json:"do_not_deploy,omitempty"`
}

type CreateDeploymentSchema struct {
	Name          string `json:"name"`
	KubeNamespace string `json:"kube_namespace"`
	UpdateDeploymentSchema
}

type GetDeploymentSchema struct {
	GetClusterSchema
	DeploymentName string `uri:"deploymentName" binding:"required"`
	KubeNamespace  string `uri:"kubeNamespace" binding:"required"`
}
