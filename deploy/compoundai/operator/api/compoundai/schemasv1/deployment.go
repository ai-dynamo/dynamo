package schemasv1

import "github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/modelschemas"

type DeploymentSchema struct {
	ResourceSchema
	Mode           *modelschemas.DeploymentMode  `json:"mode,omitempty"`
	Creator        *UserSchema                   `json:"creator"`
	Cluster        *ClusterFullSchema            `json:"cluster"`
	Status         modelschemas.DeploymentStatus `json:"status" enum:"unknown,non-deployed,running,unhealthy,failed,deploying"`
	URLs           []string                      `json:"urls"`
	LatestRevision *DeploymentRevisionSchema     `json:"latest_revision"`
	KubeNamespace  string                        `json:"kube_namespace"`
}

type DeploymentListSchema struct {
	BaseListSchema
	Items []*DeploymentSchema `json:"items"`
}

type UpdateDeploymentSchema struct {
	Mode        *modelschemas.DeploymentMode    `json:"mode,omitempty"`
	Targets     []*CreateDeploymentTargetSchema `json:"targets"`
	Labels      *modelschemas.LabelItemsSchema  `json:"labels,omitempty"`
	Description *string                         `json:"description,omitempty"`
	DoNotDeploy bool                            `json:"do_not_deploy,omitempty"`
}

type CreateDeploymentSchema struct {
	Name          string `json:"name"`
	KubeNamespace string `json:"kube_namespace"`
	UpdateDeploymentSchema
}
