package schemasv2

import "github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/schemas"

type DeploymentSchema struct {
	schemas.ResourceSchema
	Creator        *schemas.UserSchema               `json:"creator"`
	Cluster        *ClusterSchema                    `json:"cluster"`
	Status         schemas.DeploymentStatus          `json:"status" enum:"unknown,non-deployed,running,unhealthy,failed,deploying"`
	URLs           []string                          `json:"urls"`
	LatestRevision *schemas.DeploymentRevisionSchema `json:"latest_revision"`
	KubeNamespace  string                            `json:"kube_namespace"`
}

type GetDeploymentSchema struct {
	DeploymentName string `uri:"deploymentName" binding:"required"`
}

func (s *GetDeploymentSchema) ToV1(clusterName string, namespace string) *schemas.GetDeploymentSchema {
	return &schemas.GetDeploymentSchema{
		GetClusterSchema: schemas.GetClusterSchema{
			ClusterName: clusterName,
		},
		KubeNamespace:  namespace,
		DeploymentName: s.DeploymentName,
	}
}

type CreateDeploymentSchema struct {
	UpdateDeploymentSchema
	Name string `json:"name"`
}

type UpdateDeploymentSchema struct {
	DeploymentConfigSchema
	CompoundNim string `json:"bento"`
}

type DeploymentConfigSchema struct {
	AccessAuthorization bool                   `json:"access_authorization"`
	Envs                interface{}            `json:"envs,omitempty"`
	Secrets             interface{}            `json:"secrets,omitempty"`
	Services            map[string]ServiceSpec `json:"services"`
}

type ServiceSpec struct {
	Scaling          ScalingSpec                        `json:"scaling"`
	ConfigOverrides  ConfigOverridesSpec                `json:"config_overrides"`
	ExternalServices map[string]schemas.ExternalService `json:"external_services,omitempty"`
	ColdStartTimeout *int32                             `json:"cold_start_timeout,omitempty"`
}

type ScalingSpec struct {
	MinReplicas int `json:"min_replicas"`
	MaxReplicas int `json:"max_replicas"`
}

type ConfigOverridesSpec struct {
	Resources schemas.Resources `json:"resources"`
}
