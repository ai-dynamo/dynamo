package schemasv1

import (
	"github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/modelschemas"
)

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
	Organization    *OrganizationSchema                `json:"organization"`
	KubeConfig      *string                            `json:"kube_config"`
	Config          **modelschemas.ClusterConfigSchema `json:"config"`
	GrafanaRootPath string                             `json:"grafana_root_path"`
}

type UpdateClusterSchema struct {
	Description *string                            `json:"description"`
	KubeConfig  *string                            `json:"kube_config"`
	Config      **modelschemas.ClusterConfigSchema `json:"config"`
}

type CreateClusterSchema struct {
	Description string                            `json:"description"`
	KubeConfig  string                            `json:"kube_config"`
	Config      *modelschemas.ClusterConfigSchema `json:"config"`
	Name        string                            `json:"name"`
}
