package schemasv1

import (
	"time"

	"github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/modelschemas"
)

type YataiComponentSchema struct {
	ResourceSchema
	Creator           *UserSchema                                `json:"creator"`
	Cluster           *ClusterFullSchema                         `json:"cluster"`
	Description       string                                     `json:"description"`
	Version           string                                     `json:"version"`
	KubeNamespace     string                                     `json:"kube_namespace"`
	Manifest          *modelschemas.YataiComponentManifestSchema `json:"manifest"`
	LatestInstalledAt *time.Time                                 `json:"latest_installed_at"`
	LatestHeartbeatAt *time.Time                                 `json:"latest_heartbeat_at"`
}

type RegisterYataiComponentSchema struct {
	Name           modelschemas.YataiComponentName            `json:"name"`
	Version        string                                     `json:"version"`
	KubeNamespace  string                                     `json:"kube_namespace"`
	SelectorLabels map[string]string                          `json:"selector_labels,omitempty"`
	Manifest       *modelschemas.YataiComponentManifestSchema `json:"manifest"`
}
