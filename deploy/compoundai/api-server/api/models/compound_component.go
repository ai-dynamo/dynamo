package models

import (
	"time"

	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/schemas"
)

type CompoundComponent struct {
	Resource
	ClusterAssociate

	Version           string                                   `json:"version"`
	KubeNamespace     string                                   `json:"kube_namespace"`
	Description       string                                   `json:"description"`
	Manifest          *schemas.CompoundComponentManifestSchema `json:"manifest" type:"jsonb"`
	LatestInstalledAt *time.Time                               `json:"latest_installed_at"`
	LatestHeartbeatAt *time.Time                               `json:"latest_heartbeat_at"`
}

func (d *CompoundComponent) GetResourceType() schemas.ResourceType {
	return schemas.ResourceTypeCompoundAIComponent
}
