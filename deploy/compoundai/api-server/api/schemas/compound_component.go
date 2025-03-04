package schemas

import (
	"database/sql/driver"
	"encoding/json"
	"time"
)

type CompoundComponentName string

const (
	CompoundComponentNameDeployment   CompoundComponentName = "deployment"
	CompoundComponentNameImageBuilder CompoundComponentName = "image-builder"
	CompoundComponentNameFunction     CompoundComponentName = "function"
	CompoundComponentNameJob          CompoundComponentName = "job"
)

type CompoundComponentManifestSchema struct {
	SelectorLabels   map[string]string `json:"selector_labels,omitempty"`
	LatestCRDVersion string            `json:"latest_crd_version,omitempty"`
}

func (c *CompoundComponentManifestSchema) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	return json.Unmarshal(value.([]byte), c)
}

func (c *CompoundComponentManifestSchema) Value() (driver.Value, error) {
	if c == nil {
		return nil, nil
	}
	return json.Marshal(c)
}

type RegisterCompoundComponentSchema struct {
	Name           CompoundComponentName            `json:"name"`
	Version        string                           `json:"version"`
	KubeNamespace  string                           `json:"kube_namespace"`
	SelectorLabels map[string]string                `json:"selector_labels,omitempty"`
	Manifest       *CompoundComponentManifestSchema `json:"manifest"`
}

type CompoundComponentSchema struct {
	ResourceSchema
	Creator           *UserSchema                      `json:"creator"`
	Cluster           *ClusterFullSchema               `json:"cluster"`
	Description       string                           `json:"description"`
	Version           string                           `json:"version"`
	KubeNamespace     string                           `json:"kube_namespace"`
	Manifest          *CompoundComponentManifestSchema `json:"manifest"`
	LatestInstalledAt *time.Time                       `json:"latest_installed_at"`
	LatestHeartbeatAt *time.Time                       `json:"latest_heartbeat_at"`
}
