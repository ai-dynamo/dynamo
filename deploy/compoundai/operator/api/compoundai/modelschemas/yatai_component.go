package modelschemas

import (
	"database/sql/driver"
	"encoding/json"
)

type YataiComponentName string

const (
	YataiComponentNameDeployment   YataiComponentName = "deployment"
	YataiComponentNameImageBuilder YataiComponentName = "image-builder"
	YataiComponentNameServerless   YataiComponentName = "serverless"
	YataiComponentNameFunction     YataiComponentName = "function"
	YataiComponentNameJob          YataiComponentName = "job"
)

type YataiComponentManifestSchema struct {
	SelectorLabels   map[string]string `json:"selector_labels,omitempty"`
	LatestCRDVersion string            `json:"latest_crd_version,omitempty"`
}

func (c *YataiComponentManifestSchema) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	return json.Unmarshal(value.([]byte), c)
}

func (c *YataiComponentManifestSchema) Value() (driver.Value, error) {
	if c == nil {
		return nil, nil
	}
	return json.Marshal(c)
}
