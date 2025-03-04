package modelschemas

import (
	"database/sql/driver"
	"encoding/json"
)

type ModelUploadStatus string

const (
	ModelUploadStatusPending   ModelUploadStatus = "pending"
	ModelUploadStatusUploading ModelUploadStatus = "uploading"
	ModelUploadStatusSuccess   ModelUploadStatus = "success"
	ModelUploadStatusFailed    ModelUploadStatus = "failed"
)

type ModelManifestSchema struct {
	BentomlVersion string                 `json:"bentoml_version"`
	ApiVersion     string                 `json:"api_version"`
	Module         string                 `json:"module"`
	Metadata       map[string]interface{} `json:"metadata"`
	Context        map[string]interface{} `json:"context"`
	Options        map[string]interface{} `json:"options"`
	SizeBytes      uint                   `json:"size_bytes"`
}

func (c *ModelManifestSchema) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	return json.Unmarshal(value.([]byte), c)
}

func (c *ModelManifestSchema) Value() (driver.Value, error) {
	if c == nil {
		return nil, nil
	}
	return json.Marshal(c)
}
