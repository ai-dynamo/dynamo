package modelschemas

import (
	"database/sql/driver"
	"encoding/json"
)

type TransmissionStrategy string

const (
	TransmissionStrategyPresignedURL TransmissionStrategy = "presigned_url"
	TransmissionStrategyProxy        TransmissionStrategy = "proxy"
)

type BentoUploadStatus string

const (
	BentoUploadStatusPending   BentoUploadStatus = "pending"
	BentoUploadStatusUploading BentoUploadStatus = "uploading"
	BentoUploadStatusSuccess   BentoUploadStatus = "success"
	BentoUploadStatusFailed    BentoUploadStatus = "failed"
)

type ImageBuildStatus string

const (
	ImageBuildStatusPending  ImageBuildStatus = "pending"
	ImageBuildStatusBuilding ImageBuildStatus = "building"
	ImageBuildStatusSuccess  ImageBuildStatus = "success"
	ImageBuildStatusFailed   ImageBuildStatus = "failed"
)

type BentoApiSchema struct {
	Route  string `json:"route"`
	Doc    string `json:"doc"`
	Input  string `json:"input"`
	Output string `json:"output"`
}

type BentoRunnerResourceSchema struct {
	CPU             *float64           `json:"cpu"`
	NvidiaGPU       *float64           `json:"nvidia_gpu"`
	CustomResources map[string]float64 `json:"custom_resources"`
}

type BentoRunnerSchema struct {
	Name           string                     `json:"name"`
	RunnableType   string                     `json:"runnable_type"`
	Models         []string                   `json:"models"`
	ResourceConfig *BentoRunnerResourceSchema `json:"resource_config"`
}

type BentoManifestSchema struct {
	Service        string                    `json:"service"`
	BentomlVersion string                    `json:"bentoml_version"`
	Apis           map[string]BentoApiSchema `json:"apis"`
	Models         []string                  `json:"models"`
	Runners        []BentoRunnerSchema       `json:"runners"`
	SizeBytes      uint                      `json:"size_bytes"`
}

func (c *BentoManifestSchema) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	return json.Unmarshal(value.([]byte), c)
}

func (c *BentoManifestSchema) Value() (driver.Value, error) {
	if c == nil {
		return nil, nil
	}
	return json.Marshal(c)
}
