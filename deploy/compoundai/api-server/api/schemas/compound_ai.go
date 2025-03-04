package schemas

type CompoundNimApiSchema struct {
	Route  string `json:"route"`
	Doc    string `json:"doc"`
	Input  string `json:"input"`
	Output string `json:"output"`
}

type CompoundNimManifestSchema struct {
	Service           string                          `json:"service"`
	CompoundAiVersion string                          `json:"bentoml_version"`
	Apis              map[string]CompoundNimApiSchema `json:"apis"`
	SizeBytes         uint                            `json:"size_bytes"`
}

type TransmissionStrategy string

const (
	TransmissionStrategyPresignedURL TransmissionStrategy = "presigned_url"
	TransmissionStrategyProxy        TransmissionStrategy = "proxy"
)

type CompoundNimVersionUploadStatus string

const (
	CompoundNimVersionUploadStatusPending   CompoundNimVersionUploadStatus = "pending"
	CompoundNimVersionUploadStatusUploading CompoundNimVersionUploadStatus = "uploading"
	CompoundNimVersionUploadStatusSuccess   CompoundNimVersionUploadStatus = "success"
	CompoundNimVersionUploadStatusFailed    CompoundNimVersionUploadStatus = "failed"
)

type ImageBuildStatus string

const (
	ImageBuildStatusPending  ImageBuildStatus = "pending"
	ImageBuildStatusBuilding ImageBuildStatus = "building"
	ImageBuildStatusSuccess  ImageBuildStatus = "success"
	ImageBuildStatusFailed   ImageBuildStatus = "failed"
)
