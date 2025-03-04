package schemasv1

import (
	"time"

	"github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/modelschemas"
)

type ModelSchema struct {
	ResourceSchema
	ModelUid                string                             `json:"model_uid"`
	Creator                 *UserSchema                        `json:"creator"`
	Version                 string                             `json:"version"`
	Description             string                             `json:"description"`
	ImageBuildStatus        modelschemas.ImageBuildStatus      `json:"image_build_status"`
	UploadStatus            modelschemas.ModelUploadStatus     `json:"upload_status"`
	UploadStartedAt         *time.Time                         `json:"upload_started_at"`
	UploadFinishedAt        *time.Time                         `json:"upload_finished_at"`
	UploadFinishedReason    string                             `json:"upload_finished_reason"`
	PresignedUploadUrl      string                             `json:"presigned_upload_url"`
	PresignedDownloadUrl    string                             `json:"presigned_download_url"`
	PresignedUrlsDeprecated bool                               `json:"presigned_urls_deprecated"`
	TransmissionStrategy    *modelschemas.TransmissionStrategy `json:"transmission_strategy"`
	UploadId                string                             `json:"upload_id"`
	Manifest                *modelschemas.ModelManifestSchema  `json:"manifest"`
	BuildAt                 time.Time                          `json:"build_at"`
}

type ModelListSchema struct {
	BaseListSchema
	Items []*ModelSchema `json:"items"`
}

type ModelWithRepositorySchema struct {
	ModelSchema
	Repository *ModelRepositorySchema `json:"repository"`
}

type ModelWithRepositoryListSchema struct {
	BaseListSchema
	Items []*ModelWithRepositorySchema `json:"items"`
}

type ModelFullSchema struct {
	ModelWithRepositorySchema
	Bentos []*BentoSchema `json:"bentos"`
}

type CreateModelSchema struct {
	Version     string                            `json:"version"`
	Description string                            `json:"description"`
	Manifest    *modelschemas.ModelManifestSchema `json:"manifest"`
	BuildAt     string                            `json:"build_at"`
	Labels      modelschemas.LabelItemsSchema     `json:"labels"`
}

type UpdateModelSchema struct {
	Description string                            `json:"description,omitempty"`
	Manifest    *modelschemas.ModelManifestSchema `json:"manifest"`
	BuildAt     string                            `json:"build_at"`
	Labels      *modelschemas.LabelItemsSchema    `json:"labels,omitempty"`
}

type FinishUploadModelSchema struct {
	Status *modelschemas.ModelUploadStatus `json:"status"`
	Reason *string                         `json:"reason"`
}
