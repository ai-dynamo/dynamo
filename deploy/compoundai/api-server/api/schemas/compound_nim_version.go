package schemas

import "time"

type CompoundNimVersionSchema struct {
	ResourceSchema
	CompoundNimUid          string                         `json:"bento_repository_uid"`
	Creator                 *UserSchema                    `json:"creator"`
	Version                 string                         `json:"version"`
	Description             string                         `json:"description"`
	ImageBuildStatus        ImageBuildStatus               `json:"image_build_status"`
	UploadStatus            CompoundNimVersionUploadStatus `json:"upload_status"`
	UploadStartedAt         *time.Time                     `json:"upload_started_at"`
	UploadFinishedAt        *time.Time                     `json:"upload_finished_at"`
	UploadFinishedReason    string                         `json:"upload_finished_reason"`
	PresignedUploadUrl      string                         `json:"presigned_upload_url"`
	PresignedDownloadUrl    string                         `json:"presigned_download_url"`
	PresignedUrlsDeprecated bool                           `json:"presigned_urls_deprecated"`
	TransmissionStrategy    TransmissionStrategy           `json:"transmission_strategy"`
	UploadId                string                         `json:"upload_id"`
	Manifest                *CompoundNimManifestSchema     `json:"manifest"`
	BuildAt                 time.Time                      `json:"build_at"`
}

type CompoundNimVersionFullSchema struct {
	CompoundNimVersionSchema
	Repository *CompoundNimSchema `json:"repository"`
}

type GetCompoundNimVersionSchema struct {
	GetCompoundNimSchema
	CompoundNimVersion string `uri:"version" binding:"required"`
}

func (s *GetCompoundNimVersionSchema) Tag() *string {
	tag := s.CompoundNimName + ":" + s.CompoundNimVersion
	return &tag
}
