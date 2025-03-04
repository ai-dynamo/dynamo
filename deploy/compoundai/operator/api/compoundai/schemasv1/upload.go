package schemasv1

type CompletePartSchema struct {
	PartNumber int    `json:"part_number" binding:"required"`
	ETag       string `json:"etag" binding:"required"`
}

type PreSignMultipartUploadSchema struct {
	UploadId   string `json:"upload_id" binding:"required"`
	PartNumber int    `json:"part_number" binding:"required"`
}

type CompleteMultipartUploadSchema struct {
	UploadId string               `json:"upload_id" binding:"required"`
	Parts    []CompletePartSchema `json:"parts" binding:"required"`
}
