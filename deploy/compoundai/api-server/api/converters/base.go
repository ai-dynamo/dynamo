package converters

import (
	"time"

	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/models"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/schemas"
)

func ToBaseSchema(base models.BaseModel) schemas.BaseSchema {
	var deletedAt *time.Time
	deletedAt_ := base.GetDeletedAt()
	if deletedAt_.Valid {
		deletedAt = &deletedAt_.Time
	}
	return schemas.BaseSchema{
		Uid:       base.GetUid(),
		CreatedAt: base.GetCreatedAt(),
		UpdatedAt: base.GetUpdatedAt(),
		DeletedAt: deletedAt,
	}
}
