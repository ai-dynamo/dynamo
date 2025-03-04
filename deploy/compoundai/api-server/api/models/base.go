package models

import (
	"time"

	"github.com/google/uuid"
	"gorm.io/gorm"
)

type IBaseModel interface {
	GetId() uint
	GetUid() string
	GetCreatedAt() time.Time
	GetUpdatedAt() time.Time
	GetDeletedAt() gorm.DeletedAt
}

type BaseModel struct {
	gorm.Model
	Uid uuid.UUID `json:"uid" gorm:"type:uuid;default:gen_random_uuid()"`
}

func (b *BaseModel) GetId() uint {
	return b.ID
}

func (b *BaseModel) GetUid() string {
	return b.Uid.String()
}

func (b *BaseModel) GetCreatedAt() time.Time {
	return b.CreatedAt
}

func (b *BaseModel) GetUpdatedAt() time.Time {
	return b.UpdatedAt
}

func (b *BaseModel) GetDeletedAt() gorm.DeletedAt {
	return b.DeletedAt
}
