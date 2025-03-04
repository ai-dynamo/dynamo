package schemas

import "time"

type BaseSchema struct {
	Uid       string     `json:"uid"`
	CreatedAt time.Time  `json:"created_at"`
	UpdatedAt time.Time  `json:"updated_at"`
	DeletedAt *time.Time `json:"deleted_at"`
}
