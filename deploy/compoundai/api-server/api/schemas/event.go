package schemas

import (
	"time"
)

type EventStatus string

const (
	EventStatusPending EventStatus = "pending"
	EventStatusSuccess EventStatus = "success"
	EventStatusFailed  EventStatus = "failed"
)

type EventSchema struct {
	BaseSchema
	Resource        interface{} `json:"resource,omitempty"`
	Name            string      `json:"name,omitempty"`
	Status          EventStatus `json:"status,omitempty"`
	OperationName   string      `json:"operation_name,omitempty"`
	ApiTokenName    string      `json:"api_token_name,omitempty"`
	Creator         *UserSchema `json:"creator,omitempty"`
	CreatedAt       time.Time   `json:"created_at,omitempty"`
	ResourceDeleted bool        `json:"resource_deleted,omitempty"`
}

type EventListSchema struct {
	BaseListSchema
	Items []*EventSchema `json:"items"`
}
