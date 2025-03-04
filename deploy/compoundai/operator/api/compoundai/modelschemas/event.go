package modelschemas

import (
	"database/sql/driver"
	"encoding/json"
)

type EventStatus string

const (
	EventStatusPending EventStatus = "pending"
	EventStatusSuccess EventStatus = "success"
	EventStatusFailed  EventStatus = "failed"
)

func (e EventStatus) Ptr() *EventStatus {
	return &e
}

type EventInfo struct {
	ResourceName string `json:"resource_name"`
}

func (c *EventInfo) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	return json.Unmarshal([]byte(value.(string)), c)
}

func (c *EventInfo) Value() (driver.Value, error) {
	if c == nil {
		return nil, nil
	}
	return json.Marshal(c)
}
