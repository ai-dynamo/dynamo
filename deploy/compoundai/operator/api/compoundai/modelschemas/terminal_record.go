package modelschemas

import (
	"database/sql/driver"
	"encoding/json"
)

type RecordType string

const (
	RecordTypeInput  RecordType = "i"
	RecordTypeOutput RecordType = "o"
)

type TerminalRecordEnv struct {
	TERM  string `json:"term"`
	SHELL string `json:"shell"`
}

type TerminalRecordMeta struct {
	Version   uint               `json:"version"`
	Width     uint16             `json:"width"`
	Height    uint16             `json:"height"`
	Timestamp int64              `json:"timestamp"`
	Env       *TerminalRecordEnv `json:"env"`
}

func (m *TerminalRecordMeta) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	return json.Unmarshal([]byte(value.(string)), m)
}

func (m *TerminalRecordMeta) Value() (driver.Value, error) {
	if m == nil {
		return nil, nil
	}
	return json.Marshal(m)
}
