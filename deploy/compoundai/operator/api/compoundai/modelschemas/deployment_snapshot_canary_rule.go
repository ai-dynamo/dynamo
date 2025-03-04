package modelschemas

import (
	"database/sql/driver"
	"encoding/json"
)

type DeploymentTargetCanaryRuleType string

const (
	DeploymentTargetCanaryRuleTypeWeight DeploymentTargetCanaryRuleType = "weight"
	DeploymentTargetCanaryRuleTypeHeader DeploymentTargetCanaryRuleType = "header"
	DeploymentTargetCanaryRuleTypeCookie DeploymentTargetCanaryRuleType = "cookie"
)

type DeploymentTargetCanaryRule struct {
	Type DeploymentTargetCanaryRuleType `json:"type" enum:"weight,header,cookie"`

	Weight      *uint   `json:"weight"`
	Header      *string `json:"header"`
	Cookie      *string `json:"cookie"`
	HeaderValue *string `json:"header_value"`
}

type DeploymentTargetCanaryRules []*DeploymentTargetCanaryRule

func (c *DeploymentTargetCanaryRules) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	return json.Unmarshal([]byte(value.(string)), c)
}

func (c *DeploymentTargetCanaryRules) Value() (driver.Value, error) {
	if c == nil {
		return nil, nil
	}
	return json.Marshal(c)
}
