package schemas

import (
	"encoding/json"
)

type ExternalService struct {
	DeploymentSelectorKey   string `json:"-"`
	DeploymentSelectorValue string `json:"-"`
}

// UnmarshalJSON handles snake_case to struct mapping
func (e *ExternalService) UnmarshalJSON(data []byte) error {
	var temp map[string]interface{}
	if err := json.Unmarshal(data, &temp); err != nil {
		return err
	}

	if val, ok := temp["deployment_selector_key"].(string); ok {
		e.DeploymentSelectorKey = val
	}
	if val, ok := temp["deployment_selector_value"].(string); ok {
		e.DeploymentSelectorValue = val
	}
	return nil
}

// MarshalJSON converts the struct to camelCase
func (e ExternalService) MarshalJSON() ([]byte, error) {
	temp := map[string]interface{}{
		"deploymentSelectorKey":   e.DeploymentSelectorKey,
		"deploymentSelectorValue": e.DeploymentSelectorValue,
	}
	return json.Marshal(temp)
}
