package modelschemas

import (
	"database/sql/driver"
	"encoding/json"
)

type ClusterConfigAWSSchema struct {
	Region string `json:"region"`
}

type ClusterConfigSchema struct {
	DefaultDeploymentKubeNamespace string                  `json:"default_deployment_kube_namespace"`
	IngressIp                      string                  `json:"ingress_ip"`
	AWS                            *ClusterConfigAWSSchema `json:"aws"`
	ResourceInstances              []ResourceInstance      `json:"resource_instances"`
}

func (c *ClusterConfigSchema) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	return json.Unmarshal([]byte(value.(string)), c)
}

func (c *ClusterConfigSchema) Value() (driver.Value, error) {
	if c == nil {
		return nil, nil
	}
	return json.Marshal(c)
}
