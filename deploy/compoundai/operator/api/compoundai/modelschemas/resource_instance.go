package modelschemas

type ResourceInstance struct {
	ID            string                    `json:"id"`
	Name          string                    `json:"name"`
	Group         string                    `json:"group"`
	Description   string                    `json:"description"`
	NodeSelectors map[string]string         `json:"node_selectors"`
	Resources     DeploymentTargetResources `json:"resources"`
	Price         string                    `json:"price"`
}
