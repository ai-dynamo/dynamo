package schemasv1

type KubeResourceSchema struct {
	APIVersion  string            `json:"api_version"`
	Kind        string            `json:"kind"`
	Name        string            `json:"name"`
	Namespace   string            `json:"namespace"`
	MatchLabels map[string]string `json:"match_labels"`
}
