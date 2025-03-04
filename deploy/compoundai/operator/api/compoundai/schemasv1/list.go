package schemasv1

type BaseListSchema struct {
	Total uint `json:"total"`
	Start uint `json:"start"`
	Count uint `json:"count"`
}

type ListQuerySchema struct {
	Start  uint    `query:"start"`
	Count  uint    `query:"count"`
	Search *string `query:"search"`
	Q      Q       `query:"q"`
}
