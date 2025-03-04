package schemas

type BaseListSchema struct {
	Total uint `json:"total"`
	Start uint `json:"start"`
	Count uint `json:"count"`
}

type ListQuerySchema struct {
	Start  uint    `form:"start"`
	Count  uint    `form:"count"`
	Search *string `form:"search"`
	Q      string  `query:"q"`
}
