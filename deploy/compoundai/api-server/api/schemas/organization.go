package schemas

type OrganizationSchema struct {
	ResourceSchema
	Creator     *UserSchema `json:"creator"`
	Description string      `json:"description"`
}

type OrganizationFullSchema struct {
	OrganizationSchema
}

type OrganizationListSchema struct {
	BaseListSchema
	Items []*OrganizationSchema `json:"items"`
}

type UpdateOrganizationSchema struct {
	Description *string `json:"description"`
}

type CreateOrganizationSchema struct {
	Name        string `json:"name"`
	Description string `json:"description"`
}
