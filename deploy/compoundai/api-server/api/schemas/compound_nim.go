package schemas

type CompoundNimSchema struct {
	ResourceSchema
	Creator                   *UserSchema                 `json:"creator"`
	Organization              *OrganizationSchema         `json:"organization"`
	LatestCompoundNimVersion  *CompoundNimVersionSchema   `json:"latest_bento"`
	NCompoundNimVersions      uint                        `json:"n_bentos"`
	NDeployments              uint                        `json:"n_deployments"`
	LatestCompoundNimVersions []*CompoundNimVersionSchema `json:"latest_bentos"`
	Description               string                      `json:"description"`
}

type GetCompoundNimSchema struct {
	CompoundNimName string `uri:"compoundNimName" binding:"required"`
}
