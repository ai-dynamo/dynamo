package models

type VisibilityLevel string

const (
	UserLevel VisibilityLevel = "user"
	OrgLevel  VisibilityLevel = "organization"
)

type Visibility struct {
	Visibility VisibilityLevel `json:"visibility"`
}
