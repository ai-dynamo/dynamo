package schemas

type MemberRole string

const (
	MemberRoleGuest     MemberRole = "guest"
	MemberRoleDeveloper MemberRole = "developer"
	MemberRoleAdmin     MemberRole = "admin"
)

type OrganizationMemberSchema struct {
	BaseSchema
	Role         MemberRole         `json:"role"`
	Creator      *UserSchema        `json:"creator"`
	User         UserSchema         `json:"user"`
	Organization OrganizationSchema `json:"organization"`
}
