package schemasv1

import "github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/modelschemas"

type CreateMembersSchema struct {
	Usernames []string                `json:"usernames"`
	Role      modelschemas.MemberRole `json:"role" enum:"guest,developer,admin"`
}

type DeleteMemberSchema struct {
	Username string `json:"username"`
}
