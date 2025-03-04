package modelschemas

type UserPerm string

const (
	UserPermDefault UserPerm = "default"
	UserPermAdmin   UserPerm = "admin"
)

func UserPermPtr(perm UserPerm) *UserPerm {
	return &perm
}
