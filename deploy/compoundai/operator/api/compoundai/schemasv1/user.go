package schemasv1

import "github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/modelschemas"

type UserSchema struct {
	ResourceSchema
	FirstName    string `json:"first_name"`
	LastName     string `json:"last_name"`
	Email        string `json:"email"`
	AvatarUrl    string `json:"avatar_url"`
	IsSuperAdmin bool   `json:"is_super_admin"`
}

type UserListSchema struct {
	BaseListSchema
	Items []*UserSchema `json:"items"`
}

type RegisterUserSchema struct {
	Name      string `json:"name" validate:"required"`
	FirstName string `json:"first_name"`
	LastName  string `json:"last_name"`
	Email     string `json:"email" validate:"required"`
	Password  string `json:"password" validate:"required"`
}

type LoginUserSchema struct {
	NameOrEmail string `json:"name_or_email" validate:"required"`
	Password    string `json:"password" validate:"required"`
}

type UpdateUserSchema struct {
	FirstName string `json:"first_name" validate:"required"`
	LastName  string `json:"last_name" validate:"required"`
}

type ResetPasswordSchema struct {
	CurrentPassword string `json:"current_password"`
	NewPassword     string `json:"new_password"`
}

type CreateUserSchema struct {
	Name     string                  `json:"name" validate:"required"`
	Email    string                  `json:"email" validate:"required"`
	Password string                  `json:"password" validate:"required"`
	Role     modelschemas.MemberRole `json:"role" enum:"guest,developer,admin"`
}
