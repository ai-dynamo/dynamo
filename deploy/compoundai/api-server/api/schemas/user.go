package schemas

type UserSchema struct {
	ResourceSchema
	FirstName string `json:"first_name"`
	LastName  string `json:"last_name"`
	Email     string `json:"email"`
}
