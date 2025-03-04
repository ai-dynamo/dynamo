package schemasv1

import (
	"time"

	"github.com/dynemo-ai/dynemo/deploy/compoundai/operator/api/compoundai/modelschemas"
)

type ApiTokenSchema struct {
	ResourceSchema
	Description  string                       `json:"description"`
	User         *UserSchema                  `json:"user"`
	Organization *OrganizationSchema          `json:"organization"`
	Scopes       *modelschemas.ApiTokenScopes `json:"scopes"`
	ExpiredAt    *time.Time                   `json:"expired_at"`
	LastUsedAt   *time.Time                   `json:"last_used_at"`
	IsExpired    bool                         `json:"is_expired"`
}

type ApiTokenListSchema struct {
	BaseListSchema
	Items []*ApiTokenSchema `json:"items"`
}

type ApiTokenFullSchema struct {
	ApiTokenSchema
	Token string `json:"token"`
}

type UpdateApiTokenSchema struct {
	Description *string                      `json:"description"`
	Scopes      *modelschemas.ApiTokenScopes `json:"scopes"`
	ExpiredAt   *time.Time                   `json:"expired_at"`
	LastUsedAt  *time.Time                   `json:"last_used_at"`
}

type CreateApiTokenSchema struct {
	Name        string                       `json:"name"`
	Description string                       `json:"description"`
	Scopes      *modelschemas.ApiTokenScopes `json:"scopes"`
	ExpiredAt   *time.Time                   `json:"expired_at"`
}
