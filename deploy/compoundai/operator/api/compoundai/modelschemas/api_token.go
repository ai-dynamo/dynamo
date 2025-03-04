package modelschemas

import (
	"database/sql/driver"
	"encoding/json"
)

type ApiTokenScopeOp string

const (
	ApiTokenScopeOpRead    ApiTokenScopeOp = "read"
	ApiTokenScopeOpWrite   ApiTokenScopeOp = "write"
	ApiTokenScopeOpOperate ApiTokenScopeOp = "operate"
)

type ApiTokenScope string

const (
	ApiTokenScopeApi ApiTokenScope = "api"
)

type ApiTokenScopes []ApiTokenScope

func (c *ApiTokenScopes) Scan(value interface{}) error {
	if value == nil {
		return nil
	}
	return json.Unmarshal([]byte(value.(string)), c)
}

func (c *ApiTokenScopes) Value() (driver.Value, error) {
	if c == nil {
		return nil, nil
	}
	return json.Marshal(c)
}

func (c *ApiTokenScopes) Contains(scope ApiTokenScope) bool {
	for _, s := range *c {
		if s == scope {
			return true
		}
	}
	return false
}
