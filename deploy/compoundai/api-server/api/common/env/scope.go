package env

import (
	"fmt"
	"os"
	"sync"
)

type ResourceScope string

const (
	OrganizationScope ResourceScope = "organization"
	UserScope         ResourceScope = "user"
)

var (
	ApplicationScope ResourceScope
	getScopeOnce     sync.Once
)

func SetResourceScope() (ResourceScope, error) {
	var err error
	getScopeOnce.Do(func() {
		scope := os.Getenv("RESOURCE_SCOPE")
		if scope == "" {
			scope = string(UserScope)
		}

		switch ResourceScope(scope) {
		case OrganizationScope, UserScope:
			ApplicationScope = ResourceScope(scope)
		default:
			err = fmt.Errorf("invalid scope value: %s", scope)
			return
		}
	})

	if err != nil {
		return "", err
	}

	return ApplicationScope, nil
}
