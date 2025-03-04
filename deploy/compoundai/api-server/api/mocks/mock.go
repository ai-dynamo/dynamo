package mocks

import (
	"time"

	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/schemas"
)

var mockedUid = "nvid1a11-1234-5678-9abc-def012345678"

func DefaultUser() *schemas.UserSchema {
	return &schemas.UserSchema{
		ResourceSchema: schemas.ResourceSchema{
			BaseSchema: schemas.BaseSchema{
				Uid:       mockedUid,
				CreatedAt: time.Now(),
				UpdatedAt: time.Now(),
				DeletedAt: nil,
			},
			Name: "nvidia-user",
		},
		FirstName: "Compound",
		LastName:  "AI",
		Email:     "compoundai@nvidia.com",
	}
}

func DefaultOrg() *schemas.OrganizationSchema {
	return &schemas.OrganizationSchema{
		ResourceSchema: schemas.ResourceSchema{
			BaseSchema: schemas.BaseSchema{
				Uid:       mockedUid,
				CreatedAt: time.Now(),
				UpdatedAt: time.Now(),
				DeletedAt: nil,
			},
			Name:         "nvidia-org",
			ResourceType: schemas.ResourceTypeOrganization,
			Labels:       []schemas.LabelItemSchema{},
		},
		Description: "nvidia-org-desc",
	}
}

func DefaultOrgMember() *schemas.OrganizationMemberSchema {
	return &schemas.OrganizationMemberSchema{
		BaseSchema: schemas.BaseSchema{
			Uid:       mockedUid,
			CreatedAt: time.Now(),
			UpdatedAt: time.Now(),
			DeletedAt: nil,
		},
		Role:         schemas.MemberRoleAdmin,
		Creator:      DefaultUser(),
		User:         *DefaultUser(),
		Organization: *DefaultOrg(),
	}
}
