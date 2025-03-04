package services

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/rs/zerolog/log"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/common/client"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/common/env"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/schemas"
)

type datastoreService struct{}

var DatastoreService = datastoreService{}

/**
	This service connects to the Nemo Datastore Microservice

	Note: We should not do any write requests via this service as transactionality is not guaranteed in this way
**/

func (s *datastoreService) GetCompoundNimVersion(ctx context.Context, compoundNim string, compoundNimVersion string) (*schemas.CompoundNimVersionFullSchema, error) {
	ndsUrl := env.GetNdsUrl()
	getUrl := fmt.Sprintf("%s/api/v1/bento_repositories/%s/bentos/%s", ndsUrl, compoundNim, compoundNimVersion)

	_, body, err := client.SendRequestJSON(getUrl, http.MethodGet, nil)
	if err != nil {
		log.Error().Msgf("Failed to get Compound NIM version %s:%s from %s", compoundNim, compoundNimVersion, ndsUrl)
		return nil, err
	}

	var schema schemas.CompoundNimVersionFullSchema
	if err = json.Unmarshal(body, &schema); err != nil {
		log.Error().Msgf("Failed to unmarshal into a Compound NIM version schema: %s", err.Error())
		return nil, err
	}

	return &schema, nil
}
