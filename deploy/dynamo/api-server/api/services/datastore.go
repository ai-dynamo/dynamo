/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package services

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"

	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/common/client"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/common/env"
	"github.com/dynemo-ai/dynemo/deploy/dynamo/api-server/api/schemas"
	"github.com/rs/zerolog/log"
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
