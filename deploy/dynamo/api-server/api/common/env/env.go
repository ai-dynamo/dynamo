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

package env

import (
	"sync"

	"github.com/ai-dynamo/dynamo/deploy/dynamo/api-server/api/common/utils"
	"github.com/joho/godotenv"
	"github.com/rs/zerolog/log"
)

var (
	BackendUrl string
	once       sync.Once
)

func SetupEnv() {
	err := godotenv.Load()
	if err != nil {
		log.Fatal().Msgf("Failed to load env during setup %s", err.Error())
	}

	_, err = SetResourceScope()
	if err != nil {
		log.Fatal().Msgf("Failed to set resource scope during env setup %s", err.Error())
	}
}

func GetBackendUrl() string {
	// Gets the backend URL from the API_BACKEND_URL environment variable set in the Python script
	once.Do(func() { // cache and reuse
		var err error
		BackendUrl, err = utils.MustGetEnv("API_BACKEND_URL")
		if err != nil {
			log.Fatal().Msgf("Failed to get backend URL: %v", err)
		}
	})

	return BackendUrl
}
