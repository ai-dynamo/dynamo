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

package runtime

import (
	"fmt"

	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/common/env"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/database"
	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/routes"
	"github.com/rs/zerolog/log"
)

type runtime struct{}

var Runtime = runtime{}

func (r *runtime) StartServer(port int) {
	env.SetupEnv()

	database.SetupDB()
	router := routes.SetupRouter()

	log.Info().Msgf("Starting CompoundAI API server on port %d", port)

	router.Run(fmt.Sprintf("0.0.0.0:%d", port))
}
