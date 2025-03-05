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
	"fmt"
	"sync"

	"github.com/dynemo-ai/dynemo/deploy/compoundai/api-server/api/common/utils"
)

var (
	NdsHostBase string
	once        sync.Once
)

func GetNdsUrl() string {
	baseUrl := GetNdsHost()
	return fmt.Sprintf("http://%s", baseUrl)
}

func GetNdsHost() string {
	return NdsHostBase
}

func SetNdsHost() (string, error) {
	var err error
	once.Do(func() { // We cache and reuse the same NDS host
		NDS_HOST, syncErr := utils.MustGetEnv("NDS_HOST")
		if syncErr != nil {
			err = syncErr
			return
		}

		NDS_PORT, syncErr := utils.MustGetEnv("NDS_PORT")
		if syncErr != nil {
			err = syncErr
			return
		}

		NdsHostBase = fmt.Sprintf("%s:%s", NDS_HOST, NDS_PORT)
	})

	if err != nil {
		return "", err
	}

	return NdsHostBase, nil
}
