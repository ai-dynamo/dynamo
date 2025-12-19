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
package common

import (
	"fmt"
	"net/url"
	"strings"
)

func GetHost(someURL string) (string, error) {
	// Add scheme if not present
	if !strings.Contains(someURL, "://") {
		someURL = "dummy://" + someURL
	}
	url, err := url.Parse(someURL)
	if err != nil {
		return "", err
	}
	if url.Host == "" {
		return "", fmt.Errorf("no host found in URL %q", someURL)
	}
	return url.Host, nil
}
