/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

package defaulting

import (
	"errors"
	"fmt"

	"github.com/ai-dynamo/dynamo/deploy/operator/internal/versionoverride"
)

// ValidateAISimulateDefault rejects an operator configuration whose default AISimulate artifact
// validation would reject on every Request that receives it: an image without a semantic-version
// tag needs the override that names its compatibility version. An override without an image is
// rejected too, because it would silently pin the version of the image derived from the
// operator version.
func ValidateAISimulateDefault(image, versionOverride string) error {
	if image == "" {
		if versionOverride != "" {
			return errors.New("the AISimulate default version override needs the AISimulate default image")
		}
		return nil
	}
	if _, err := versionoverride.Resolve(image, versionOverride); err != nil {
		return fmt.Errorf("the AISimulate default image %q cannot be used as it is: %w; "+
			"set the AISimulate default version override", image, err)
	}
	return nil
}
