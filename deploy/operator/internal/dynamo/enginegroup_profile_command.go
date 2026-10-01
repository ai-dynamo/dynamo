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

package dynamo

import "strings"

func isSupportedProfilePythonExecutable(executable string) bool {
	// Keep the invocation contract to interpreters used by Dynamo's shipped workloads.
	switch executable {
	case "python", "python3":
		return true
	default:
		return false
	}
}

func isProfileShellOrWrapper(executable string) bool {
	// These executables can evaluate or replace argv after Kubernetes creates the container.
	switch executable {
	case "sh", "bash", "dash", "ash", "zsh", "ksh", "fish", "env", "uv":
		return true
	default:
		return false
	}
}

func hasKubernetesArgumentExpansion(value string) bool {
	// Match kubelet's dollar escaping so $$(NAME) remains a literal while $(NAME) is dynamic.
	for index := 0; index+1 < len(value); index++ {
		if value[index] != '$' {
			continue
		}
		if value[index+1] == '$' {
			index++
			continue
		}
		if value[index+1] == '(' && strings.Contains(value[index+2:], ")") {
			return true
		}
	}

	return false
}
