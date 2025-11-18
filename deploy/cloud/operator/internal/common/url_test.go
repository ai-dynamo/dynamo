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

import "testing"

func TestGetHost(t *testing.T) {
	type args struct {
		someURL string
	}
	tests := []struct {
		name    string
		args    args
		want    string
		wantErr bool
	}{
		{
			name: "docker.io",
			args: args{
				someURL: "docker.io",
			},
			want:    "docker.io",
			wantErr: false,
		},
		{
			name: "gitlab-master.nvidia.com:5005",
			args: args{
				someURL: "gitlab-master.nvidia.com:5005",
			},
			want:    "gitlab-master.nvidia.com:5005",
			wantErr: false,
		},
		{
			name: "gitlab-master.nvidia.com:5005/registry",
			args: args{
				someURL: "gitlab-master.nvidia.com:5005/registry",
			},
			want:    "gitlab-master.nvidia.com:5005",
			wantErr: false,
		},
		{
			name: "https://gitlab-master.nvidia.com",
			args: args{
				someURL: "https://gitlab-master.nvidia.com",
			},
			want:    "gitlab-master.nvidia.com",
			wantErr: false,
		},
		{
			name: "https://gitlab-master.nvidia.com:5005/registry",
			args: args{
				someURL: "https://gitlab-master.nvidia.com:5005/registry",
			},
			want:    "gitlab-master.nvidia.com:5005",
			wantErr: false,
		},
		{
			name: "empty",
			args: args{
				someURL: "",
			},
			want:    "",
			wantErr: true,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := GetHost(tt.args.someURL)
			if (err != nil) != tt.wantErr {
				t.Errorf("GetHost() error = %v, wantErr %v", err, tt.wantErr)
				return
			}
			if got != tt.want {
				t.Errorf("GetHost() = %v, want %v", got, tt.want)
			}
		})
	}
}
