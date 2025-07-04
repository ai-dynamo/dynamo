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
