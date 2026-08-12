package types

import (
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func validAgentConfig() *AgentConfig {
	return &AgentConfig{
		Storage: StorageSpec{
			Type:     "pvc",
			BasePath: "/checkpoints",
		},
		Restore: RestoreSpec{
			RestoreTimeoutSeconds: 60,
		},
	}
}

func TestAgentConfigValidate_StorageBasePath(t *testing.T) {
	t.Parallel()

	tests := []struct {
		name     string
		basePath string
		wantErr  bool
		want     string
	}{
		{
			name:     "surrounding whitespace is trimmed",
			basePath: " /checkpoints ",
			want:     "/checkpoints",
		},
		{
			name:     "relative path is rejected",
			basePath: "checkpoints",
			wantErr:  true,
		},
		{
			// The base path is the containment boundary for every artifact the
			// agent mounts, so an unclean one is refused rather than cleaned.
			name:     "unclean path is rejected",
			basePath: "/checkpoints/../etc",
			wantErr:  true,
		},
		{
			name:     "trailing slash is unclean",
			basePath: "/checkpoints/",
			wantErr:  true,
		},
		{
			name:     "empty path is rejected",
			basePath: "   ",
			wantErr:  true,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			cfg := validAgentConfig()
			cfg.Storage.BasePath = tc.basePath

			err := cfg.Validate()
			if tc.wantErr {
				require.Error(t, err)
				return
			}
			require.NoError(t, err)
			assert.Equal(t, tc.want, cfg.Storage.BasePath)
		})
	}
}
