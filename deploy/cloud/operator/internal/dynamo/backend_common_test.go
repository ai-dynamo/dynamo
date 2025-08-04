package dynamo

import (
	"testing"

	"github.com/onsi/gomega"
	ptr "k8s.io/utils/ptr"
)

func TestApplyFlagOverridesAndExtraArgs(t *testing.T) {
	tests := []struct {
		name          string
		baseFlags     map[string]*string
		flagOverrides map[string]*string
		extraArgs     []string
		expected      []string
	}{
		{
			name: "no overrides or extra args",
			baseFlags: map[string]*string{
				"model": ptr.To("test-model"),
				"port":  ptr.To("8000"),
			},
			flagOverrides: nil,
			extraArgs:     nil,
			expected:      []string{"--model test-model", "--port 8000"},
		},
		{
			name: "with flag overrides",
			baseFlags: map[string]*string{
				"model": ptr.To("test-model"),
				"port":  ptr.To("8000"),
			},
			flagOverrides: map[string]*string{
				"model": ptr.To("overridden-model"),
				"new":   ptr.To("new-value"),
			},
			extraArgs: nil,
			expected:  []string{"--model overridden-model", "--port 8000", "--new new-value"},
		},
		{
			name: "with flag removal",
			baseFlags: map[string]*string{
				"model": ptr.To("test-model"),
				"port":  ptr.To("8000"),
			},
			flagOverrides: map[string]*string{
				"port": nil, // Remove this flag
			},
			extraArgs: nil,
			expected:  []string{"--model test-model"},
		},
		{
			name: "with extra args",
			baseFlags: map[string]*string{
				"model": ptr.To("test-model"),
			},
			flagOverrides: nil,
			extraArgs:     []string{"--extra", "value", "--another"},
			expected:      []string{"--model test-model", "--extra", "value", "--another"},
		},
		{
			name: "with everything",
			baseFlags: map[string]*string{
				"model": ptr.To("test-model"),
				"port":  ptr.To("8000"),
				"gpu":   ptr.To("1"),
			},
			flagOverrides: map[string]*string{
				"model": ptr.To("new-model"),
				"port":  nil,          // Remove
				"batch": ptr.To("32"), // Add new
			},
			extraArgs: []string{"--custom", "arg"},
			expected:  []string{"--model new-model", "--gpu 1", "--batch 32", "--custom", "arg"},
		},
		{
			name:          "empty inputs",
			baseFlags:     map[string]*string{},
			flagOverrides: map[string]*string{},
			extraArgs:     []string{},
			expected:      []string{},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			g := gomega.NewGomegaWithT(t)
			result := applyFlagOverridesAndExtraArgs(tt.baseFlags, tt.flagOverrides, tt.extraArgs)

			// Since map iteration is not deterministic, we use ConsistOf to check elements match
			g.Expect(result).To(gomega.ConsistOf(tt.expected))
		})
	}
}
