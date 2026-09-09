/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

package lpx

import "testing"

func TestParse(t *testing.T) {
	t.Log("Define modern, legacy, and malformed topology strings")
	tests := []struct {
		name    string
		raw     string
		want    Topology
		wantErr bool
	}{
		{
			name: "without optional fields",
			raw:  "URSA_V2_1__INT8__32C__G_128_128__FEC_A__GHZ_1_7__FPGA_X",
			want: Topology{
				ChipCount:        32,
				Raw:              "URSA_V2_1__INT8__32C__G_128_128__FEC_A__GHZ_1_7__FPGA_X",
				compatibilityKey: "URSA_V2_1__INT8__C__G_128_128__FEC_A__GHZ_1_7__FPGA_X",
			},
		},
		{
			name: "with optional fields",
			raw:  "URSA_V2_1__FP8__64C__G_256_256__FEC_B__GHZ_2_0__FPGA_Y__G_512_512__BETA",
			want: Topology{
				ChipCount:        64,
				Raw:              "URSA_V2_1__FP8__64C__G_256_256__FEC_B__GHZ_2_0__FPGA_Y__G_512_512__BETA",
				compatibilityKey: "URSA_V2_1__FP8__C__G_256_256__FEC_B__GHZ_2_0__FPGA_Y__G_512_512__BETA",
			},
		},
		{
			name: "legacy",
			raw:  "RT_A14_8_CHIP_PS2_NOFEC",
			want: Topology{
				ChipCount: 8,
				Raw:       "RT_A14_8_CHIP_PS2_NOFEC",
			},
		},
		{
			name:    "invalid format",
			raw:     "not-a-topology",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Logf("Parse topology %q", tt.raw)
			got, err := parse(tt.raw)
			if tt.wantErr {
				t.Log("Verify malformed topology input is rejected")
				if err == nil {
					t.Fatalf("parse(%q) expected error, got nil", tt.raw)
				}
				return
			}

			t.Log("Verify valid topology input produces the expected normalized shape")
			if err != nil {
				t.Fatalf("parse(%q) unexpected error: %v", tt.raw, err)
			}

			if got != tt.want {
				t.Fatalf("parse(%q) = %#v, want %#v", tt.raw, got, tt.want)
			}
		})
	}
}

func TestTopologyCompatibleWith(t *testing.T) {
	t.Log("Define modern, legacy, and empty topology compatibility cases")
	const modern = "A__Q__8C__G_1__F__GHZ_1__P"
	tests := []struct {
		name     string
		leftRaw  string
		rightRaw string
		want     bool
	}{
		{name: "modern chip count differs", leftRaw: modern, rightRaw: "A__Q__16C__G_1__F__GHZ_1__P", want: true},
		{name: "modern non-count segment differs", leftRaw: modern, rightRaw: "A__R__8C__G_1__F__GHZ_1__P"},
		{name: "modern optional field differs", leftRaw: modern, rightRaw: modern + "__G_2__S"},
		{name: "legacy raw and chip count differ", leftRaw: "A_8_CHIP_X", rightRaw: "B_16_CHIP_Y", want: true},
		{name: "modern and legacy", leftRaw: modern, rightRaw: "A_8_CHIP_X"},
		{name: "legacy and zero-field topology", leftRaw: "A_8_CHIP_X", want: true},
		{name: "zero-field topology and legacy", rightRaw: "A_8_CHIP_X", want: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Parse topology compatibility state and compare it with the requested peer")
			left := Topology{}
			var err error
			if tt.leftRaw != "" {
				left, err = parse(tt.leftRaw)
				if err != nil {
					t.Fatalf("parse(%q) unexpected error: %v", tt.leftRaw, err)
				}
			}
			right := Topology{}
			if tt.rightRaw != "" {
				right, err = parse(tt.rightRaw)
				if err != nil {
					t.Fatalf("parse(%q) unexpected error: %v", tt.rightRaw, err)
				}
			}

			t.Log("Verify the compatibility result")
			if got := left.compatibleWith(right); got != tt.want {
				t.Fatalf("compatibleWith() = %t, want %t", got, tt.want)
			}
		})
	}

	t.Run("withChipCount preserves compatibility", func(t *testing.T) {
		t.Log("Resize one modern topology and compare its retained compatibility state")
		topology, err := parse(modern)
		if err != nil {
			t.Fatalf("parse(%q) unexpected error: %v", modern, err)
		}

		t.Log("Resize the parsed topology")
		resized, err := topology.withChipCount(64)
		if err != nil {
			t.Fatalf("withChipCount() unexpected error: %v", err)
		}

		t.Log("Verify resizing preserves compatibility")
		if !topology.compatibleWith(resized) {
			t.Fatal("withChipCount() changed topology compatibility")
		}
	})
}

func TestTopologyReplicas(t *testing.T) {
	t.Log("Define topology chip counts and expected host replica counts")
	tests := []struct {
		name string
		topo Topology
		want int
	}{
		{
			name: "exact multiple",
			topo: Topology{ChipCount: 32},
			want: 4,
		},
		{
			name: "single node",
			topo: Topology{ChipCount: 8},
			want: 1,
		},
		{
			name: "integer division",
			topo: Topology{ChipCount: 30},
			want: 3,
		},
		{
			name: "single chip",
			topo: Topology{ChipCount: 1},
			want: 1,
		},
	}

	for _, tt := range tests {
		tt := tt
		t.Run(tt.name, func(t *testing.T) {
			t.Log("Derive and verify the topology replica count")
			got := tt.topo.Replicas()
			if got != tt.want {
				t.Fatalf("Replicas() = %d, want %d", got, tt.want)
			}
		})
	}
}
