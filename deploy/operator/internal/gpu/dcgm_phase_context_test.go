package gpu

import (
	"context"
	"testing"
	"time"
)

func TestDCGMPhaseContextReservesTimeForTheFallback(t *testing.T) {
	outer, cancelOuter := context.WithTimeout(t.Context(), 4*time.Second)
	defer cancelOuter()
	outerDeadline, _ := outer.Deadline()

	t.Log("With the fallback enabled the DCGM phase ends well before the outer deadline")
	phase, cancel := dcgmPhaseContext(outer, true)
	defer cancel()
	phaseDeadline, ok := phase.Deadline()
	if !ok {
		t.Fatal("phase context has no deadline")
	}
	if remaining := outerDeadline.Sub(phaseDeadline); remaining < time.Second {
		t.Fatalf("only %v is left for the fallback, want at least 1s", remaining)
	}

	t.Log("Without the fallback the whole context is used")
	same, cancelSame := dcgmPhaseContext(outer, false)
	defer cancelSame()
	if same != outer {
		t.Fatal("context was narrowed although there is no fallback to reserve time for")
	}

	t.Log("A context without a deadline is left alone")
	bare, cancelBare := dcgmPhaseContext(t.Context(), true)
	defer cancelBare()
	if _, hasDeadline := bare.Deadline(); hasDeadline {
		t.Fatal("a deadline was invented for a context that had none")
	}
}
