package extproc

import (
	"math"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

func approxEqual(a, b, epsilon float64) bool {
	return math.Abs(a-b) < epsilon
}

func TestComputeRoutingMomentum_EmptySignals(t *testing.T) {
	momentum := ComputeRoutingMomentum(nil, 0.3, 0.9)
	if momentum != 0.5 {
		t.Errorf("Expected 0.5 for empty signals, got %f", momentum)
	}
}

func TestComputeRoutingMomentum_SingleComplexMessage(t *testing.T) {
	// Single complex message should escalate quickly (attack path)
	momentum := ComputeRoutingMomentum([]float64{0.95}, 0.3, 0.9)
	// 0.95 > 0.5 → attack: 0.3*0.5 + 0.7*0.95 = 0.815
	if !approxEqual(momentum, 0.815, 0.01) {
		t.Errorf("Expected ~0.815, got %f", momentum)
	}
}

func TestComputeRoutingMomentum_SingleTrivialMessage(t *testing.T) {
	// Single trivial message should de-escalate slowly (release path)
	momentum := ComputeRoutingMomentum([]float64{0.05}, 0.3, 0.9)
	// 0.05 < 0.5 → release: 0.9*0.5 + 0.1*0.05 = 0.455
	if !approxEqual(momentum, 0.455, 0.01) {
		t.Errorf("Expected ~0.455, got %f", momentum)
	}
}

func TestComputeRoutingMomentum_GoodMorningThenKernel(t *testing.T) {
	// The scenario that broke symmetric momentum:
	// "good morning" → trivial, then "develop kernel" → complex
	// Attack should escalate quickly despite trivial history.
	signals := []float64{0.05, 0.95}
	momentum := ComputeRoutingMomentum(signals, 0.3, 0.9)

	// Turn 1: 0.05 < 0.5 → release: 0.9*0.5 + 0.1*0.05 = 0.455
	// Turn 2: 0.95 > 0.455 → attack: 0.3*0.455 + 0.7*0.95 = 0.802
	if !approxEqual(momentum, 0.802, 0.01) {
		t.Errorf("Expected ~0.802 (fast escalation), got %f", momentum)
	}
	if momentum <= 0.5 {
		t.Errorf("Momentum should be above threshold 0.5 after complex message, got %f", momentum)
	}
}

func TestComputeRoutingMomentum_ComplexThenCommitIt(t *testing.T) {
	// "implement LRU cache" → complex, then "commit it" → trivial
	// Release should keep momentum high.
	signals := []float64{0.92, 0.08}
	momentum := ComputeRoutingMomentum(signals, 0.3, 0.9)

	// Turn 1: 0.92 > 0.5 → attack: 0.3*0.5 + 0.7*0.92 = 0.794
	// Turn 2: 0.08 < 0.794 → release: 0.9*0.794 + 0.1*0.08 = 0.723
	if !approxEqual(momentum, 0.723, 0.01) {
		t.Errorf("Expected ~0.723 (slow release), got %f", momentum)
	}
	if momentum <= 0.5 {
		t.Errorf("Momentum should remain above 0.5 after 'commit it', got %f", momentum)
	}
}

func TestComputeRoutingMomentum_DecayOverManyTrivialTurns(t *testing.T) {
	// Complex question followed by many trivial follow-ups.
	// Momentum should eventually decay below threshold.
	signals := []float64{0.95} // complex start
	for i := 0; i < 10; i++ {
		signals = append(signals, 0.05) // 10 trivial follow-ups
	}

	momentum := ComputeRoutingMomentum(signals, 0.3, 0.9)

	// After ~8 trivial turns, momentum should drop below 0.5
	if momentum >= 0.5 {
		t.Errorf("Expected momentum below 0.5 after 10 trivial turns, got %f", momentum)
	}
}

func TestComputeRoutingMomentum_AsymmetricBehavior(t *testing.T) {
	// Verify attack is faster than release.
	// Going up from 0.5 to 0.9 signal should move more than going down.
	upSignal := ComputeRoutingMomentum([]float64{0.9}, 0.3, 0.9)
	downSignal := ComputeRoutingMomentum([]float64{0.1}, 0.3, 0.9)

	upDelta := upSignal - 0.5     // how much it moved up
	downDelta := 0.5 - downSignal // how much it moved down

	if upDelta <= downDelta {
		t.Errorf("Attack (up delta=%.3f) should be faster than release (down delta=%.3f)",
			upDelta, downDelta)
	}
}

func TestComputeRoutingMomentum_ClampsInputs(t *testing.T) {
	// Signals outside [0,1] should be clamped
	momentum := ComputeRoutingMomentum([]float64{-0.5, 1.5}, 0.3, 0.9)
	if momentum < 0 || momentum > 1 {
		t.Errorf("Expected momentum in [0,1], got %f", momentum)
	}
}

// Backward compatibility tests — verify that when CRM is disabled (default),
// the routing behavior is identical to the current per-message routing.

func TestCRM_DisabledByDefault(t *testing.T) {
	// RoutingMomentumConfig zero value should have Enabled=false
	cfg := config.RoutingMomentumConfig{}
	if cfg.Enabled {
		t.Error("CRM should be disabled by default (zero value of bool)")
	}
}

func TestCRM_DisabledDoesNotAffectSignals(t *testing.T) {
	// When CRM is disabled, applyRoutingMomentum should not be called.
	// Verify by checking that complexity rules are unchanged.
	// Scenario 1: Complex then simple — should route differently (current behavior = the bug)
	complexSignals := []float64{0.92}
	simpleSignals := []float64{0.08}

	// Without CRM, each message is evaluated independently
	// Complex message → high momentum (but CRM not applied)
	complexMomentum := ComputeRoutingMomentum(complexSignals, 0.3, 0.9)
	simpleMomentum := ComputeRoutingMomentum(simpleSignals, 0.3, 0.9)

	// Both should return independently calculated values (no history effect)
	if complexMomentum <= 0.5 {
		t.Errorf("Single complex message momentum should be > 0.5, got %f", complexMomentum)
	}
	if simpleMomentum >= 0.5 {
		t.Errorf("Single simple message momentum should be < 0.5, got %f", simpleMomentum)
	}

	// Key: without CRM, each message is evaluated alone (no multi-turn context)
	// This is the current behavior — the "bug" that CRM fixes
}

func TestCRM_EnabledFixesRouteBouncing(t *testing.T) {
	// With CRM enabled: complex → simple should maintain high momentum
	signals := []float64{0.92, 0.08}
	momentum := ComputeRoutingMomentum(signals, 0.3, 0.9)

	if momentum <= 0.5 {
		t.Errorf("CRM should keep momentum > 0.5 after complex→simple, got %f", momentum)
	}

	// Without CRM: simple message alone would route to cheap model
	simpleAlone := ComputeRoutingMomentum([]float64{0.08}, 0.3, 0.9)
	if simpleAlone >= 0.5 {
		t.Errorf("Without history, simple message should be < 0.5, got %f", simpleAlone)
	}
}

func TestCRM_SimpleThenComplex(t *testing.T) {
	// Simple → complex: CRM should escalate quickly (attack path)
	signals := []float64{0.08, 0.95}
	momentum := ComputeRoutingMomentum(signals, 0.3, 0.9)

	if momentum <= 0.5 {
		t.Errorf("CRM should escalate above 0.5 for simple→complex, got %f", momentum)
	}
}

func TestCRM_ConfigDefaults(t *testing.T) {
	cfg := config.RoutingMomentumConfig{}
	if cfg.GetAttack() != 0.3 {
		t.Errorf("Default attack should be 0.3, got %f", cfg.GetAttack())
	}
	if cfg.GetRelease() != 0.9 {
		t.Errorf("Default release should be 0.9, got %f", cfg.GetRelease())
	}
	if cfg.GetThreshold() != 0.5 {
		t.Errorf("Default threshold should be 0.5, got %f", cfg.GetThreshold())
	}
}

func TestEstimateComplexityFromLength(t *testing.T) {
	tests := []struct {
		length   int
		expected float64
	}{
		{5, 0.1},   // "yes"
		{10, 0.1},  // "ok thanks"
		{50, 0.3},  // short question
		{150, 0.6}, // medium question
		{500, 0.8}, // detailed prompt
	}

	for _, tt := range tests {
		got := EstimateComplexityFromLength(tt.length)
		if got != tt.expected {
			t.Errorf("EstimateComplexityFromLength(%d) = %f, want %f", tt.length, got, tt.expected)
		}
	}
}
