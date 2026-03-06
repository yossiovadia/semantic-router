package extproc

import (
	"fmt"
	"strings"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/classification"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
)

// Conversational Routing Momentum (CRM)
//
// An asymmetric low-pass filter for LLM routing signals, inspired by audio
// compressor attack/release dynamics. Prevents model bouncing in multi-turn
// conversations by applying different time constants for escalation vs
// de-escalation:
//
//   - Fast attack: quickly respond to complexity increases (don't give
//     hard questions to weak models)
//   - Slow release: gradually decay after complexity drops (don't bounce
//     a conversation to a different model mid-flow)
//
// The algorithm is stateless — momentum is computed from the conversation
// history present in each Chat Completions request.

// ComputeRoutingMomentum applies the asymmetric low-pass filter to a
// sequence of complexity signals and returns the current momentum value.
//
// Parameters:
//   - signals: per-turn complexity scores (0.0=trivial, 1.0=complex)
//   - attack: escalation coefficient (lower = faster, e.g. 0.3)
//   - release: de-escalation coefficient (higher = slower, e.g. 0.9)
//
// Returns momentum in [0.0, 1.0]. Compare against threshold to decide routing.
func ComputeRoutingMomentum(signals []float64, attack, release float64) float64 {
	if len(signals) == 0 {
		return 0.5 // neutral starting point
	}

	momentum := 0.5
	for _, signal := range signals {
		// Clamp signal to [0, 1]
		if signal < 0 {
			signal = 0
		} else if signal > 1 {
			signal = 1
		}

		if signal > momentum {
			// Escalation: fast attack — respond quickly to complexity increase
			momentum = attack*momentum + (1-attack)*signal
		} else {
			// De-escalation: slow release — resist dropping to cheaper model
			momentum = release*momentum + (1-release)*signal
		}
	}

	return momentum
}

// EstimateComplexityFromLength returns a cheap complexity proxy based on
// message character length. Used for historical messages in the conversation
// to avoid expensive embedding computation on every turn.
//
// The mapping is a simple piecewise linear function:
//
//	len < 30   → 0.1  (trivial: "yes", "ok", "thanks")
//	len 30-100 → 0.3  (short: "can you explain that?")
//	len 100-300 → 0.6 (medium: brief technical questions)
//	len > 300  → 0.8  (long: detailed technical prompts)
func EstimateComplexityFromLength(messageLen int) float64 {
	switch {
	case messageLen < 30:
		return 0.1
	case messageLen < 100:
		return 0.3
	case messageLen < 300:
		return 0.6
	default:
		return 0.8
	}
}

// applyRoutingMomentum adjusts the complexity signal in SignalResults based on
// the conversation's routing momentum. This modifies MatchedComplexityRules
// to reflect the momentum-adjusted difficulty, allowing the decision engine
// to route based on conversation trajectory rather than per-message signal.
func (r *OpenAIRouter) applyRoutingMomentum(signals *classification.SignalResults, ctx *RequestContext) {
	allMsgs := ctx.AllUserMessages
	if len(allMsgs) <= 1 {
		return // single message, no history to compute momentum from
	}

	cfg := r.Config.RoutingMomentum
	attack := cfg.GetAttack()
	release := cfg.GetRelease()
	threshold := cfg.GetThreshold()

	// Extract the current message's real complexity score from SignalConfidences.
	// Fall back to length-based estimate if no score available.
	currentScore := EstimateComplexityFromLength(len(allMsgs[len(allMsgs)-1]))
	for key, score := range signals.SignalConfidences {
		if strings.HasPrefix(key, "complexity:") {
			currentScore = score
			break
		}
	}

	// Build signal history: length-based proxy for historical, real score for current
	signalHistory := make([]float64, len(allMsgs))
	for i, msg := range allMsgs {
		if i == len(allMsgs)-1 {
			signalHistory[i] = currentScore
		} else {
			signalHistory[i] = EstimateComplexityFromLength(len(msg))
		}
	}

	momentum := ComputeRoutingMomentum(signalHistory, attack, release)

	logging.Infof("[CRM] Routing momentum: %.3f (threshold=%.2f, attack=%.2f, release=%.2f, turns=%d, current_signal=%.3f)",
		momentum, threshold, attack, release, len(allMsgs), currentScore)

	// Store momentum for response header (observability)
	ctx.RoutingMomentum = momentum

	// Adjust complexity rules based on momentum.
	// If momentum says "hard" but current signal says "easy", override to "hard".
	// This is the core of CRM: the conversation's trajectory overrides per-message noise.
	momentumDifficulty := "easy"
	if momentum > threshold {
		momentumDifficulty = "hard"
	}

	// Rebuild MatchedComplexityRules with momentum-adjusted difficulty
	adjusted := make([]string, 0, len(signals.MatchedComplexityRules))
	for _, rule := range signals.MatchedComplexityRules {
		parts := strings.SplitN(rule, ":", 2)
		if len(parts) == 2 {
			adjusted = append(adjusted, fmt.Sprintf("%s:%s", parts[0], momentumDifficulty))
		} else {
			adjusted = append(adjusted, rule)
		}
	}

	// If no complexity rules matched but momentum is active, inject one
	if len(adjusted) == 0 && momentum != 0.5 {
		adjusted = append(adjusted, fmt.Sprintf("crm_momentum:%s", momentumDifficulty))
	}

	signals.MatchedComplexityRules = adjusted
	ctx.VSRMatchedComplexity = adjusted
}
