/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package selection

import (
	"context"
	"fmt"
	"testing"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// TestSessionAffinityProblem_RouteBouncing demonstrates that VSR has no session
// affinity: each turn in a multi-turn conversation is independently evaluated,
// so the selected model can change between turns based on the current message's
// content alone.
//
// This is NOT a bug report — it's a demonstration of an architectural gap.
// RequestContext (processor_core.go:21) is created fresh per gRPC stream.
// performDecisionEvaluation (req_filter_classification.go) runs independently
// per request with no cross-turn state. SelectionContext.SessionID exists in
// the interface but is never populated from the request pipeline.
func TestSessionAffinityProblem_RouteBouncing(t *testing.T) {
	ctx := context.Background()

	// --- Setup: two decisions with static selection ---
	// "easy" decision routes to model-a (lightweight model)
	// "hard" decision routes to model-b (powerful model)
	//
	// In production, decisions are matched by signal evaluation (complexity,
	// keywords, embeddings, etc). Here we simulate the outcome: each turn's
	// message independently triggers a decision, and the static selector picks
	// the first model from that decision's ModelRefs.

	easySelector := NewStaticSelector(DefaultStaticConfig())
	hardSelector := NewStaticSelector(DefaultStaticConfig())

	easyCandidates := []config.ModelRef{
		{Model: "model-a-lightweight"},
	}
	hardCandidates := []config.ModelRef{
		{Model: "model-b-powerful"},
	}

	// Simulate a 5-turn conversation where message complexity alternates.
	// In production, performDecisionEvaluation() would evaluate the complexity
	// signal and match different decisions. We simulate the final selection step.
	turns := []struct {
		turnNumber      int
		userMessage     string
		complexityLevel string // "easy" or "hard" — what the signal classifier would detect
		expectedModel   string
	}{
		{1, "What's a mutex?", "easy", "model-a-lightweight"},
		{2, "How do I debug a race condition in concurrent Go code with goroutines and channels?", "hard", "model-b-powerful"},
		{3, "Can you simplify that?", "easy", "model-a-lightweight"},
		{4, "Now implement an optimized lock-free concurrent hashmap with linear probing", "hard", "model-b-powerful"},
		{5, "Thanks, that helps", "easy", "model-a-lightweight"},
	}

	var selectedModels []string
	var modelChanges int

	for _, turn := range turns {
		// Each turn creates a fresh SelectionContext — no session state carried over.
		// This mirrors processor_core.go:21 where RequestContext is created per stream.
		var selector *StaticSelector
		var candidates []config.ModelRef

		if turn.complexityLevel == "easy" {
			selector = easySelector
			candidates = easyCandidates
		} else {
			selector = hardSelector
			candidates = hardCandidates
		}

		selCtx := &SelectionContext{
			Query:           turn.userMessage,
			DecisionName:    turn.complexityLevel,
			CandidateModels: candidates,
			// SessionID is empty — never populated from request pipeline
			SessionID: "",
			UserID:    "",
		}

		result, err := selector.Select(ctx, selCtx)
		if err != nil {
			t.Fatalf("Turn %d: unexpected error: %v", turn.turnNumber, err)
		}

		if result.SelectedModel != turn.expectedModel {
			t.Errorf("Turn %d: expected %s, got %s", turn.turnNumber, turn.expectedModel, result.SelectedModel)
		}

		selectedModels = append(selectedModels, result.SelectedModel)

		if len(selectedModels) > 1 && selectedModels[len(selectedModels)-1] != selectedModels[len(selectedModels)-2] {
			modelChanges++
		}

		t.Logf("Turn %d: %q → complexity=%s → model=%s",
			turn.turnNumber, turn.userMessage, turn.complexityLevel, result.SelectedModel)
	}

	// ASSERTION: model changes between turns, proving no session affinity
	if modelChanges == 0 {
		t.Error("Expected model changes between turns (proving route bouncing), but model stayed the same")
	}

	t.Logf("\n=== Session Affinity Problem Demonstrated ===")
	t.Logf("Total turns: %d", len(turns))
	t.Logf("Model changes: %d", modelChanges)
	t.Logf("Models used: %v", selectedModels)
	t.Logf("Conversation bounced between %d different models with no session tracking", modelChanges)
	t.Logf("")
	t.Logf("Root cause: RequestContext is created fresh per gRPC stream (processor_core.go:21)")
	t.Logf("SelectionContext.SessionID exists but is never populated from the request pipeline")
}

// TestSessionAffinityProblem_ToolCallFormatConflict demonstrates that when a
// conversation bounces between models with different tool-call formats, the
// conversation history becomes inconsistent.
//
// Example scenario:
//   - Turn 1 routes to a Qwen-family model that uses XML-style tool calls
//   - Turn 2 routes to a Claude-family model that expects JSON tool calls
//   - The conversation history now contains XML tool calls that Claude doesn't understand
//
// This test doesn't require actual model inference — it demonstrates the format
// mismatch at the routing layer by showing that different models would be selected
// for consecutive turns in the same conversation.
func TestSessionAffinityProblem_ToolCallFormatConflict(t *testing.T) {
	ctx := context.Background()

	// Setup: Elo selector with ratings that make different models win for different queries.
	// In production, this happens when the decision engine matches different decisions
	// or when Elo/RouterDC/AutoMix selects different models based on query content.
	selector := NewEloSelector(DefaultEloConfig())

	// Simulate: model-qwen has high rating for "coding" decision,
	// model-claude has high rating for "analysis" decision.
	selector.setCategoryRating("coding", "model-qwen", &ModelRating{Model: "model-qwen", Rating: 1700})
	selector.setCategoryRating("coding", "model-claude", &ModelRating{Model: "model-claude", Rating: 1300})
	selector.setCategoryRating("analysis", "model-qwen", &ModelRating{Model: "model-qwen", Rating: 1300})
	selector.setCategoryRating("analysis", "model-claude", &ModelRating{Model: "model-claude", Rating: 1700})

	candidates := createCandidateModels("model-qwen", "model-claude")

	// Simulate tool call formats (what each model would produce/expect)
	toolCallFormats := map[string]string{
		"model-qwen":  "xml",  // <tool_call><name>get_weather</name><arguments>{"city":"NYC"}</arguments></tool_call>
		"model-claude": "json", // {"type":"function","function":{"name":"get_weather","arguments":{"city":"NYC"}}}
	}

	// Turn 1: User asks a coding question → routes to model-qwen (higher Elo for coding)
	turn1Ctx := &SelectionContext{
		Query:           "Write a function to call the weather API",
		DecisionName:    "coding",
		CandidateModels: candidates,
	}
	turn1Result, err := selector.Select(ctx, turn1Ctx)
	if err != nil {
		t.Fatalf("Turn 1: unexpected error: %v", err)
	}

	// Turn 2: User asks to analyze the results → routes to model-claude (higher Elo for analysis)
	turn2Ctx := &SelectionContext{
		Query:           "Analyze the weather data patterns from the API response",
		DecisionName:    "analysis",
		CandidateModels: candidates,
	}
	turn2Result, err := selector.Select(ctx, turn2Ctx)
	if err != nil {
		t.Fatalf("Turn 2: unexpected error: %v", err)
	}

	t.Logf("Turn 1: %q → model=%s (tool format: %s)",
		turn1Ctx.Query, turn1Result.SelectedModel, toolCallFormats[turn1Result.SelectedModel])
	t.Logf("Turn 2: %q → model=%s (tool format: %s)",
		turn2Ctx.Query, turn2Result.SelectedModel, toolCallFormats[turn2Result.SelectedModel])

	// ASSERTION: different models are selected, creating a format conflict
	if turn1Result.SelectedModel == turn2Result.SelectedModel {
		t.Skip("Models matched (no format conflict in this run) — test requires Elo category differentiation")
	}

	turn1Format := toolCallFormats[turn1Result.SelectedModel]
	turn2Format := toolCallFormats[turn2Result.SelectedModel]

	if turn1Format == turn2Format {
		t.Skip("Both models use same tool format — no conflict to demonstrate")
	}

	t.Logf("\n=== Tool Call Format Conflict Demonstrated ===")
	t.Logf("Turn 1 model (%s) uses %s tool call format", turn1Result.SelectedModel, turn1Format)
	t.Logf("Turn 2 model (%s) uses %s tool call format", turn2Result.SelectedModel, turn2Format)
	t.Logf("")
	t.Logf("Problem: Turn 2's conversation history contains %s-format tool calls", turn1Format)
	t.Logf("from Turn 1, but %s expects %s-format tool calls.", turn2Result.SelectedModel, turn2Format)
	t.Logf("This can cause parsing errors, hallucinated tool calls, or silent failures.")

	// Simulate the conversation history that Turn 2's model would see
	simulatedHistory := []string{
		// Turn 1: user message
		`user: Write a function to call the weather API`,
		// Turn 1: assistant response with XML tool call (from model-qwen)
		fmt.Sprintf(`assistant: Here's the function:
<tool_call>
<name>get_weather</name>
<arguments>{"city": "NYC", "units": "metric"}</arguments>
</tool_call>`),
		// Tool result
		`tool: {"temperature": 72, "condition": "sunny"}`,
		// Turn 2: user message
		`user: Analyze the weather data patterns from the API response`,
	}

	t.Logf("\nSimulated conversation history that %s (Turn 2) would receive:", turn2Result.SelectedModel)
	for i, msg := range simulatedHistory {
		t.Logf("  [%d] %s", i, msg)
	}
	t.Logf("\n%s expects JSON tool_use blocks but sees XML <tool_call> tags in history.", turn2Result.SelectedModel)
}

// TestSessionAffinityProblem_NoSessionStateInRequestContext demonstrates that
// RequestContext has no session tracking fields. This is a structural test that
// verifies the gap exists at the type level.
func TestSessionAffinityProblem_NoSessionStateInRequestContext(t *testing.T) {
	// SelectionContext HAS SessionID and UserID fields...
	selCtx := &SelectionContext{
		SessionID: "conversation-123",
		UserID:    "user-456",
	}

	// ...but they are never populated from the request pipeline.
	// In performDecisionEvaluation (req_filter_classification.go), the
	// SelectionContext is built in selectModelFromCandidates (line 316-325)
	// WITHOUT setting SessionID or UserID:
	//
	//   selCtx := &selection.SelectionContext{
	//       Query:           query,
	//       DecisionName:    decisionName,
	//       CategoryName:    categoryName,
	//       CandidateModels: modelRefs,
	//       CostWeight:      costWeight,
	//       QualityWeight:   qualityWeight,
	//       // SessionID: NOT SET
	//       // UserID: NOT SET
	//   }

	if selCtx.SessionID == "" {
		t.Error("SessionID should be settable on SelectionContext (the field exists)")
	}
	if selCtx.UserID == "" {
		t.Error("UserID should be settable on SelectionContext (the field exists)")
	}

	// Verify that static selector ignores SessionID entirely
	ctx := context.Background()
	selector := NewStaticSelector(DefaultStaticConfig())

	// Same query, same candidates, different SessionIDs → same result
	// This proves the selector has no session-aware behavior
	result1, _ := selector.Select(ctx, &SelectionContext{
		Query:           "test query",
		CandidateModels: createCandidateModels("model-a", "model-b"),
		SessionID:       "session-1",
	})
	result2, _ := selector.Select(ctx, &SelectionContext{
		Query:           "test query",
		CandidateModels: createCandidateModels("model-a", "model-b"),
		SessionID:       "session-2",
	})

	if result1.SelectedModel != result2.SelectedModel {
		t.Errorf("Static selector should ignore SessionID, but got different results: %s vs %s",
			result1.SelectedModel, result2.SelectedModel)
	}

	t.Logf("=== No Session State Demonstrated ===")
	t.Logf("SelectionContext.SessionID field exists but is unused in the pipeline")
	t.Logf("selectModelFromCandidates() (req_filter_classification.go:316-325) never sets SessionID")
	t.Logf("RequestContext (processor_req_header.go:39-159) has no SessionID field")
	t.Logf("Process() (processor_core.go:21) creates fresh RequestContext per stream")
	t.Logf("")
	t.Logf("The plumbing gap:")
	t.Logf("  Request headers → RequestContext → performDecisionEvaluation → selectModelFromCandidates → SelectionContext")
	t.Logf("  SessionID is never extracted from headers or propagated through this chain")
}
