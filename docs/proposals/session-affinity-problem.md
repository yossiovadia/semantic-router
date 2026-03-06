# Session Affinity Problem in VSR

## Summary

VSR has no session affinity for multi-turn conversations. Each request is independently evaluated by `performDecisionEvaluation()`, meaning consecutive turns in the same conversation can be routed to different models. This causes two concrete problems:

1. **Route bouncing** — the selected model changes between turns based on per-message content
2. **Tool call format conflicts** — different models use incompatible tool call formats (XML vs JSON), creating broken conversation histories

## Evidence

### Architecture: No Cross-Turn State

1. **`Process()` creates fresh context per stream** (`processor_core.go:21`):
   ```go
   ctx := &RequestContext{
       Headers: make(map[string]string),
   }
   ```
   No session ID, no previous-turn model, no conversation tracking.

2. **`performDecisionEvaluation()` runs independently per request** (`req_filter_classification.go:21`):
   - Evaluates signals (complexity, keywords, embeddings) on the current message only
   - Matches a decision based on the current message's content
   - Selects a model from the matched decision's `ModelRefs`
   - No awareness of which model handled previous turns

3. **`selectModelFromCandidates()` never sets SessionID** (`req_filter_classification.go:316-325`):
   ```go
   selCtx := &selection.SelectionContext{
       Query:           query,
       DecisionName:    decisionName,
       CategoryName:    categoryName,
       CandidateModels: modelRefs,
       CostWeight:      costWeight,
       QualityWeight:   qualityWeight,
       // SessionID: NOT SET
       // UserID: NOT SET
   }
   ```

4. **`SelectionContext` has `SessionID`/`UserID` fields** (`selector.go:127-129`) but they are never populated from the request pipeline. The plumbing is missing:
   ```
   Request headers → RequestContext → performDecisionEvaluation → selectModelFromCandidates → SelectionContext
                                                                                               ↑ SessionID never set
   ```

### Test Results

Tests in `src/semantic-router/pkg/selection/session_affinity_test.go` demonstrate:

#### Test 1: Route Bouncing (`TestSessionAffinityProblem_RouteBouncing`)

A 5-turn conversation where message complexity alternates:

| Turn | Message | Complexity | Selected Model |
|------|---------|-----------|----------------|
| 1 | "What's a mutex?" | easy | model-a-lightweight |
| 2 | "How do I debug a race condition in concurrent Go code?" | hard | model-b-powerful |
| 3 | "Can you simplify that?" | easy | model-a-lightweight |
| 4 | "Implement an optimized lock-free concurrent hashmap" | hard | model-b-powerful |
| 5 | "Thanks, that helps" | easy | model-a-lightweight |

The model changes **4 times** across 5 turns. Each turn is evaluated in isolation.

#### Test 2: Tool Call Format Conflict (`TestSessionAffinityProblem_ToolCallFormatConflict`)

When Elo category ratings differ per decision:
- Turn 1 (coding query) → routes to `model-qwen` (XML tool calls)
- Turn 2 (analysis query) → routes to `model-claude` (JSON tool calls)

Turn 2's model receives conversation history containing XML `<tool_call>` tags from Turn 1, but expects JSON `tool_use` blocks. This causes parsing errors or hallucinated tool calls.

#### Test 3: No Session State (`TestSessionAffinityProblem_NoSessionStateInRequestContext`)

- `RequestContext` has no `SessionID` field
- `SelectionContext.SessionID` exists but is never set in the routing pipeline
- Static selector produces identical results regardless of `SessionID` value

## Scope of Impact

This affects any deployment where:
- Multiple decisions map to different models (the common case for intelligent routing)
- Users have multi-turn conversations (the common case for chat)
- Models use different tool call formats (Qwen XML vs OpenAI JSON vs Anthropic JSON)
- Conversations mix simple and complex messages (natural conversation flow)

## What This Proposal Does NOT Do

- Does not implement a fix
- Does not modify any existing routing code
- Does not add dependencies
- Only demonstrates the problem with tests and documents findings
