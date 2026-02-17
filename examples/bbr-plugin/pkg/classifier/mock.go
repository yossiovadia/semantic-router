package classifier

import "strings"

type MockSemanticRouterClassifier struct{}

func NewMockSemanticRouterClassifier() *MockSemanticRouterClassifier {
	return &MockSemanticRouterClassifier{}
}

func (m *MockSemanticRouterClassifier) ProcessAll(text string) (CategoryResult, PIIResult, JailbreakResult, error) {
	lower := strings.ToLower(text)
	cat := CategoryResult{Category: "other", Confidence: 0.5, LatencyMs: 1}
	if strings.Contains(lower, "math") || strings.Contains(lower, "derivative") || strings.Contains(lower, "2+2") {
		cat = CategoryResult{Category: "math", Confidence: 0.95, LatencyMs: 1}
	} else if strings.Contains(lower, "code") || strings.Contains(lower, "function") || strings.Contains(lower, "python") {
		cat = CategoryResult{Category: "computer_science", Confidence: 0.90, LatencyMs: 1}
	}
	return cat, PIIResult{LatencyMs: 1}, JailbreakResult{LatencyMs: 1}, nil
}
