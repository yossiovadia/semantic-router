package classifier

type CategoryResult struct {
	Category   string  `json:"category"`
	Confidence float32 `json:"confidence"`
	LatencyMs  int64   `json:"latency_ms"`
}

type PIIResult struct {
	HasPII    bool     `json:"has_pii"`
	PIITypes  []string `json:"pii_types"`
	LatencyMs int64    `json:"latency_ms"`
}

type JailbreakResult struct {
	IsJailbreak bool    `json:"is_jailbreak"`
	ThreatType  string  `json:"threat_type"`
	Confidence  float32 `json:"confidence"`
	LatencyMs   int64   `json:"latency_ms"`
}

type SemanticRouterClassifier interface {
	ProcessAll(text string) (CategoryResult, PIIResult, JailbreakResult, error)
}
