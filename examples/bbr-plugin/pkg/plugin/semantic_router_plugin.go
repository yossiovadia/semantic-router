package plugin

import (
	"encoding/json"
	"fmt"
	"strings"
	"sync"
	"time"

	"github.com/vllm-project/semantic-router/examples/bbr-plugin/pkg/classifier"
	bbrplugins "sigs.k8s.io/gateway-api-inference-extension/pkg/bbr/plugins"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
)

const (
	PluginType = "Guardrail"
	PluginName = "vsr-semantic-router"

	HeaderModelName        = "X-Gateway-Model-Name"
	HeaderIntentCategory   = "X-Gateway-Intent-Category"
	HeaderIntentConfidence = "X-Gateway-Intent-Confidence"
	HeaderPluginLatencyMs  = "X-Gateway-Semantic-Router-Latency-Ms"
)

type ModelEndpoint struct {
	Host      string `json:"host"`
	APIFormat string `json:"api_format"`
}

type TierPolicy struct {
	AllowedModels []string `json:"allowed_models"`
}

type Config struct {
	ClassifierEnabled bool                    `json:"classifier_enabled"`
	Models            map[string]ModelEndpoint `json:"models"`
	TierPolicy        map[string]TierPolicy   `json:"tier_policy"`
}

type Stats struct {
	TotalRequests      int64
	TranslatedRequests int64
	BlockedRequests    int64
	TotalLatencyMs     int64
}

type SemanticRouterPlugin struct {
	typedName  plugin.TypedName
	config     Config
	classifier classifier.SemanticRouterClassifier
	stats      Stats
	mu         sync.RWMutex
}

func DefaultConfig() Config {
	return Config{
		Models: map[string]ModelEndpoint{
			"qwen2.5:1.5b": {Host: "llm-katan-external:8000", APIFormat: "openai"},
			"claude-sonnet": {Host: "mock-anthropic:8003", APIFormat: "anthropic"},
			"mock-llama3":   {Host: "llm-katan-internal:8000", APIFormat: "openai"},
		},
		TierPolicy: map[string]TierPolicy{
			"free":       {AllowedModels: []string{"mock-llama3"}},
			"premium":    {AllowedModels: []string{"mock-llama3", "qwen2.5:1.5b", "claude-sonnet"}},
			"enterprise": {AllowedModels: []string{"*"}},
		},
	}
}

func NewSemanticRouterPlugin() bbrplugins.BBRPlugin {
	return NewSemanticRouterPluginWithConfig(DefaultConfig())
}

func NewSemanticRouterPluginWithConfig(config Config) *SemanticRouterPlugin {
	return &SemanticRouterPlugin{
		typedName:  plugin.TypedName{Type: PluginType, Name: PluginName},
		config:     config,
		classifier: classifier.NewMockSemanticRouterClassifier(),
	}
}

func NewSemanticRouterPluginFactory(name string, parameters json.RawMessage) (bbrplugins.BBRPlugin, error) {
	config := DefaultConfig()
	if len(parameters) > 0 {
		if err := json.Unmarshal(parameters, &config); err != nil {
			return nil, fmt.Errorf("failed to parse config: %w", err)
		}
	}
	p := NewSemanticRouterPluginWithConfig(config)
	p.typedName.Name = name
	return p, nil
}

func (p *SemanticRouterPlugin) TypedName() plugin.TypedName { return p.typedName }

func (p *SemanticRouterPlugin) Execute(requestBodyBytes []byte) (mutatedBodyBytes []byte, headers map[string][]string, err error) {
	start := time.Now()
	headers = make(map[string][]string)
	mutatedBodyBytes = requestBodyBytes

	p.mu.Lock()
	p.stats.TotalRequests++
	p.mu.Unlock()

	var req struct {
		Model    string `json:"model"`
		Messages []struct {
			Role    string `json:"role"`
			Content string `json:"content"`
		} `json:"messages"`
	}
	if err := json.Unmarshal(requestBodyBytes, &req); err != nil || req.Model == "" {
		return requestBodyBytes, headers, nil
	}

	headers[HeaderModelName] = []string{req.Model}

	// Note: tier-based access control requires request headers which are not
	// available in the current BBR plugin interface (Execute takes body only).
	// Tier enforcement is handled by the standalone ExtProc (Phase A).
	// When the BBR interface adds request headers support, tier check will be added here.

	if ep, ok := p.config.Models[req.Model]; ok && ep.APIFormat == "anthropic" {
		if translated, err := p.translateToAnthropic(requestBodyBytes); err == nil {
			mutatedBodyBytes = translated
			// Set Anthropic-specific headers
			headers[":path"] = []string{"/v1/messages"}
			headers["x-api-key"] = []string{"mock-anthropic-key"}
			headers["anthropic-version"] = []string{"2023-06-01"}
			p.mu.Lock()
			p.stats.TranslatedRequests++
			p.mu.Unlock()
		}
	}

	if p.config.ClassifierEnabled && p.classifier != nil {
		var parts []string
		for _, m := range req.Messages {
			if m.Role == "user" { parts = append(parts, m.Content) }
		}
		if uc := strings.Join(parts, " "); uc != "" {
			cat, _, _, _ := p.classifier.ProcessAll(uc)
			if cat.Category != "" {
				headers[HeaderIntentCategory] = []string{cat.Category}
				headers[HeaderIntentConfidence] = []string{fmt.Sprintf("%.4f", cat.Confidence)}
			}
		}
	}

	latency := time.Since(start).Milliseconds()
	headers[HeaderPluginLatencyMs] = []string{fmt.Sprintf("%d", latency)}
	p.mu.Lock()
	p.stats.TotalLatencyMs += latency
	p.mu.Unlock()

	return mutatedBodyBytes, headers, nil
}

func (p *SemanticRouterPlugin) isModelAllowedForTier(tier, model string) bool {
	if len(p.config.TierPolicy) == 0 { return true }
	policy, ok := p.config.TierPolicy[tier]
	if !ok { return false }
	for _, a := range policy.AllowedModels {
		if a == "*" || a == model { return true }
	}
	return false
}

func (p *SemanticRouterPlugin) translateToAnthropic(body []byte) ([]byte, error) {
	var req struct {
		Model       string            `json:"model"`
		Messages    []json.RawMessage `json:"messages"`
		MaxTokens   int               `json:"max_tokens,omitempty"`
		Temperature float64           `json:"temperature,omitempty"`
	}
	if err := json.Unmarshal(body, &req); err != nil { return nil, err }

	var system string
	var msgs []json.RawMessage
	for _, raw := range req.Messages {
		var m struct{ Role string `json:"role"` }
		json.Unmarshal(raw, &m)
		if m.Role == "system" {
			var sm struct{ Content string `json:"content"` }
			json.Unmarshal(raw, &sm)
			system = sm.Content
		} else {
			msgs = append(msgs, raw)
		}
	}
	out := map[string]interface{}{"model": req.Model, "messages": msgs, "max_tokens": req.MaxTokens}
	if system != "" { out["system"] = system }
	if req.Temperature > 0 { out["temperature"] = req.Temperature }
	return json.Marshal(out)
}

func (p *SemanticRouterPlugin) GetStats() Stats {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return p.stats
}

func init() {
	bbrplugins.Register(PluginType, NewSemanticRouterPluginFactory)
}
