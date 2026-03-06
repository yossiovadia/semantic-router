//go:build windows || !cgo

package cache

import (
	"context"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/config"
)

// HybridCache combines in-memory HNSW index with external Milvus storage
type HybridCache struct {
	enabled bool
}

// HybridCacheOptions contains configuration for the hybrid cache
type HybridCacheOptions struct {
	Enabled                 bool
	SimilarityThreshold     float32
	TTLSeconds              int
	MaxMemoryEntries        int
	HNSWM                   int
	HNSWEfConstruction      int
	Milvus                  *config.MilvusConfig
	MilvusConfigPath        string
	DisableRebuildOnStartup bool
}

// NewHybridCache creates a new hybrid cache instance
func NewHybridCache(options HybridCacheOptions) (*HybridCache, error) {
	return &HybridCache{
		enabled: options.Enabled,
	}, nil
}

// IsEnabled returns whether the cache is active
func (h *HybridCache) IsEnabled() bool {
	return h.enabled
}

// AddPendingRequest stores a request awaiting its response
func (h *HybridCache) AddPendingRequest(_ string, _ string, _ string, _ []byte, _ int, _ string) error {
	return nil
}

// UpdateWithResponse completes a pending request with its response
func (h *HybridCache) UpdateWithResponse(_ string, _ []byte, _ int) error {
	return nil
}

// AddEntry stores a complete request-response pair
func (h *HybridCache) AddEntry(_ string, _ string, _ string, _, _ []byte, _ int, _ string) error {
	return nil
}

// AddEntriesBatch stores multiple request-response pairs efficiently
func (h *HybridCache) AddEntriesBatch(_ []CacheEntry) error {
	return nil
}

// FindSimilar searches for semantically similar cached requests
func (h *HybridCache) FindSimilar(_ string, _ string, _ string) ([]byte, bool, error) {
	return nil, false, nil
}

// FindSimilarWithThreshold searches for semantically similar cached requests with custom threshold
func (h *HybridCache) FindSimilarWithThreshold(_ string, _ string, _ float32, _ string) ([]byte, bool, error) {
	return nil, false, nil
}

// RebuildFromMilvus rebuilds the in-memory HNSW index
func (h *HybridCache) RebuildFromMilvus(_ context.Context) error {
	return nil
}

// Flush forces persistence
func (h *HybridCache) Flush() error {
	return nil
}

// Close releases all resources
func (h *HybridCache) Close() error {
	return nil
}

// GetStats returns cache statistics
func (h *HybridCache) GetStats() CacheStats {
	return CacheStats{}
}

// CheckConnection checks if the cache backend is reachable
func (h *HybridCache) CheckConnection() error {
	return nil
}
