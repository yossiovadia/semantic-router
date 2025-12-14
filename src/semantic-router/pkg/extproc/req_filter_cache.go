package extproc

import (
	"time"

	ext_proc "github.com/envoyproxy/go-control-plane/envoy/service/ext_proc/v3"
	"go.opentelemetry.io/otel/attribute"

	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/cache"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/logging"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/observability/tracing"
	"github.com/vllm-project/semantic-router/src/semantic-router/pkg/utils/http"
)

// handleCaching handles cache lookup and storage with category-specific settings
func (r *OpenAIRouter) handleCaching(ctx *RequestContext, categoryName string) (*ext_proc.ProcessingResponse, bool) {
	// Extract the model and query for cache lookup
	requestModel, requestQuery, err := cache.ExtractQueryFromOpenAIRequest(ctx.OriginalRequestBody)
	if err != nil {
		logging.Errorf("Error extracting query from request: %v", err)
		// Continue without caching
		return nil, false
	}

	ctx.RequestModel = requestModel
	ctx.RequestQuery = requestQuery

	// Determine domain from category (default to "general" if no category)
	domain := categoryName
	if domain == "" {
		domain = "general"
	}

	// Check if caching is enabled for this domain
	cacheEnabled := r.CacheManager.IsEnabled(domain)
	if !cacheEnabled {
		logging.Debugf("Caching is disabled for domain '%s'", domain)
		return nil, false
	}

	// Get domain-specific cache
	domainCache, err := r.CacheManager.GetCache(domain)
	if err != nil {
		logging.Errorf("Failed to get cache for domain '%s': %v", domain, err)
		// Try fallback to general cache
		domainCache, err = r.CacheManager.GetCache("general")
		if err != nil {
			logging.Errorf("Failed to get fallback general cache: %v", err)
			return nil, false
		}
		logging.Infof("Using fallback general cache for domain '%s'", domain)
	}

	logging.Infof("handleCaching: requestQuery='%s' (len=%d), domain='%s', cacheEnabled=%v",
		requestQuery, len(requestQuery), domain, cacheEnabled)

	if requestQuery != "" && domainCache.IsEnabled() {
		// Get decision-specific threshold
		threshold := r.Config.GetCacheSimilarityThreshold()
		if categoryName != "" {
			threshold = r.Config.GetCacheSimilarityThresholdForDecision(categoryName)
		}

		logging.Infof("handleCaching: Performing cache lookup - domain=%s, model=%s, query='%s', threshold=%.2f",
			domain, requestModel, requestQuery, threshold)

		// Start cache lookup span
		spanCtx, span := tracing.StartSpan(ctx.TraceContext, tracing.SpanCacheLookup)
		defer span.End()

		startTime := time.Now()
		// Use domain as namespace for cache isolation
		namespace := domain
		// Try to find a similar cached response using category-specific threshold and namespace
		cachedResponse, found, cacheErr := domainCache.FindSimilarWithThreshold(namespace, requestModel, requestQuery, threshold)
		lookupTime := time.Since(startTime).Milliseconds()

		logging.Infof("FindSimilarWithThreshold returned: found=%v, error=%v, lookupTime=%dms", found, cacheErr, lookupTime)

		tracing.SetSpanAttributes(span,
			attribute.String(tracing.AttrCacheKey, requestQuery),
			attribute.Bool(tracing.AttrCacheHit, found),
			attribute.Int64(tracing.AttrCacheLookupTimeMs, lookupTime),
			attribute.String(tracing.AttrCategoryName, categoryName),
			attribute.String("cache.domain", domain),
			attribute.Float64("cache.threshold", float64(threshold)))

		if cacheErr != nil {
			logging.Errorf("Error searching cache: %v", cacheErr)
			tracing.RecordError(span, cacheErr)
		} else if found {
			// Mark this request as a cache hit
			ctx.VSRCacheHit = true
			// Log cache hit
			logging.LogEvent("cache_hit", map[string]interface{}{
				"request_id": ctx.RequestID,
				"model":      requestModel,
				"query":      requestQuery,
				"category":   categoryName,
				"domain":     domain,
				"threshold":  threshold,
			})
			// Return immediate response from cache
			response := http.CreateCacheHitResponse(cachedResponse, ctx.ExpectStreamingResponse)
			ctx.TraceContext = spanCtx
			return response, true
		}
		ctx.TraceContext = spanCtx
	}

	// Cache miss, store the request for later with namespace
	namespace := domain
	err = domainCache.AddPendingRequest(ctx.RequestID, namespace, requestModel, requestQuery, ctx.OriginalRequestBody)
	if err != nil {
		logging.Errorf("Error adding pending request to cache: %v", err)
		// Continue without caching
	}

	return nil, false
}
