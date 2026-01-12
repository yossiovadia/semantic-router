# Cache Embedding Multi-Domain LoRA Architecture Decisions

**Date**: 2025-01-12
**Status**: Decision Pending
**Context**: Designing multi-domain cache embedding system with LoRA adapters

---

## Problem Statement

We need to support **13+ domain-specific cache embedding models** using LoRA adapters. Each domain (medical, programming, law, etc.) needs its own specialized embedding model for semantic cache lookup.

### Key Questions

1. **Can we combine multiple LoRA adapters** (base + medical + programming)?
2. **How do we handle memory efficiently** with 13+ domains?
3. **How do we handle concurrent requests** across different domains?
4. **What are the latency/throughput trade-offs?**

---

## Answer: Can We Stack Multiple LoRA Adapters?

**Short Answer**: No, standard LoRA doesn't support stacking (base + medical + programming).

### Options for Multi-Domain

| Approach | Description | Pros | Cons |
|----------|-------------|------|------|
| **Multi-LoRA Router** ✅ | Base model + N adapters, swap per domain | Memory efficient, domain specialization | Need switching logic |
| **Single Multi-Domain LoRA** | Train one LoRA on all domains combined | Simplest | Loses domain specialization |
| **Sequential Stacking** | Research technique, not production-ready | Theoretical modularity | Not supported in PEFT |

**Recommendation**: Use Multi-LoRA Router pattern (already implemented for Qwen3 classifiers).

---

## Memory Architecture

### Current Problem (Each Cache Loads Separate Model)

```
Medical cache:     base_model + medical_lora     = 584MB
Programming cache: base_model + programming_lora = 584MB
Law cache:         base_model + law_lora         = 584MB
... (10 more domains)
─────────────────────────────────────────────────────────
Total: 13 × 584MB = 7.6GB ❌ UNACCEPTABLE
```

### Solution: Shared Base Model with LoRA Switching

```
┌─────────────────────────────────────────────────────┐
│   SINGLE Embedding Model Manager (Singleton)        │
│   - Base Model: all-MiniLM-L12-v2        (584MB)   │
│   - LoRA Adapters:                                  │
│     * medical       (~10MB)                         │
│     * programming   (~10MB)                         │
│     * law           (~10MB)                         │
│     * ... (10 more) (~100MB)                        │
│   ─────────────────────────────────────────────     │
│   Total Memory: 584MB + 130MB = ~714MB ✅           │
└─────────────────────────────────────────────────────┘
```

**Key Insight**: LoRA adapters are TINY (rank-8 matrices), only ~5-10MB each!

---

## Concurrency & Performance: THE CRITICAL DECISION

### The Challenge: 100 Concurrent Requests

With 100 concurrent requests arriving simultaneously, how do we handle them?

### Option 1: Serialize with Locking (Simple)

**Architecture**:
- Single model instance with mutex lock
- Requests queue up, processed one at a time

**Performance**:
```
Request 1:   0-5ms
Request 2:   5-10ms (waits for Req 1)
Request 3:   10-15ms (waits for Req 1, 2)
...
Request 100: 495-500ms (waits for 99 requests)
```

| Metric | Value |
|--------|-------|
| Memory | 714MB |
| Avg Latency | ~250ms |
| P99 Latency | ~500ms |
| Throughput | ~200 req/sec |
| **Production Ready?** | ❌ **NO** (only for low-concurrency) |

---

### Option 2: Pool of N Model Instances (Balanced)

**Architecture**:
- N instances of (base + all adapters)
- Load balancer distributes requests round-robin

**Performance** (with 4 instances):
```
Instance 1: Req 1, 5, 9, 13, ...
Instance 2: Req 2, 6, 10, 14, ...
Instance 3: Req 3, 7, 11, 15, ...
Instance 4: Req 4, 8, 12, 16, ...

Request 1:   5ms
Request 25:  ~35ms (waits for 6 requests in queue)
Request 100: ~125ms (waits for 24 requests in queue)
```

| Metric | Value |
|--------|-------|
| Memory | 4 × 714MB = **2.8GB** |
| Avg Latency | ~60ms |
| P99 Latency | ~125ms |
| Throughput | ~800 req/sec |
| **Production Ready?** | ⚠️ **MAYBE** (good for < 500 req/sec) |

---

### Option 3: Batch Processing by Domain (Optimal) ✅

**Architecture**:
- Single model instance
- Batch queue per domain (medical, programming, etc.)
- Background thread processes batches every 10ms

**Key Insight**: GPUs can process **multiple texts in parallel**!

```python
# Serial (BAD):
for text in 100_texts:
    embedding = model.encode(text)  # 5ms × 100 = 500ms

# Batched (GOOD):
embeddings = model.encode(100_texts)  # 20ms for ALL 100!
```

**GPU Batch Performance**:
- 1 text: 5ms
- 10 texts: 8ms (only 60% slower for 10× data)
- 100 texts: 20ms (only 4× slower for 100× data)

**Processing Flow** (100 requests across 5 domains):
```
T=0ms:   Requests arrive, queued by domain
         - Medical: 25 requests
         - Programming: 30 requests
         - Law: 15 requests
         - Geography: 10 requests
         - Biology: 20 requests

T=10ms:  Batch processor wakes up
         - Swap to medical → encode 25 texts (12ms)
T=22ms:  - Swap to programming → encode 30 texts (14ms)
T=36ms:  - Swap to law → encode 15 texts (10ms)
T=46ms:  - Swap to geography → encode 10 texts (8ms)
T=54ms:  - Swap to biology → encode 20 texts (11ms)
T=65ms:  ✅ ALL 100 requests complete!
```

| Metric | Value |
|--------|-------|
| Memory | 714MB (single instance) |
| Avg Latency | ~35ms (10ms queue + 25ms processing) |
| P99 Latency | ~65ms |
| Throughput | **1500+ req/sec** |
| **Production Ready?** | ✅ **YES** (production-grade) |

---

## Comparison Table

| Option | Memory | Avg Latency | P99 Latency | Throughput | Complexity | Recommended? |
|--------|--------|-------------|-------------|------------|------------|--------------|
| **1. Serial Lock** | 714MB | 250ms | 500ms | 200 req/sec | Low (1 day) | ❌ No |
| **2. Model Pool (N=4)** | 2.8GB | 60ms | 125ms | 800 req/sec | Medium (2-3 days) | ⚠️ Maybe |
| **3. Batch by Domain** | 714MB | 35ms | 65ms | 1500+ req/sec | High (1 week) | ✅ **YES** |

---

## Implementation Roadmap

### Phase 1: Proof of Concept (Option 2 - Model Pool)
**Timeline**: 2-3 days
**Why**: Quick to implement, validates architecture

1. Create `SharedEmbeddingManager` singleton
2. Load base model + all LoRA adapters once
3. Implement pool of 2-4 model instances
4. Update cache backends to use shared manager

**Acceptance Criteria**:
- Memory usage < 3GB
- P99 latency < 150ms under 100 concurrent requests

### Phase 2: Production Optimization (Option 3 - Batching)
**Timeline**: 1 week
**When**: If throughput > 500 req/sec needed

1. Implement domain-based batch queues
2. Background batch processor (10ms tick)
3. GPU batch encoding per domain
4. Dynamic batch sizing based on queue depth

**Acceptance Criteria**:
- Memory usage < 1GB
- P99 latency < 100ms under 500 concurrent requests
- Throughput > 1500 req/sec

---

## Open Questions

### 1. Expected Concurrent Load?
- [ ] What's the peak concurrent request rate?
- [ ] What's the acceptable P99 latency?
- [ ] Can we start with Option 2 and upgrade later?

### 2. GPU Availability
- [ ] Do we have GPU access for batch processing?
- [ ] What's the GPU memory available (affects batch size)?
- [ ] CPU-only fallback acceptable for low-traffic?

### 3. Domain Distribution
- [ ] Are requests evenly distributed across domains?
- [ ] Can we predict which domains are hot/cold?
- [ ] Should we lazy-load cold domain adapters?

---

## Decision Log

| Date | Decision | Rationale |
|------|----------|-----------|
| 2025-01-12 | Use Multi-LoRA Router pattern | Memory efficient, preserves domain specialization |
| 2025-01-12 | Start with Option 2 (Model Pool) | Quick validation, good enough for < 500 req/sec |
| TBD | Upgrade to Option 3 if needed | Only if profiling shows throughput bottleneck |

---

## References

- Multi-LoRA implementation: `candle-binding/semantic-router.go:2061` (Qwen3MultiLoRAClassifier)
- Cache manager: `src/semantic-router/pkg/cache/cache_manager.go`
- Embedding generation: `src/semantic-router/pkg/cache/inmemory_cache.go:166`
- LoRA training: `src/training/cache_embeddings/lora_trainer.py`

---

## Next Steps

1. **Immediate**: Choose next domain for LoRA training (see domain research below)
2. **Short-term**: Train and validate new domain model
3. **Medium-term**: Implement Option 2 (Model Pool) architecture
4. **Long-term**: Profile and decide on Option 3 (Batching) if needed
