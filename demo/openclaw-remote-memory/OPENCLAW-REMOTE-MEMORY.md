# OpenClaw + vLLM Semantic Router: Remote Memory Demo

Demonstrates VSR's Vector Store API ([PR #1311](https://github.com/vllm-project/semantic-router/pull/1311)) as a remote memory backend for [OpenClaw](https://github.com/openclaw/openclaw).

OpenClaw syncs workspace files to VSR, which chunks and embeds them using mmBERT (768-dim). On each agent turn, relevant memory chunks are retrieved via semantic search and injected into context.

## Quick Start

```bash
# 1. Build VSR (one-time)
make rust-ci && make build-router

# 2. Start VSR with vector store only
./vsr-openclaw.sh

# 3. Configure OpenClaw (add to openclaw.json)
# {
#   "memory": {
#     "backend": "remote",
#     "remote": {
#       "baseUrl": "http://127.0.0.1:8080",
#       "vectorStoreName": "openclaw-memory",
#       "syncIntervalMs": 30000,
#       "searchMaxResults": 5,
#       "searchScoreThreshold": 0.3
#     }
#   }
# }

# 4. Restart OpenClaw gateway — files sync automatically

# 5. Run benchmarks
./benchmark-hybrid-memory.sh   # Retrieval benchmark (20 questions)
./e2e-benchmark.sh             # E2E benchmark (8 questions, real LLM)
```

## Files

| File | Purpose |
|------|---------|
| `vsr-openclaw.sh` | Start VSR with vector-store-only config |
| `config/openclaw-memory-only.yaml` | Minimal VSR config (no routing, no vLLM) |
| `benchmark-hybrid-memory.sh` | Retrieval benchmark: old vs hybrid |
| `e2e-benchmark.sh` | E2E benchmark: real LLM responses |
| `record-demo.sh` | Record asciinema demo |
| `openclaw-vsr-hybrid-memory.cast` | Recorded demo |

## Results

- **Retrieval recall**: 70% (old) → 85% (hybrid)
- **E2E accuracy**: 50% (old) → 88% (hybrid)
- **Tokens per turn**: 35% reduction

## Credits

- **OpenClaw remote memory client**: [Huamin Chen (@rootfs)](https://github.com/rootfs)
- **VSR Vector Store API**: [Yossi Ovadia (@yossiovadia)](https://github.com/yossiovadia)
