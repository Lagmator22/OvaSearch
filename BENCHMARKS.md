# OvaSearch — Performance Benchmark Report

**System**: Native C++ Multimodal RAG Engine  
**Runtime**: Intel OpenVINO (CPU inference, no Python runtime)  
**Date**: March 12, 2026  
**Platform**: macOS, Apple Silicon (M-series)

---

## System Configuration

| Component | Model | Precision | Device |
|-----------|-------|-----------|--------|
| Text Embeddings | bge-small-en-v1.5 | FP32 | CPU |
| Vision Model | Qwen2-VL-2B-Instruct | INT4 | CPU |
| Language Model | Llama-3.2-3B-Instruct | INT4 | CPU |
| Vector Index | USearch HNSW | 384-dim cosine | Memory |

**Language**: C++17 — zero Python interpreter overhead, no GIL  
**Vector Index**: 384-dimensional dense HNSW (Hierarchical Navigable Small World)  
**Chunking**: 1000-char sliding window, 200-char overlap

---

## 100-Query Stress Test Results

The benchmark runs **100 consecutive prompts** (20 unique queries × 5 iterations) covering text retrieval, image understanding, and multimodal reasoning. The engine loads all 3 models once and handles all 100 queries in a single persistent process — no restarts, no memory reloads.

### End-to-End Latency (Retrieval + Generation)

| Metric | Value |
|--------|-------|
| **Total queries** | 100 |
| **Total time** | 708.3s |
| **Average latency** | 7.08s |
| **Median (P50)** | 4.78s |
| **P90** | 15.74s |
| **P95** | 17.67s |
| **P99** | 19.43s |
| **Min** | 2.50s |
| **Max** | 26.73s |
| **Crashes** | 0 (100% stability) |

### Vector Search (Retrieval Only)

| Metric | Value |
|--------|-------|
| **Average** | 19.7ms |
| **Median (P50)** | 6ms |
| **P90** | 37ms |
| **Max** | 344ms |
| **Min** | 0ms (cached) |

> **Sub-20ms median retrieval** using in-process USearch with cosine similarity. No network hops, no serialization — just direct pointer access to the HNSW graph.

### LLM Generation (Llama 3.2-3B INT4)

| Metric | Value |
|--------|-------|
| **Average** | 7.06s |
| **Median (P50)** | 4.87s |
| **P90** | 16.28s |
| **Min** | 2.50s |
| **Max** | 26.7s |

---

## Why C++ Matters for AIPC Deployment

### Performance Advantages over Python

| Factor | C++ (OvaSearch) | Python (typical RAG) |
|--------|----------------|---------------------|
| **GIL** | No GIL — true parallel execution | GIL blocks multi-threaded inference |
| **Memory overhead** | Single binary, ~direct model loading | Python interpreter + pip packages |
| **Startup** | Models loaded once, cached embeddings | Re-imports on every run |
| **Inference call path** | Direct OpenVINO C++ API calls | Python → C++ bridge overhead per call |
| **Resource control** | Manual memory management, RAII | GC pauses, unpredictable memory spikes |
| **AIPC suitability** | ✅ Strict resource budgets | ❌ Hard to guarantee memory bounds |

### Key Architectural Decisions

1. **Single-binary deployment** — No Python runtime, no pip, no virtualenv. Just the executable + model files.
2. **Persistent model loading** — All 3 models (embedding, VLM, LLM) loaded once at startup. 100 queries served without any reload.
3. **In-process vector search** — USearch runs in the same address space. Zero serialization, zero network latency.
4. **Granular file-level caching** — Only re-embeds modified/new documents. Unchanged files load instantly from binary cache.
5. **Multimodal pipeline** — Text, images (JPG/PNG/BMP/WEBP), and documents (PDF/DOCX/PPTX) all handled natively.

---

## Stability Results

| Metric | Result |
|--------|--------|
| **Queries executed** | 100/100 |
| **Segfaults** | 0 |
| **Memory leaks** | 0 (RAII-managed) |
| **Model reload failures** | 0 |
| **Process restarts needed** | 0 |

The engine ran all 100 queries in a **single continuous session** with no crashes, no hangs, and no degradation.

---

## Features Demonstrated

- ✅ **Text retrieval** — Direct filename-aware + semantic search
- ✅ **Image understanding** — VLM-based image captioning and search
- ✅ **Multimodal RAG** — Combine text + image sources in single answer
- ✅ **Document ingestion** — PDF/DOCX/PPTX auto-extraction
- ✅ **Persistent caching** — Binary embedding cache with file-level invalidation
- ✅ **Hot reload** — `reload` command re-indexes without restart
- ✅ **Source citation** — Every answer cites source files

---

## Reproducing These Results

```bash
# Build
mkdir build && cd build
cmake ..
make -j4

# Run the 100-query stress test
cd ..
python3 run_stress_test.py

# Results saved to stress_test_results.json
```

---

*OvaSearch — Native C++ Multimodal RAG, powered by Intel OpenVINO.*
