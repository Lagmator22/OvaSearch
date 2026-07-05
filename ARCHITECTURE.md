# OvaSearch Architecture

## Flowchart 1 — Current State: How OvaSearch Works Today

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'primaryColor': '#7C4DFF', 'primaryTextColor': '#F5F0E8', 'primaryBorderColor': '#9B7FFF', 'lineColor': '#C8A030', 'secondaryColor': '#2D2D3F', 'tertiaryColor': '#1A1A2E'}}}%%
flowchart LR
    subgraph INPUT["Data Ingestion"]
        direction TB
        A1["Text: .txt .md"] --> C1["Sliding Window Chunker<br/>1000 chars, 200 overlap"]
        A2["Docs: .pdf .docx"] -->|prepare_documents.py| A1
        A3["Images: .jpg .png"] -->|stb_image load| B1["VLMPipeline<br/>Qwen2-VL-2B INT4<br/>Generates text caption"]
        B1 --> C1
    end 

    subgraph STORE["Storage"]
        direction TB
        D1["TextEmbeddingPipeline<br/>bge-small-en-v1.5<br/>384-dim vectors"]
        D2["USearch HNSW Index<br/>Cosine similarity<br/>6ms P50 retrieval"]
        D3["Flat-File Cache<br/>manifest.txt + chunks.dat<br/>+ sources.dat + embeddings.bin"]
        D1 --> D2
        D1 --> D3
    end

    subgraph QUERY["Query Pipeline"]
        direction TB
        E1["User types query in CLI"]
        E2["Embed query<br/>bge-small 384-dim"]
        E3["USearch top-5<br/>distance threshold 0.45"]
        E4["Context Assembly<br/>Chunks + source citations"]
        E5["LLMPipeline<br/>Llama-3.2-3B INT4<br/>Streaming response"]
        E6["Answer displayed in terminal"]
        E1 --> E2 --> E3 --> E4 --> E5 --> E6
    end

    C1 --> D1
    D2 --> E3

    classDef purple fill:#7C4DFF,stroke:#9B7FFF,stroke-width:2px,color:#F5F0E8
    classDef gunmetal fill:#2D2D3F,stroke:#4A4A6A,stroke-width:1px,color:#F5F0E8
    classDef cream fill:#3D3520,stroke:#C8A030,stroke-width:2px,color:#F0D060

    class A1,A2,A3,C1 gunmetal
    class D1,D2,E2,E3,E5,B1 purple
    class D3,E1,E4,E6 cream
```

**What each component does:**

| Component | Role | Code Reference |
|-----------|------|---------------|
| TextEmbeddingPipeline | Converts text to 384-dim vectors | main.cpp L688 |
| VLMPipeline | Describes images as text captions | main.cpp L695, L127-146 |
| LLMPipeline | Generates answers from context | main.cpp L703, L907-910 |
| USearch HNSW | Finds nearest vectors by cosine | main.cpp L680-683 |
| Flat-File Cache | Stores chunks/embeddings on disk | main.cpp L166-258 |
| Sliding Window | Splits text into overlapping chunks | main.cpp L148-164 |

**Limitations of current state:**
- CLI only, no GUI
- All models hardcoded to CPU
- No video or audio support
- No keyword search, pure semantic only
- Flat-file cache has no filtering or metadata

---

## Flowchart 2 — Phase 1-2: Local Application Upgrades

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'primaryColor': '#7C4DFF', 'primaryTextColor': '#F5F0E8', 'primaryBorderColor': '#9B7FFF', 'lineColor': '#C8A030', 'secondaryColor': '#2D2D3F', 'tertiaryColor': '#1A1A2E'}}}%%
flowchart LR
    subgraph NEW_INPUT["Expanded Ingestion"]
        direction TB
        N1["Video: .mp4 .avi .webm"]
        N2["Audio: .mp3 .wav .m4a"]
        N3["All existing formats"]
        N1 -->|"cv::VideoCapture<br/>keyframe detection"| N4["VLMPipeline<br/>Caption per keyframe"]
        N2 -->|"16kHz float samples"| N5["WhisperPipeline<br/>ov::genai first-party"]
        N4 --> N6["Chunker"]
        N5 -->|transcript| N6
        N3 --> N6
    end

    subgraph BETTER_STORE["Upgraded Storage"]
        direction TB
        S1["Upgraded Embedding Model<br/>bge-m3 1024-dim<br/>or bge-base 768-dim"]
        S2["SQLite Database<br/>Modality, timestamps<br/>Filtered queries"]
        S3["USearch HNSW<br/>Higher dimension index"]
        S4["BM25 Keyword Index<br/>Exact term matching"]
        S1 --> S3
        S1 --> S2
        S1 --> S4
    end

    subgraph BETTER_QUERY["Upgraded Retrieval"]
        direction TB
        Q1["Embed query"]
        Q2["Hybrid Search<br/>Dense + Sparse + RRF"]
        Q3["TextRerankPipeline<br/>Cross-encoder precision"]
        Q4["Top-5 best results"]
        Q5["LLMPipeline<br/>Streaming answer"]
        Q1 --> Q2 --> Q3 --> Q4 --> Q5
    end

    subgraph APP["Standalone Application"]
        direction TB
        G1["Native C++ Binary<br/>Single executable"]
        G2["Embedded HTTP Server<br/>cpp-httplib localhost only"]
        G3["HTML/JS Frontend<br/>Opens in system browser"]
        G4["System Tray Icon<br/>Background indexing"]
        G1 --> G2 --> G3
        G1 --> G4
    end

    N6 --> S1
    S3 --> Q2
    S4 --> Q2
    Q5 --> G2

    classDef purple fill:#7C4DFF,stroke:#9B7FFF,stroke-width:2px,color:#F5F0E8
    classDef gunmetal fill:#2D2D3F,stroke:#4A4A6A,stroke-width:1px,color:#F5F0E8
    classDef cream fill:#3D3520,stroke:#C8A030,stroke-width:2px,color:#F0D060
    classDef newfeature fill:#1A2E1A,stroke:#60C860,stroke-width:2px,color:#C0F0C0

    class S1,S3,Q1,Q2,Q5,G1 purple
    class N3,N6,Q4 gunmetal
    class S2,S4,Q3 cream
    class N1,N2,N4,N5,G2,G3,G4 newfeature
```

**Key upgrades over current state:**

| Upgrade | Why | Swappable? |
|---------|-----|-----------|
| SQLite replaces flat files | Filtered queries, metadata, modality tagging | Config: database path |
| BM25 added alongside vectors | Catches exact IDs, names, codes that vectors miss | Config: enable/disable |
| TextRerankPipeline | 47% fewer retrieval failures via precise scoring | Config: reranker model path |
| WhisperPipeline | Audio transcription, first-party OpenVINO API | Config: whisper model path |
| Video keyframes | Scene detection + VLM caption per frame | Config: enable/disable |
| Embedding upgrade | Better vectors = better everything downstream | Config: model path + dimension |
| Standalone app with embedded server | Runs as single binary, opens browser tab on localhost | Always standalone, never deployed |

**Standalone vs Web: it is a standalone app.** The binary runs on your machine, serves `localhost:8080`, opens your browser. No server deployment, no cloud hosting. Same pattern as Jupyter Notebook or Ollama — native binary, browser-based UI. The CLI stays as an alternative interface.

---

## Flowchart 3 — Phase 3: AIPC Optimization + Cloud Bridge (If Time Allows)

```mermaid
%%{init: {'theme': 'dark', 'themeVariables': {'primaryColor': '#7C4DFF', 'primaryTextColor': '#F5F0E8', 'primaryBorderColor': '#9B7FFF', 'lineColor': '#C8A030', 'secondaryColor': '#2D2D3F', 'tertiaryColor': '#1A1A2E'}}}%%
flowchart LR
    subgraph SETTINGS["User Settings Panel"]
        direction TB
        SET1["Local Model Selection<br/>Pick LLM: Llama/Qwen/Phi<br/>Pick Embedding: bge-small/m3<br/>Pick Vision: Qwen2-VL/InternVL"]
        SET2["Device Assignment<br/>Embeddings: CPU/NPU<br/>LLM: CPU/GPU<br/>Vision: CPU/GPU"]
        SET3["Cloud Config<br/>API Key: OpenAI/Anthropic<br/>Monthly budget cap<br/>Enable/Disable toggle"]
    end

    subgraph DEVICE["AIPC Hardware Routing"]
        direction TB
        DET["ov::Core::get_available_devices<br/>Auto-detect at startup"]
        NPU["NPU: Embeddings + Reranker<br/>Low power, always-on"]
        GPU["GPU: LLM + VLM<br/>Parallel decoding"]
        CPUF["CPU: Fallback for all<br/>INT4 quantization"]
        DET --> NPU
        DET --> GPU
        DET --> CPUF
    end

    subgraph SPEED["Generation Speedup"]
        direction TB
        SPEC["Speculative Decoding<br/>Small draft model guesses<br/>Large model verifies in batch<br/>2-3x faster generation"]
        LRU["LRU Memory Manager<br/>Caps RAM usage<br/>Evicts cold data first"]
    end

    subgraph CLOUD["Cloud Bridge — Optional"]
        direction TB
        ROUTER{"Query Complexity<br/>Simple or Complex?"}
        LOC["Local: 95% of queries<br/>Zero cost, zero latency"]
        CLD["Cloud: Complex reasoning<br/>Opus / GPT-5 class models"]
        PII["PII Sanitizer<br/>Strip personal data<br/>Send only text chunks"]
        CACHE["Response Cache<br/>Hash-based dedup<br/>No repeat API calls"]
        ROUTER -->|simple| LOC
        ROUTER -->|complex| CLD
        CLD --> PII --> CACHE
    end

    SET1 -->|model paths| DEVICE
    SET2 -->|device strings| DEVICE
    SET3 -->|api config| CLOUD
    DEVICE --> SPEED
    LOC --> DEVICE
    CACHE --> CPUF

    classDef purple fill:#7C4DFF,stroke:#9B7FFF,stroke-width:2px,color:#F5F0E8
    classDef gunmetal fill:#2D2D3F,stroke:#4A4A6A,stroke-width:1px,color:#F5F0E8
    classDef cream fill:#3D3520,stroke:#C8A030,stroke-width:2px,color:#F0D060
    classDef optional fill:#2E1A1A,stroke:#C86060,stroke-width:2px,color:#F0C0C0,stroke-dasharray: 5 5
    classDef hw fill:#1A2E1A,stroke:#60C860,stroke-width:2px,color:#C0F0C0

    class DET,SPEC,SET1 purple
    class LOC,CPUF,LRU gunmetal
    class SET2,SET3,ROUTER cream
    class CLD,PII,CACHE optional
    class NPU,GPU hw
```

**Can you swap architecture via settings?**

Yes — and it is not as complex as it sounds. The pattern:

```cpp
// Config-driven model loading — change path = change model
std::string llm_path   = config.get("llm_model",   "models/Llama-3.2-3B-INT4");
std::string embed_path = config.get("embed_model",  "models/bge-small-en-v1.5");
std::string vlm_path   = config.get("vlm_model",    "models/Qwen2-VL-2B-INT4");
std::string llm_device = config.get("llm_device",   "CPU");

// Same pipeline API, different model
ov::genai::LLMPipeline gen_pipe(llm_path, llm_device);
```

OpenVINO GenAI pipelines accept any compatible model path. Swapping Llama for Qwen or Phi is literally changing a string. The API stays the same. This is how LM Studio works — same inference engine, different model files.

**What is swappable without code changes:**
- LLM model (any OpenVINO-compatible text model)
- Embedding model (any model producing fixed-dim vectors)
- Vision model (any VLM compatible with VLMPipeline)
- Whisper model (different sizes: tiny/base/small/medium)
- Target device per model (CPU/GPU/NPU string)
- Cloud provider (different API endpoint URL)

**What needs a code change to swap:**
- Vector search library (USearch to FAISS — different API)
- Database backend (SQLite to something else)
- Chunking strategy (would need new code, not just config)

These are acceptable constraints. USearch and SQLite are both lightweight, battle-tested, and have no reason to swap.

---

## Progression Summary

```
FLOWCHART 1                 FLOWCHART 2                    FLOWCHART 3
Current Demo                Local Product                  AIPC + Cloud
                                                           
CLI interface          -->  Standalone app + CLI      -->  Settings panel
Text + Images + PDF    -->  + Video + Audio           -->  Same + cloud models  
bge-small 384-dim      -->  bge-m3 1024-dim           -->  User-selectable
Pure vector search     -->  Hybrid + Reranking        -->  Same + cloud fallback
Flat-file cache        -->  SQLite + BM25             -->  + LRU memory manager
CPU only               -->  CPU (tested)              -->  NPU/GPU/CPU routing
No streaming UI        -->  WebSocket streaming       -->  + speculative decoding
```

> Each column builds on the previous. Nothing is thrown away.
> Components are independently upgradeable via config.
> Cloud is always optional — the local system must be complete first.
