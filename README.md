# OvaSearch | Samsung PRISM GenAI Hackathon 2026, Theme 1 (Agentic Code Intelligence)

**Demo video:** https://drive.google.com/file/d/1lrPoPA8nCbHOSsjEkVldNYA3m0ZxI8St/view?usp=drive_link

**Slides:** [Thapar_Institute_of_Engineering_and_Technology_Romantic_Shers_Submission.pptx](Thapar_Institute_of_Engineering_and_Technology_Romantic_Shers_Submission.pptx) ([PDF](Thapar_Institute_of_Engineering_and_Technology_Romantic_Shers_Submission.pdf))

**Release with the MTEB result JSON:** https://github.com/Lagmator22/OvaSearch/releases/tag/PRISM_GENAI_HACKATHON_Y2026

## Team

**Romantic Shers**, Thapar Institute of Engineering and Technology

- Gurman Singh
- Harkamal Singh
- Manraj Singh
- Dhruv Srivastava

## The problem (Theme 1)

Given a natural language query, rank code snippets by how relevant they are. Only retrieval is scored; generation is out of scope. Screening uses NDCG@10 and MRR on the CoIR **AppsRetrieval** test split through MTEB. In AppsRetrieval the queries are competitive programming problem statements (3765 of them) and the corpus is 8765 Python solutions. The hands-on round runs our code on similar queries and checks:

- **P1:** rebuild the index for any version of the code in reasonable time.
- **Bonus:** retrieval across versions, and handling near-duplicate snippets that exist in several versions.

## What we built

This repo has two parts:

1. **`prism/`** (new, Python): the code retrieval part for this theme. It has the MTEB AppsRetrieval evaluation and a CLI that indexes a git repo at any revision and searches it. Everything runs on CPU.
2. **The C++ OvaSearch engine** (the original project, unchanged, see [Base engine](#base-engine-c-ovasearch) below): a native C++ multimodal RAG app on Intel OpenVINO for documents and images.

## Benchmark results (AppsRetrieval, MTEB, CPU only)

| Setup | NDCG@10 | MRR@10 |
|---|---|---|
| Baseline: BAAI/bge-small-en-v1.5 | 0.05638 | 0.04820 |
| jinaai/jina-embeddings-v2-base-code | 0.16340 | 0.13905 |
| nomic-ai/CodeRankEmbed + query cleanup | 0.24177 | 0.21247 |
| Salesforce/SFR-Embedding-Code-400M_R | 0.50800 | 0.46187 |
| **Final: SFR-Embedding-Code-400M_R + query cleanup** | **0.51391** | **0.46956** |

All runs are on CPU only (Apple M2 Pro). The final setup is about 9x the baseline on both metrics.

Full table with every run we did, including the ones that did not help, is in [prism/eval/RESULTS.md](prism/eval/RESULTS.md). The final JSON is [appsretrieval_results.json](appsretrieval_results.json) (also attached to the release). Our re-run of the bge-small baseline is kept in [prism/eval/baseline_bge_small.json](prism/eval/baseline_bge_small.json).

Reproduce the final number (about 1 hour on an Apple M2 Pro CPU; downloads the model and dataset on first run; prints NDCG@10 and MRR@10 and writes the JSON with `json.dump(..., default=str)`):

```bash
source .venv-prism/bin/activate   # after the Setup section below
python -m prism.eval.run_apps_eval --model sfr-400m --query-mode no_examples_compact --out appsretrieval_results.json
```

### What we tried and what we kept

- **Model choice (biggest win).** A code-trained embedding model matters much more than anything else. bge-small is a general English model and scores 0.056. The code models we could run on CPU with transformers 4.56 (all with raw queries): jina-embeddings-v2-base-code 0.163, CodeRankEmbed 0.235, **SFR-Embedding-Code-400M_R 0.508**. codesage-small-v2 did not load with our transformers version, so it was not measured.
- **Query cleanup (kept).** APPS problem statements end with "-----Examples-----" blocks of raw sample input/output. Those numbers do not describe what the code does, so we cut everything from the Examples/Note header onward and squash repeated whitespace. On CodeRankEmbed this gave +0.006 NDCG@10 (0.23539 to 0.24177); changing only whitespace without removing examples gave less (0.23934). On SFR it gave +0.006 as well (0.50800 to 0.51391), so it is part of the final setup.
- **Model prefixes (kept).** Each model gets the query prefix from its model card (SFR: "Instruct: Given Code or Text, retrieval relevant content\nQuery: ", CodeRankEmbed: "Represent this query for searching relevant code: "). Code documents get no prefix.
- **Max length 1024 tokens.** Long enough for nearly all APPS solutions, and keeps CPU time reasonable.
- **BM25 + dense fusion (tested, not kept for the benchmark).** We checked offline with the cached CodeRankEmbed embeddings (before the SFR run finished) (`prism/eval/hybrid_check.py`). BM25 alone gets NDCG@10 0.015 because problem statements and Python solutions share almost no words. RRF fusion made the score worse at every weight we tried (0.228, 0.213, 0.173 for BM25 weights 0.2, 0.5, 1.0, versus 0.235 dense only). So the MTEB submission is dense only. BM25 is still used in the CLI, where queries often contain real identifiers.
- **Code cleanup (not measured).** Stripping blank lines or comments from the solutions is implemented (`--doc-mode strip|nocomment`) but we ran out of time to measure it, so it is off.

## How it works

### Index path

```
git revision --> read files with git (ls-tree + cat-file, no checkout)
            --> tree-sitter chunker (Python, JavaScript): one chunk per function / class / method,
                with the JSDoc or comment block above it; code between definitions goes into
                40 line windows; other text files and unparseable files use 40 line windows
            --> chunk text = "# file: <path> | <kind>: <symbol>" header + code
            --> sha1 of chunk text --> already in the vector cache?  yes: reuse   no: embed
            --> SFR-Embedding-Code-400M_R on CPU (sentence-transformers), 1024 dim, L2 normalised
            --> revs/<commit>.json (chunk list) + vectors.npy (one row per unique chunk hash)
```

### Query path

```
query --> classify: identifier-like (computeTotal, res.send, parse_args) or natural language
      --> dense: SFR query embedding (with its instruction prefix), cosine vs all chunks
      --> sparse: BM25 over identifier tokens (camelCase and snake_case are split too)
      --> reciprocal rank fusion (k=60); identifier-like queries give BM25 weight 2, else 1
      --> top k: file:start-end, symbol, score, 3 line preview, latency in ms
```

With `--all-versions` the search runs once over the union of unique chunks from every indexed revision. Hits are then grouped into families: same symbol name, and cosine similarity of at least 0.95 to the best scoring version (same path is also required for module level windows). Each family shows the best version first and then every revision that has it, marked as identical or as a near-duplicate with its similarity.

### Why SFR-Embedding-Code-400M_R

It had by far the best NDCG@10 of the models we measured (0.508 vs 0.235 for the next best), and at 400M parameters it still runs on a laptop CPU: the full AppsRetrieval run (12.5k texts) took 58 minutes, and indexing a repo like Express (about 900 chunks) takes a few minutes. It is trained for text to code and code to code retrieval and only needs an instruction prefix on the query. If CPU time matters more than accuracy, `--model nomic-ai/CodeRankEmbed` (137M) is about 3x faster and is supported too.

## Setup

Tested with Python 3.11 on macOS (Apple M2 Pro, CPU only). The requirements pin torch from the CPU wheel index, so no GPU or CUDA is needed.

```bash
git clone https://github.com/Lagmator22/OvaSearch.git
cd OvaSearch
python3.11 -m venv .venv-prism
source .venv-prism/bin/activate
pip install -r requirements.txt
```

The first run downloads `Salesforce/SFR-Embedding-Code-400M_R` plus its model code from `Alibaba-NLP/new-impl` (829 MB in total) from Hugging Face. It uses custom model code, so it is loaded with `trust_remote_code=True`.

Run all commands below from the repo root (`python -m prism` imports the `prism/` folder from the current directory). We checked this whole section in a fresh venv with an empty Hugging Face cache: install took about 2 minutes and the test suite passed (17 tests, about 2.5 minutes including the model download).

Run the tests (builds a small two-commit git repo from `tests/fixtures/sample_repo`, indexes both commits and runs searches, about 15 seconds once the model is downloaded):

```bash
python -m pytest -q
```

## Using the CLI

```bash
# index one or more revisions (tags, branches or commit hashes); default is HEAD
python -m prism index <repo_path> --rev <tag-or-commit> [--rev <another>]

# search the last indexed revision, or a specific one
python -m prism search "<query>" [--rev <tag-or-commit>] [-k 10] [--mode hybrid|dense|bm25]

# search every indexed revision and group near-duplicate snippets
python -m prism search "<query>" --all-versions

# show what is indexed
python -m prism list
```

The index lives in `./.prism_index` by default. Use `--index-dir <dir>` (before the subcommand) or the `PRISM_INDEX` environment variable to put it somewhere else. A folder that is not a git repo is indexed from the working tree.

### Example on a real repo (Express.js, three releases)

We cloned [expressjs/express](https://github.com/expressjs/express) and indexed three releases. These are real outputs from our machine (Apple M2 Pro, CPU). The timings were taken while an MTEB run was using the CPU at the same time, so on an idle machine they are lower.

```text
$ git clone https://github.com/expressjs/express.git /tmp/express
$ python -m prism index /tmp/express --rev 4.18.2 --rev 4.21.2 --rev v5.1.0
model Salesforce/SFR-Embedding-Code-400M_R loaded on CPU via sentence-transformers (PyTorch CPU) in 9.5s
indexed 4.18.2 (8368dc178a): 180 files, 888 chunks
  reused 0 cached chunk vectors, embedded 888 new chunks (0 duplicates inside this revision)
  chunking 0.19s | embedding 317.47s | total rebuild 317.67s
indexed 4.21.2 (1faf228935): 182 files, 910 chunks
  reused 668 cached chunk vectors, embedded 242 new chunks (0 duplicates inside this revision)
  chunking 0.18s | embedding 121.23s | total rebuild 121.43s
indexed v5.1.0 (cd7d4397c3): 173 files, 836 chunks
  reused 358 cached chunk vectors, embedded 478 new chunks (0 duplicates inside this revision)
  chunking 0.15s | embedding 214.20s | total rebuild 214.38s
$ python -m prism index /tmp/express --rev 4.21.2
model Salesforce/SFR-Embedding-Code-400M_R loaded on CPU via sentence-transformers (PyTorch CPU) in 9.9s
indexed 4.21.2 (1faf228935): 182 files, 910 chunks
  reused 910 cached chunk vectors, embedded 0 new chunks (0 duplicates inside this revision)
  chunking 0.17s | embedding 0.00s | total rebuild 0.19s
```

The first line after each revision is the P1 part: 4.21.2 reuses 668 of its 910 chunk vectors from 4.18.2 and only embeds the 242 chunks that changed, so its rebuild takes 121 s instead of 318 s for a cold index. Re-indexing a revision that is already there embeds nothing (0.19 s).

```text
$ python -m prism search "set the Content-Type header and send a JSON response" -k 3
query: "set the Content-Type header and send a JSON response"  (type: natural, mode: hybrid)
revision 4.21.2 (1faf228935), 910 chunks

 1. lib/response.js:601-625  res.contentType  score 0.0328 (cos 0.769)
      | /**
      |  * Set _Content-Type_ response header with `type` through `mime.lookup()`
      |  * when it does not contain "/", or set the Content-Type to `type` otherwise.
 2. lib/response.js:758-801  res.set  score 0.0318 (cos 0.699)
      | /**
      |  * Set header `field` to `val`, or pass
      |  * an object of header fields.
 3. lib/response.js:238-279  res.json  score 0.0306 (cos 0.684)
      | /**
      |  * Send JSON response.
      |  *
latency: 321.1 ms total (250.6 ms query embedding)
```

```text
$ python -m prism search "redirect to another url with a status code" --all-versions -k 3
query: "redirect to another url with a status code"  (type: natural, mode: hybrid)
searched 3 revisions: 4.18.2, 4.21.2, v5.1.0

 1. lib/response.js:918-980  res.redirect  score 0.0325 (cos 0.799)
      | /**
      |  * Redirect to the given `url` with optional response `status`
      |  * defaulting to 302.
      in 4.18.2: lib/response.js:918-980 [best]
      in 4.21.2: lib/response.js:928-990 [near-duplicate, sim 1.000]
      in v5.1.0: lib/response.js:803-856 [near-duplicate, sim 0.988]

 2. test/res.redirect.js:41-80  <module>  score 0.0286 (cos 0.713)
      |       request(app)
      |       .get('/')
      |       .expect('Location', 'https://google.com?q=%A710')
      in 4.18.2, 4.21.2: test/res.redirect.js:41-80 [best]
      in v5.1.0: test/res.redirect.js:41-80 [near-duplicate, sim 0.995]

 3. test/res.redirect.js:1-40  <module>  score 0.0267 (cos 0.724)
      | 'use strict'
      | 
      | var express = require('..');
      in 4.18.2, 4.21.2, v5.1.0: test/res.redirect.js:1-40 [best]

latency: 406.1 ms total (255.1 ms query embedding)
```

The latency line is measured inside the CLI after the model is loaded (model load is about 8 s and printed separately). Most of it is embedding the query with the 400M model on CPU.

## P1 and Bonus

**P1 (rebuild for any version).** `--rev` accepts any tag, branch or commit. Files are read straight from git objects, so there is no checkout and the working tree is not touched. Each chunk is keyed by the sha1 of its text, and the vector cache is shared by all revisions in an index. A new revision therefore only embeds chunks whose text changed; everything else is reused. The output prints how many chunks were reused and embedded and the total rebuild time. Chunking itself is fast (about 0.2 s for Express); almost all rebuild time is embedding new chunks on CPU. Re-indexing a revision that is already indexed embeds nothing.

**Bonus (across versions + near-duplicates).** See the `--all-versions` example above: `res.redirect` exists in all three releases with slightly different text. 4.21.2 is almost the same as 4.18.2 (similarity 1.000 after rounding) and the rewritten v5.1.0 version is still grouped (0.988). You get one result that lists all versions instead of three separate hits, and each revision appears at most once per result. Identical chunks are stored and embedded once (the "in 4.18.2, 4.21.2" lines), which is also why rebuilds are cheap.

## Tests

`tests/test_prism.py` (17 tests) covers:

- chunking: functions, classes, methods, arrow functions, chained assignments like `res.type = res.contentType = function ...`, JSDoc attached to the function, Python classes and methods, line window fallback for unparseable JS and for Markdown, every non-blank line covered
- BM25 tokenisation and query classification
- indexing: first index embeds everything; second commit (one function edited, one file added) embeds at most 4 chunks and reuses the rest; re-indexing the same revision embeds nothing
- search: 6 hand-written queries must return the expected file and symbol in the top 3; `--rev v1` must not return a file that only exists in v2
- all-versions: the edited function is one family with both revisions; an unchanged function is one version present in both revisions

## Repo layout (new parts)

```
prism/
  cli.py              python -m prism index | search | list
  chunker.py          tree-sitter chunking + line window fallback
  embedder.py         CodeRankEmbed on CPU, query prefixes
  index.py            per-revision index, hash vector cache, BM25, RRF, all-versions grouping
  eval/
    run_apps_eval.py  MTEB AppsRetrieval with PrePostPipelineEncoder (mteb AbsEncoder subclass)
    hybrid_check.py   offline BM25 / RRF check on AppsRetrieval
    results_table.py  builds the table in RESULTS.md from runs/*.json
    RESULTS.md        every run we did
    runs/             JSON of every run
tests/                pytest suite + sample repo
appsretrieval_results.json   final MTEB result (also on the release)
requirements.txt
```

## Honest limitations

- **The Python side does not use OpenVINO yet.** We tried with CodeRankEmbed first: optimum-intel cannot export it (custom `nomic_bert` architecture), and a direct `openvino.convert_model` of a traced model converted but its embeddings only had cosine 0.95 to 0.98 against PyTorch, so we did not ship it. SFR also uses custom model code and we did not get to try exporting it. prism runs the model with PyTorch on CPU. The C++ engine still uses OpenVINO.
- **The C++ engine and prism are separate programs.** prism is not wired into the C++ app yet.
- **AST chunking covers Python and JavaScript only.** TypeScript, Java, Go and others are indexed with 40 line windows.
- **No reranker and no query expansion.** Query "classification" is a simple rule (identifier-like or not) that changes BM25 weight in the CLI only.
- **Near-duplicate grouping is by symbol name plus embedding similarity.** A function that is renamed between versions shows up as a separate family.
- **Index storage is simple** (JSON + one numpy matrix, brute force cosine). Fine for repos with tens of thousands of chunks; very large monorepos would need an ANN index.
- **Numbers are from one machine** (Apple M2 Pro, CPU). Wall times will differ on other CPUs. Several eval runs shared the CPU with other runs, so their wall times are not clean timings.
- **SFR is slower than small models.** 400M parameters on CPU: about 58 minutes for the full benchmark and a few minutes to index a mid-size repo from scratch. Incremental re-indexing keeps later revisions cheap.
- **Baseline JSON.** The team's original bge-small baseline JSON was not in the repo when we finished, so `prism/eval/baseline_bge_small.json` is our own re-run. With bge's standard query instruction it reproduces the original numbers exactly (NDCG@10 0.05638, MRR@10 0.04820); without the instruction it gives 0.05142.
- Windows is not tested.

---

# Base engine (C++ OvaSearch)

This is the original OvaSearch project that the hackathon work builds on. Its code (main.cpp, CMakeLists.txt, the Python helper scripts) was not changed for the hackathon work; the instructions below are the original ones.

### Features

- **Multimodal Search**: Seamlessly query across text documents and images
- **Vision-Language Understanding**: Automatic image analysis and captioning using Qwen2-VL
- **Document Support**: Automatic extraction from PDF, DOCX, and PPTX files
- **Intelligent Caching**: Persistent embedding cache for instant startup
- **Hardware Acceleration**: Optimized for Intel CPUs and GPUs via OpenVINO
- **Native Performance**: Pure C++ implementation with zero Python runtime overhead

### Architecture

OvaSearch combines several state-of-the-art technologies:

- **Vector Search**: USearch for high-performance similarity search
- **Text Embeddings**: bge-small-en-v1.5 for semantic text understanding
- **Vision Model**: Qwen2-VL-2B-Instruct for image analysis
- **Language Model**: Llama-3.2-3B-Instruct (INT4) for response generation
- **Framework**: Intel OpenVINO for optimized inference

### Prerequisites

- macOS or Linux (Windows support coming soon)
- CMake 3.18+
- C++17 compatible compiler
- Python 3.8+ (for model downloads only)
- 8GB+ RAM recommended

### Installation

#### 1. Clone the repository

```bash
git clone https://github.com/lagmator22/OvaSearch.git
cd OvaSearch
```

#### 2. Set up Python environment

```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install openvino openvino-genai optimum-intel
```

#### 3. Download models

```bash
python pull_model.py
```

This downloads and converts the required models to OpenVINO format (~2GB total).

#### 4. Build the application

The build system auto-detects your Python environment, OpenVINO installation, and required libraries. No manual path configuration needed.

```bash
mkdir build && cd build
cmake ..
make -j4  # Or: ninja
```

> **Note:** On first build, CMake will automatically clone the OpenVINO GenAI C++ headers (~shallow clone) if they are not already present.

### Usage

#### Basic Usage
```bash
./ovasearch
```

#### Adding Documents
Place your documents in the `data/` folder:
- **Text**: `.txt`, `.md` files
- **Images**: `.jpg`, `.png`, `.jpeg`, `.bmp`, `.webp`
- **Documents**: `.pdf`, `.docx`, `.pptx` (auto-converted)

#### Commands
- Type your query and press Enter to search
- `reload` - Refresh the knowledge base with new documents
- `exit` - Quit the application

#### Example Queries
```
❯ What images contain animals?
❯ Summarize the key points from my documents
❯ What's in the dog image?
```

### Performance

OvaSearch leverages several optimizations for superior performance:

- **Embedding Cache**: First-run processes documents, subsequent runs load instantly
- **Sliding Window Chunking**: Efficient document segmentation with configurable overlap
- **Batch Processing**: Vectorized operations for embedding generation
- **Native C++**: Direct memory management and zero interpreter overhead

Typical performance metrics:
- Document indexing: ~50-100 docs/second
- Query latency: <100ms for search, 1-3s for generation
- Memory usage: ~500MB base + document embeddings

### Project Structure

```
OvaSearch/
├── main.cpp              # Core application
├── prepare_documents.py  # Document extraction utility
├── pull_model.py        # Model download script
├── CMakeLists.txt       # Build configuration
├── include/
│   └── stb_image.h      # Image loading library
├── models/              # Downloaded OpenVINO models
├── data/                # Your documents go here
└── .ovasearch_cache/    # Persistent embedding cache
```

### Technical Details

#### Embedding Dimensions
- Text/Image embeddings: 384-dimensional vectors
- Similarity metric: Cosine distance
- Index type: Dense HNSW (Hierarchical Navigable Small World)

#### Chunking Strategy
- Default chunk size: 1000 characters
- Overlap: 200 characters (20%)
- Automatic validation to prevent edge cases

#### Cache System
The cache stores:
- Document chunks and metadata
- Computed embedding vectors
- File modification timestamps

Cache invalidates when files are added, modified, or removed.

### Development

#### Building with Debug Symbols
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

#### Running Tests
```bash
# Add test documents
echo "Test content" > data/test.txt
./ovasearch
# Query: "test"
```

### Troubleshooting

#### Models not loading

Ensure models are downloaded:

```bash
ls models/
# Should show: Llama-3.2-3B-Instruct-INT4, bge-small-en-v1.5, Qwen2-VL-2B-Instruct-INT4
```

#### Build errors

Check OpenVINO installation:

```bash
python -c "import openvino; print(openvino.__version__)"
python -c "import openvino_genai; print('GenAI OK')"
```

If CMake can't find packages, ensure you activated the venv before running cmake.

#### Memory issues

Reduce chunk size in `main.cpp`:

```cpp
auto chunks = create_sliding_window_chunks(content, 500, 100);  // Smaller chunks
```

### Contributing

Contributions welcome! Areas of interest:
- GPU acceleration support
- Additional model backends
- Web API interface
- Distributed indexing

### License

Apache License 2.0 - See LICENSE file for details.

### Acknowledgments

- Intel OpenVINO team for the inference framework
- USearch for the vector search engine
- Google Summer of Code 2026 program
- Open-source model contributors

### Author

Built with Intel OpenVINO

---

*Built with performance and simplicity in mind.*