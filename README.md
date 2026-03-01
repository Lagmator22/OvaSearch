# OvaSearch

A high-performance multimodal RAG (Retrieval-Augmented Generation) system built in native C++ with Intel OpenVINO for hardware acceleration.

## Features

- **Multimodal Search**: Seamlessly query across text documents and images
- **Vision-Language Understanding**: Automatic image analysis and captioning using Qwen2-VL
- **Document Support**: Automatic extraction from PDF, DOCX, and PPTX files
- **Intelligent Caching**: Persistent embedding cache for instant startup
- **Hardware Acceleration**: Optimized for Intel CPUs and GPUs via OpenVINO
- **Native Performance**: Pure C++ implementation with zero Python runtime overhead

## Architecture

OvaSearch combines several state-of-the-art technologies:

- **Vector Search**: USearch for high-performance similarity search
- **Text Embeddings**: bge-small-en-v1.5 for semantic text understanding
- **Vision Model**: Qwen2-VL-2B-Instruct for image analysis
- **Language Model**: Llama-3.2-3B-Instruct (INT4) for response generation
- **Framework**: Intel OpenVINO for optimized inference

## Prerequisites

- macOS or Linux (Windows support coming soon)
- CMake 3.18+
- C++17 compatible compiler
- Python 3.8+ (for model downloads only)
- 8GB+ RAM recommended

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/lagmator22/OvaSearch.git
cd OvaSearch
```

### 2. Set up Python environment

```bash
python3 -m venv env
source env/bin/activate  # On Windows: env\Scripts\activate
pip install openvino openvino-genai optimum-intel
```

### 3. Download models

```bash
python pull_model.py
```

This downloads and converts the required models to OpenVINO format (~2GB total).

### 4. Build the application

The build system auto-detects your Python environment, OpenVINO installation, and required libraries. No manual path configuration needed.

```bash
mkdir build && cd build
cmake ..
make -j4  # Or: ninja
```

> **Note:** On first build, CMake will automatically clone the OpenVINO GenAI C++ headers (~shallow clone) if they are not already present.

## Usage

### Basic Usage
```bash
./ovasearch
```

### Adding Documents
Place your documents in the `data/` folder:
- **Text**: `.txt`, `.md` files
- **Images**: `.jpg`, `.png`, `.jpeg`, `.bmp`, `.webp`
- **Documents**: `.pdf`, `.docx`, `.pptx` (auto-converted)

### Commands
- Type your query and press Enter to search
- `reload` - Refresh the knowledge base with new documents
- `exit` - Quit the application

### Example Queries
```
❯ What images contain animals?
❯ Summarize the key points from my documents
❯ What's in the dog image?
```

## Performance

OvaSearch leverages several optimizations for superior performance:

- **Embedding Cache**: First-run processes documents, subsequent runs load instantly
- **Sliding Window Chunking**: Efficient document segmentation with configurable overlap
- **Batch Processing**: Vectorized operations for embedding generation
- **Native C++**: Direct memory management and zero interpreter overhead

Typical performance metrics:
- Document indexing: ~50-100 docs/second
- Query latency: <100ms for search, 1-3s for generation
- Memory usage: ~500MB base + document embeddings

## Project Structure

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

## Technical Details

### Embedding Dimensions
- Text/Image embeddings: 384-dimensional vectors
- Similarity metric: Cosine distance
- Index type: Dense HNSW (Hierarchical Navigable Small World)

### Chunking Strategy
- Default chunk size: 1000 characters
- Overlap: 200 characters (20%)
- Automatic validation to prevent edge cases

### Cache System
The cache stores:
- Document chunks and metadata
- Computed embedding vectors
- File modification timestamps

Cache invalidates when files are added, modified, or removed.

## Development

### Building with Debug Symbols
```bash
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

### Running Tests
```bash
# Add test documents
echo "Test content" > data/test.txt
./ovasearch
# Query: "test"
```

## Troubleshooting

### Models not loading

Ensure models are downloaded:

```bash
ls models/
# Should show: Llama-3.2-3B-Instruct-INT4, bge-small-en-v1.5, Qwen2-VL-2B-Instruct-INT4
```

### Build errors

Check OpenVINO installation:

```bash
python -c "import openvino; print(openvino.__version__)"
python -c "import openvino_genai; print('GenAI OK')"
```

If CMake can't find packages, ensure you activated the venv before running cmake.

### Memory issues

Reduce chunk size in `main.cpp`:

```cpp
auto chunks = create_sliding_window_chunks(content, 500, 100);  // Smaller chunks
```

## Contributing

Contributions welcome! Areas of interest:
- GPU acceleration support
- Additional model backends
- Web API interface
- Distributed indexing

## License

Apache License 2.0 - See LICENSE file for details.

## Acknowledgments

- Intel OpenVINO team for the inference framework
- USearch for the vector search engine
- Google Summer of Code 2026 program
- Open-source model contributors

## Author

Created for GSoC 2026 - OpenVINO Organization

---

*Built with performance and simplicity in mind.*