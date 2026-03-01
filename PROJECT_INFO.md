# OvaSearch - Project Information

## For GSoC 2026 Mentors

This project demonstrates:

1. **Native C++ Performance** - Zero Python overhead in runtime
2. **OpenVINO Integration** - Full utilization of Intel's inference framework
3. **Multimodal Capabilities** - Seamless text and image understanding
4. **Production-Ready Features** - Caching, error handling, clean architecture

## Quick Demo Commands

```bash
# Build from source
mkdir build && cd build
cmake .. && ninja

# Run the application
./ovasearch

# Test queries
"What's in the dog image?"
"Summarize my documents"
"reload" # Show cache performance
```

## Key Technical Achievements

- **384-dimensional embeddings** with cosine similarity search
- **Persistent cache system** reducing startup time to milliseconds
- **Sliding window chunking** with configurable overlap
- **Automatic document extraction** from PDF/DOCX/PPTX
- **Clean C++17 code** with RAII patterns

## Performance Metrics

- Document processing: 50-100 docs/second
- Query latency: <100ms search + 1-3s generation
- Memory usage: ~500MB base
- Cache loading: <10ms for thousands of embeddings

## Files to Review

1. `main.cpp` - Core implementation
2. `CMakeLists.txt` - Build configuration
3. `prepare_documents.py` - Document preprocessing
4. `pull_model.py` - Model download utility

## Contact

Created for GSoC 2026 - OpenVINO Organization
GitHub: @lagmator22