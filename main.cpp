// OvaSearch - Multimodal RAG System
// GSoC 2026 - OpenVINO Project

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#include <usearch/index_dense.hpp>

#include "openvino/genai/llm_pipeline.hpp"
#include "openvino/genai/rag/text_embedding_pipeline.hpp"
#include "openvino/genai/visual_language/pipeline.hpp"
#include <openvino/openvino.hpp>

namespace fs = std::filesystem;

#define RESET "\033[0m"
#define BOLD "\033[1m"
#define DIM "\033[2m"
#define BLUE "\033[38;2;0;113;197m"
#define SKY "\033[38;2;0;174;239m"
#define PURPLE "\033[38;2;124;77;255m"
#define TEAL "\033[38;2;0;210;190m"
#define WHITE "\033[38;2;230;230;235m"
#define GRAY "\033[38;2;120;120;130m"
#define DARK "\033[38;2;60;60;70m"
#define GREEN "\033[38;2;60;210;120m"
#define GOLD "\033[38;2;240;190;50m"
#define ROSE "\033[38;2;230;100;120m"
#define LAVENDER "\033[38;2;170;140;255m"

void print_banner() {
  std::cout << BLUE << BOLD << R"(
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                                                                              ║
    ║    ██████╗ ██╗   ██╗ █████╗ ███████╗███████╗ █████╗ ██████╗  ██████╗██╗  ██╗ ║
    ║   ██╔═══██╗██║   ██║██╔══██╗██╔════╝██╔════╝██╔══██╗██╔══██╗██╔════╝██║  ██║ ║
    ║   ██║   ██║██║   ██║███████║███████╗█████╗  ███████║██████╔╝██║     ███████║ ║
    ║   ██║   ██║╚██╗ ██╔╝██╔══██║╚════██║██╔══╝  ██╔══██║██╔══██╗██║     ██╔══██║ ║
    ║   ╚██████╔╝ ╚████╔╝ ██║  ██║███████║███████╗██║  ██║██║  ██║╚██████╗██║  ██║ ║
    ║    ╚═════╝   ╚═══╝  ╚═╝  ╚═╝╚══════╝╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝╚═╝  ╚═╝ ║
    ║                                                                              ║
    ╚══════════════════════════════════════════════════════════════════════════════╝)"
            << RESET << "\n";

  std::cout << WHITE << "    " << DIM
            << "Native C++ Multimodal RAG Engine | Built for GSoC 2026" << RESET
            << "\n";
  std::cout << WHITE << "    " << DIM << "Powered by " << RESET << BLUE << BOLD
            << "Intel" << RESET << WHITE << DIM << " OpenVINO" << RESET
            << "\n\n";
}

void print_divider() {
  std::cout
      << DARK
      << "  "
         "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94"
         "\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2"
         "\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
         "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94"
         "\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2"
         "\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
         "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94"
         "\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2"
         "\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80"
         "\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94"
         "\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2\x94\x80\xe2"
         "\x94\x80\xe2\x94\x80\xe2\x94\x80"
      << RESET << "\n";
}

std::string get_source_filename(const fs::path &filepath) {
  std::string fname = filepath.filename().string();
  const std::string suffix = ".extracted.txt";
  if (fname.size() > suffix.size() &&
      fname.compare(fname.size() - suffix.size(), suffix.size(), suffix) == 0)
    return fname.substr(0, fname.size() - suffix.size());
  return fname;
}

ov::Tensor load_image_as_tensor(const fs::path &image_path) {
  int x = 0, y = 0, ch = 0;
  unsigned char *img = stbi_load(image_path.string().c_str(), &x, &y, &ch, 3);
  if (!img)
    throw std::runtime_error("Failed to load: " + image_path.string());

  struct ImageAllocator {
    unsigned char *data;
    size_t size;

    void *allocate(size_t bytes, size_t) const {
      if (data && bytes == size)
        return data;
      throw std::runtime_error("Invalid allocation size");
    }

    void deallocate(void *ptr, size_t, size_t) noexcept {
      if (ptr == data) {
        stbi_image_free(data);
      }
    }

    bool is_equal(const ImageAllocator &other) const noexcept {
      return this == &other;
    }
  };

  size_t total_size = 3 * y * x;
  return ov::Tensor(ov::element::u8, {1, (size_t)y, (size_t)x, 3},
                    ImageAllocator{img, total_size});
}

std::string analyze_image_with_vlm(ov::genai::VLMPipeline &vlm,
                                   const fs::path &path) {
  try {
    ov::Tensor t = load_image_as_tensor(path);
    ov::genai::GenerationConfig cfg;
    cfg.max_new_tokens = 200;
    auto r =
        vlm.generate("Describe ONLY what you see in THIS specific image. "
                     "List: 1) Main objects/subjects 2) Dominant colors 3) "
                     "Setting/background. "
                     "Be factual and specific. Do not add information not "
                     "visible in the image.",
                     ov::genai::image(t), ov::genai::generation_config(cfg));
    if (!r.texts.empty() && !r.texts[0].empty())
      return r.texts[0];
    return "An image file containing visual content.";
  } catch (const std::exception &e) {
    return std::string("Image file (VLM unavailable: ") + e.what() + ")";
  }
}

std::vector<std::string> create_sliding_window_chunks(const std::string &text,
                                                      size_t chunk_size = 1000,
                                                      size_t overlap = 200) {
  std::vector<std::string> chunks;
  if (text.empty())
    return chunks;
  if (overlap >= chunk_size) {
    overlap = chunk_size / 2;
  }
  for (size_t i = 0; i < text.size(); i += (chunk_size - overlap)) {
    size_t end = std::min(text.size(), i + chunk_size);
    chunks.push_back(text.substr(i, end - i));
    if (end == text.size())
      break;
  }
  return chunks;
}

// Cache system - per-file granular caching

const std::string CACHE_DIR_NAME = ".ovasearch_cache";
const std::string MANIFEST_FILE = "manifest.txt";
const std::string CHUNKS_FILE = "chunks.dat";
const std::string SOURCES_FILE = "sources.dat";
const std::string EMBEDDINGS_FILE = "embeddings.bin";
const std::string SEPARATOR = "\x1F\x1E\x1D";
const size_t EMB_DIM = 384;

std::map<std::string, long long> load_manifest(const std::string &cache_dir) {
  std::map<std::string, long long> manifest;
  std::ifstream f(cache_dir + "/" + MANIFEST_FILE);
  if (!f.is_open())
    return manifest;
  std::string line;
  while (std::getline(f, line)) {
    auto sep = line.find('\t');
    if (sep == std::string::npos)
      continue;
    manifest[line.substr(0, sep)] = std::stoll(line.substr(sep + 1));
  }
  return manifest;
}

void save_manifest(const std::string &cache_dir,
                   const std::map<std::string, long long> &manifest) {
  std::ofstream f(cache_dir + "/" + MANIFEST_FILE);
  for (const auto &[path, mtime] : manifest)
    f << path << "\t" << mtime << "\n";
}

std::vector<std::string> split_by_sep(const std::string &data,
                                      const std::string &delim) {
  std::vector<std::string> parts;
  size_t pos = 0, prev = 0;
  while ((pos = data.find(delim, prev)) != std::string::npos) {
    if (pos > prev)
      parts.push_back(data.substr(prev, pos - prev));
    prev = pos + delim.size();
  }
  return parts;
}

void save_cache(const std::string &cache_dir,
                const std::vector<std::string> &chunks,
                const std::vector<std::string> &sources,
                const std::vector<std::vector<float>> &embeddings) {
  std::ofstream cf(cache_dir + "/" + CHUNKS_FILE, std::ios::binary);
  for (const auto &c : chunks)
    cf << c << SEPARATOR;
  std::ofstream sf(cache_dir + "/" + SOURCES_FILE, std::ios::binary);
  for (const auto &s : sources)
    sf << s << SEPARATOR;
  std::ofstream ef(cache_dir + "/" + EMBEDDINGS_FILE, std::ios::binary);
  uint32_t count = (uint32_t)embeddings.size();
  ef.write(reinterpret_cast<const char *>(&count), sizeof(count));
  for (const auto &emb : embeddings)
    ef.write(reinterpret_cast<const char *>(emb.data()),
             EMB_DIM * sizeof(float));
}

bool load_cache(const std::string &cache_dir, std::vector<std::string> &chunks,
                std::vector<std::string> &sources,
                std::vector<std::vector<float>> &embeddings) {
  std::string emb_path = cache_dir + "/" + EMBEDDINGS_FILE;
  if (!fs::exists(emb_path))
    return false;
  std::ifstream cf(cache_dir + "/" + CHUNKS_FILE, std::ios::binary);
  if (!cf.is_open())
    return false;
  std::string cdata((std::istreambuf_iterator<char>(cf)),
                    std::istreambuf_iterator<char>());
  chunks = split_by_sep(cdata, SEPARATOR);
  std::ifstream sf(cache_dir + "/" + SOURCES_FILE, std::ios::binary);
  if (!sf.is_open())
    return false;
  std::string sdata((std::istreambuf_iterator<char>(sf)),
                    std::istreambuf_iterator<char>());
  sources = split_by_sep(sdata, SEPARATOR);
  std::ifstream ef(emb_path, std::ios::binary);
  if (!ef.is_open())
    return false;
  uint32_t count = 0;
  ef.read(reinterpret_cast<char *>(&count), sizeof(count));
  embeddings.resize(count);
  for (uint32_t i = 0; i < count; ++i) {
    embeddings[i].resize(EMB_DIM);
    ef.read(reinterpret_cast<char *>(embeddings[i].data()),
            EMB_DIM * sizeof(float));
  }
  return chunks.size() == count && sources.size() == count;
}

long long get_file_mtime(const fs::path &p) {
  auto ftime = fs::last_write_time(p);
  auto sctp = std::chrono::time_point_cast<std::chrono::seconds>(
      ftime - fs::file_time_type::clock::now() +
      std::chrono::system_clock::now());
  return sctp.time_since_epoch().count();
}

// Convert a source name to lowercase for comparison
std::string to_lower(const std::string &s) {
  std::string out = s;
  std::transform(out.begin(), out.end(), out.begin(),
                 [](unsigned char c) { return std::tolower(c); });
  return out;
}

void load_documents(const std::string &data_folder,
                    std::vector<std::string> &document_chunks,
                    std::vector<std::string> &chunk_sources,
                    ov::genai::TextEmbeddingPipeline &embed_pipe,
                    unum::usearch::index_dense_t &index,
                    ov::genai::VLMPipeline &vlm_pipe, bool verbose) {
  auto t0 = std::chrono::high_resolution_clock::now();
  std::string cache_dir = data_folder + "/" + CACHE_DIR_NAME;
  fs::create_directories(cache_dir);

  // Scan current files in the data folder
  std::map<std::string, long long> new_manifest;
  struct FileEntry {
    fs::path path;
    std::string ext, fname;
  };
  std::vector<FileEntry> current_files;

  for (const auto &entry : fs::recursive_directory_iterator(data_folder)) {
    if (!entry.is_regular_file())
      continue;
    std::string fpath = entry.path().string();
    if (fpath.find(CACHE_DIR_NAME) != std::string::npos)
      continue;
    std::string ext = entry.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    std::string fname = entry.path().filename().string();
    if (fname[0] == '.')
      continue;
    bool is_processable =
        (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".bmp" ||
         ext == ".webp" || ext == ".txt" || ext == ".md");
    if (!is_processable)
      continue;
    new_manifest[fpath] = get_file_mtime(entry.path());
    current_files.push_back({entry.path(), ext, fname});
  }

  // Load old manifest and existing cache
  auto old_manifest = load_manifest(cache_dir);
  std::vector<std::vector<float>> all_embeddings;

  // Find deleted and modified files
  std::set<std::string> deleted_files;
  for (const auto &[old_path, old_mtime] : old_manifest) {
    if (new_manifest.find(old_path) == new_manifest.end()) {
      deleted_files.insert(old_path);
      if (verbose) {
        std::cout << ROSE << "    \xe2\x9c\x97 " << RESET << GRAY
                  << fs::path(old_path).filename().string() << " (removed)"
                  << RESET << "\n";
      }
    }
  }

  // Try to load cache
  bool cache_loaded =
      !old_manifest.empty() &&
      load_cache(cache_dir, document_chunks, chunk_sources, all_embeddings);

  if (cache_loaded && !deleted_files.empty()) {
    // Per-file granular removal: remove only the deleted files' chunks
    // Build a map of source filename -> set of file paths for that source
    std::map<std::string, std::string> path_to_source;
    for (const auto &[path, mtime] : old_manifest) {
      fs::path p(path);
      path_to_source[path] = get_source_filename(p);
    }

    // Identify which source names were deleted
    std::set<std::string> deleted_sources;
    for (const auto &dp : deleted_files) {
      fs::path p(dp);
      deleted_sources.insert(get_source_filename(p));
    }

    // Remove chunks belonging to deleted sources
    std::vector<std::string> kept_chunks;
    std::vector<std::string> kept_sources;
    std::vector<std::vector<float>> kept_embeddings;
    for (size_t i = 0; i < document_chunks.size(); ++i) {
      if (i < chunk_sources.size() &&
          deleted_sources.find(chunk_sources[i]) == deleted_sources.end()) {
        kept_chunks.push_back(document_chunks[i]);
        kept_sources.push_back(chunk_sources[i]);
        if (i < all_embeddings.size())
          kept_embeddings.push_back(all_embeddings[i]);
      }
    }
    document_chunks = kept_chunks;
    chunk_sources = kept_sources;
    all_embeddings = kept_embeddings;

    // Rebuild the index with kept embeddings
    index.reset();
    index.reserve(all_embeddings.size() + 100);
    for (size_t i = 0; i < all_embeddings.size(); ++i)
      index.add(i, all_embeddings[i].data());

    // Remove deleted entries from old_manifest so they dont block new file
    // processing
    for (const auto &dp : deleted_files)
      old_manifest.erase(dp);

    if (verbose)
      std::cout << GOLD << "    \xe2\x86\xbb " << RESET << GRAY << "Removed "
                << deleted_sources.size() << " deleted file(s), kept "
                << document_chunks.size() << " cached chunks" << RESET << "\n";
  } else if (cache_loaded) {
    // Cache is fully valid, just rebuild the index
    index.reset();
    index.reserve(all_embeddings.size() + 100);
    for (size_t i = 0; i < all_embeddings.size(); ++i)
      index.add(i, all_embeddings[i].data());
    if (verbose)
      std::cout << GREEN << "    \xe2\x9c\x93 " << RESET << WHITE << "Loaded "
                << document_chunks.size() << " cached chunks" << RESET << GRAY
                << " (instant, no recomputation)" << RESET << "\n";
  } else if (!old_manifest.empty()) {
    // Cache file corrupted or missing, full rebuild
    if (verbose)
      std::cout << GOLD << "    \xe2\x86\xbb " << RESET << GRAY
                << "Cache corrupted, rebuilding..." << RESET << "\n";
    document_chunks.clear();
    chunk_sources.clear();
    all_embeddings.clear();
    index.reset();
    old_manifest.clear();
  }

  // Process new or modified files only
  int n_text = 0, n_img = 0, n_doc = 0;
  int n_cached = (int)document_chunks.size();
  int n_new = 0;

  for (const auto &fe : current_files) {
    const auto &ext = fe.ext;
    const auto &fname = fe.fname;
    std::string fpath = fe.path.string();

    auto it = old_manifest.find(fpath);
    if (cache_loaded && it != old_manifest.end() &&
        it->second == new_manifest[fpath]) {
      // This file is already cached and unchanged, just count it
      bool is_image = (ext == ".jpg" || ext == ".png" || ext == ".jpeg" ||
                       ext == ".bmp" || ext == ".webp");
      if (is_image)
        n_img++;
      else {
        bool is_extracted = fname.find(".extracted.txt") != std::string::npos;
        if (is_extracted)
          n_doc++;
        else
          n_text++;
      }
      continue;
    }

    n_new++;
    if (ext == ".jpg" || ext == ".png" || ext == ".jpeg" || ext == ".bmp" ||
        ext == ".webp") {
      if (verbose)
        std::cout << GRAY << "    \xe2\x97\x8b " << RESET << SKY << fname
                  << RESET << GRAY << " analyzing..." << RESET << std::flush;
      std::string caption = analyze_image_with_vlm(vlm_pipe, fe.path);
      size_t new_id = document_chunks.size();
      document_chunks.push_back(caption);
      chunk_sources.push_back(fname);
      auto results = embed_pipe.embed_documents({document_chunks.back()});
      auto emb = std::get<std::vector<std::vector<float>>>(results)[0];
      all_embeddings.push_back(emb);
      index.reserve(new_id + 1);
      index.add(new_id, emb.data());
      n_img++;
      if (verbose)
        std::cout << "\r" << GREEN << "    \xe2\x9c\x93 " << RESET << WHITE
                  << fname << RESET << GOLD << " (new)" << RESET
                  << "                         \n";
    } else if (ext == ".txt" || ext == ".md") {
      std::string source = get_source_filename(fe.path);
      std::ifstream file(fe.path);
      std::string content((std::istreambuf_iterator<char>(file)),
                          std::istreambuf_iterator<char>());
      if (content.empty())
        continue;
      auto chunks = create_sliding_window_chunks(content);
      for (const auto &c : chunks) {
        size_t new_id = document_chunks.size();
        document_chunks.push_back(c);
        chunk_sources.push_back(source);
        auto results = embed_pipe.embed_documents({document_chunks.back()});
        auto emb = std::get<std::vector<std::vector<float>>>(results)[0];
        all_embeddings.push_back(emb);
        index.reserve(new_id + 1);
        index.add(new_id, emb.data());
      }
      bool is_doc = fname.find(".extracted.txt") != std::string::npos;
      if (is_doc)
        n_doc++;
      else
        n_text++;
      if (verbose) {
        std::string color = is_doc ? LAVENDER : SKY;
        std::cout << GREEN << "    \xe2\x9c\x93 " << RESET << color << source
                  << RESET << GRAY << " (" << chunks.size() << " chunks)"
                  << RESET << GOLD << " (new)" << RESET << "\n";
      }
    }
  }

  if (document_chunks.empty()) {
    document_chunks.push_back("OvaSearch is a native C++ multimodal RAG system "
                              "powered by Intel OpenVINO.");
    chunk_sources.push_back("system");
    index.reserve(1);
    auto results = embed_pipe.embed_documents({document_chunks[0]});
    auto emb = std::get<std::vector<std::vector<float>>>(results)[0];
    all_embeddings.push_back(emb);
    index.add(0, emb.data());
  }

  save_manifest(cache_dir, new_manifest);
  save_cache(cache_dir, document_chunks, chunk_sources, all_embeddings);

  auto t1 = std::chrono::high_resolution_clock::now();
  double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
  if (verbose) {
    std::cout << "\n";
    print_divider();
    std::cout << GREEN << BOLD << "  \xe2\x9c\x93 Knowledge base ready" << RESET
              << "  " << GRAY << document_chunks.size() << " chunks" << RESET
              << DARK << " \xe2\x94\x82 " << RESET << GRAY << n_text << " text"
              << RESET << DARK << " \xe2\x94\x82 " << RESET << GRAY << n_img
              << " images" << RESET << DARK << " \xe2\x94\x82 " << RESET << GRAY
              << n_doc << " docs" << RESET;
    if (n_new > 0)
      std::cout << DARK << " \xe2\x94\x82 " << RESET << GOLD << n_new << " new"
                << RESET;
    if (n_cached > 0)
      std::cout << DARK << " \xe2\x94\x82 " << RESET << TEAL << n_cached
                << " cached" << RESET;
    std::cout << DARK << " \xe2\x94\x82 " << RESET << GOLD << std::fixed
              << std::setprecision(0) << ms << "ms" << RESET << "\n";
    print_divider();
    std::cout << "\n";
  }
}

void auto_convert_documents(const fs::path &base_path,
                            const std::string &data_folder) {
  fs::path backup_dir = base_path / "data_backup";
  const std::string ext_suffix = ".extracted.txt";

  // Step 1: Clean up orphaned .extracted.txt files (original was deleted)
  {
    std::vector<fs::path> orphans;
    for (const auto &entry : fs::recursive_directory_iterator(data_folder)) {
      if (!entry.is_regular_file())
        continue;
      std::string fname = entry.path().filename().string();
      if (fname.size() > ext_suffix.size() &&
          fname.compare(fname.size() - ext_suffix.size(), ext_suffix.size(),
                        ext_suffix) == 0) {
        std::string orig_name =
            fname.substr(0, fname.size() - ext_suffix.size());
        fs::path orig_in_data = entry.path().parent_path() / orig_name;
        fs::path orig_in_backup = backup_dir / orig_name;
        if (!fs::exists(orig_in_data) && !fs::exists(orig_in_backup))
          orphans.push_back(entry.path());
      }
    }
    for (const auto &orphan : orphans) {
      fs::remove(orphan);
      std::cout << ROSE << "    \xe2\x9c\x97 " << RESET << GRAY
                << orphan.filename().string() << " (orphaned, removed)"
                << RESET << "\n";
    }
  }

  // Step 2: Check for documents needing extraction
  bool needs_preprocessing = false;
  std::vector<fs::path> doc_files;
  for (const auto &entry : fs::recursive_directory_iterator(data_folder)) {
    if (!entry.is_regular_file())
      continue;
    std::string ext = entry.path().extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (ext == ".pdf" || ext == ".docx" || ext == ".pptx") {
      doc_files.push_back(entry.path());
      std::string extracted = entry.path().string() + ".extracted.txt";
      if (!fs::exists(extracted))
        needs_preprocessing = true;
    }
  }
  if (needs_preprocessing) {
    std::cout << "\n";
    std::cout << LAVENDER << BOLD << "  \xe2\x96\xb8 Auto-converting documents"
              << RESET << "\n";
    std::cout << GRAY << "    Detected PDF/DOCX/PPTX files, extracting text..."
              << RESET << "\n";
    std::string cmd =
        "cd " + base_path.string() + " && python3 prepare_documents.py 2>&1";
    int ret = system(cmd.c_str());
    if (ret == 0)
      std::cout << GREEN << "    \xe2\x9c\x93 Document extraction complete"
                << RESET << "\n";
    else
      std::cout << ROSE
                << "    ! Document extraction had issues (install: pip3 "
                   "install pdfplumber python-docx python-pptx)"
                << RESET << "\n";
  }

  // Step 3: Move successfully extracted originals to data_backup/
  for (const auto &doc : doc_files) {
    std::string extracted = doc.string() + ".extracted.txt";
    if (fs::exists(extracted) && fs::file_size(extracted) > 0) {
      fs::create_directories(backup_dir);
      fs::path dest = backup_dir / doc.filename();
      try {
        fs::rename(doc, dest);
      } catch (...) {
        try {
          fs::copy(doc, dest, fs::copy_options::overwrite_existing);
          fs::remove(doc);
        } catch (...) {
          continue;
        }
      }
      std::cout << GREEN << "    \xe2\x86\xa6 " << RESET << GRAY
                << doc.filename().string()
                << " \xe2\x86\x92 data_backup/" << RESET << "\n";
    }
  }
}

bool is_image_source(const std::string &src) {
  std::string s = to_lower(src);
  size_t len = s.size();
  return (len > 4 && (s.substr(len - 4) == ".jpg" ||
                      s.substr(len - 4) == ".png" ||
                      s.substr(len - 4) == ".bmp")) ||
         (len > 5 && (s.substr(len - 5) == ".jpeg" ||
                      s.substr(len - 5) == ".webp"));
}

// Extract mentioned filenames from the user query for source-targeted retrieval
std::vector<std::string>
extract_mentioned_files(const std::string &query,
                        const std::vector<std::string> &chunk_sources) {
  std::string q_lower = to_lower(query);
  std::vector<std::string> mentioned;
  std::set<std::string> unique_sources(chunk_sources.begin(),
                                       chunk_sources.end());
  for (const auto &src : unique_sources) {
    if (src == "system")
      continue;
    std::string src_lower = to_lower(src);
    // Check for exact filename match
    if (q_lower.find(src_lower) != std::string::npos) {
      mentioned.push_back(src);
      continue;
    }
    // Check without extension
    size_t dot = src_lower.rfind('.');
    if (dot != std::string::npos) {
      std::string name_only = src_lower.substr(0, dot);
      if (name_only.size() > 2) {
        size_t pos = q_lower.find(name_only);
        if (pos != std::string::npos) {
          // Ensure it's not actually referring to a different extension
          bool followed_by_dot = (pos + name_only.size() < q_lower.size() &&
                                  q_lower[pos + name_only.size()] == '.');
          if (!followed_by_dot) {
            mentioned.push_back(src);
          }
        }
      }
    }
  }
  return mentioned;
}

int main(int argc, char *argv[]) {
  print_banner();
  fs::path base_path = fs::current_path();
  if (!fs::exists(base_path / "models")) {
    std::cerr << ROSE << "  Error: models/ directory not found.\n"
              << "  Run from the OvaSearch project directory or ensure models are downloaded.\n"
              << RESET;
    return 1;
  }
  std::string models_dir = (base_path / "models").string();
  std::string data_folder = (base_path / "data").string();

  print_divider();
  std::cout << BLUE << BOLD << "  \xe2\x96\xb8 Loading models" << RESET << "\n";
  print_divider();
  std::cout << "\n";

  size_t dimensions = 384;
  unum::usearch::metric_punned_t metric(dimensions,
                                        unum::usearch::metric_kind_t::cos_k);
  unum::usearch::index_dense_t index =
      unum::usearch::index_dense_t::make(metric);
  std::cout << GREEN << "    \xe2\x9c\x93 " << RESET << WHITE << "Vector index"
            << RESET << GRAY << "   384-dim cosine" << RESET << "\n";

  std::string emb_path = models_dir + "/bge-small-en-v1.5";
  ov::genai::TextEmbeddingPipeline embed_pipe(emb_path, "CPU");
  std::cout << GREEN << "    \xe2\x9c\x93 " << RESET << WHITE << "Embeddings"
            << RESET << GRAY << "     bge-small-en-v1.5" << RESET << "\n";

  std::string vlm_path = models_dir + "/Qwen2-VL-2B-Instruct-INT4";
  std::cout << GRAY << "    \xe2\x97\x8b Vision model   loading..." << RESET
            << std::flush;
  ov::genai::VLMPipeline vlm_pipe(vlm_path, "CPU");
  std::cout << "\r" << GREEN << "    \xe2\x9c\x93 " << RESET << WHITE
            << "Vision" << RESET << GRAY << "         Qwen2-VL-2B-INT4" << RESET
            << "         \n";

  std::string llm_path = models_dir + "/Llama-3.2-3B-Instruct-INT4";
  std::cout << GRAY << "    \xe2\x97\x8b Language       loading..." << RESET
            << std::flush;
  ov::genai::LLMPipeline gen_pipe(llm_path, "CPU");
  std::cout << "\r" << GREEN << "    \xe2\x9c\x93 " << RESET << WHITE
            << "Language" << RESET << GRAY << "       Llama-3.2-3B-INT4"
            << RESET << "         \n";

  auto streamer = [](std::string word) {
    std::cout << WHITE << word << RESET << std::flush;
    return ov::genai::StreamingStatus::RUNNING;
  };

  fs::create_directories(data_folder);
  std::vector<std::string> document_chunks;
  std::vector<std::string> chunk_sources;

  auto_convert_documents(base_path, data_folder);

  std::cout << "\n";
  print_divider();
  std::cout << BLUE << BOLD << "  \xe2\x96\xb8 Indexing knowledge base" << RESET
            << "\n";
  print_divider();
  std::cout << "\n";
  load_documents(data_folder, document_chunks, chunk_sources, embed_pipe, index,
                 vlm_pipe, true);

  std::cout << GRAY << "    Formats  " << RESET << SKY
            << ".txt .md .jpg .png .jpeg" << RESET << DARK << "  \xe2\x94\x82  "
            << RESET << LAVENDER << ".pdf .docx .pptx" << RESET << GRAY
            << " (auto-converted)" << RESET << "\n";
  std::cout << GRAY << "    Commands " << RESET << TEAL << "reload" << RESET
            << DARK << " \xe2\x94\x82 " << RESET << TEAL << "exit" << RESET
            << "\n";
  std::cout << "\n";
  print_divider();
  std::cout << "\n";
  while (true) {
    std::cout << SKY << BOLD << "  \xe2\x9d\xaf " << RESET;
    std::string query;
    std::getline(std::cin, query);
    if (query == "exit" || query.empty())
      break;
    if (query == "reload") {
      std::cout << "\n";
      print_divider();
      std::cout << BLUE << BOLD << "  \xe2\x96\xb8 Reloading knowledge base"
                << RESET << "\n";
      print_divider();
      std::cout << "\n";
      auto_convert_documents(base_path, data_folder);
      load_documents(data_folder, document_chunks, chunk_sources, embed_pipe,
                     index, vlm_pipe, true);
      continue;
    }
    std::cout << "\n";

    // -- Retrieval with filename-aware filtering --
    auto t0 = std::chrono::high_resolution_clock::now();

    // Check if the user mentioned a specific file by name
    auto mentioned_files = extract_mentioned_files(query, chunk_sources);

    std::string context_block;
    std::set<std::string> cited_sources;
    int relevant_count = 0;

    if (!mentioned_files.empty()) {
      // Filename-targeted retrieval: directly use chunks from mentioned files
      for (size_t i = 0; i < document_chunks.size() && relevant_count < 3;
           ++i) {
        for (const auto &mf : mentioned_files) {
          if (chunk_sources[i] == mf) {
            {
              std::string label = "[Source " +
                                  std::to_string(relevant_count + 1) + ": " +
                                  chunk_sources[i] + "]";
              if (is_image_source(chunk_sources[i]))
                context_block +=
                    label + " (Image)\nDescription: " +
                    document_chunks[i] + "\n\n";
              else
                context_block +=
                    label + " (Text file)\nContent: " +
                    document_chunks[i] + "\n\n";
            }
            cited_sources.insert(chunk_sources[i]);
            relevant_count++;
            break;
          }
        }
      }
    }

    // If no specific file was mentioned, fall back to semantic search
    if (mentioned_files.empty()) {
      auto q_res = embed_pipe.embed_query(query);
      std::vector<float> query_emb = std::get<std::vector<float>>(q_res);
      auto search_results = index.search(query_emb.data(), 5);
      for (std::size_t i = 0; i < search_results.size(); ++i) {
        if (relevant_count >= 3)
          break;
        float distance = search_results[i].distance;
        if (distance > 0.45f)
          continue;
        std::size_t doc_id = search_results[i].member.key;
        if (doc_id < document_chunks.size()) {
          // Skip if we already have this source from filename matching
          if (cited_sources.count(chunk_sources[doc_id]))
            continue;
          {
            std::string label = "[Source " +
                                std::to_string(relevant_count + 1) + ": " +
                                chunk_sources[doc_id] + "]";
            if (is_image_source(chunk_sources[doc_id]))
              context_block +=
                  label + " (Image)\nDescription: " +
                  document_chunks[doc_id] + "\n\n";
            else
              context_block +=
                  label + " (Text file)\nContent: " +
                  document_chunks[doc_id] + "\n\n";
          }
          cited_sources.insert(chunk_sources[doc_id]);
          relevant_count++;
        }
      }
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double search_ms =
        std::chrono::duration<double, std::milli>(t1 - t0).count();

    if (!cited_sources.empty()) {
      std::cout << GRAY << "    sources " << RESET;
      bool first = true;
      for (const auto &src : cited_sources) {
        if (!first)
          std::cout << DARK << " \xe2\x94\x82 " << RESET;
        std::cout << LAVENDER << src << RESET;
        first = false;
      }
      std::cout << DARK << "  " << std::fixed << std::setprecision(0)
                << search_ms << "ms" << RESET << "\n\n";
    }

    // Build a prompt with proper structure for Llama 3.2
    std::string sys_msg =
        "You are OvaSearch, a precise retrieval assistant. Answer questions "
        "using ONLY the provided context.\n"
        "Rules:\n"
        "1. Read ALL provided sources carefully. Each is labeled [Source N: "
        "filename].\n"
        "2. For text files: ALWAYS quote or paraphrase the actual file "
        "content. Never claim content is missing when it is provided "
        "above.\n"
        "3. For images: describe what is shown based on the provided "
        "description.\n"
        "4. When multiple sources are provided, ALWAYS address the content "
        "of EACH source in your answer.\n"
        "5. Always cite the source filename in brackets, e.g. [filename].\n"
        "6. If no source contains relevant information, say \"I don't have "
        "information on this.\"\n"
        "7. Be concise and factual. Do not speculate beyond what is in the "
        "sources.";

    std::string prompt;
    if (context_block.empty()) {
      prompt = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
               "You are OvaSearch. The knowledge base has no relevant "
               "information for this question. Respond that you don't have "
               "information on this topic.<|eot_id|>"
               "<|start_header_id|>user<|end_header_id|>\n\n" +
               query +
               "<|eot_id|>"
               "<|start_header_id|>assistant<|end_header_id|>\n\n";
    } else {
      prompt =
          "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n" +
          sys_msg +
          "<|eot_id|>"
          "<|start_header_id|>user<|end_header_id|>\n\n"
          "Context:\n" +
          context_block +
          [&]() -> std::string {
            if (cited_sources.size() > 1) {
              std::string hint = "Retrieved sources: ";
              bool first = true;
              for (const auto &src : cited_sources) {
                if (!first)
                  hint += ", ";
                hint += src;
                hint += is_image_source(src) ? " (image)" : " (text)";
                first = false;
              }
              return hint + ". Address each source.\n";
            }
            return std::string();
          }() +
          "\nQuestion: " + query +
          "<|eot_id|>"
          "<|start_header_id|>assistant<|end_header_id|>\n\n";
    }

    std::cout << "    ";
    auto gen_start = std::chrono::high_resolution_clock::now();
    ov::genai::GenerationConfig gen_config;
    gen_config.max_new_tokens = 250;
    gen_config.repetition_penalty = 1.1f;
    gen_pipe.generate(prompt, gen_config, streamer);
    auto gen_end = std::chrono::high_resolution_clock::now();
    double gen_ms =
        std::chrono::duration<double, std::milli>(gen_end - gen_start).count();
    std::cout << "\n"
              << DARK << "    " << std::fixed << std::setprecision(0) << gen_ms
              << "ms" << RESET << "\n\n";
  }
  std::cout << "\n" << GRAY << "  Goodbye." << RESET << "\n\n";
  return 0;
}
