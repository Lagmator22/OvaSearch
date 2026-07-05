# OvaSearch code tour (plain language)

Everything lives in one file: `main.cpp` (~950 lines). This doc walks
through it top to bottom so you always know where you are. For diagrams,
see `ARCHITECTURE.md` in the repo root.

## The one-sentence version

OvaSearch reads your documents and images, turns each piece into a list of
384 numbers (an "embedding"), stores those in a fast similarity index, and
when you ask a question it finds the closest pieces and hands them to a
local Llama model to write the answer. All on your machine, all in C++.

## The three models (loaded at startup)

| Model | Job | Folder in models/ |
|---|---|---|
| bge-small-en-v1.5 | Turns text into 384-dim embedding vectors | `bge-small-en-v1.5` |
| Qwen2-VL-2B INT4 | Looks at images and writes a caption | `Qwen2-VL-2B-Instruct-INT4` |
| Llama-3.2-3B INT4 | Writes the final answer from retrieved context | `Llama-3.2-3B-Instruct-INT4` |

All three run through OpenVINO GenAI pipelines on CPU. If any folder is
missing, startup now fails with a readable message telling you to run
`pull_model.py`.

## Walk through main.cpp

**Banner and colors (top of file).** ANSI escape codes for the pretty
terminal output. Nothing functional.

**`load_image_as_tensor()`.** Loads an image with stb_image and wraps the
raw pixels in an OpenVINO tensor *without copying them* (the custom
`ImageAllocator` hands OpenVINO the same buffer stb_image allocated, and
frees it when OpenVINO is done). This is the "zero-copy ingestion" trick.

**`analyze_image_with_vlm()`.** Asks Qwen2-VL to describe an image
(objects, colors, setting). That caption becomes the searchable text for
the image.

**`create_sliding_window_chunks()`.** Cuts long text into 1000-character
chunks that overlap by 200 characters, so a sentence cut in half by a
chunk border still appears whole in the neighbor chunk.

**The cache (`.ovasearch_cache/` inside your data folder).** Four files:

- `manifest.txt`: every indexed file path + its modification time.
- `chunks.dat` / `sources.dat`: the text chunks and which file each came
  from, separated by an unusual byte sequence.
- `embeddings.bin`: a count, then count x 384 floats.

On startup, `load_documents()` compares the manifest against the folder:

- unchanged file: reuse its cached chunks (instant),
- deleted file: drop its chunks,
- **modified file: drop its old chunks, then re-process it** (previously
  modified files were re-added without dropping the stale copies, which
  duplicated content),
- new file: chunk it, embed it, add it to the index.

The cache loader also refuses to load a corrupted `embeddings.bin`: the
count in the header must exactly match the file's real size, otherwise a
damaged cache could request absurd memory.

**`auto_convert_documents()`.** PDFs, DOCX and PPTX can't be read
directly, so this shells out to `prepare_documents.py` (pdfplumber etc.),
which writes `<name>.extracted.txt` next to each document. Originals get
moved to `data_backup/`. Orphaned extracted files (original deleted) are
cleaned up. The path is shell-quoted.

**The REPL (bottom of main).** For each question you type:

1. `extract_mentioned_files()` checks if you named a specific file
   ("what does report.pdf say?"). If yes, its chunks are used directly.
2. Otherwise: embed the question, ask the USearch index for the 5 nearest
   chunks by cosine distance, keep up to 3 that score under the 0.45
   relevance threshold.
3. Retrieved chunks are pasted into a Llama-3.2 chat prompt with strict
   rules (answer only from sources, cite filenames). If nothing relevant
   was found, a special "say you don't know" prompt is used instead.
4. The answer streams token by token, followed by timing info.

Commands: `reload` re-scans the data folder, `exit` (or empty line) quits.

## Known limits (honest list)

- Everything is one file; fine at this size, worth splitting if it grows.
- Chunks are embedded one at a time during indexing; batching them would
  speed up first-time indexing considerably.
- The chunk separator is a byte string; a binary-ish .txt containing that
  exact sequence would confuse the cache (it would just rebuild).
- Filename-targeted retrieval takes the first 3 matching chunks in index
  order; it doesn't yet rank within the file or balance across multiple
  mentioned files.
