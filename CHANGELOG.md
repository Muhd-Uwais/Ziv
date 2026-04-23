# Changelog

All notable changes to Ziv are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning follows [Semantic Versioning](https://semver.org/).

---

## [0.3.0] — 2026-04-23

_Initial public release._

### Added
- Local semantic code search using FAISS (IndexFlatIP) with L2-normalized embeddings
- ONNX Runtime inference (`all-MiniLM-L6-v2`) for fully local, CPU-only embedding
- Incremental indexing — re-embeds only new or modified chunks
- Content-hash based cache stored in `.ziv/cache/`
- Persistent embedding server (eliminates model reload per query)
- CLI commands: `init`, `start`, `stop`, `status`, `build-index`, `search`, `feedback`
- Rich CLI interface (progress bars, tables, panels)
- Global model storage at `~/.ziv/models/` (shared across projects)

### Improved
- Reduced peak RAM usage during indexing from multi-GB spikes to <120 MB through batching and pipeline optimization
- Stabilized memory usage (no linear growth with codebase size)

### Performance (observed, CPU-only)
- Indexing: ~0.05–0.06 sec per chunk
- Query latency: ~2–3 seconds
- Peak RAM: ~80–120 MB during indexing (~1.4k chunks, batch-dependent)

### Known Limitations
- Python-only (`.py`) support
- Uses a general-purpose embedding model (not code-aware)
- Large repositories may require tuning `--batch-size` for optimal performance

---

## [Unreleased]

### Planned
- Code-aware embedding model
- AST-based chunking for improved retrieval accuracy
- Multi-language support