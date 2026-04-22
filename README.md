<div align="center">
  <img src="https://github.com/Muhd-Uwais/Pong/blob/master/ziv_logo.svg?raw=true" alt="Ziv logo" width="120"/>
  <h1>Ziv</h1>
  <p><strong>Local semantic code search for Python repositories — no cloud, no API keys.</strong></p>
</div>

<div align="center">

[![PyPI version](https://img.shields.io/pypi/v/ziv?color=blue&label=pypi)](https://pypi.org/project/ziv/)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-beta-yellow)]()
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey)]()

</div>

<div align="center">
  <img src="https://github.com/Muhd-Uwais/Pong/blob/master/ziv_demo.gif?raw=true" alt="ziv demo" width="700"/>
</div>

---

## What is Ziv?

Most codebases are navigated with grep and guesswork. Ziv fixes that.

Instead of hunting for exact keywords, ask questions like *"where is authentication handled?"* and get the most relevant files back based on meaning, not string matches. No more reading 50 files to find the one that matters.

Everything runs on your machine — no cloud, no API keys, no code leaving your environment.

---

## Who Is It For?

- Developers joining a new codebase at a new job who need to understand the project quickly
- Open source contributors navigating an unfamiliar project before submitting a fix
- Developers working on large Python projects with many files spread across many modules
- Anyone who has spent more time grepping than coding
- Teams wanting fast, private, offline code intelligence with zero infrastructure

---

## Features

- **Natural language search** — query your code with plain English using semantic similarity (cosine distance)
- **Fully local & private** — the embedding model and FAISS index run entirely on your machine; no data is sent anywhere
- **ONNX-accelerated inference** — uses a compact `all-MiniLM-L6-v2`-based ONNX model for fast CPU inference with multi-threaded ONNX Runtime
- **Incremental indexing** — chunks are content-hashed; re-running `build-index` only embeds files that changed, not the entire codebase
- **Smart directory pruning** — automatically skips `.git`, `__pycache__`, virtual environments, build artefacts, and more
- **Background embedding server** — the model loads once into a persistent FastAPI process; all index and search calls reuse it with no per-query startup cost
- **Rich CLI output** — color-coded tables, progress bars, spinners, and panels powered by [Rich](https://github.com/Textualize/rich)
- **Port auto-selection** — the embedding server picks a free port automatically if you don't specify one

---

## Installation

```bash
pip install ziv
```

On first run, `ziv init` downloads the ONNX embedding model (~90 MB) from Hugging Face automatically. Requires Python 3.10+.

> **Warning:** Current version supports Python files only. Support for JavaScript, TypeScript, Rust, and other languages is planned for future versions.

---

## Quickstart

```bash
# Step 1: Go to your Python project
cd your-python-project

# Step 2: Initialize ziv (downloads model on first run — ~15s)
ziv init

# Step 3: Start the background server
ziv start

# Step 4: Build the index (run once, or after significant changes)
ziv build-index

# Step 5: Search your codebase
ziv search "where is request context handled?"

# Step 6: Stop the server when done
ziv stop
```

Example output:

```
🔍 Query: "where is request context handled?" — 3 results found

╭───┬──────────────────────────────────────┬───────╮
│ # │ File                                 │ Score │
├───┼──────────────────────────────────────┼───────┤
│ 1 │ flask/ctx.py                         │ 0.505 │
│ 2 │ flask/sansio/scaffold.py             │ 0.479 │
│ 3 │ flask/helpers.py                     │ 0.477 │
╰───┴──────────────────────────────────────┴───────╯
```

---

## Requirements

- Python 3.10+
- OS: Linux, Windows (macOS: SIGTERM-based process handling is present in the code, but end-to-end testing on macOS has not been done — community feedback welcome)
- Disk: ~100 MB baseline (model).
  Additional storage scales with indexed code (~1–2 KB per chunk for embeddings + metadata).
- RAM: ~150–200 MB baseline (model + runtime).
  Can increase during indexing depending on batch size and workload,
  but remains stable (no linear growth with codebase size).
  Run `ziv stop` to fully release memory.

---

## Commands Reference

| Command              | Description                                           | When to use it                                        |
| -------------------- | ----------------------------------------------------- | ----------------------------------------------------- |
| `ziv init`           | Download the ONNX embedding model to `~/.ziv/models/` | Once per machine, before first use                    |
| `ziv start`          | Launch the background embedding server                | Before running `build-index` or `search`              |
| `ziv stop`           | Stop the background embedding server                  | When done searching                                   |
| `ziv status`         | Show server status, PID, and model readiness          | To check if the server is running                     |
| `ziv build-index`    | Embed all `.py` files and build the FAISS index       | Once after setup, then after significant code changes |
| `ziv search <query>` | Search the codebase with a natural language query     | Any time the server is running and index is built     |
| `ziv feedback`       | Open the built-in feedback form in the browser        | To report bugs or request features                    |

### Command Details

### `ziv init`

Download the embedding model to local storage. Run this **once** before using any other command.

```
ziv init [OPTIONS]
```

The model is stored in `~/.ziv/models/embedder-fast-onnx/`. If the model is already fully installed, the command exits immediately without re-downloading.

| Flag        | Short | Default | Description                                                    |
| ----------- | ----- | ------- | -------------------------------------------------------------- |
| `--model`   | `-m`  | `fast`  | Model variant to download. Only `fast` is currently available. |
| `--verbose` | `-v`  | `False` | Show detailed download logs.                                   |
| `--help`    | `-h`  | —       | Show help for this command.                                    |

**Example:**

```bash
ziv init
ziv init --model fast --verbose
```

---

### `ziv start`

Start the background embedding server. The server loads the ONNX model into memory once and keeps it resident so `build-index` and `search` share the same model instance without any per-call startup cost.

```
ziv start [OPTIONS]
```

The server runs on `127.0.0.1` and writes its PID, port, and URL to `~/.ziv/server.instance`. If a server is already running, the command reports the existing URL and exits.

| Flag        | Short | Default | Description                                                                     |
| ----------- | ----- | ------- | ------------------------------------------------------------------------------- |
| `--port`    | `-p`  | auto    | Port to bind the server on. Automatically selects a free port if not specified. |
| `--verbose` | `-v`  | `False` | Show detailed startup logs.                                                     |
| `--help`    | `-h`  | —       | Show help for this command.                                                     |

**Examples:**

```bash
ziv start              # auto-select a free port
ziv start --port 8765  # bind to a specific port
```

---

### `ziv build-index`

Scan a directory for Python source files, chunk them into overlapping line windows, embed the chunks via the running server, and save a FAISS vector index to `.ziv/` in the current directory.

Re-running this command is safe and **incremental**: previously embedded chunks are loaded from the cache in `.ziv/cache/` and only new or changed chunks are sent to the embedding server.

```
ziv build-index [PATH] [OPTIONS]
```

| Argument / Flag | Short | Default                 | Description                                                                 |
| --------------- | ----- | ----------------------- | --------------------------------------------------------------------------- |
| `PATH`          | —     | `.` (current directory) | Root directory to index.                                                    |
| `--batch-size`  | `-b`  | `128`                   | Number of chunks per embedding request. Accepted values: `32`, `64`, `128`. |
| `--verbose`     | `-v`  | `False`                 | Show internal pipeline logs.                                                |
| `--help`        | `-h`  | —                       | Show help for this command.                                                 |

**Examples:**

```bash
ziv build-index                        # index current directory
ziv build-index ./src                  # index a subdirectory
ziv build-index /path/to/project -b 64 # custom batch size
ziv build-index . --verbose            # with detailed logging
```

> **Note:** The embedding server (`ziv start`) must be running before calling `build-index`.

---

### `ziv search`

Search the indexed codebase using a natural language query. The query is embedded by the running server and compared against the FAISS index using cosine similarity. Results are ranked by score (higher = more relevant).

```
ziv search QUERY [OPTIONS]
```

| Argument / Flag | Short | Default      | Description                    |
| --------------- | ----- | ------------ | ------------------------------ |
| `QUERY`         | —     | *(required)* | Natural language search query. |
| `--limit`       | `-l`  | `3`          | Number of results to return.   |
| `--verbose`     | `-v`  | `False`      | Show internal logs.            |
| `--help`        | `-h`  | —            | Show help for this command.    |

**Examples:**

```bash
ziv search "where is authentication handled?"
ziv search "session management" --limit 5
ziv search "where is request context handled?" -l 10 --verbose
```

> **Note:** Both the embedding server (`ziv start`) and a built index (`ziv build-index`) must exist before searching.

---

### `ziv stop`

Stop the background embedding server. Sends `SIGTERM` to the server process and cleans up the `~/.ziv/server.instance` file.

```
ziv stop [OPTIONS]
```

| Flag        | Short | Default | Description                  |
| ----------- | ----- | ------- | ---------------------------- |
| `--verbose` | `-v`  | `False` | Show detailed shutdown logs. |
| `--help`    | `-h`  | —       | Show help for this command.  |

**Example:**

```bash
ziv stop
```

---

### `ziv status`

Display the current state of the background embedding server — whether it is running, its PID, the port it is bound to, and whether the model is fully loaded and ready.

```
ziv status
```

No flags. Output example:

```
╭─────── ziv · server status ────────╮
│  Process       ● running           │
│  PID           12345               │
│  Model Status  ✔ ready             │
│  Model         embedder-fast-onnx  │
╰────────────────────────────────────╯
```

---

### `ziv feedback`

Open an in-browser feedback form served locally. The form runs on `127.0.0.1` on a random free port. Submitting feedback saves a JSON file to `~/.ziv/feedback/` and optionally forwards it to the project's feedback endpoint. The server shuts down automatically after submission or after a 300-second timeout.

```
ziv feedback
```

No flags. Press `Ctrl+C` to cancel without submitting.

> **RAM usage:** Ziv is designed to limit RAM consumption — the model is loaded once in the background server and stays resident only while the server is running. If RAM issues occur, run `ziv stop` to unload the model, or open an issue / use `ziv feedback` to report the problem.

---

### Global Flags

These flags are available on the root `ziv` command:

| Flag        | Short | Description                                    |
| ----------- | ----- | ---------------------------------------------- |
| `--version` | `-V`  | Print the installed version of ziv and exit.   |
| `--help`    | `-h`  | Show help for any command or the root command. |

**Examples:**

```bash
ziv --version
ziv -V
ziv --help
ziv search --help
```

---

## Configuration

ziv has no configuration file. All runtime state lives in two locations:

| Path                                | Contents                                                                           |
| ----------------------------------- | ---------------------------------------------------------------------------------- |
| `~/.ziv/models/embedder-fast-onnx/` | Downloaded ONNX model files (`model.onnx`, tokenizer, config)                      |
| `~/.ziv/server.instance`            | PID, port, and URL of the running embedding server                                 |
| `~/.ziv/server.log`                 | Embedding server stdout/stderr log                                                 |
| `~/.ziv/feedback/`                  | JSON files saved from `ziv feedback` submissions                                   |
| `.ziv/` *(project-local)*           | FAISS index (`index.faiss`), metadata map (`id_map.json`)                          |
| `.ziv/cache/` *(project-local)*     | Embedding cache (`embeddings.npy`, `cache_manifest.json`) for incremental indexing |

No environment variables are required. The `~/.ziv` home directory and the `.ziv` project directory are created automatically on first use.

---

## How It Works

ziv's pipeline has four stages:

```
 Source files
      │
      ▼
 ┌─────────────┐
 │ file_loader │  Walk the directory tree, skip VCS/build/dep dirs,
 │             │  read .py files as UTF-8 text records.
 └──────┬──────┘
        │
        ▼
 ┌─────────────┐
 │   chunker   │  Split each file into 40-line sliding-window chunks
 │             │  with 10-line overlap. Each chunk gets a content hash.
 └──────┬──────┘
        │
        ▼
 ┌──────────────────────────────────────────────────────────────┐
 │  Embedding server (FastAPI + uvicorn, background process)    │
 │                                                              │
 │  LightEmbedder (ONNX Runtime + all-MiniLM-L6-v2)             │
 │    tokenize → ONNX inference → mean pool → L2 normalize      │
 │    → 384-dimensional float32 vectors                         │
 └──────────────────┬───────────────────────────────────────────┘
                    │  POST /encode-chunks  (batched)
                    │  POST /encode-query   (search time)
        ┌───────────┘
        │
        ▼
 ┌─────────────┐
 │ vector_store│  Build FAISS IndexFlatIP, L2-normalize vectors,
 │             │  persist index.faiss + id_map.json to .ziv/.
 └──────┬──────┘
        │
        ▼
    ziv search
        │
        └─ embed query → cosine similarity → top-k ranked results
```

A FastAPI server runs as a background process and keeps the ONNX model loaded in memory for the duration of the session. Embeddings are generated in batches (default 128 chunks per batch) via the local HTTP API. The resulting vectors are L2-normalised and stored in a FAISS `IndexFlatIP` index on disk in `.ziv/`. Incremental indexing compares chunk content hashes against a cache manifest, so only files that have changed since the last `build-index` run are re-embedded — the rest are reused from cache.

---

## Performance

| Repo                    | Files  | Chunks | Index Time (first run) | Query Time |
| ----------------------- | ------ | ------ | ---------------------- | ---------- |
| Flask (pallets/flask)   | 24     | 327    | ~31s                   | ~2s        |
| Large repo (~55k lines) | varies | ~1,395 | ~130s                  | ~2s        |

> Incremental re-indexing (after small changes) is significantly faster — only modified files are re-embedded.
>
> Future versions will include Rust-based parallel processing for faster embedding (planned for 0.5.0+).

---

## Known Limitations

**Model quality:**
Ziv uses `all-MiniLM-L6-v2`, a general-purpose English sentence model, not a code-specific model. It was trained on natural language text, not source code. As a result:
- Search quality is useful for codebase navigation but not precise for complex code semantics
- Very short queries (under 4 words) may return less relevant results

A code-aware model is the top priority for version 0.4.0 and will significantly improve accuracy.

**Language support:** Only Python files (`.py`) are indexed in this version.

**Query time:** Queries take ~3 seconds — a known bottleneck being addressed in the roadmap.

**Beta status:** APIs and CLI flags may change before stable release (1.0.0).

---

## Roadmap

| Version    | Status           | Focus                                                               |
| ---------- | ---------------- | ------------------------------------------------------------------- |
| **0.3.0**  | ✅ Current (Beta) | FAISS retrieval, ONNX model, incremental indexing, improved CLI     |
| **0.4.0**  | 🔜 Next           | Code-aware embedding model + AST-based chunking for better accuracy |
| **0.5.0+** | 🔜 Planned        | Multi-language support, Rust-based performance improvements         |

The primary goal of 0.4.0 is replacing the general-purpose model with a code-aware one and switching from line-based chunking to AST-level parsing — this is expected to significantly improve retrieval quality.

---

## Project Structure

```
ziv/
├── src/
│   └── ziv/
│       ├── __init__.py
│       ├── api/
│       │   ├── __init__.py
│       │   ├── embed_server.py      # FastAPI app — ONNX inference endpoints
│       │   ├── embedder.py          # ONNX Runtime session and tokenizer
│       │   └── process_manager.py   # Server start/stop/status, PID/port management
│       ├── cli/
│       │   ├── __init__.py
│       │   ├── feedback.py          # Browser-based feedback form server
│       │   ├── main.py              # Typer CLI — all commands
│       │   └── templates/           # HTML templates for feedback form
│       ├── core/
│       │   ├── __init__.py
│       │   ├── chunker.py           # Sliding-window line-based file chunker
│       │   ├── downloader.py        # HuggingFace model downloader
│       │   ├── file_loader.py       # .py file discovery and loading
│       │   └── vector_store.py      # FAISS index build, load, and search
│       ├── pipelines/
│       │   ├── __init__.py
│       │   ├── index_builder.py     # Full build-index pipeline
│       │   └── retriever.py         # Query pipeline
│       ├── static/                  # Static assets for feedback UI
│       └── utils/
│           ├── __init__.py
│           └── hash_utils.py        # Deterministic SHA hash for chunk IDs
├── assets/
│   ├── ziv_demo.gif
│   └── ziv_logo.svg
├── benchmarks/
├── scripts/
├── tests/
├── pyproject.toml
├── LICENSE
├── NOTICE
├── SECURITY.md
└── README.md
```

---

## Security

For security vulnerabilities, see [SECURITY.md](SECURITY.md). Do not open public issues for security concerns. Contact **ziv.ai.786@gmail.com** to report vulnerabilities privately.

---

## License

Apache License, Version 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for details.

---

## Contributing

Ziv is in active development and welcomes contributions. Open an issue before submitting a large pull request so the direction can be discussed first. Bug reports with clear reproduction steps are especially valuable during the beta — they directly influence what gets fixed in the next release. Feature requests are tracked on [GitHub Issues](https://github.com/Muhd-Uwais/ziv/issues). Code style follows standard Python conventions; run existing tests before submitting.

---

## Contact & Feedback

**Muhd Uwais** — Project Author

- 📝 **Contact Form**: [Send a Message](https://nox-uwi.github.io/Form/)
- 🐛 **Bug Reports**: [Open an Issue](https://github.com/Muhd-Uwais/ziv/issues)
- 💬 **Questions**: [Start a Discussion](https://github.com/Muhd-Uwais/ziv/discussions)
- 📟 **In-terminal feedback**: Run `ziv feedback`

---

<p align="center">
  If Ziv saved you time navigating a codebase, consider giving it a ⭐
</p>
