<div align="center">
  <img src="https://github.com/Muhd-Uwais/Pong/blob/master/ziv_logo.svg?raw=true" alt="Ziv logo" width="120" />
  <h1>Ziv</h1>
  <p><strong>Local semantic code search for Python repositories — no cloud, no API keys.</strong></p>
</div>

<div align="center">

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-Apache%202.0-green)](LICENSE)
[![Status](https://img.shields.io/badge/status-beta-yellow)]()
[![Platform](https://img.shields.io/badge/platform-Linux%20%7C%20Windows-lightgrey)]()

</div>

---

Source Code: https://github.com/Muhd-Uwais/ziv

---

## What is Ziv?

Most codebases are navigated with grep and guesswork. Ziv fixes that.

Instead of hunting for exact keywords, ask questions like *"where is authentication handled?"* and get the most relevant files back based on meaning, not string matches.

Everything runs on your machine — no cloud, no API keys, no code leaving your environment.

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

---

## Requirements

- Python 3.10+
- OS: Linux, Windows (macOS: SIGTERM-based process handling is present in the code, but end-to-end testing on macOS has not been done — community feedback welcome)
- Disk: ~100 MB baseline (model).
  Additional storage scales with indexed code (~1–2 KB per chunk for embeddings + metadata).
- Baseline RAM usage around 150 to 200 MB while the embedding server is running

RAM can increase during indexing depending on batch size and workload. Run `ziv stop` to fully unload the model.

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

Download the embedding model to local storage.

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

> **Note:** `ziv start` and `ziv build-index` must both be completed before searching.

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

Display the current state of the background embedding server — whether it is running, its PID, and whether the model is fully loaded and ready.

```
ziv status
```

No flags.

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

Ziv's pipeline has four stages:

1. **Load files**  
   Walk the directory tree, skip dependency/build/cache directories, and read `.py` files as UTF-8 text.

2. **Chunk files**  
   Split each file into overlapping line-based chunks. Each chunk gets a deterministic content hash.

3. **Generate embeddings**  
   A local FastAPI server loads the ONNX model once and exposes:
   - `POST /encode-chunks` for batch indexing
   - `POST /encode-query` for search-time query encoding

4. **Build and search the FAISS index**  
   Chunk embeddings are L2-normalized, stored in a FAISS `IndexFlatIP`, and searched with cosine similarity.

In practice, `ziv build-index` prepares the chunk embeddings and writes the FAISS index to `.ziv/`, while `ziv search` embeds the user query and retrieves the top matching chunks from that local index.

---

## Performance

| Repo                    | Files  | Chunks | Index Time (first run) | Query Time |
| ----------------------- | ------ | ------ | ---------------------- | ---------- |
| Flask (pallets/flask)   | 24     | 327    | ~31s                   | ~2s        |
| Large repo (~55k lines) | varies | ~1,395 | ~130s                  | ~2s        |

> Incremental re-indexing (after small changes) is significantly faster — only modified files are re-embedded.

---

## Known Limitations

### Model quality

Ziv currently uses `all-MiniLM-L6-v2`, which is a general-purpose English sentence model rather than a code-specific model.

As a result:
- Search quality is useful for codebase navigation, but not ideal for deep code semantics
- Very short queries may return weaker results

A code-aware model is the top priority for the next major improvement.

### Language support

Only Python files (`.py`) are indexed in the current release.

### Query time

Queries still take around 2 to 3 seconds and this remains an active optimization target.

### Beta status

APIs and CLI flags may still change before 1.0.

---

## License

This project is licensed under the terms of the Apache License, Version 2.0.

---

## Contact & Feedback

**Muhd Uwais** — Project Author

- 📝 **Contact Form**: [Send a Message](https://nox-uwi.github.io/Form/)
- 🐛 **Bug Reports**: [Open an Issue](https://github.com/Muhd-Uwais/ziv/issues)
- 💬 **Questions**: [Start a Discussion](https://github.com/Muhd-Uwais/ziv/discussions)
- 📟 **In-terminal feedback**: Run `ziv feedback`

---