import os
import pytest
from ziv.pipelines.index_builder import BuildIndex
from ziv.pipelines.retriever import Retriever


CORPUS = {
    "auth.py":    "def login(username, password):\n    # verify user credentials\n    pass",
    "database.py": "def connect_db(host, port):\n    # establish database connection\n    return connection",
    "utils.py":   "def format_date(timestamp):\n    # convert unix timestamp to readable date\n    return str(timestamp)",
    "logger.py":  "def log_error(message, level):\n    # write error to log file\n    pass",
    "parser.py":  "def parse_config(filepath):\n    # read and parse yaml config file\n    return config",
}


@pytest.fixture(scope="module")
def build_index(tmp_path_factory):
    temp = tmp_path_factory.mktemp("corpus")
    for filename, content in CORPUS.items():
        (temp / filename).write_text(content)
    index_dir = tmp_path_factory.mktemp("index")
    builder = BuildIndex()
    builder.build_index(root_path=str(temp), output_dir=str(index_dir))
    return str(index_dir)


def top_files(index_dir, query, k=3):
    retriever = Retriever(index_path=index_dir)
    results = retriever.search(query, top_k=k)
    return [os.path.basename(r["file_path"]) for r in results]


def test_auth_query(build_index):
    results = top_files(build_index, "user login authentication")
    assert "auth.py" in results, f"Expected auth.py in top-3, got: {results}"


def test_database_query(build_index):
    results = top_files(build_index, "connect to database")
    assert "database.py" in results, f"Expected database.py in top-3 got: {results}"


def test_config_query(build_index):
    results = top_files(build_index, "read config file yaml")
    assert "parser.py" in results, f"Expected parser.py in top-3, got: {results}"


def test_nonsense_query_does_not_crash(build_index):
    # FAISS always returns k results - just verify no exception
    results = top_files(build_index, "xkqzwpf random gibberish abc123")
    assert isinstance(results, list)
