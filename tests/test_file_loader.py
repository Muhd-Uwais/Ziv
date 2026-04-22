import os
import tempfile
from ziv.core.file_loader import load_files_from_directory


def test_loads_python_files():
    with tempfile.TemporaryDirectory() as tmp:
        (open(os.path.join(tmp, "main.py"), "w")).write("def hello(): pass")
        files = load_files_from_directory(tmp)
        assert len(files) == 1
        assert "main.py" in files[0]["file_path"]


def test_skips_venv():
    with tempfile.TemporaryDirectory() as tmp:
        venv = os.path.join(tmp, ".venv", "lib")
        os.makedirs(venv)
        open(os.path.join(venv, "site.py"), "w").write("x = 1")
        open(os.path.join(tmp, "app.py"), "w").write("x = 1")
        files = load_files_from_directory(tmp)
        assert len(files) == 1
        print(files[0])
        assert "app.py" in files[0]["file_path"]


def test_skips_pycache():
    with tempfile.TemporaryDirectory() as tmp:
        cache = os.path.join(tmp, "__pycache__")
        os.makedirs(cache)
        open(os.path.join(cache, "main.cpython-312.pyc"), "w").write("")
        open(os.path.join(tmp, "app.py"), "w").write("x = 1")
        files = load_files_from_directory(tmp)
        assert len(files) == 1


def test_empty_directory_returns_empty_list():
    with tempfile.TemporaryDirectory() as tmp:
        assert load_files_from_directory(tmp) == []
