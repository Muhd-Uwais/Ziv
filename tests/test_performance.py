import os, time, tempfile
from ziv.pipelines.index_builder import BuildIndex

def _make_corpus(tmp_dir, n=300):
    for i in range(n):
        with open(os.path.join(tmp_dir, f"module_{i}.py"), "w") as f:
            f.write(f"def process_{i}(data):\n    # process input data\n    return data\n" * 10)

def test_indexing_speed_baseline():
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as idx:
        _make_corpus(src, n=300)
        start = time.time()
        BuildIndex().build_index(root_path=src, output_dir=idx)
        elapsed = time.time() - start
        assert elapsed < 45, f"Indexing regressed: {elapsed:.1f}s for 300 files (expected < 45s)"