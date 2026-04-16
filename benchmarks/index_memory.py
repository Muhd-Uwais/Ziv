import os
import time
import json
import psutil

try:
    import resource
except ImportError:
    resource = None


class MemoryProbe:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.samples = []
        self.start_time = time.time()

    def _rss_mb(self) -> float:
        return self.process.memory_info().rss / (1024 * 1024)

    def _peak_rss_mb(self):
        if resource is None:
            return None

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # Linux reports KB, macOS reports bytes
        if os.uname().sysname == "Darwin":
            return peak / (1024 * 1024)
        return peak / 1024

    def mark(self, label: str):
        self.samples.append({
            "label": label,
            "elapsed_sec": round(time.time() - self.start_time, 3),
            "rss_mb": round(self._rss_mb(), 2),
            "peak_rss_mb": round(self._peak_rss_mb(), 2) if self._peak_rss_mb() is not None else None,
        })

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.samples, f, indent=2)