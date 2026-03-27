import subprocess
import os
import signal
import logging


logger = logging.getLogger(__name__)

PID_FILE = ".lfit/server.pid"
LOG_FILE = ".lfit/server.log"


def start_server():
    os.makedirs(".lfit", exist_ok=True)
    if os.path.exists(PID_FILE):
        logger.info("Server already running")
        return

    with open(LOG_FILE, "a") as log_out:
        process = subprocess.Popen(
            ["uvicorn", "lfit.api.embed_server:app", "--port", "8000"],
            stdout=log_out,
            stderr=log_out
        )

    with open(PID_FILE, "w") as f:
        f.write(str(process.pid))

    logger.info("Server started!")


def stop_server():
    if not os.path.exists(PID_FILE):
        logger.info("Server not running")
        return
    
    with open(PID_FILE, "r") as f:
        pid = int(f.read())

    os.kill(pid, signal.SIGTERM)
    os.remove(PID_FILE)

    logger.info("Server stopped")    