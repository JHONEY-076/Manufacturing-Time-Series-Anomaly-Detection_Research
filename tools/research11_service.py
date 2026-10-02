"""Start/stop only this workspace's background Uvicorn server (Windows)."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import time
import urllib.error
import urllib.request
import webbrowser

import psutil

ROOT = Path(__file__).resolve().parents[1]
PYTHON = ROOT / ".venv-api/Scripts/python.exe"
DIRECTORY = ROOT / "data/research11/service"
STATE = DIRECTORY / "process.json"
URL = "http://127.0.0.1:8000"
COMMAND = [str(PYTHON), "-m", "uvicorn", "fmd_api.app:app", "--host", "127.0.0.1", "--port", "8000"]


def read_state():
    return json.loads(STATE.read_text(encoding="utf-8-sig")) if STATE.exists() else {}


def owned_process():
    record = read_state()
    if not record.get("pid") or record.get("status") == "stopped":
        return None
    try:
        process = psutil.Process(int(record["pid"]))
        command = process.cmdline()
        if not command or Path(command[0]).resolve() != PYTHON.resolve() or command[1:] != COMMAND[1:]:
            raise RuntimeError("Recorded PID is not this workspace's API server. It will not be stopped.")
        if "process_created" in record and abs(process.create_time() - record["process_created"]) > 0.01:
            raise RuntimeError("The recorded PID was reused. It will not be stopped.")
        return process
    except psutil.NoSuchProcess:
        return None


def health():
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(URL + "/health", timeout=1) as response:
            return json.load(response)
    except (urllib.error.URLError, OSError, ValueError):
        return None


def stop():
    process = owned_process()
    if process is None:
        print("No managed API server is running.")
        return
    # The full command, workspace interpreter, and creation time above establish ownership.
    processes = process.children(recursive=True) + [process]
    for child in processes:
        try:
            child.terminate()
        except psutil.NoSuchProcess:
            pass
    _, alive = psutil.wait_procs(processes, timeout=5)
    for child in alive:
        child.kill()
    psutil.wait_procs(alive, timeout=5)
    record = read_state()
    record.update(status="stopped", stopped_utc=datetime.now(timezone.utc).isoformat())
    STATE.write_text(json.dumps(record, indent=2), encoding="utf-8")
    print("API server stopped.")


def start(open_browser):
    process = owned_process()
    if process is not None:
        if health() is None:
            raise RuntimeError("Managed process exists but is not responding. Use restart, or inspect stderr.log.")
        print("API server is already running: " + URL)
    else:
        import socket
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", 8000))
            except OSError as error:
                raise RuntimeError("Port 8000 is occupied by another process. Close its own server terminal first.") from error
        if not PYTHON.is_file():
            raise RuntimeError("Missing .venv-api. See fmd_api/README.md for installation.")
        DIRECTORY.mkdir(parents=True, exist_ok=True)
        with (DIRECTORY / "stdout.log").open("a", encoding="utf-8") as stdout, (DIRECTORY / "stderr.log").open("a", encoding="utf-8") as stderr:
            child = subprocess.Popen(COMMAND, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr,
                                     creationflags=subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP)
        record = {"pid": child.pid, "process_created": psutil.Process(child.pid).create_time(),
                  "started_utc": datetime.now(timezone.utc).isoformat(), "status": "starting",
                  "url": URL, "docs_url": URL + "/docs", "dashboard_url": URL + "/", "command": COMMAND}
        STATE.write_text(json.dumps(record, indent=2), encoding="utf-8")
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if child.poll() is not None:
                raise RuntimeError("Server startup failed. Read data/research11/service/stderr.log.")
            result = health()
            if result and set(result.get("models", [])) == {"baseline", "magnitude_warping", "fmd"}:
                record["status"] = "running"
                STATE.write_text(json.dumps(record, indent=2), encoding="utf-8")
                print("API server ready: " + URL)
                break
            time.sleep(0.2)
        else:
            stop()
            raise RuntimeError("Startup timeout. Read data/research11/service/stderr.log.")
    if open_browser:
        webbrowser.open(URL + "/")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["start", "stop", "restart", "status"])
    parser.add_argument("--open", action="store_true", help="Open the browser dashboard after startup")
    args = parser.parse_args()
    if args.action in ("stop", "restart"):
        stop()
    if args.action in ("start", "restart"):
        start(args.open)
    if args.action == "status":
        process = owned_process()
        print(json.dumps({"managed_pid": process.pid if process else None, "health": health(), "url": URL}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, psutil.Error) as error:
        raise SystemExit(str(error))
