"""Sync notebook cells into the live Colab frontend if connected."""

import json
from pathlib import Path
import subprocess
import sys
import urllib.request
import os

NOTEBOOK_PATH = Path("/content/video-generation-prototype/VideoGenerationTest.ipynb")
SKILL_SCRIPT = Path("/usr/local/share/colab/agy/.agents/skills/colab_notebook/scripts/notebook.py")


def check_frontend() -> bool:
    host = os.environ.get("COLAB_JUPYTER_IP") or "127.0.0.1"
    port = os.environ.get("KMP_TARGET_PORT") or "9000"
    try:
        req = urllib.request.Request(f"http://{host}:{port}/api/kernels")
        with urllib.request.urlopen(req, timeout=2) as resp:
            data = json.load(resp)
            if data and data[0].get("connections", 0) > 0:
                return True
    except Exception:
        pass
    return False


def main():
    if not NOTEBOOK_PATH.exists():
        print(f"Error: {NOTEBOOK_PATH} does not exist.")
        sys.exit(1)

    with open(NOTEBOOK_PATH, "r", encoding="utf-8") as f:
        nb_data = json.load(f)

    ops = []
    for cell in nb_data.get("cells", []):
        cell_type = cell.get("cell_type")
        source = "".join(cell.get("source", []))
        if cell_type == "markdown":
            ops.append({
                "action": "add_text_cell",
                "params": {"content": source, "cellIndex": 99999},
            })
        elif cell_type == "code":
            ops.append({
                "action": "add_code_cell",
                "params": {"code": source, "cellIndex": 99999, "language": "python"},
            })

    ops_path = Path("/tmp/colab_ops.json")
    ops_path.write_text(json.dumps(ops), encoding="utf-8")
    print(f"Prepared {len(ops)} operations in {ops_path}.")

    if not check_frontend():
        print("Notice: No live Colab frontend connection currently detected (connections=0).")
        print("The notebook file is saved to /content/VideoGenerationTest.ipynb and in git.")
        return

    print("Live frontend detected. Applying batch ops via notebook.py...")
    cmd = [
        "/usr/bin/python3",
        "-I",
        str(SKILL_SCRIPT),
        "call",
        "--batch",
    ]
    with open(ops_path, "r", encoding="utf-8") as f_in:
        res = subprocess.run(cmd, stdin=f_in, capture_output=True, text=True)
    print("STDOUT:", res.stdout)
    print("STDERR:", res.stderr)


if __name__ == "__main__":
    main()
