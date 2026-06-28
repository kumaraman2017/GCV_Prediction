import subprocess
import sys
from pathlib import Path

NOTEBOOK_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "01_eda.ipynb"
BUILD_SCRIPT = Path(__file__).resolve().parents[1] / "notebooks" / "build_eda_notebook.py"


def test_eda_notebook_builds_and_executes_without_error():
    subprocess.run([sys.executable, str(BUILD_SCRIPT)], check=True)
    assert NOTEBOOK_PATH.exists()

    result = subprocess.run(
        [
            sys.executable, "-m", "nbconvert",
            "--to", "notebook", "--execute", "--output", "01_eda.ipynb",
            str(NOTEBOOK_PATH),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
