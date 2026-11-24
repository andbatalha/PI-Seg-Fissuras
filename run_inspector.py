import sys
import os
from pathlib import Path
from streamlit.web import cli as stcli


def main():
    root = Path(__file__).resolve().parent
    os.chdir(root)

    sys.argv = [
        "streamlit",
        "run",
        str(root / "visualization" / "app.py"),
        "--server.address",
        "127.0.0.1",
    ]

    sys.exit(stcli.main())


if __name__ == "__main__":
    main()