"""Root entry point — forwards to src/main.py.

Run from the project root:
    python main.py              # all stages (W3 + W4 + W5)
    python main.py --w3         # Monte Carlo engine only
    python main.py --w4         # COS engine only
    python main.py --w5         # data pipeline only
    python main.py --log-level DEBUG

All parameters are read from .env in this directory.
See .env.example for the full list of configurable variables.
"""

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.main import main  # noqa: E402

if __name__ == "__main__":
    main()
