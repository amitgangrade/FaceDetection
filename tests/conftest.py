import sys
from pathlib import Path

# Make `src/` importable without installing the package.
_root = Path(__file__).resolve().parent.parent
_src = _root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))
