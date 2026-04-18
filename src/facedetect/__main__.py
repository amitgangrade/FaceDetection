from __future__ import annotations

from pathlib import Path

from .app import build_and_run


def main() -> None:
    build_and_run(project_root=Path.cwd())


if __name__ == "__main__":
    main()
