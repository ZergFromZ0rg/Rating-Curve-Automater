"""``rca`` — one entry point for the Rating Curve Automater command-line tools.

    rca validate  gaugings.xlsx --output-csv cleaned.csv
    rca fit       --segments auto --bootstrap 2000
    rca report    --rating-table-csv rating_table.csv
    rca app       # launch the Streamlit UI (needs the 'app' extra, from a source checkout)

Each subcommand forwards its remaining arguments to the matching module, so
``rca fit --help`` shows that command's full option list.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from rating_curve_automater import __version__

_REPO_ROOT = Path(__file__).resolve().parents[1]

_USAGE = """usage: rca <command> [options]

commands:
  validate   clean & validate a gauging workbook to CSV
  fit        fit a rating curve from a cleaned CSV
  report     fit and write the multi-sheet Excel report (+ rating table)
  app        launch the Streamlit web UI (needs the 'app' extra + a source checkout)

  rca <command> --help   for that command's options
  rca --version
"""


def _delegate(module: str, argv: list[str], prog: str) -> None:
    sys.argv = [prog, *argv]
    import importlib

    importlib.import_module(module).main()


def _launch_streamlit() -> int:
    script = _REPO_ROOT / "app.py"
    if not script.is_file():
        print(
            "app.py is not next to the installed package. The web UI runs from a "
            "source checkout: clone the repo and run\n  streamlit run app.py\nfrom its root.",
            file=sys.stderr,
        )
        return 1
    try:
        import streamlit  # noqa: F401
    except ImportError:
        print("The web UI needs the 'app' extra:  pip install 'rating-curve-automater[app]'", file=sys.stderr)
        return 1
    return subprocess.call([sys.executable, "-m", "streamlit", "run", str(script)])


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(_USAGE)
        return
    if argv[0] in ("-V", "--version", "version"):
        print(f"rca {__version__}")
        return

    command, rest = argv[0], argv[1:]
    delegates = {
        "validate": "rating_curve_automater.field_measurement_validation",
        "fit": "rating_curve_automater.rating_curve_fitting",
        "report": "rating_curve_automater.rating_curve_report",
    }
    if command in delegates:
        _delegate(delegates[command], rest, f"rca {command}")
    elif command == "app":
        raise SystemExit(_launch_streamlit())
    else:
        print(f"unknown command: {command}\n\n{_USAGE}", file=sys.stderr)
        raise SystemExit(2)


if __name__ == "__main__":
    main()
