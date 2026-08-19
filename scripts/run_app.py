#!/usr/bin/env python
"""Run the Axial Screening Assistant web application.

Usage:
    python scripts/run_app.py
    python scripts/run_app.py --port 8000 --debug
    python scripts/run_app.py --host 0.0.0.0        # accessible on the local network
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from neuroscan.config import load_config
from neuroscan.utils import setup_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the Axial Screening Assistant web application")
    parser.add_argument("--config", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    setup_logging(args.log_level)

    overrides: dict = {}
    web: dict = {}
    if args.host:
        web["host"] = args.host
    if args.port:
        web["port"] = args.port
    if args.debug:
        web["debug"] = True
    if web:
        overrides["web"] = web

    cfg = load_config(args.config, overrides=overrides)
    cfg.paths.ensure_all()

    from neuroscan.web.app import create_app

    app = create_app(cfg)
    service = app.config["NEUROSCAN_SERVICE"]
    status = service.status

    print("\n" + "=" * 70)
    print("  Axial Screening Assistant")
    print("=" * 70)
    print(f"  Classifier     : {'ready' if status['model'] else 'NOT LOADED'}")
    print(f"  Knowledge base : {'ready' if status['rag'] else 'NOT LOADED'}")
    print(f"  Language model : {status['llm_provider'] or 'none'}"
          f"{'' if status['llm'] else '  (degraded: source text shown directly)'}")
    for error in status.get("errors", []):
        print(f"\n  ! {error}")
    print(f"\n  http://{cfg.web.host}:{cfg.web.port}")
    print("=" * 70 + "\n")

    if cfg.web.debug:
        app.run(host=cfg.web.host, port=cfg.web.port, debug=True)
    else:
        # Waitress on Windows, since gunicorn is POSIX-only. Flask's built-in
        # server is explicitly not for anything but debugging.
        try:
            from waitress import serve

            serve(app, host=cfg.web.host, port=cfg.web.port, threads=4)
        except ImportError:
            print("waitress not installed - falling back to the Flask dev server.\n")
            app.run(host=cfg.web.host, port=cfg.web.port, debug=False)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("\nStopped.")
        raise SystemExit(0) from None
