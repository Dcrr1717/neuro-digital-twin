"""Command-line interface: ``python -m neurotwin run --config configs/small.yaml``."""

from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="neurotwin", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="simulate data, train, evaluate and write the report")
    run_p.add_argument("--config", default="configs/small.yaml", help="YAML config file")
    run_p.add_argument("--out", default=None, help="output directory (default results/<name>)")
    run_p.add_argument("--figdir", default=None, help="figure directory (default <out>/figures)")
    run_p.add_argument("--no-figures", action="store_true", help="skip figure generation")
    args = parser.parse_args(argv)

    if args.command == "run":
        from .pipeline import run

        cfg = load_config(args.config)
        out = Path(args.out) if args.out else Path("results") / cfg.name
        figdir = Path(args.figdir) if args.figdir else None
        run(cfg, out, figdir, make_figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
