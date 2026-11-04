"""Command-line interface.

    python -m neurotwin run   --config configs/small.yaml [--device cuda]
    python -m neurotwin sweep --config configs/full.yaml --out results/full --patches 5 10 20 50
"""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

from .config import Config, load_config


def _common(p: argparse.ArgumentParser, default_config: str) -> None:
    p.add_argument("--config", default=default_config, help="YAML config file")
    p.add_argument("--out", default=None, help="output directory (default results/<name>)")
    p.add_argument("--figdir", default=None, help="figure directory (default <out>/figures)")
    p.add_argument("--device", default=None,
                   help="override the config's device: auto, cpu, cuda or cuda:N")


def _load(args: argparse.Namespace) -> tuple[Config, Path, Path | None]:
    cfg = load_config(args.config)
    if args.device:
        cfg = replace(cfg, device=args.device)
    out = Path(args.out) if args.out else Path("results") / cfg.name
    figdir = Path(args.figdir) if args.figdir else None
    return cfg, out, figdir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="neurotwin", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    run_p = sub.add_parser("run", help="simulate data, train, evaluate and write the report")
    _common(run_p, "configs/small.yaml")
    run_p.add_argument("--no-figures", action="store_true", help="skip figure generation")
    sweep_p = sub.add_parser("sweep", help="GRU patch-size sweep: accuracy vs inference speed")
    _common(sweep_p, "configs/small.yaml")
    sweep_p.add_argument("--patches", type=int, nargs="+", default=[5, 10, 20, 50],
                         help="samples per GRU step to compare (must divide the trace length)")
    args = parser.parse_args(argv)

    cfg, out, figdir = _load(args)
    if args.command == "run":
        from .pipeline import run

        run(cfg, out, figdir, make_figures=not args.no_figures)
    else:
        from .sweep import run_patch_sweep

        run_patch_sweep(cfg, out, args.patches, figdir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
