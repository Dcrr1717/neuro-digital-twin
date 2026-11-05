"""Head-to-head inference timing of two (or more) saved twins, in one process.

Speed measurements drift by tens of percent between runs on the same machine,
so twins are compared here interleaved, on identical inputs, best of several
repeats. Example:

    python scripts/benchmark_twins.py results/full/twin old_twin/ --labels "1 ms" "0.5 ms"
"""

from __future__ import annotations

import argparse
import glob
import json
import time
from collections.abc import Callable
from functools import partial

import numpy as np
import torch

from neurotwin.data import Dataset
from neurotwin.evaluate import tile_inputs
from neurotwin.models import predict_mv
from neurotwin.pipeline import load_twin


def best_time(fn: Callable[[], object], repeats: int) -> float:
    fn()  # warm-up
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("twins", nargs="+", help="twin directories (written by neurotwin run)")
    p.add_argument("--labels", nargs="+", default=None)
    p.add_argument("--data", default="results/full/data", help="cached datasets of a run")
    p.add_argument("--batch", type=int, nargs="+", default=[1, 1024])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--rounds", type=int, default=3, help="interleaved rounds (best is kept)")
    args = p.parse_args()
    labels = args.labels or args.twins
    test = Dataset.load(glob.glob(f"{args.data}/test_id-*.npz")[0])
    twins = [load_twin(t) for t in args.twins]
    res: dict[str, dict[str, float]] = {lab: {} for lab in labels}
    for _ in range(args.rounds):
        for lab, twin in zip(labels, twins, strict=True):
            for bs in args.batch:
                th, cur = tile_inputs(test, bs)
                one = best_time(partial(predict_mv, twin.members[0], twin.normalizer, th, cur, bs),
                                args.repeats)
                ens = best_time(partial(twin.predict, th, cur, bs), max(1, args.repeats // 2))
                for key, t in ((f"one_b{bs}", one), (f"ensemble_b{bs}", ens)):
                    res[lab][key] = min(res[lab].get(key, float("inf")), 1e3 * t / bs)
    print(json.dumps({"threads": torch.get_num_threads(),
                      "ms_per_trace": {k: {m: round(v, 4) for m, v in d.items()}
                                       for k, d in res.items()}}, indent=2))
    base = labels[-1]
    for lab in labels[:-1]:
        ratios = {k: round(res[base][k] / v, 2) for k, v in res[lab].items()}
        print(f"speed-up of {lab!r} over {base!r}: {ratios}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
