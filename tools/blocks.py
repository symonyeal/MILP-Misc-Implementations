"""Find block structure in a case's constraint matrix by hypergraph partitioning.

The row-net hypergraph of A (`mmi hgr`: columns are vertices, each row a hyperedge) is split into
k blocks of near-equal column count by Mt-KaHyPar (github.com/kahypar/mt-kahypar),
minimising the cut-net metric. A row inside one block belongs to that block's subproblem; a cut
row links blocks. Few linking rows mean the matrix is close to bordered block-diagonal, the form a
Dantzig-Wolfe decomposition needs (Bergner et al., Math. Programming 149, 2015, detect it the
same way with hMETIS).

Usage: python tools/blocks.py CASE.json [-k K] [--eps EPS] [--seed S]
The mmi binary is $MMI, else target/release/mmi under $CARGO_TARGET_DIR or the repository.
"""

import argparse
import multiprocessing as mp
import os
import subprocess
from pathlib import Path

import mtkahypar


def mmi():
    if "MMI" in os.environ:
        return os.environ["MMI"]
    t = Path(os.environ.get("CARGO_TARGET_DIR", Path(__file__).resolve().parents[1] / "target"))
    exe = t / "release" / ("mmi.exe" if os.name == "nt" else "mmi")
    if not exe.exists():
        raise SystemExit(f"{exe} not found: run cargo build --release or set MMI")
    return str(exe)


def hgr(case):
    """(n, nets) from `mmi hgr`, vertices 0-based."""
    s = subprocess.run([mmi(), "hgr", case], check=True, capture_output=True, text=True).stdout.split("\n")
    m, n = map(int, s[0].split())
    return n, [[int(v) - 1 for v in ln.split()] for ln in s[1 : m + 1]]


def main():
    a = argparse.ArgumentParser(description="Partition a case's row-net hypergraph with Mt-KaHyPar.")
    a.add_argument("case")
    a.add_argument("-k", type=int, default=2)
    a.add_argument(
        "--eps", type=float, default=0.03, help="imbalance: each block holds at most (1 + eps) n / k columns"
    )
    a.add_argument("--seed", type=int, default=0)
    o = a.parse_args()
    n, nets = hgr(o.case)
    nets = [e for e in nets if e]
    # Mt-KaHyPar 1.7's Windows wheel fails fast (0xC0000409) as the process exits after any
    # partition (measured 2026-11-03), so a child partitions and the parent reads its answer.
    q = mp.Queue()
    c = mp.Process(target=part, args=(n, nets, o.k, o.eps, o.seed, q))
    c.start()
    b = q.get(timeout=600)
    c.join()
    lam = [len({b[v] for v in e}) for e in nets]
    cut, km1 = sum(x > 1 for x in lam), sum(x - 1 for x in lam)
    inner = [sum(x == 1 and b[e[0]] == k for x, e in zip(lam, nets)) for k in range(o.k)]
    print(f"{o.case}: {n} columns, {len(nets)} rows, k = {o.k}, eps = {o.eps}, seed = {o.seed}")
    print(
        f"linking rows {cut} ({100 * cut / len(nets):.1f}%), km1 {km1}, largest block {max(map(b.count, range(o.k)))}"
    )
    for k in range(o.k):
        print(f"  block {k}: {b.count(k)} columns, {inner[k]} rows")


def part(n, nets, k, eps, seed, q):
    """Block of each vertex, minimising the cut-net metric, onto queue q."""
    mt = mtkahypar.initialize(1)
    mtkahypar.set_seed(seed)
    ctx = mt.context_from_preset(mtkahypar.PresetType.DEFAULT)
    ctx.set_partitioning_parameters(k, eps, mtkahypar.Objective.CUT)
    ctx.logging = False
    p = mt.create_hypergraph(ctx, n, len(nets), nets).partition(ctx)
    q.put([p.block_id(v) for v in range(n)])


if __name__ == "__main__":
    main()
