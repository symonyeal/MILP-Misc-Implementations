"""Write data/*.json from a checkout of Gurobi/modeling-examples (Apache-2.0).

Each notebook's parameter cells run against a stub `gurobipy` (multidict, tuplelist) up to the
first cell that builds a model, so the instance data are taken from the source, not retyped.
The Gurobi objective each notebook's saved output records becomes the case's oracle.

Usage: python tools/import_gurobi.py <modeling-examples checkout> [out dir, default data]
"""

import csv
import json
import os
import sys
import types
from collections import Counter
from pathlib import Path

SRC = "https://github.com/Gurobi/modeling-examples"


def stub():
    def multidict(d):
        k = list(d)
        v = list(d.values())
        if v and isinstance(v[0], list):
            return [k] + [{a: x[i] for a, x in zip(k, v)} for i in range(len(v[0]))]
        return [k, dict(d)]

    # Any object in sys.modules satisfies `import gurobipy as gp`.
    sys.modules["gurobipy"] = types.SimpleNamespace(multidict=multidict, tuplelist=list, GRB=types.SimpleNamespace())  # type: ignore[assignment]


def params(nb):
    """Globals after the code cells that precede the first model, run in the notebook's folder."""
    g = {}
    here = os.getcwd()
    os.chdir(nb.parent)
    try:
        for c in json.loads(nb.read_text(encoding="utf-8"))["cells"]:
            s = "".join(c["source"])
            if c["cell_type"] != "code" or s.lstrip().startswith("%"):
                continue
            if "Model(" in s:
                break
            exec(s, g)  # noqa: S102 -- the notebook's own parameter cells are the data source
    finally:
        os.chdir(here)
    return g


def oracle(nb):
    """Recorded Gurobi objectives in log order: a number for an "Optimal objective" line, the pair
    [incumbent, best bound] for a "Best objective a, best bound b" line (equal when proven)."""
    out = []
    for c in json.loads(nb.read_text(encoding="utf-8"))["cells"]:
        for o in c.get("outputs", []):
            for ln in "".join(o.get("text", "")).splitlines():
                w = ln.replace(",", " ").split()
                if ln.startswith("Optimal objective"):
                    out.append(float(w[2]))
                elif ln.startswith("Best objective"):
                    a, b = float(w[2]), float(w[5])
                    out.append(a if a == b else [a, b])
    return out


def case(name, nb, data, orc, note=None, rtol=None):
    d = {"case": name, "source": f"{SRC}/blob/master/{nb}", "oracle": orc}
    if rtol:
        d["rtol"] = rtol
    if note:
        d["note"] = note
    d["data"] = data
    return d


def cell_tower(r):
    nb = "cell_tower_coverage/cell_tower.ipynb"
    g = params(r / nb)
    towers = [{"e": sorted(g["coverage"][t]), "c": g["cost"][t]} for t in g["sites"]]
    data = {"p": [g["population"][i] for i in g["regions"]], "towers": towers, "budget": g["budget"]}
    return case("cover", nb, data, oracle(r / nb))


def tic_tac_toe(r):
    nb = "3d_tic_tac_toe/3d_tic_tac_toe.ipynb"
    return case("colour", nb, {"size": 3, "x": 14}, oracle(r / nb))


def workforce(r):
    nb = "workforce/workforce_scheduling.ipynb"
    g = params(r / nb)
    data = {
        "shifts": [{"name": s, "need": g["shiftRequirements"][s]} for s in g["shifts"]],
        "workers": g["workers"],
        "avail": [list(a) for a in g["availability"]],
        "reltol": 0.2,
    }
    return case("workforce", nb, data, oracle(r / nb))


def factory(r, k):
    nb = f"factory_planning/factory_planning_{k}.ipynb"
    g = params(r / nb)
    T, P, M = g["months"], g["products"], g["machines"]
    data = {
        "months": T,
        "products": [{"name": p, "profit": g["profit"][p]} for p in P],
        "machines": [{"name": m, "installed": g["installed"][m]} for m in M],
        "time": [[g["time_req"][m].get(p, 0) for p in P] for m in M],
        "sales": [[g["max_sales"][t, p] for p in P] for t in T],
        "hold": g["holding_cost"],
        "smax": g["max_inventory"],
        "target": g["store_target"],
        "hours": g["hours_per_month"],
    }
    if "down" in g:
        data["down"] = [[g["down"].get((t, m), 0) for m in M] for t in T]
    else:
        data["maint"] = [g["down_req"][m] for m in M]
    return case("factory", nb, data, oracle(r / nb))


def food(r, k):
    nb = f"food_manufacturing/food_manufacture_{k}.ipynb"
    g = params(r / nb)
    T, O = g["months"], g["oils"]
    data = {
        "months": T,
        "oils": [{"name": o, "veg": "VEG" in o, "hard": g["hardness"][o]} for o in O],
        "cost": [[g["cost"][t, o] for o in O] for t in T],
        "price": g["price"],
        "s0": g["init_store"],
        "target": g.get("target_store", g["init_store"]),
        "cap_veg": g["veg_cap"],
        "cap_oil": g["oil_cap"],
        "hmin": g["min_hardness"],
        "hmax": g["max_hardness"],
        "hold": g["holding_cost"],
    }
    if "max_ingredients" in g:
        # Notebook constraints 7-9: at most 3 oils, each used at >= 20 tons, VEG1 or VEG2 => OIL3.
        data["recipe"] = {
            "k": g["max_ingredients"],
            "min": g["min_consume"],
            "implies": [["VEG1", "OIL3"], ["VEG2", "OIL3"]],
        }
    return case("food", nb, data, oracle(r / nb))


def opencast(r):
    nb = "opencast_mining/opencast_mining.ipynb"
    g = params(r / nb)
    data = {"blocks": [{"name": b, "p": g["profit"][b]} for b in g["blocks"]], "arcs": [list(a) for a in g["arcs"]]}
    return case("opencast", nb, data, oracle(r / nb))


def curve_fitting(r):
    nb = "curve_fitting/curve_fitting.ipynb"
    g = params(r / nb)
    return case("fit", nb, {"pts": [[g["x"][i], g["y"][i]] for i in g["observations"]]}, oracle(r / nb))


def facility(r):
    nb = "facility_location/facility_location.ipynb"
    g = params(r / nb)
    sites = [{"at": list(a), "f": f} for a, f in zip(g["facilities"], g["setup_cost"])]
    data = {"customers": [list(c) for c in g["customers"]], "sites": sites, "rate": g["cost_per_mile"]}
    return case("facility", nb, data, oracle(r / nb))


def network(r, k):
    nb = f"supply_network_design/supply_network_design_{k}.ipynb"
    g = params(r / nb)
    oc = g.get("opencost")
    depots = [
        {"name": d, "cap": c, "open": oc[d] if oc else None, "must": d in ("Birmingham", "London") if oc else False}
        for d, c in g["through"].items()
    ]
    data = {
        "factories": [{"name": f, "supply": s} for f, s in g["supply"].items()],
        "depots": depots,
        "customers": [{"name": c, "demand": q} for c, q in g["demand"].items()],
        "arcs": [[a, b, g["cost"][a, b]] for a, b in g["arcs"]],
    }
    note = None
    if oc:
        # Notebook 2: Birmingham may expand by 20000 for 3000; at most 4 depots; the objective
        # constant credits the open costs of the existing Newcastle and Exeter depots.
        data |= {
            "expand": [{"depot": "Birmingham", "cap": 20000, "cost": 3000}],
            "max_open": 4,
            "c0": -(oc["Newcastle"] + oc["Exeter"]),
        }
        note = "objective constant -(open cost of Newcastle + Exeter), as the notebook's objcon"
    return case("network", nb, data, oracle(r / nb), note)


def tsp(r):
    nb = "traveling_salesman/tsp.ipynb"
    j = json.loads((r / "traveling_salesman/capitals.json").read_text(encoding="utf-8"))
    cities = [
        {"name": v["capital"], "at": [float(v["lat"]), float(v["long"])]} for s, v in j.items() if s not in ("AK", "HI")
    ]
    return case("tsp", nb, {"cities": cities}, [], "the notebook records no objective")


def cutting_stock(r):
    nb = "colgen-cutting_stock/colgen-cutting_stock.ipynb"
    # The orders cell follows the class definitions; its literal is copied here.
    orders = {
        0: [1380, 22],
        1: [1520, 25],
        2: [1560, 12],
        3: [1710, 14],
        4: [1820, 18],
        5: [1880, 18],
        6: [1930, 20],
        7: [2000, 10],
        8: [2050, 12],
        9: [2100, 14],
        10: [2140, 16],
        11: [2150, 18],
        12: [2200, 20],
    }
    data = {"L": 5600, "pieces": [{"l": l, "d": d} for l, d in orders.values()]}
    return case("cutstock", nb, data, [], "the notebook records no objective")


def playlist(r):
    nb = "music_recommendation/music_recommendation.ipynb"
    with open(r / "music_recommendation/artist_data.csv", encoding="utf-8", newline="") as f:
        a = sorted(csv.DictReader(f), key=lambda x: int(x["artist_id"]))
    assert [int(x["artist_id"]) for x in a] == list(range(len(a)))
    with open(r / "music_recommendation/user_playlist_data.csv", encoding="utf-8", newline="") as f:
        n = Counter((int(x["user_id"]), int(x["artist_id"])) for x in csv.DictReader(f))
    data = {
        "user": 0,
        "alpha": 0.9,
        "top": 1000,
        "n": 30,
        "pmax": 500000,
        "cmin": 20,
        "artists": [{"name": x["artist"], "country": x["country"], "listeners": float(x["listeners"])} for x in a],
        "plays": [[u, v, k] for (u, v), k in sorted(n.items())],
    }
    note = (
        "preference scores are Zhou et al.'s hypergraph ranking, not the notebook's LightFM factorisation, "
        "so the notebook's objectives are not oracles; plays aggregates the 42 MB playlist file to (user, artist, tracks)"
    )
    return case("playlist", nb, data, [], note)


def pooling(r):
    nb = "pooling/std_pooling.ipynb"
    g = params(r / nb)
    at = sorted(g["attrs"])
    data = {
        "attrs": at,
        "sources": [
            {"name": i, "cost": g["cost"][i], "supply": g["supply"][i], "q": [g["content"][i][a] for a in at]}
            for i in g["sources"]
        ],
        "targets": [
            {"name": k, "price": g["price"][k], "demand": g["demand"][k], "lo": g["min_tol"][k], "hi": g["max_tol"][k]}
            for k in g["targets"]
        ],
        "pools": [{"name": j, "cap": g["cap"][j]} for j in g["pools"]],
        "direct": [[i, k, 750.0 if (i, k) in {("s1", "t2"), ("s3", "t1")} else None] for i, k in sorted(g["s2t"])],
    }
    # The notebook's first (P) run stopped at its 300 s limit at 411530.7 with a 103% gap; the
    # second (Q) run is proven optimal, and the two formulations share their optimum.
    q = oracle(r / nb)[1]
    note = "the notebook's P-formulation run hit its time limit at 411530.70 (gap 103%); both oracles are the Q optimum"
    return case("pooling", nb, data, [q, q], note)


def agricultural(r):
    nb = "agricultural_pricing/agricultural_pricing.ipynb"
    g = params(r / nb)
    D, C = g["dairy"], g["components"]
    data = {
        "products": [{"name": d, "q0": g["consumption"][d], "p0": g["price"][d], "e": g["elasticity"][d]} for d in D],
        "components": [{"name": c, "cap": g["capacity"][c], "share": [g["qtyper"][c, d] for d in D]} for c in C],
        # Cross elasticities between the two cheeses: e12 of cheese 1 to cheese 2's price, e21 the reverse.
        "cross": [["cheese1", "cheese2", g["elasticity12"]], ["cheese2", "cheese1", g["elasticity21"]]],
        "index": g["priceIndex"],
    }
    return case("agriprice", nb, data, oracle(r / nb))


def portfolio(r):
    nb = "portfolio_selection_optimization/portfolio_selection_optimization.ipynb"
    g = params(r / nb)
    data = {
        "stocks": g["stocks"],
        "mu": [float(v) for v in g["delta"]],
        "sigma": [[float(v) for v in row] for row in g["sigma"]],
        "frontier": 25,
    }
    note = (
        "mu and sigma are the notebook's np.mean and np.cov of daily relative price changes in dummy_yfinance_2y.csv; "
        "the oracles are Gurobi barrier values, two of which HiGHS's exactly feasible solutions beat by 1.2e-9 (4.6e-5 relative)"
    )
    return case("portfolio", nb, data, oracle(r / nb), note, rtol=5e-5)


def dump(v, pad=""):
    """JSON with one key per line at the top two levels and one record per line in lists."""
    if isinstance(v, dict) and len(pad) < 2:
        body = ",\n".join(f"{pad} {json.dumps(k)}: {dump(x, pad + ' ')}" for k, x in v.items())
        return "{\n" + body + "\n" + pad + "}"
    if isinstance(v, list) and v and isinstance(v[0], (list, dict)):
        return "[\n" + ",\n".join(f"{pad}  {json.dumps(x, ensure_ascii=False)}" for x in v) + "\n" + pad + " ]"
    return json.dumps(v, ensure_ascii=False)


def main():
    r, out = Path(sys.argv[1]), Path(sys.argv[2] if len(sys.argv) > 2 else "data")
    stub()
    out.mkdir(exist_ok=True)
    cases = {
        "cell_tower": cell_tower(r),
        "tic_tac_toe": tic_tac_toe(r),
        "workforce": workforce(r),
        "factory_planning_1": factory(r, 1),
        "factory_planning_2": factory(r, 2),
        "food_manufacture_1": food(r, 1),
        "food_manufacture_2": food(r, 2),
        "opencast_mining": opencast(r),
        "curve_fitting": curve_fitting(r),
        "facility_location": facility(r),
        "supply_network_1": network(r, 1),
        "supply_network_2": network(r, 2),
        "tsp": tsp(r),
        "cutting_stock": cutting_stock(r),
        "playlist": playlist(r),
        "pooling": pooling(r),
        "agricultural_pricing": agricultural(r),
        "portfolio": portfolio(r),
    }
    for k, d in cases.items():
        (out / f"{k}.json").write_text(dump(d) + "\n", encoding="utf-8", newline="\n")
        print(k, d["oracle"])


if __name__ == "__main__":
    main()
