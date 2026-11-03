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


def avocado(r):
    """The notebook's demand model, refit: OLS of units_sold (millions) on price, year_index,
    peak (months 2-7) and region with treatment coding (base Great_Lakes), over conventional
    avocados in its eight regions, both CSV files. statsmodels' `smf.ols` fits the same model."""
    import numpy as np

    nb = "price_optimization/price_optimization.ipynb"
    regions = [
        "Great_Lakes",
        "Midsouth",
        "Northeast",
        "Northern_New_England",
        "SouthCentral",
        "Southeast",
        "West",
        "Plains",
    ]
    rows = []
    for f in ("HABdata_2019_2022.csv", "kaggledata_till2018.csv"):
        with open(r / "price_optimization" / f, encoding="utf-8", newline="") as h:
            for x in csv.DictReader(h):
                if x["type"] != "Conventional" or x["region"] not in regions:
                    continue
                mo, _, yy = x["date"].split()[0].split("/")
                rows.append(
                    (
                        float(x["units_sold"]) / 1e6,
                        float(x["price"]),
                        int(yy) + 2000 - 2015,
                        2 <= int(mo) <= 7,
                        x["region"],
                    )
                )
    base = sorted(regions)
    X = np.array([[1, p, yi, pk] + [float(g == b) for b in base[1:]] for _, p, yi, pk, g in rows])
    y = np.array([u for u, *_ in rows])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    lo = {g: min(u for u, *_, h in rows if h == g) for g in regions}
    hi = {g: max(u for u, *_, h in rows if h == g) for g in regions}
    shift = dict(zip(base, [0.0] + list(beta[4:])))
    transport = {
        "Great_Lakes": 0.3,
        "Midsouth": 0.1,
        "Northeast": 0.4,
        "Northern_New_England": 0.5,
        "SouthCentral": 0.3,
        "Southeast": 0.2,
        "West": 0.2,
        "Plains": 0.2,
    }
    data = {
        "supply": 30,
        "waste": 0.1,
        "pmax": 2,
        "year": 2022,
        "peak": 1,
        "fit": {"intercept": beta[0], "price": beta[1], "year": beta[2], "peak": beta[3]},
        "regions": [
            {"name": g, "shift": shift[g], "transport": transport[g], "lo": lo[g], "hi": hi[g]} for g in regions
        ],
    }
    note = f"demand coefficients refit by OLS over {len(rows)} weekly rows (numpy lstsq), as the notebook's smf.ols on the full data"
    return case("avocado", nb, data, oracle(r / nb)[:1], note)


def best_subset(r):
    """Boston housing, response MV on the 13 other columns. The notebook records no objective, so
    the oracle is computed here: for each k, the least residual sum of squares over every subset of
    k features (with an intercept), by numpy least squares over all 2^13 - 1 subsets."""
    from itertools import combinations

    import numpy as np

    nb = "linear_regression/l0_regression.ipynb"
    with open(r / "linear_regression/boston.csv", encoding="utf-8", newline="") as h:
        rows = list(csv.reader(h))
    names, X = rows[0], np.array(rows[1:], dtype=float)
    A, y = X[:, :-1], X[:, -1]
    best = []
    for k in range(1, A.shape[1] + 1):
        rss = []
        for S in combinations(range(A.shape[1]), k):
            Z = np.column_stack([np.ones(len(y)), A[:, S]])
            e = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
            rss.append(float(e @ e))
        best.append(min(rss))
    data = {"features": names[:-1], "response": names[-1], "x": A.tolist(), "y": y.tolist()}
    note = "oracle: least RSS for k = 1..13 features by exhaustive enumeration (numpy lstsq), not a Gurobi log"
    return case("subset", nb, data, best, note)


def economic(r):
    nb = "economic_planning/economic_planning.ipynb"
    g = params(r / nb)
    I = list(g["industries"])
    data = {
        "industries": I,
        "a": [[g["inout_prod"][i, j] for j in I] for i in I],
        "b": [[g["inout_cap"][i, j] for j in I] for i in I],
        "la": [g["labor_prod"][i] for i in I],
        "lb": [g["labor_extra_cap"][i] for i in I],
        "s0": [g["stock0"][i] for i in I],
        "k0": [g["capacity0"][i] for i in I],
        "d": [g["demand"][i] for i in I],
    }
    return case("economic", nb, data, oracle(r / nb))


def manpower(r):
    nb = "manpower_planning/manpower_planning.ipynb"
    g = params(r / nb)
    T, S = g["years"], g["skills"]
    data = {
        "skills": S,
        "w0": [g["curr_workforce"][s] for s in S],
        "need": [[g["demand"][t, s] for s in S] for t in T],
        "hire": [[g["max_hiring"][t, s] for s in S] for t in T],
        "rookie": [g["rookie_attrition"][s] for s in S],
        "veteran": [g["veteran_attrition"][s] for s in S],
        "demoted": g["demoted_attrition"],
        "over": g["max_overmanning"],
        "part": g["max_parttime"],
        "part_w": g["parttime_cap"],
        "train12": g["max_train_unskilled"],
        "train23": g["max_train_semiskilled"],
    }
    return case("manpower", nb, data, oracle(r / nb))


def farm(r):
    nb = "farm_planning/farm_planning.ipynb"
    g = params(r / nb)
    keys = [
        "sb_yield",
        "housing_cap",
        "gr_intake",
        "sb_intake",
        "hf_land",
        "land_cap",
        "hf_labor",
        "cow_labor",
        "gr_labor",
        "sb_labor",
        "labor_cap",
        "cow_decay",
        "hf_decay",
        "initial_hf",
        "initial_cows",
        "birthrate",
        "min_final_cows",
        "max_final_cows",
        "bl_price",
        "hf_price",
        "cow_price",
        "milk_price",
        "gr_price",
        "sb_price",
        "gr_cost",
        "sb_cost",
        "overtime_cost",
        "regular_time_cost",
        "hf_cost",
        "cow_cost",
        "gr_land_cost",
        "sb_land_cost",
        "installment",
    ]
    data = {k: g[k] for k in keys}
    data |= {
        "years": len(g["years"]),
        "ages": len(g["ages"]),
        "gr_area": [g["gr_area"][l] for l in g["lands"]],
        "gr_yield": [g["gr_yield"][l] for l in g["lands"]],
    }
    return case("farm", nb, data, oracle(r / nb))


def mining(r):
    nb = "mining/mining.ipynb"
    g = params(r / nb)
    M = g["mines"]
    data = {
        "mines": [{"royalty": g["royalties"][k], "cap": g["capacity"][k], "quality": g["quality"][k]} for k in M],
        "target": [g["target"][t] for t in g["years"]],
        "rate": 0.1,
        "max_open": g["max_mines"],
        "price": g["price"],
    }
    return case("mining", nb, data, oracle(r / nb))


def refinery(r):
    nb = "refinery/refinery.ipynb"
    g = params(r / nb)
    keys = [
        "buy_limit",
        "lbo_min",
        "lbo_max",
        "distill_cap",
        "reform_cap",
        "crack_cap",
        "end_product_profit",
        "blending_coefficients",
        "lube_oil_factor",
        "pmf_rmf_ratio",
        "octance_number_coefficients",
        "octance_number_fuel",
        "vapor_pressure_constants",
        "reforming_splitting_coefficients",
    ]
    data = {k: g[k] for k in keys}
    data["buy_limit"] = [g["buy_limit"][c] for c in g["crude_numbers"]]
    data["distillation"] = {k: list(v) for k, v in g["distillation_splitting_coefficients"].items()}
    data["cracking"] = [[a, b, v] for (a, b), v in g["cracking_splitting_coefficients"].items()]
    return case("refinery", nb, data, oracle(r / nb))


def logical(r):
    nb = "logical_design/logical_design.ipynb"
    g = params(r / nb)
    rows = ["1", "2", "3", "4"]
    data = {
        "gates": len(g["gates"]),
        "a": [g["valueA"]["1", k] for k in rows],
        "b": [g["valueB"]["1", k] for k in rows],
        # Gate 1's output is fixed by the notebook's bounds: 0, 1, 1, 0 (exclusive or).
        "out": [0, 1, 1, 0],
    }
    return case("logic", nb, data, oracle(r / nb))


def market(r):
    nb = "market_sharing/market_sharing.ipynb"
    g = params(r / nb)
    R = list(g["retailers"])
    goal = [
        ("delivery points", {k: g["deliveryPoints"][k] for k in R}, "deliveryPoints40", "deliveryPoints5"),
        ("spirit market", {k: g["spiritMarket"][k] for k in R}, "spiritMarket40", "spiritMarket5"),
        ("oil market 1", g["oilMarket1"], "oilMarket1_40", "oilMarket1_5"),
        ("oil market 2", g["oilMarket2"], "oilMarket2_40", "oilMarket2_5"),
        ("oil market 3", g["oilMarket3"], "oilMarket3_40", "oilMarket3_5"),
        ("group A", g["retailerA"], "retailerA40", "retailerA5"),
        ("group B", g["retailerB"], "retailerB40", "retailerB5"),
    ]
    data = {
        "retailers": len(R),
        "goals": [{"name": n, "w": [[k, v] for k, v in w.items()], "target": g[t], "tol": g[s]} for n, w, t, s in goal],
    }
    return case("market", nb, data, oracle(r / nb))


def decentral(r):
    nb = "decentralization_planning/decentralization_planning.ipynb"
    g = params(r / nb)
    data = {
        "departments": g["Deparments"],
        "cities": g["Cities"],
        "benefit": [[d, c, v] for (d, c), v in g["benefit"].items()],
        "comm": [[d, c, e, f, v] for (d, c, e, f), v in g["communicationCost"].items()],
        "max_per_city": 3,
    }
    return case("decentral", nb, data, oracle(r / nb))


def dea(r):
    """The garage data are a literal inside the notebook's solve_DEA function; the oracle is the
    printed efficiency of each garage, rounded to 3 decimals, as the interval it rounds from."""
    import ast
    import re

    nb = "efficiency_analysis/efficiency_analysis.ipynb"
    cells = json.loads((r / nb).read_text(encoding="utf-8"))["cells"]
    src = next("".join(c["source"]) for c in cells if "def solve_DEA" in "".join(c["source"]))
    call = next(
        n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.Call) and getattr(n.func, "attr", "") == "multidict"
    )
    units = ast.literal_eval(call.args[0])
    ins = ["staff", "showRoom", "Population1", "Population2", "alphaEnquiries", "betaEnquiries"]
    outs = ["alphaSales", "BetaSales", "profit"]
    text = "".join("".join(o.get("text", "")) for c in cells for o in c.get("outputs", []))
    score = {m[1]: float(m[2]) for m in re.finditer(r"performance value of DMU (\w+) is: ([0-9.]+)", text)}
    data = {
        "inputs": ins,
        "outputs": outs,
        "units": [{"name": k, "x": [v[0][i] for i in ins], "y": [v[1][o] for o in outs]} for k, v in units.items()],
    }
    orc = [[round(score[k] - 5e-4, 4), round(score[k] + 5e-4, 4)] for k in units]
    return case("dea", nb, data, orc, "oracle: the notebook's printed efficiencies (3 decimals) as rounding intervals")


def constraint_opt(r):
    nb = "constraint_optimization/constraint_optimization.ipynb"
    src = "".join(
        "".join(c["source"])
        for c in json.loads((r / nb).read_text(encoding="utf-8"))["cells"]
        if c["cell_type"] == "code"
    )
    import re

    def rows(kind):
        # Each row reads Σ b_i (i in the list) <= b0 or >= b0 + 1; the regex also catches the b0.
        found = re.finditer(kind + r"\w* = model\.addConstr\( (.*?) ,", src)
        return [[int(i) for i in re.findall(r"b\[(\d)\]", m[1])][:-1] for m in found]

    data = {"n": 9, "le": rows("C"), "ge": rows("R"), "obj": [[0, 1], [3, -1], [5, -1]]}
    return case("constraint", nb, data, oracle(r / nb))


def wind(r):
    nb = "offshore_wind_farming/offshore_wind_farming.ipynb"
    g = params(r / nb)
    data = {
        "supply": [g["vertices"][v] for v in sorted(g["vertices"])],
        "edges": [[a, b, g["cap"][a, b], g["flow_cost"][a, b], g["fixed_cost"][a, b]] for a, b in g["edges"]],
    }
    return case("wind", nb, data, oracle(r / nb))


def folding(r):
    nb = "protein_folding/protein_folding.ipynb"
    g = params(r / nb)
    return case("folding", nb, {"n": len(g["acids"]), "phobic": g["h_phobic"]}, oracle(r / nb))


def comparison(r):
    nb = "protein_comparison/protein_comparison.ipynb"
    g = params(r / nb)
    data = {"n1": len(g["nodes1"]), "e1": g["edges1"], "n2": len(g["nodes2"]), "e2": g["edges2"]}
    return case("comparison", nb, data, oracle(r / nb))


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
        "avocado_pricing": avocado(r),
        "best_subset": best_subset(r),
        "economic_planning": economic(r),
        "manpower_planning": manpower(r),
        "farm_planning": farm(r),
        "mining": mining(r),
        "refinery": refinery(r),
        "logical_design": logical(r),
        "market_sharing": market(r),
        "decentralization": decentral(r),
        "efficiency_analysis": dea(r),
        "constraint_optimization": constraint_opt(r),
        "offshore_wind": wind(r),
        "protein_folding": folding(r),
        "protein_comparison": comparison(r),
    }
    for k, d in cases.items():
        (out / f"{k}.json").write_text(dump(d) + "\n", encoding="utf-8", newline="\n")
        print(k, d["oracle"])


if __name__ == "__main__":
    main()
