# MILP Misc Implementations

Thirty-eight optimisation cases from Gurobi's [modeling-examples](https://github.com/Gurobi/modeling-examples)
notebooks, rewritten in Rust on open-source solvers: [HiGHS](https://github.com/ERGO-Code/HiGHS) 1.12.0 for
linear and mixed-integer linear models and convex QPs, and [SCIP](https://scipopt.org) 10.0.2 for models with
quadratic terms, including nonconvex ones, which it solves to global optimality. The covering, colouring and
recommendation cases are modelled on hypergraphs, and any case's constraint matrix can be split into blocks
with the hypergraph partitioner [Mt-KaHyPar](https://github.com/kahypar/mt-kahypar).

Every case reads one data file, solves, checks its solution against every row, bound and integrality
condition of its model, and compares each objective with the one Gurobi recorded in the notebook.

## Cases

| Data file | Problem | Solver | Objective | Gurobi oracle |
| --- | --- | --- | --- | --- |
| `cell_tower` | Budgeted maximum coverage on a hypergraph | HiGHS | 7051 | met |
| `tic_tac_toe` | 3D tic-tac-toe: fewest monochromatic lines | HiGHS | 4 | met |
| `playlist` | Hypergraph ranking, then a playlist MILP | HiGHS | 6.95612, 6.37597 | none (see below) |
| `workforce` | Shift assignment, two ranked objectives | HiGHS | 6, 1 | met |
| `resource_assignment` | Resource assignment with a budget | HiGHS | 52 | met |
| `factory_planning_1`, `_2` | Williams: factory planning, maintenance | HiGHS | 93715.2; 108855 | met |
| `food_manufacture_1`, `_2` | Williams: blending, recipe logic | HiGHS | 107843; 100279 | met |
| `farm_planning` | Williams: five-year dairy farm | HiGHS | 121719 | met |
| `economic_planning` | Williams: dynamic Leontief economy | HiGHS | 0, 1902.22 | met |
| `manpower_planning` | Williams: hiring, training, redundancy | HiGHS | 841.797 | met |
| `mining` | Williams: mine scheduling | HiGHS | 1.46862e8 | met |
| `refinery` | Williams: refinery optimisation | HiGHS | 211365 | met |
| `opencast_mining` | Williams: maximum-weight closure | HiGHS | 17500 | met |
| `curve_fitting` | L1 and L∞ line fits | HiGHS | 11.4662, 1.725 | met |
| `logical_design` | Williams: fewest NOR gates for XOR | HiGHS | 5 | met |
| `market_sharing` | Williams: goal programming | HiGHS | 7.6 | met |
| `constraint_optimization` | Williams: simplest equivalent knapsack row | HiGHS | 25 | met |
| `efficiency_analysis` | Data envelopment analysis, 28 units | HiGHS | 28 LPs | met (printed to 3 decimals) |
| `protein_folding` | Williams: hydrophobic contacts | HiGHS | 10 | met |
| `electrical_power_1`, `_2` | Williams: unit commitment, hydro | HiGHS | 1.00254e6; 1.00063e6 | met |
| `facility_location` | Uncapacitated facility location | HiGHS | 4.72371 | met |
| `covid19_facility_1`, `_2` | Temporary facility location, two scenarios | HiGHS | 1.52165e6; 6.70045e10 | met |
| `supply_network_1`, `_2` | Williams: distribution, depot location | HiGHS | 198500; 174000 | met |
| `offshore_wind` | Fixed-charge cable network | HiGHS | 17 | met |
| `tsp` | Travelling salesman, 48 US capitals | HiGHS | 175.449 | none recorded |
| `cutting_stock` | Column generation, then an integer master | HiGHS | LP 72.9167, 73 rolls | none recorded |
| `portfolio` | Markowitz minimum risk and frontier | HiGHS QP | 26 QPs | met (rtol 5e-5) |
| `pooling` | Pooling, P and Q bilinear forms | SCIP | 439183 (both) | met |
| `agricultural_pricing` | Williams: pricing, bilinear revenue | SCIP | 2.06641 | met |
| `avocado_pricing` | Price and allocation, bilinear revenue | SCIP | 42.5083 | met |
| `decentralization` | Williams: binary quadratic placement | SCIP | 14.9 | met |
| `protein_comparison` | Contact map overlap, binary quadratic | SCIP | 5 | met |
| `best_subset` | L0 regression for k = 1..13 (MIQP) | SCIP | 13 MIQPs | met (exhaustive enumeration) |

Measured on 2026-11-03 with `mmi run data/*.json`: all 38 pass, in 70 s in total; best subset (38 s) and pooling
(18 s) take the longest, and every other case takes under 4 s. A Gurobi oracle is the objective in the notebook's
log, or the interval [incumbent, bound] when Gurobi stopped within its MIP gap. Points of note:

- **Pooling:** SCIP proves the global optimum of Haverly's P-formulation, which Gurobi 9.1 left at 411531, a 103% gap,
  after 300 s.
- **Agricultural pricing:** SCIP's 2.0664076 lies inside Gurobi's interval [2.0663983, 2.0664931].
- **Cutting stock:** the 73-roll plan equals the rounded-up LP bound, so it is optimal.
- **Best subset:** the notebook records no objective, so `tools/import_gurobi.py` enumerates all 8191 feature subsets.
- **Playlist:** its scores come from hypergraph ranking rather than the notebook's LightFM model, so its objectives
  cannot be compared with Gurobi's.

[docs/DECISIONS.md](docs/DECISIONS.md) lists the notebooks still to port and the reasons for every reformulation.

## Layout

| Path | Contents |
| --- | --- |
| `crates/milp` | `Model`: sparse rows with optional quadratic terms; `solve` (HiGHS when linear, SCIP otherwise), `solve_on`, `lex`; `viol`, the solution check |
| `crates/hyper` | `Hg`: weighted hypergraph in DeepHypergraph's convention; Zhou's operator Θ and ranking; hMETIS export |
| `crates/models` | One module per formulation, and the `mmi` command |
| `data/` | One JSON file per case: source notebook, Gurobi oracle, data |
| `tools/import_gurobi.py` | Rebuilds `data/` from a modeling-examples checkout |
| `tools/blocks.py` | Partitions a case's constraint matrix with Mt-KaHyPar |

## Running

Requires Rust 1.88 or later, CMake and a C++ compiler (for HiGHS). The first build downloads the prebuilt SCIP
10.0.2 that the `scip-sys` crate provides (46 MB on Linux, 67 MB on Windows). Python 3 with `mtkahypar` runs `blocks.py`.

```
git clone --recursive <repository>
cargo build --release
target/release/mmi run data/*.json         # solve, check, compare with Gurobi; exit 1 on any failure
target/release/mmi hgr data/tsp.json       # the constraint matrix as an hMETIS hypergraph
python -m pip install -r requirements.txt
python tools/blocks.py data/food_manufacture_2.json -k 6
cargo test --release                       # includes a check that HiGHS and SCIP agree on every linear case
```

To change a case, edit its data file. Each module in `crates/models/src` states its formulation and
source at the top. Set `MILP_LOG=1` to see the solver logs.

## Limits

- HiGHS has no lazy-constraint callback, so the TSP re-solves the whole MIP after each round of subtour rows.
- SCIP needs finite bounds on the variables in nonconvex terms. On a convex objective whose variables are free, it
  did not terminate (measured), so every case bounds them.
- SCIP satisfies a quadratic objective's epigraph row only to 1e-6 absolute, so `portfolio`, whose risks are about
  1e-5, uses HiGHS's QP solver.
- The cutting stock integer plan uses only the columns generated at the root (price-and-branch, as in the notebook).
- On Windows, the Mt-KaHyPar 1.7 wheel crashes as its process exits, so `blocks.py` partitions in a child process.

## Licence

Apache-2.0. The data and problem statements come from Gurobi's modeling-examples (Apache-2.0); see `NOTICE`.
