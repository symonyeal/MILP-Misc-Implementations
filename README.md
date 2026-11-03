# MILP Misc Implementations

Fifteen optimisation cases from Gurobi's [modeling-examples](https://github.com/Gurobi/modeling-examples)
notebooks, rewritten in Rust on the open-source solver [HiGHS](https://github.com/ERGO-Code/HiGHS) 1.12.0.
The covering, colouring and recommendation cases are modelled on hypergraphs, and any case's
constraint matrix can be split into blocks with the hypergraph partitioner
[Mt-KaHyPar](https://github.com/kahypar/mt-kahypar).

Every case reads one data file, solves, checks its solution against every row, bound and integrality
condition of its model, and compares the objective with the one Gurobi recorded in the notebook.

## Cases

| Data file | Problem | Formulation | Gurobi | HiGHS |
| --- | --- | --- | --- | --- |
| `cell_tower` | Cell tower coverage | Budgeted maximum coverage on a hypergraph | 7051 | 7051 |
| `tic_tac_toe` | 3D tic-tac-toe | Fewest monochromatic hyperedges in a 2-colouring | 4 | 4 |
| `playlist` | Music recommendation | Hypergraph ranking, then a selection MILP | (not comparable) | 6.956, 6.376 |
| `workforce` | Workforce scheduling | Assignment with two ranked objectives | 6, 1 | 6, 1 |
| `factory_planning_1` | Factory planning I | Multi-period production LP | 93715.18 | 93715.18 |
| `factory_planning_2` | Factory planning II | Same, with maintenance scheduling (MILP) | 108855 | 108855 |
| `food_manufacture_1` | Food manufacture I | Multi-period blending LP | 107842.59 | 107842.59 |
| `food_manufacture_2` | Food manufacture II | Same, with recipe logic (MILP) | 100278.70 | 100278.70 |
| `opencast_mining` | Opencast mining | Maximum-weight closure LP | 17500 | 17500 |
| `curve_fitting` | Curve fitting | L1 and L∞ line fits as LPs | 11.46625, 1.725 | 11.46625, 1.725 |
| `facility_location` | Facility location | Uncapacitated facility location | 4.723713 | 4.723713 |
| `supply_network_1` | Supply network design 1 | Transshipment LP | 198500 | 198500 |
| `supply_network_2` | Supply network design 2 | Depot opening and expansion (MILP) | 174000 | 174000 |
| `tsp` | Travelling salesman, 48 US capitals | Subtour rows added round by round | (not recorded) | 175.449037 |
| `cutting_stock` | Cutting stock | Column generation, then an integer master | (not recorded) | LP 72.9167, 73 rolls |

The HiGHS column was measured on 2026-11-03 with `mmi run data/*.json`. Each case solves in under one second.
The 73-roll cutting plan equals the rounded-up LP bound, so it is optimal. The playlist scores come from
hypergraph ranking rather than the notebook's LightFM model, so its objectives cannot be compared with
Gurobi's. [docs/DECISIONS.md](docs/DECISIONS.md) lists the notebooks that were left out and why.

## Layout

| Path | Contents |
| --- | --- |
| `crates/milp` | `Model`: sparse rows, bounds, integrality; `solve` and `lex` on HiGHS (vendored as a git submodule); `viol`, the solution check |
| `crates/hyper` | `Hg`: weighted hypergraph in DeepHypergraph's convention; Zhou's operator Θ and ranking; hMETIS export |
| `crates/models` | One module per formulation, and the `mmi` command |
| `data/` | One JSON file per case: source notebook, recorded Gurobi objectives, data |
| `tools/import_gurobi.py` | Rebuilds `data/` from a modeling-examples checkout |
| `tools/blocks.py` | Partitions a case's constraint matrix with Mt-KaHyPar |

## Running

Requires Rust 1.88 or later, CMake and a C++ compiler (for HiGHS), and Python 3 with `mtkahypar` for `blocks.py`.

```
git clone --recursive <repository>
cargo build --release
target/release/mmi run data/*.json         # solve, check, compare with Gurobi; exit 1 on any failure
target/release/mmi hgr data/tsp.json       # the constraint matrix as an hMETIS hypergraph
python -m pip install -r requirements.txt
python tools/blocks.py data/food_manufacture_2.json -k 6
cargo test --release
```

To change a case, edit its data file. Each module in `crates/models/src` states its formulation and
source at the top.

## Limits

- HiGHS has no lazy-constraint callback. The TSP therefore re-solves the whole MIP after each round of
  subtour rows (7 rounds for the 48 capitals) instead of adding rows inside one branch-and-bound.
- HiGHS solves continuous QPs but not MIQPs, so the notebooks that need a quadratic integer model were
  left out.
- The cutting stock integer plan uses only the columns generated at the root (price-and-branch, as in
  the notebook). It is proven optimal only when it reaches the rounded-up LP bound, as it does here.
- On Windows, the Mt-KaHyPar 1.7 wheel crashes as its process exits, so `blocks.py` runs the
  partitioner in a child process.

## Licence

Apache-2.0. The data and problem statements come from Gurobi's modeling-examples (Apache-2.0); see `NOTICE`.
