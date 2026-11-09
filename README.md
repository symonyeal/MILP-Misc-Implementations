# MILP Misc Implementations

## Problem

- Thirty-eight optimisation cases from Gurobi's [modeling-examples](https://github.com/Gurobi/modeling-examples)
  notebooks, rewritten in Rust on open-source solvers.
- The covering, colouring and recommendation cases are modelled on hypergraphs.
- Any case's constraint matrix can be split into blocks with the hypergraph partitioner
  [Mt-KaHyPar](https://github.com/kahypar/mt-kahypar).

## Methods

- **Solvers**: [HiGHS](https://github.com/ERGO-Code/HiGHS) 1.12.0 for linear and mixed-integer linear models
  and convex QPs; [SCIP](https://scipopt.org) 10.0.2 for models with quadratic terms, including nonconvex
  ones, which it solves to global optimality.
- **Case procedure**: each case reads one data file, solves, checks its solution against every row, bound
  and integrality condition of its model, and compares each objective with the one Gurobi recorded in the
  notebook.
- **Gurobi oracle**: the objective in the notebook's log, or the interval [incumbent, bound] when Gurobi
  stopped within its MIP gap. Exceptions:
  - `tsp` and `cutting_stock` have no recorded objective.
  - `playlist` takes its scores from hypergraph ranking rather than the notebook's LightFM model, so its
    objectives cannot be compared with Gurobi's.
  - `best_subset`: the notebook records no objective, so `tools/import_gurobi.py` enumerates all 8191
    feature subsets.
  - `efficiency_analysis`: the oracle is printed to 3 decimals.
  - `portfolio` compares with rtol 5e-5.
- **Reformulations and unported notebooks**: [docs/DECISIONS.md](docs/DECISIONS.md) lists the notebooks
  still to port and the reason for every reformulation.
- **Sources**: [docs/SOURCES.md](docs/SOURCES.md) lists the references for each formulation.

### Cases

HiGHS:

- `cell_tower`: budgeted maximum coverage on a hypergraph.
- `tic_tac_toe`: 3D tic-tac-toe, fewest monochromatic lines.
- `playlist`: hypergraph ranking, then a playlist MILP.
- `workforce`: shift assignment with two ranked objectives.
- `resource_assignment`: resource assignment with a budget.
- `factory_planning_1`, `_2` (Williams): factory planning, maintenance.
- `food_manufacture_1`, `_2` (Williams): blending, recipe logic.
- `farm_planning` (Williams): five-year dairy farm.
- `economic_planning` (Williams): dynamic Leontief economy.
- `manpower_planning` (Williams): hiring, training, redundancy.
- `mining` (Williams): mine scheduling.
- `refinery` (Williams): refinery optimisation.
- `opencast_mining` (Williams): maximum-weight closure.
- `curve_fitting`: L1 and L∞ line fits.
- `logical_design` (Williams): fewest NOR gates for XOR.
- `market_sharing` (Williams): goal programming.
- `constraint_optimization` (Williams): simplest equivalent knapsack row.
- `efficiency_analysis`: data envelopment analysis, 28 units (28 LPs).
- `protein_folding` (Williams): hydrophobic contacts.
- `electrical_power_1`, `_2` (Williams): unit commitment, hydro.
- `facility_location`: uncapacitated facility location.
- `covid19_facility_1`, `_2`: temporary facility location, two scenarios.
- `supply_network_1`, `_2` (Williams): distribution, depot location.
- `offshore_wind`: fixed-charge cable network.
- `tsp`: travelling salesman, 48 US capitals.
- `cutting_stock`: column generation, then an integer master.
- `portfolio` (HiGHS QP): Markowitz minimum risk and frontier, 26 QPs.

SCIP:

- `pooling`: pooling, P and Q bilinear forms.
- `agricultural_pricing` (Williams): pricing with bilinear revenue.
- `avocado_pricing`: price and allocation with bilinear revenue.
- `decentralization` (Williams): binary quadratic placement.
- `protein_comparison`: contact map overlap, binary quadratic.
- `best_subset`: L0 regression for k = 1..13, 13 MIQPs.

"Williams" marks cases from H. P. Williams, *Model Building in Mathematical Programming*.

## Layout

- `crates/milp`: `Model`, sparse rows with optional quadratic terms; `solve` (HiGHS when linear, SCIP
  otherwise), `solve_on`, `lex`; `viol`, the solution check.
- `crates/hyper`: `Hg`, a weighted hypergraph in DeepHypergraph's convention; Zhou's operator Θ and
  ranking; hMETIS export.
- `crates/models`: one module per formulation, and the `mmi` command. Each module states its formulation
  and source at the top.
- `data/`: one JSON file per case: source notebook, Gurobi oracle, data.
- `tools/import_gurobi.py`: rebuilds `data/` from a modeling-examples checkout.
- `tools/blocks.py`: partitions a case's constraint matrix with Mt-KaHyPar.

## Execution

Requires Rust 1.88 or later, CMake and a C++ compiler (for HiGHS). The first build downloads the prebuilt
SCIP 10.0.2 that the `scip-sys` crate provides (46 MB on Linux, 67 MB on Windows). Python 3 with
`mtkahypar` runs `blocks.py`.

```
git clone --recursive <repository>
cargo build --release
target/release/mmi run data/*.json         # solve, check, compare with Gurobi; exit 1 on any failure
target/release/mmi hgr data/tsp.json       # the constraint matrix as an hMETIS hypergraph
python -m pip install -r requirements.txt
python tools/blocks.py data/food_manufacture_2.json -k 6
cargo test --release                       # includes a check that HiGHS and SCIP agree on every linear case
```

To change a case, edit its data file. Set `MILP_LOG=1` to see the solver logs.

## Limits

- HiGHS has no lazy-constraint callback, so the TSP re-solves the whole MIP after each round of subtour rows.
- SCIP needs finite bounds on the variables in nonconvex terms. On a convex objective whose variables are free, it
  did not terminate (measured), so every case bounds them.
- SCIP satisfies a quadratic objective's epigraph row only to 1e-6 absolute, so `portfolio`, whose risks are about
  1e-5, uses HiGHS's QP solver.
- The cutting stock integer plan uses only the columns generated at the root (price-and-branch, as in the notebook).
- On Windows, the Mt-KaHyPar 1.7 wheel crashes as its process exits, so `blocks.py` partitions in a child process.

## Results

`mmi run data/*.json` prints each case's objective beside its Gurobi oracle. The oracle objectives are in
the `oracle` field of each file in `data/`.

## Licence

Apache-2.0. The data and problem statements come from Gurobi's modeling-examples (Apache-2.0); see `NOTICE`.
