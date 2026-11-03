# Decisions

## Coverage

The source has 60 notebooks in 50 folders. Thirty-eight cases from 37 notebooks are ported. Thirty-five meet the
objective their notebook recorded or an independent oracle. The TSP, cutting stock and playlist cases have no
recorded objective; their methods are unit-tested instead. Still to port:

| Notebooks | What they need |
| --- | --- |
| car_rental 1-2, yield_management, marketing_campaign_optimization, food_program, power_generation, battery_scheduling, burrito_optimization_game, aviation_planning, railway_dispatching | LP/MILP; mostly larger data in CSV files, which the importer can read |
| technician_routing_scheduling, lost_luggage_distribution, milk_collection, drone_network | Routing and queueing MILPs; lost luggage and milk collection use lazy-constraint callbacks, which become row generation as in `tsp` |
| fantasy_basketball 2 | A MILP on the precomputed predictions in `results_target_advanced.csv` |
| customer_assignment, fantasy_basketball 1, index_tracking, pricing_competing_products, price_optimization_gurobiML (2) | A scikit-learn or XGBoost model embedded with `gurobi_ml`; porting needs the fitted model's equations written as rows |
| intro_to_gurobipy, milp_tutorial (introduction, RAP 1-2), optimization101/201/202/301 | Tutorials on the Gurobi API; their models are covered by the cases above |
| text_dissimilarity | The Google News word2vec vectors, a multi-gigabyte download |

## Solvers

- **Two backends, one model type.** `milp::Model` holds linear rows plus optional quadratic terms. `solve` sends a
  linear model to HiGHS and anything with a quadratic term to SCIP. `solve_on` picks one explicitly, and a test runs
  every linear case on both solvers and requires equal optima, so each solver checks the other.
- **SCIP through russcip's bundled build.** The `scip-sys` crate ships pregenerated bindings and downloads the
  prebuilt SCIP 10.0.2, so the build needs neither libclang nor a local SCIP install, and it works the same on Linux
  CI and on Windows. On Windows, `milp`'s build script copies the SCIP DLLs next to the binaries, because Windows has
  no rpath.
- **Only quadratic terms.** A survey of the notebooks (2026-11-03) found that every nonlinear term is quadratic (a
  square or a bilinear product), so the model has no general expression type.
- **HiGHS solves MIPs to a 1e-9 relative gap**, not its default 1e-4, so that "optimal" means optimal. SCIP's
  default gap is already 0. The whole suite still runs in 70 s.
- **HiGHS QP objectives are scaled by a power of two.** On the portfolio covariance, with Hessian entries from 7e-8
  to 1e-3, HiGHS 1.12's QP solver cycled without progress for 3 million iterations. Scaled so that its largest entry
  lies in [1, 2), it converged, and the unscaled values match Gurobi's to 9 digits. A power of two scales exactly.
- **Convex QPs that need precision go to HiGHS.** SCIP meets a quadratic objective's epigraph row only to 1e-6
  absolute, so the portfolio case, with risks around 1e-5, calls `solve_on(Backend::Highs)`.

## Oracles

- **An oracle is an interval.** Gurobi's "Best objective a, best bound b" line gives [a, b]; an exact "Optimal
  objective" gives [a, a]. A case passes when it is optimal, its solution replays within 1e-6 and each objective
  lies within max(1e-6 |a|, 1e-9) of its interval. Agricultural pricing needs the interval: SCIP finds 2.0664076,
  better than Gurobi's incumbent of 2.0663983, which stopped at its gap.
- **A wider tolerance is stated in the case file.** Portfolio sets `rtol` 5e-5 because two of Gurobi's barrier
  values are 1.2e-9 higher than HiGHS's exactly feasible solutions (violation 2e-16).
- **When the notebook records nothing usable, the oracle is built independently.** Best subset: exhaustive
  enumeration of all 8191 subsets. DEA: the printed 3-decimal efficiencies, taken as rounding intervals. Pooling: the
  proven Q-formulation optimum for both forms, because Gurobi's P run was time-limited. The TSP and cutting stock
  have their own unit tests (Held–Karp dynamic programming; the LP over all maximal patterns).
- **Oracles are never edited by hand.** `tools/import_gurobi.py` writes them from the notebooks' saved logs.

## Formulation choices

- **Indicator and general constraints become exact linear rows.** These are the big-M rows of 3D tic-tac-toe
  (M = 1) and Food Manufacture II (M = the class capacity), and the max/min bounds of workforce. The big-M values
  are already bounds on the expressions, so the feasible sets are unchanged.
- **Ranked objectives follow Gurobi's rule.** `Model::lex` allows each objective to degrade by
  max(reltol·|z|, 1e-6), the MIP rule of `setObjectiveN`.
- **Faithful to the notebook, not to the textbook.** Economic planning passes `ub=capacity0`, a dict keyed by
  industry for columns keyed by (industry, year), and gurobipy applied no bound from it. Gurobi's log shows bounds
  range [0, 0], and its plan makes 961 units of coal in a year against a capacity of 300. The port applies no bound
  either; with the bound, the optimum falls to 1506. Farm planning keeps gurobipy's default profit ≥ 0.
- **Bounds that change no optimum are added where SCIP needs them.** Pool qualities lie in [min λ, max λ]. In best
  subset, |β| ≤ M = 10 max |β̂_OLS|, and each solve reports whether that bound was active; it never was. The intercept
  lies in [min y, max y].
- **Quadratic programs stay quadratic.** Williams linearises decentralization and protein comparison, and
  substitutes demand out of agricultural pricing; the notebooks keep the products, and so does SCIP.
- **Row generation in place of lazy constraints.** HiGHS has no lazy-row callback, so the TSP re-solves after
  adding a subtour row for every cycle of the incumbent.
- **Playlist scores come from hypergraph ranking.** LightFM is randomly initialised and the notebook records no
  seed. Zhou's ranking is deterministic, on the same play data; α = 0.9 is this repository's choice.
- **Unreproducible instances are dropped.** The resource-assignment notebook's large instance draws scores over a
  Python set of random names, whose order depends on the interpreter's hash seed; only its recorded small instance
  is kept.

## Engineering choices

- **HiGHS 1.12.0 is built from source (git submodule) with hand-written C bindings**, following the author's
  Sparse-Portfolio-Optimization (`pf-highs`), including the optimisation flags without which HiGHS builds unoptimised.
- **Rows are stored by row (CSR) and passed to HiGHS that way**; `mmi hgr` reads the same rows.
- **Instance data are extracted, not retyped.** `tools/import_gurobi.py` runs each notebook's parameter cells
  against a stub `gurobipy`. Data built inside a solve function are captured from that function's local variables
  when its stub `gp.Model` is called; they are deep-copied, because later cells change them in place. Data that are
  derived (the portfolio covariance, the avocado demand regression, the playlist play counts) are computed exactly
  as the notebook computes them, and each case file's note says so. Rerunning the importer reproduces `data/` byte
  for byte.
- **Cases with several solves replay every one** (`Out::all`), not only the last.
