# Decisions

## Which notebooks were kept

The source has 60 notebooks in 50 folders. A notebook was kept if it met three conditions: HiGHS can
solve its model exactly (it is linear, or can be made linear without changing it); it adds a
formulation or method that the other kept cases do not already cover; and its data are in the
repository or small enough to store. Problems already covered by other repositories of this author
(portfolio and index tracking, unit commitment, vehicle routing) were not repeated.

| Kept | Reason |
| --- | --- |
| cell_tower_coverage, 3d_tic_tac_toe, music_recommendation | Hypergraph structure: coverage sets, lines of the cube, users' play sets |
| workforce | Ranked (hierarchical) objectives |
| factory_planning 1-2, food_manufacturing 1-2 | Williams's multi-period planning models; they share the stock balance rows |
| opencast_mining | Maximum-weight closure; a totally unimodular LP |
| curve_fitting | L1 and L∞ regression as LPs |
| facility_location, supply_network_design 1-2 | Location and network design |
| traveling_salesman | Row generation |
| colgen-cutting_stock | Column generation using LP duals |

| Left out | Reason |
| --- | --- |
| agricultural_pricing, portfolio_selection_optimization, pooling, covid19_facility_location, drone_network, pricing_competing_products, price_optimization (3 notebooks) | Quadratic or bilinear terms. Pooling is nonconvex, and the others mix quadratic terms with integers or use Gurobi's nonconvex solver |
| customer_assignment, fantasy_basketball, index_tracking, linear_regression (L0 regression) | Built around a scikit-learn model or `gurobi_ml` predictor constraints. L0 regression is a mixed-integer QP |
| intro_to_gurobipy, milp_tutorial, optimization101/201/202/301 | Tutorials on the Gurobi API, whose models the kept cases cover |
| electrical_power_generation 1-2, power_generation, battery_scheduling | Unit commitment and storage dispatch, which the author's Electricity-Market-Model covers |
| milk_collection, technician_routing_scheduling, lost_luggage_distribution, burrito_optimization_game, aviation_planning, railway_dispatching, yield_management, car_rental 1-2, marketing_campaign_optimization | Routing and scheduling variants. They need Gurobi callbacks or multiple objectives on top of what the TSP, network and workforce cases show. Several also read larger CSV files |
| text_dissimilarity | Needs the Google News word2vec vectors, a multi-gigabyte download |
| economic_planning, farm_planning, manpower_planning, mining, refinery, logical_design, market_sharing, decentralization_planning, efficiency_analysis, constraint_optimization, protein_comparison, protein_folding, offshore_wind_farming | More small Williams-style or textbook models. Each can be added as a new module without changing the shared crates |

## Formulation choices

- **Indicator constraints become linear rows.** Gurobi's indicators in 3D tic-tac-toe (bound M = 1) and in
  Food Manufacture II (M = the oil's refining capacity) are replaced by big-M rows. In both cases M is
  already an upper bound on the expression, so the feasible set does not change. Both cases still reach
  Gurobi's optimum.
- **Max and min general constraints become bounds.** In the workforce model, rows M ≥ n_w ≥ μ replace
  `addGenConstrMax` and `addGenConstrMin`. The second objective pushes M down and μ up, so they are tight
  at its optimum.
- **Ranked objectives use Gurobi's rule.** `Model::lex` solves the objectives in order. Each objective
  may get worse by at most max(reltol·|z|, 10⁻⁶), the MIP rule of `setObjectiveN` with `ObjNAbsTol` at
  its default.
- **TSP subtour rows are added between solves.** Because HiGHS has no lazy-row callback, each round adds a
  subtour row for every cycle in the current solution, not only the shortest cycle as the notebook does.
  This reaches the optimum in fewer rounds.
- **Playlist scores come from hypergraph ranking.** LightFM is a randomly initialised model trained with
  a stochastic gradient method, and the notebook records no seed, so its scores cannot be reproduced.
  Zhou's hypergraph ranking is deterministic and depends on the same data, the users' play sets. The
  selection model, its data and its parameters (1000 candidates, N = 30, average listeners ≤ 500 000,
  ≥ 20 countries) are the notebook's. α = 0.9 was chosen by this repository; the source does not set it.
- **Factory Planning I is Factory Planning II with the downtime fixed.** One module covers both, so the
  difference between the two models is one set of variable bounds.

## Engineering choices

- **HiGHS 1.12.0 is built from source as a git submodule, with hand-written C bindings.** The `highs-sys`
  crate needs bindgen and libclang, which the author's machine does not have, and its releases lag
  HiGHS. The build script and bindings follow those of the author's Sparse-Portfolio-Optimization
  repository (`pf-highs`), including the optimisation flags. Without them HiGHS builds unoptimised.
- **Rows are stored row-wise (CSR) and passed to HiGHS in that form.** Models are built one row at a
  time, and the row-net hypergraph of `mmi hgr` reads the rows directly.
- **There is no solver interface layer.** Only HiGHS is used, so a trait with a single implementation
  would add code without adding a choice.
- **Instance data are extracted, not retyped.** `tools/import_gurobi.py` runs each notebook's parameter
  cells against a stub `gurobipy`. The cutting stock orders are the one exception: they are defined in
  the cell after the class definitions, so they are copied by hand. The 42 MB playlist file is reduced to
  (user, artist, track count) triples.
