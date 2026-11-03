# Sources

## Data and problem statements

Gurobi Optimization, *modeling-examples*, https://github.com/Gurobi/modeling-examples (Apache-2.0), local
copy dated 2025-09-15. Each data file names its notebook. Its `oracle` lists the objectives that the
notebook's saved Gurobi log reports, in the order they were solved.

## Formulations

| Case | Reference |
| --- | --- |
| cover | R. Church, C. ReVelle, "The maximal covering location problem", *Papers of the Regional Science Association* 32 (1974) 101–118; S. Khuller, A. Moss, J. Naor, "The budgeted maximum coverage problem", *Information Processing Letters* 70 (1999) 39–45 |
| factory, food, opencast, fit, network, farm, economic, manpower, mining, refinery, logic, market, constraint, dea, folding, comparison, power, decentral, agriprice | H. P. Williams, *Model Building in Mathematical Programming*, 5th ed., Wiley 2013: the problems of the same names |
| opencast | J.-C. Picard, "Maximal closure of a graph and applications to combinatorial problems", *Management Science* 22 (1976) 1268–1272 |
| fit | A. Charnes, W. W. Cooper, R. O. Ferguson, "Optimal estimation of executive compensation by linear programming", *Management Science* 1 (1955) 138–151 |
| facility | G. Cornuéjols, G. L. Nemhauser, L. A. Wolsey, "The uncapacitated facility location problem", in P. Mirchandani, R. Francis (eds.), *Discrete Location Theory*, Wiley 1990 |
| tsp | G. Dantzig, R. Fulkerson, S. Johnson, "Solution of a large-scale traveling-salesman problem", *J. Operations Research Society of America* 2 (1954) 393–410; test oracle: M. Held, R. M. Karp, "A dynamic programming approach to sequencing problems", *J. SIAM* 10 (1962) 196–210 |
| cutstock | P. C. Gilmore, R. E. Gomory, "A linear programming approach to the cutting-stock problem", *Operations Research* 9 (1961) 849–859 |
| playlist, hyper | D. Zhou, J. Huang, B. Schölkopf, "Learning with hypergraphs: clustering, classification, and embedding", *NIPS* 19 (2006) |
| workforce | Gurobi Optimizer Reference Manual, multiple objectives (`setObjectiveN`, `ObjNRelTol`, `ObjNAbsTol`) |
| pooling | C. A. Haverly, "Studies of the behavior of recursion for the pooling problem", *ACM SIGMAP Bulletin* 25 (1978) 19–28 (P-formulation); A. Ben-Tal, G. Eiger, V. Gershovitz, "Global minimization by reducing the duality gap", *Math. Programming* 63 (1994) 193–212 (Q-formulation) |
| portfolio | H. Markowitz, "Portfolio selection", *Journal of Finance* 7 (1952) 77–91 |
| subset | D. Bertsimas, A. King, R. Mazumder, "Best subset selection via a modern optimization lens", *Annals of Statistics* 44 (2016) 813–852 |
| dea | A. Charnes, W. W. Cooper, E. Rhodes, "Measuring the efficiency of decision making units", *European J. Operational Research* 2 (1978) 429–444 |
| wind | S. Fischetti, D. Pisinger, "Optimizing wind farm cable routing considering power losses", *European J. Operational Research* 270 (2018) 917–930 |
| avocado, covid, rap, colour | The Gurobi notebooks named in each data file; no further source |

## Software

| Component | Source | Use |
| --- | --- | --- |
| SCIP 10.0.2 | https://scipopt.org (Apache-2.0), through russcip 0.11 and scip-sys 0.1.28 (`bundled`: scipopt/scipoptsuite-deploy v0.12.0) | Every model with a quadratic term; the cross-check of linear cases |
| HiGHS 1.12.0 | https://github.com/ERGO-Code/HiGHS (MIT); Q. Huangfu, J. A. J. Hall, "Parallelizing the dual revised simplex method", *Math. Programming Computation* 10 (2018) 119–142 | LP, MILP and convex QP solves |
| DeepHypergraph | https://github.com/iMoonLab/DeepHypergraph (`dhg/structure/hypergraphs/hypergraph.py`, read 2026-11-03) | Conventions of `Hg`: `Hypergraph(num_v, e_list, e_weight)`, `nbr_e`, `L_HGNN` = Θ, `L_sym` = I − Θ, `from_bigraph` |
| Mt-KaHyPar 1.7 | https://github.com/kahypar/mt-kahypar, PyPI `mtkahypar` | `tools/blocks.py`, cut-net partitioning |
| Row-net model | C. Aykanat, A. Pinar, Ü. V. Çatalyürek, "Permuting sparse rectangular matrices into block-diagonal form", *SIAM J. Sci. Comput.* 25 (2004) 1860–1879 | `mmi hgr` |
| Block detection for Dantzig–Wolfe | M. Bergner, A. Caprara, A. Ceselli, F. Furini, M. Lübbecke, E. Malaguti, E. Traversi, "Automatic Dantzig–Wolfe reformulation of mixed integer programs", *Math. Programming* 149 (2015) 391–424 | Why `blocks.py` counts linking rows |
